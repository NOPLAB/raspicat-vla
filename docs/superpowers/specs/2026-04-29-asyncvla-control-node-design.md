# AsyncVLA 制御ノード — 設計ドキュメント

- 日付: 2026-04-29
- 対象リポジトリ: `raspicat-async-vla`
- 対象モデル: [NHirose/AsyncVLA_release](https://huggingface.co/NHirose/AsyncVLA_release)
  （論文: [AsyncVLA: An Asynchronous VLA for Fast and Robust Navigation on the Edge](https://arxiv.org/abs/2602.13476)、コード: <https://github.com/NHirose/AsyncVLA>、プロジェクトページ: <https://asyncvla.github.io/>）
- 対象ロボット: rt-net Raspberry Pi Cat（raspicat、ROS2 Humble）
- スコープ: Gazebo シミュレーション先行、後続で実機

## 1. 目的と背景

raspicat（Raspberry Pi Cat）に AsyncVLA を載せ、視覚ナビゲーションを行う制御ノード一式を作る。AsyncVLA は遅延 0.2〜5 秒の通信下でも従来手法比で 40% 高い成功率を示す、エッジ向けの分割推論を前提とした VLA である。本プロジェクトでは、この分割推論アーキテクチャを ROS2 Humble 上に実装する。

ベースモデルである OmniVLA は SigLIP + DINOv2 + LLaMA2-7B（合計 7.5B パラメータ）であり、Raspberry Pi 単体では推論不能である。したがって以下の構成を採る:

- **Remote**: GPU ワークステーションで OmniVLA 本体を推論。入力ごとに圧縮アクション埋め込み（`8 × 1024 dim`）を返す。
- **Edge**: Raspberry Pi 上で軽量 Edge Adapter を高頻度で動かし、最新画像と最新埋め込みから軌道を生成。

エッジとリモートは gRPC bidi stream で疎結合に接続し、Edge 側は通信が遅延・断絶しても safe-stop 閾値までは動作を継続する。

## 2. 要件

### 2.1 機能要件

- **F1**: ROS2 Humble 上で動作する制御ノード `asyncvla_edge_node` を提供する。
- **F2**: GPU ワークステーション上で動作する gRPC サーバ `asyncvla_remote_server` を提供する。
- **F3**: 3 種のゴールモダリティに対応: 2D goal pose、言語指示、ゴール画像。
- **F4**: ノードは `/asyncvla/goal`（topic）と `SetGoal.srv` および `NavigateAsync.action` の 3 系統でゴール受付可能。
- **F5**: 出力は `nav_msgs/Path`（軌道）。別ノード `path_follower_node` で `geometry_msgs/Twist`（`/cmd_vel`）に変換。
- **F6**: Gazebo（raspicat_sim）と実機（raspicat_ros）の双方で起動できる launch を提供。
- **F7**: 通信遅延 6 秒程度までは安全に動作継続、それ以上は safe-stop。

### 2.2 非機能要件

- **N1**: Edge 側のアクション生成ループは 10 Hz を維持する（CPU 推論を許容）。
- **N2**: 通信帯域は片道 100 KB/s 以下を目標。
- **N3**: 各パッケージ単独でビルド・テスト可能（colcon の依存最小化）。
- **N4**: 公式 AsyncVLA コードを git submodule として `external/AsyncVLA` に取り込み、再現性を担保。
- **N5**: 安全停止閾値・通信エンドポイント等は ROS2 params/YAML で外部化する。

### 2.3 スコープ外

- 障害物回避（nav2_collision_monitor 等の別系統に委ねる、初期実装では扱わない）。
- マルチロボット運用。
- ROS1 互換性（公式 edge adapter は ROS1 だが、本プロジェクトは ROS2 ネイティブで再実装）。
- Prometheus メトリクスエクスポート（任意機能、初期実装では入れない）。

## 3. アーキテクチャ全体像

```
┌─────────────────────────────────────────────────────────┐
│ Remote Workstation (GPU)                                │
│  asyncvla_remote_server (Python, gRPC)                  │
│   - HF: NHirose/AsyncVLA_release をロード               │
│   - OmniVLA 本体 (SigLIP+DINOv2+LLaMA2-7B)              │
│   - 入力: image + goal(pose|text|image) → 推論          │
│   - 出力: 圧縮 action embedding (1024-dim × N)          │
└──────────────────────────┬──────────────────────────────┘
                           │ gRPC (bidi stream) over TCP/TLS
┌──────────────────────────┴──────────────────────────────┐
│ Edge (Raspberry Pi / dev PC for sim)                    │
│ ROS2 Humble                                             │
│  ┌──────────────┐   ┌──────────────────────────────┐    │
│  │ camera_node  │──▶│ asyncvla_edge_node           │    │
│  └──────────────┘   │  - 画像 downsample/JPEG      │    │
│  ┌──────────────┐   │  - リモートへ非同期送信      │    │
│  │ goal_input   │──▶│  - 最新 embedding キャッシュ │    │
│  └──────────────┘   │  - Edge Adapter 推論 (高頻度)│    │
│                     │  - Path 出力                 │    │
│                     └──────────┬───────────────────┘    │
│                                │ nav_msgs/Path          │
│                                ▼                        │
│                    ┌──────────────────────┐             │
│                    │ path_follower_node   │             │
│                    │ (Pure Pursuit)       │             │
│                    └──────────┬───────────┘             │
│                               │ geometry_msgs/Twist     │
│                               ▼                         │
│                    ┌──────────────────────┐             │
│                    │ raspicat /cmd_vel    │             │
│                    └──────────────────────┘             │
└─────────────────────────────────────────────────────────┘
```

### Async 性のキモ

- リモート推論は重い（数 100 ms 〜数秒）。Edge は送信→受信を non-blocking で回す。
- Edge はキャッシュした最新 embedding で常時 10 Hz の軌道生成を継続。
- 通信遅延が大きくても軌道生成自体は止まらない。

## 4. プロジェクト構造

```
raspicat-async-vla/
├── docker/
│   ├── Dockerfile.sim         # 既存
│   ├── Dockerfile.real        # 既存
│   └── Dockerfile.remote      # 新規 (CUDA + torch + transformers)
├── proto/
│   └── asyncvla.proto
├── external/
│   └── AsyncVLA/              # git submodule: NHirose/AsyncVLA
├── src/
│   ├── raspicat_async_vla_msgs/
│   │   ├── msg/
│   │   │   ├── ActionEmbedding.msg
│   │   │   └── GoalSpec.msg
│   │   ├── srv/
│   │   │   └── SetGoal.srv
│   │   └── action/
│   │       └── NavigateAsync.action
│   ├── raspicat_async_vla_edge/
│   │   ├── nodes/
│   │   │   ├── asyncvla_edge_node.py
│   │   │   ├── path_follower_node.py
│   │   │   └── goal_input_node.py
│   │   ├── asyncvla_edge/
│   │   │   ├── grpc_client.py
│   │   │   ├── edge_adapter.py
│   │   │   ├── preprocess.py
│   │   │   └── pure_pursuit.py
│   │   ├── launch/
│   │   │   ├── edge_sim.launch.py
│   │   │   └── edge_real.launch.py
│   │   └── config/edge_params.yaml
│   ├── raspicat_async_vla_remote/
│   │   ├── asyncvla_remote/
│   │   │   ├── server.py
│   │   │   ├── model_loader.py
│   │   │   ├── inference.py
│   │   │   └── compress.py
│   │   ├── scripts/run_server.py
│   │   └── config/remote_params.yaml
│   └── raspicat_async_vla_bringup/
│       ├── launch/
│       │   ├── sim_full.launch.py
│       │   └── real_full.launch.py
│       └── config/topic_remap.yaml
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── tools/
│   └── benchmark.py           # RTT/throughput/遅延注入ベンチ
├── docs/superpowers/specs/
├── .gitignore
├── .gitmodules
└── README.md
```

### 主要な選択

- **Edge ノードは Python (rclpy)**: 公式 AsyncVLA コードが PyTorch ベースで、Edge Adapter を Python で動かすのが素直。Pure Pursuit はパフォーマンス上問題なし。
- **Edge Adapter 実装方針**: `external/AsyncVLA` 内の Edge Adapter PyTorch モジュールを Python レベルで再利用（薄いラッパとして `edge_adapter.py` を用意）。ROS1 依存ノード部分は採用せず、本リポジトリで ROS2 ノードを新規実装。重み（state_dict）は `NHirose/AsyncVLA_release` を直接ロード。
- **`external/AsyncVLA` を git submodule**: 公式コードの参照と更新を明示管理。`raspicat_async_vla_remote` および `raspicat_async_vla_edge` が必要なモジュールを `PYTHONPATH` 経由で参照。
- **`proto/` をリポジトリ直下**: Python (grpcio-tools) / C++ どちらからも `protoc` で生成可能にする。`optional` フィールドを使うため `protoc >= 3.15` を前提（grpcio-tools 1.50+ で十分）。
- **`raspicat_async_vla_msgs` と `proto/` は役割を分離**: ROS2 系統と gRPC 系統で別管理。

## 5. インターフェース定義

### 5.1 gRPC proto（Edge ↔ Remote）

`proto/asyncvla.proto`:

```proto
syntax = "proto3";
package asyncvla.v1;

service AsyncVLAService {
  rpc StreamInfer(stream Observation) returns (stream ActionEmbedding);
  rpc GetModelInfo(ModelInfoRequest) returns (ModelInfo);
}

message Observation {
  uint64 frame_id        = 1;
  uint64 capture_time_ns = 2;
  bytes  image_jpeg      = 3;
  uint32 image_width     = 4;        // JPEG 内の解像度（リサイズ後）
  uint32 image_height    = 5;        // JPEG 内の解像度（リサイズ後）
  GoalSpec goal          = 6;
  optional Pose2D current_pose = 7;
}

message GoalSpec {
  enum Mode { POSE = 0; TEXT = 1; IMAGE = 2; }
  Mode mode = 1;
  oneof goal {
    Pose2D pose       = 2;
    string text       = 3;
    bytes  image_jpeg = 4;
  }
  // POSE 時: pose の参照フレーム（"odom" / "map" / "base_link" 等）。
  // TEXT / IMAGE 時: 空文字列でよい（無視される）。
  string frame_id = 5;
}

message Pose2D {
  double x     = 1;
  double y     = 2;
  double theta = 3;
}

message ActionEmbedding {
  uint64 frame_id        = 1;
  uint64 server_time_ns  = 2;
  uint32 num_tokens      = 3;
  uint32 embed_dim       = 4;
  bytes  embedding_fp16  = 5;   // num_tokens * embed_dim * 2 bytes (fp16 LE)
  float  inference_ms    = 6;
  optional string model_version = 7;
}

message ModelInfoRequest {}
message ModelInfo {
  string model_name      = 1;
  string model_version   = 2;
  uint32 num_tokens      = 3;
  uint32 embed_dim       = 4;
  string device          = 5;
  bool   ready           = 6;
}
```

設計上の要点:

- `frame_id` で Observation と ActionEmbedding を対応付け、古い結果を捨てる根拠にする。
- `image_jpeg` は Edge 側でエンコード。圧縮率と帯域のトレードオフは config で `jpeg_quality` を可変。
- `embedding_fp16` で帯域半減。論文の遅延 6 秒耐性なら fp16 で十分。
- bidi stream のため、Edge 側送信頻度（例: 2Hz）と Remote 返却頻度は独立に動く。

### 5.2 ROS2 メッセージ／サービス／アクション

`raspicat_async_vla_msgs/msg/GoalSpec.msg`:

```
uint8 MODE_POSE  = 0
uint8 MODE_TEXT  = 1
uint8 MODE_IMAGE = 2
uint8 mode

geometry_msgs/PoseStamped pose
string                    text
sensor_msgs/CompressedImage image
```

`raspicat_async_vla_msgs/msg/ActionEmbedding.msg`:

```
std_msgs/Header header
uint64 frame_id
uint32 num_tokens
uint32 embed_dim
float32[] embedding              # gRPC 受信時に fp16→fp32 へ展開した、長さ num_tokens*embed_dim の配列
float32 inference_ms
string model_version
```

備考: gRPC 経路（proto）では帯域効率のため `embedding_fp16` を bytes で運び、Edge ノード内で fp32 に展開して ROS2 トピックへ流す。ROS2 トピック側で fp16 を扱わないのは、`float16` が標準型に無く、デバッグ時の可読性が損なわれるため。

`raspicat_async_vla_msgs/srv/SetGoal.srv`:

```
GoalSpec goal
---
bool success
string message
```

`raspicat_async_vla_msgs/action/NavigateAsync.action`:

```
GoalSpec goal
float32 timeout_sec
float32 goal_tolerance_m
---
bool success
string message
float32 final_distance_m
---
float32 distance_remaining_m
geometry_msgs/PoseStamped current_pose
nav_msgs/Path predicted_path
uint32 remote_inferences_completed
float32 last_round_trip_ms
```

### 5.3 ROS2 トピック規約

| トピック | 型 | 方向 | 備考 |
|---|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | ← camera | sim/real 共通名 |
| `/asyncvla/goal` | `raspicat_async_vla_msgs/GoalSpec` | ← user | latched, depth=1 |
| `/asyncvla/predicted_path` | `nav_msgs/Path` | edge_node → follower | 軌道 |
| `/asyncvla/embedding` | `ActionEmbedding` | edge_node → /debug | 観測用、任意 |
| `/cmd_vel` | `geometry_msgs/Twist` | follower → robot | raspicat 既存規約 |
| `/asyncvla/status` | `diagnostic_msgs/DiagnosticArray` | edge_node | 通信遅延・FPS |

## 6. Edge ノード仕様

### 6.1 Lifecycle Node 構成

ROS2 `LifecycleNode` を採用。configure 時にモデルと通信を初期化、activate でループを起動。

```
[unconfigured] → configure → [inactive] → activate → [active]
                  ↑                                      │
                  └────── deactivate (safe-stop) ────────┘
```

### 6.2 内部の 3 ループ

`MultiThreadedExecutor` + `asyncio` で構成。役割を分離して相互ブロックを避ける。

```
① Observation 送信ループ (2 Hz)
   - 最新 image + 最新 goal → JPEG エンコード → gRPC 書き込み
   - frame_id = monotonic counter
② Embedding 受信ループ (イベント駆動)
   - gRPC から ActionEmbedding 読み取り → EmbeddingCache へ
   - 古い frame_id は捨てる
③ Action 生成ループ (10 Hz)
   - 最新 image + EmbeddingCache → edge adapter forward → 軌道
   - nav_msgs/Path を /asyncvla/predicted_path に publish
   - 到達判定 (goal_tolerance_m)
```

### 6.3 EmbeddingCache 挙動

```python
class EmbeddingCache:
    latest: Optional[ActionEmbedding]
    last_update_ns: int
    max_age_ns: int  # 既定 6e9 (= 6 秒)

    def put(self, emb): ...       # frame_id が新しい場合のみ更新
    def get_fresh(self) -> Optional[ActionEmbedding]:
        # max_age を超えていれば None
```

フォールバック方針:

| 条件 | 挙動 |
|---|---|
| `latest is None`（起動直後） | Path publish せず、status=`WAITING_REMOTE` |
| `age < max_age` | 通常運転、`age` を diagnostic に出す |
| `max_age <= age < hard_timeout` | publish 継続、warning。`degraded=true` |
| `age >= hard_timeout`（既定 15 秒） | safe-stop（0 速度）、status=`STALE`、action abort |

設計意図: 論文の「遅延 6 秒で 40% 高い成功率」を信じて `max_age=6s` を既定値にし、それを超えても即停止せず `hard_timeout` で最終的な安全停止ラインを設定する二段構え。

### 6.4 ゴール購読と切替

- `/asyncvla/goal` topic と `SetGoal.srv` の双方を受け付ける。
- 新しいゴールが来たら:
  - 進行中の `NavigateAsync` action があれば preempt
  - EmbeddingCache を invalidate
  - frame_id カウンタは継続（ストリーム自体は切らない）

### 6.5 内部状態遷移

```
        ┌──────────┐  goal受信   ┌─────────┐
        │   IDLE   │────────────▶│ ACTIVE  │
        └──────────┘             └────┬────┘
              ▲                       │
              │ goal達成 / cancel     │ embedding stale > hard_timeout
              │                       ▼
              │                  ┌─────────┐
              └──────────────────│ BLOCKED │
                                 └─────────┘
                                      │ embedding復活
                                      ▼
                                  ACTIVE へ戻る
```

`ERROR` は configure 失敗・gRPC 再接続不能時に遷移し、lifecycle を `inactive` に戻す。

### 6.6 パラメータ

| 名前 | 既定 | 説明 |
|---|---|---|
| `remote_address` | `localhost:50051` | gRPC エンドポイント |
| `obs_publish_rate_hz` | `2.0` | ① ループ周期 |
| `action_rate_hz` | `10.0` | ③ ループ周期 |
| `image_size` | `[224, 224]` | リサイズ先 |
| `jpeg_quality` | `85` | エンコード品質 |
| `embedding_max_age_sec` | `6.0` | フォールバック閾値（degrade） |
| `embedding_hard_timeout_sec` | `15.0` | safe-stop 閾値 |
| `goal_tolerance_m` | `0.3` | 到達判定 |
| `device` | `cpu` | edge adapter 推論デバイス |
| `tls_enabled` / `tls_ca_cert_path` | `false` / `""` | gRPC TLS |

## 7. Remote サーバ仕様

### 7.1 構成

純粋な Python gRPC サーバ（ROS 非依存）。`Dockerfile.remote` から起動する。

```
asyncvla_remote_server (process)
├── ModelLoader
│   ├── HF: NHirose/AsyncVLA_release を取得
│   ├── external/AsyncVLA/ の OmniVLA 実装で重みをロード
│   ├── device 配置（cuda:0 既定）+ bf16 / fp16
│   └── warmup: ダミー画像で 1 回 forward
├── InferenceEngine
│   ├── preprocess: JPEG → tensor (224x224, normalize)
│   ├── encode goal:
│   │     POSE  → 2D pose 埋め込み
│   │     TEXT  → tokenizer
│   │     IMAGE → 別ブランチで画像エンコード
│   ├── forward: OmniVLA 本体
│   └── compress: 8x4x4096 → 1024 投影（Token Projector）
└── GrpcServer
    ├── StreamInfer: 受信 Observation を InferenceQueue に投入、推論完了次第返信
    └── GetModelInfo
```

### 7.2 推論キューと並列化

```
┌──────────────┐  put   ┌──────────────────┐   pop   ┌────────┐
│ stream(s) RX │───────▶│ InferenceQueue   │────────▶│ worker │
│ (per client) │        │ (max_size=8,     │ batched │ (GPU)  │
└──────────────┘        │  drop_oldest)    │         └────────┘
                        └──────────────────┘
```

- **drop_oldest**: 推論より入力速度が速いとき、古い Observation を捨てる（リアルタイム性優先）。
- **batch=1 既定**、必要に応じて micro-batch 可。
- worker thread は 1 GPU あたり 1。CPU 側 pre/post は別スレッドプール。

### 7.3 パラメータ

```yaml
server:
  host: 0.0.0.0
  port: 50051
  tls:
    enabled: false
    cert_path: ""
    key_path: ""
  max_concurrent_streams: 4

model:
  hf_repo: "NHirose/AsyncVLA_release"
  hf_revision: null
  cache_dir: "/root/.cache/huggingface"
  device: "cuda:0"
  dtype: "bfloat16"

inference:
  queue_max_size: 8
  drop_policy: "oldest"
  warmup_iters: 3
  num_action_tokens: 8
  embed_dim: 1024
```

### 7.4 ヘルス

- `GetModelInfo` で `ready` フラグを返す（モデルロード中は false）。

## 8. データフローと非同期戦略

### 8.1 タイムライン例

```
時刻(ms) Edge①送信   Remote推論       Edge②受信    Edge③Path生成(10Hz)
   0     F0送信(画像A)
   100                                                ├─ 既存emb=null → publishなし
   500   F1送信(画像B)
   600                F0処理開始
  1400                F0完了(emb_v0)
  1450                                emb_v0受信       ├─ adapter(画像最新, emb_v0) → Path
  1500                                                ├─
  1700                F1完了(emb_v1)
  1750                                emb_v1受信(更新) ├─ adapter(画像最新, emb_v1) → Path
```

Path 出力は 10 Hz で連続的に走り、embedding は数百 ms ごとに更新される。Edge Adapter が最新画像を毎回入力に取るため、embedding が少々古くても近時点の画像で軌道は更新される。

### 8.2 順序保証と無効化

- **frame_id monotonic**: 受信側で `if recv.frame_id <= cache.frame_id: drop`
- **goal 切替時**: cache 全部 invalidate、新ゴール由来の embedding が届くまで Path publish 停止
- **gRPC reconnect**: stream 切断時は exponential backoff で再接続。再接続中は frame_id を継続

### 8.3 帯域試算

- 画像: 224x224 RGB → JPEG q=85 で約 12-20 KB
- Observation @ 2Hz: 約 40 KB/s 上り
- Embedding (8×1024×fp16 = 16 KB) @ 2Hz: 約 32 KB/s 下り
- 合計 < 100 KB/s（Wi-Fi/有線で十分）

## 9. エラーハンドリングと安全機構

### 9.1 安全機構の階層

| レイヤ | 検出 | 対応 |
|---|---|---|
| 1. ハード | e-stop ボタン | raspicat 既存機能で停止 |
| 2. ベース速度 | path_follower の上限値 | `max_v=0.4`, `max_w=1.0` に clip |
| 3. ノード状態 | embedding_hard_timeout | safe-stop publish (Twist=0) + lifecycle deactivate |
| 4. 通信 | gRPC エラー | 指数バックオフ再接続、失敗中は status=DEGRADED |
| 5. 障害物 | （初期スコープ外） | 将来 nav2_collision_monitor を検討 |

### 9.2 異常系一覧

| 異常 | 検出 | 復旧 |
|---|---|---|
| カメラ停止 | image トピック無受信 1 秒 | safe-stop, action abort |
| Remote 未起動 | gRPC connect refused | バックオフ再接続、ユーザに通知 |
| Remote モデル未ロード | `ready=false` | embedding を使わず待機 |
| GPU OOM (Remote) | 推論例外 | クライアントへ status=UNAVAILABLE、ログ出力 |
| 入力画像サイズ不一致 | preprocess時 | resize で吸収（warning） |
| ゴールフレーム未解決 | tf 引けない | エラー応答（SetGoal: success=false） |
| 軌道がロボット背後を指す | follower で検出 | 旋回優先、後退はしない |

### 9.3 ロギング

- ROS2 ノード: `rclpy` の logger（INFO/WARN/ERROR）
- リモート: 標準 logging + `--log-level` フラグ
- 主要 KPI: `obs_send_rate`, `emb_recv_rate`, `roundtrip_p50/p95`, `embedding_age_p95`

## 10. テスト戦略

### 10.1 階層

- **Unit (pytest, モデル/通信/ROS いずれも不要)**
  - preprocess: 任意 size 画像 → 正しい tensor
  - EmbeddingCache: frame_id 順序、max_age 判定
  - PurePursuit: 既知 Path → 期待する Twist 範囲
  - GoalSpec encoder: 各モダリティの proto 変換
- **Integration (Edge ↔ Remote の gRPC を実通信、モデルはダミー)**
  - DummyRemoteServer: 入力に依存せず固定 embedding を返す
  - Edge 全フローを実トピック上で確認
  - 遅延注入: server 側に sleep を仕込んで 6 秒遅延でも Path publish が止まらないことを確認
- **E2E (Gazebo + Edge + Remote, 実モデル, GPU 必須)**
  - `sim_full.launch.py` で raspicat を 5m 直進ナビ
  - 言語ゴール / 画像ゴール / pose ゴール 各 1 ケース

### 10.2 CI

- Unit: GitHub Actions 上で colcon test + pytest
- Integration: Dockerfile.sim ベースのコンテナで起動して接続テスト
- E2E: GPU が必要なため CI ではスキップ、開発機での手動実施

### 10.3 ベンチ

`tools/benchmark.py`:

- 固定画像を N 枚送って RTT, throughput を測定
- 遅延注入 0/0.5/2/5 秒での Path 連続性を確認

## 11. デプロイ

- **Sim 環境**: `Dockerfile.sim` を拡張し、`raspicat_async_vla_*` パッケージをマウント。Gazebo + Edge ノード一式を `sim_full.launch.py` で起動。Remote は別コンテナ（`Dockerfile.remote`）または別マシンで起動。
- **実機環境**: `Dockerfile.real` を拡張し、Edge パッケージのみインストール。Remote はワークステーションで別途起動。`real_full.launch.py` で `/camera/image_raw` 等を実機トピックに remap。
- **Remote**: `Dockerfile.remote` で CUDA + torch + transformers + AsyncVLA submodule を含めビルド。GPU を持つマシンで `docker run --gpus all` で起動。

## 12. オープンクエスチョン（将来検討）

- 障害物回避の組み込み（nav2_collision_monitor との連携）。
- 言語ゴールにおける tokenizer の選択（公式コードに準拠予定）。
- 画像ゴールの座標系定義（base_link 視点 / map 上の任意視点）。
- マルチクライアント時の GPU スケジューリング（現状は drop_oldest で逐次）。

## 13. 参考資料

- 論文: <https://arxiv.org/abs/2602.13476>
- プロジェクトページ: <https://asyncvla.github.io/>
- 公式コード: <https://github.com/NHirose/AsyncVLA>
- HF チェックポイント: <https://huggingface.co/NHirose/AsyncVLA_release>
- 関連基盤モデル: OmniVLA（SigLIP + DINOv2 + LLaMA2-7B）
- raspicat ROS2: <https://github.com/rt-net/raspicat_ros>
