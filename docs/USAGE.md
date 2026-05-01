# USAGE

ワークステーション、実機 Raspberry Pi Cat、または Gazebo シミュレーションで
`raspicat-vla` を実際に動かすための手順書。本ドキュメントは `README.md` の
続きという位置づけで、README がアーキテクチャと colcon ベースのビルドを扱う
のに対し、本ファイルは `docker/run.sh` を一次入口として具体的な運用シナリオを
追う。

サブコマンドの正確な一覧は `docker/run.sh --help` を参照。

## 1. 概要

システムは gRPC でつながる 2 ホストに分かれている:

```
         camera/goal                         action
           ----->                             <-----
   ┌──────────────────┐   gRPC StreamInfer   ┌──────────────────┐
   │  Edge (raspicat) │ ───────────────────▶ │  Remote (workstn) │
   │  ROS2 Humble     │ ◀─────────────────── │  VLA backbone     │
   └──────────────────┘                      └──────────────────┘
```

* **エッジ側**は ROS2 (`raspicat_vla_edge`) を実行し、カメラフレームを取得
  して JPEG エンコード、`Observation` メッセージとしてリモートへストリーム
  する。
* **リモート側**は gRPC サーバ (`raspicat_vla_remote`) を立てる。バックエンド
  は `dummy` (CI/MVP)・`asyncvla`・`omnivla` から選ぶ。
* すべて Docker イメージとして提供。`docker/run.sh` が必要なマウント・ネット
  ワーク・エントリコマンドを設定したうえで build/run する。

`README.md` の非 Docker な colcon フローも開発用途として完全にサポートして
いる。§3.4 を参照。

## 2. 前提条件

ホスト要件:

* **ワークステーション (リモート側)** — Docker。`--remote --gpu` を使う場合
  は NVIDIA Container Toolkit も必要。`asyncvla`/`omnivla` イメージは大きな
  モデルを取得するため (AsyncVLA は約 15 GB)、ディスクと帯域に余裕を見て
  おくこと。
* **ロボット (エッジ側)** — Pi (またはその他 ROS2 対応ホスト) 上の Docker。
  `real` イメージには rt-net の `raspicat_ros` パッケージが組み込まれている。
* **単一ホスト (loopback)** — 開発用。`localhost` 経由でリモートとエッジを
  同一マシンで動作可能。

Docker フローならホスト側に ROS2 をインストールする必要はない (イメージに
ROS2 Humble が同梱)。ホスト側 ROS2 が必要になるのは §3.4 (colcon) のみ。

ネットワーク: エッジホストから所定の gRPC ポート (デフォルト `50051`) で
リモートへ到達できる必要がある。`run.sh` の各起動は `--network host` を使う
ため、Linux ではポートフォワード設定は不要。

## 3. 初回セットアップ

クローン直後に一度だけ実行する作業。互いに独立しており順序は問わないが、
`run` サブコマンドはそれぞれ対応するイメージを必要とする。

### 3.1 Docker イメージの build

```bash
docker/run.sh build --all              # すべて
docker/run.sh build asyncvla           # リモート側 AsyncVLA
docker/run.sh build omnivla            # リモート側 OmniVLA
docker/run.sh build real               # エッジ側フル (raspicat_ros 同梱)
docker/run.sh build sim                # エッジ側 + Gazebo
docker/run.sh build test               # CPU のみのテスト用イメージ
```

最低限便利な構成は `test` (pytest と fallback エッジが動く) に加えて、
リモート用に `asyncvla` か `omnivla` のいずれか。`real` と `sim` は実機
スタックや Gazebo が本当に必要になってから build すれば良い。

### 3.2 モデルチェックポイントのダウンロード

リモートのバックエンドはどちらも `./models/` から重みをロードする。
ダウンロードスクリプトは `huggingface_hub.snapshot_download` を使い、
ホストの `~/.cache/huggingface` を経由するため再実行は安価。

```bash
scripts/download_asyncvla_checkpoints.sh   # → models/AsyncVLA_release/   (~15 GB)
scripts/download_omnivla_checkpoints.sh    # → models/omnivla-original/
```

HuggingFace 上のリポジトリは公開設定なのでトークンは不要。実際に使う
バックエンドの分だけ落とせば良い。`dummy` バックエンドはチェックポイント
不要。

### 3.3 (任意) gRPC スタブの再生成

`proto/raspicat_vla.proto` を編集したときだけ実行する:

```bash
scripts/gen_proto.sh
```

`src/raspicat_vla_proto/raspicat_vla_proto/raspicat_vla_pb2*.py` が再生成
される。proto 変更とあわせてコミットすること。

### 3.4 (任意) ネイティブ colcon ビルド

Docker を使わず開発したい場合は `README.md` の Build 節に従う:

```bash
source /opt/ros/humble/setup.bash
vcs import src < raspicat.repos
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

manifest 変更時は `vcs import src < raspicat.repos` を再実行する。Docker
イメージは内部で同等のビルドを実行するので、`run.sh` フローではこの作業は
必要ない。

## 4. モデルとモード

### 4.1 バックエンド

| Backend     | 用途                | デバイス        | 重み                                | イメージ               |
|-------------|---------------------|-----------------|-------------------------------------|------------------------|
| `dummy`     | CI / MVP / loopback | CPU のみ        | なし                                | `raspicat-vla-test`    |
| `asyncvla`  | AsyncVLA 推論       | CPU (低速)・GPU | `models/AsyncVLA_release`           | `raspicat-vla-asyncvla`|
| `omnivla`   | OmniVLA 推論        | CPU (低速)・GPU | `models/omnivla-original`           | `raspicat-vla-omnivla` |

resume step は `docker/run.sh` に固定で書かれている: AsyncVLA `750000`、
OmniVLA `120000`。変更する場合はスクリプト中の `RESUME_STEP` 連想配列を
書き換える。

### 4.2 実行モード

| モード    | イメージ              | コンテナ内で動くもの                                          |
|-----------|-----------------------|---------------------------------------------------------------|
| `--remote`| `asyncvla`/`omnivla`  | gRPC サーバ (`raspicat_vla_remote.server_main`)              |
| `--real`  | `real`                | rt-net 実機向け `edge_only.launch.py`                         |
| `--sim`   | `sim`                 | `mvp_sim.launch.py` (Gazebo + エッジ + path follower)         |
| `test`    | `test`                | pytest                                                        |

`--real` と `--sim` は対応する完全イメージが build されていない場合、
警告のうえ `test` イメージにフォールバックする。フォールバックではエッジ
ノードと path follower は動くが、rt-net 実機ブリングアップ・Gazebo・
(AsyncVLA エッジに必要な) torch は使えない。動作確認程度の用途。

## 5. 実行レシピ集

代表的な 5 シナリオ。コマンドはすべてリポジトリルートから実行可能で、
`--network host` を使う。IP やポートは適宜置き換えて使うこと。

### 5.1 シングルホスト loopback (dummy バックエンド)

gRPC の疎通確認に最速のフロー。2 ターミナル使用:

```bash
# T1 — リモート (dummy バックエンド、CPU、ポート 50051)
docker/run.sh run omnivla --remote --cpu          # run.sh の remote 系で起動
                                                  # (MODEL フラグに対応する
                                                  #  バックエンドが起動する。
                                                  #  純粋な dummy は下を参照)

# T2 — localhost に対してエッジを fallback で起動
docker/run.sh run omnivla --real --host localhost
```

純粋な `dummy` バックエンド (モデルロードなし) で立てる場合は、`run.sh` を
迂回して test イメージ内で server モジュールを直接呼ぶ:

```bash
docker run --rm --network host \
    -v "$PWD:/workspace" raspicat-vla-test bash -lc \
    'cd /workspace && pip install -e src/raspicat_vla_proto src/raspicat_vla_remote && \
     python3 -m raspicat_vla_remote.server_main --backend dummy --port 50051'
```

実カメラがない環境では `tools/publish_fake_image.py` で擬似的な画像 + ゴール
ストリームを送れる:

```bash
ros2 run raspicat_vla_edge ... # 別シェルで:
python3 tools/publish_fake_image.py
```

### 5.2 リモート ワークステーション + Pi 実機エッジ

ワークステーション (`10.0.0.5`) で GPU ポリシーを動かし、Pi でエッジを実行
する構成。

```bash
# ワークステーション
docker/run.sh run asyncvla --remote --gpu --host 10.0.0.5
# 10.0.0.5:50051 に bind。CUDA がなければ --gpu の代わりに --cpu

# Pi
docker/run.sh run asyncvla --real --host 10.0.0.5
# ポートはデフォルト 50051。非デフォルトなら :PORT を付ける
```

任意: ポート明示指定 (ファイアウォール、マルチテナント環境など):

```bash
# ワークステーション: 全 IF にバインドしつつポート 9000
docker/run.sh run asyncvla --remote --gpu --host :9000

# Pi
docker/run.sh run asyncvla --real --host 10.0.0.5:9000
```

### 5.3 Sim (Gazebo) + リモート ワークステーション

```bash
# ワークステーション (リモート)
docker/run.sh run omnivla --remote --gpu --host 10.0.0.5

# Sim ホスト (X11 が動くマシンならどこでも)
docker/run.sh run omnivla --sim --host 10.0.0.5
```

Sim 起動側は `image_topic:=/camera/color/image_raw` (raspicat の RealSense
トピック) に remap し、`gzclient` がホストで描画できるよう `DISPLAY` を
forward する。あわせてホスト UID 用の `/etc/passwd` エントリを合成して
Gazebo のユーザ情報欠落警告を抑える — 利用側で設定するものはない。

### 5.4 Localhost loopback (単一マシンで Sim、強力なホスト想定)

GPU と Gazebo を同居させたワークステーション向け:

```bash
# T1
docker/run.sh run omnivla --remote --gpu --host localhost
# T2
docker/run.sh run omnivla --sim    --host localhost
```

両コンテナとも host network namespace 上で動くため、コンテナ間でも
`localhost` で疎通する。

### 5.5 OmniVLA を CPU で動かす (調査用)

GPU 無しのワークステーションで実バックエンドの挙動を確認したいときに有用:

```bash
docker/run.sh run omnivla --remote --cpu --host 127.0.0.1
docker/run.sh run omnivla --real   --host 127.0.0.1
```

推論は遅くなるので、`embedding_max_age_sec` (デフォルト 6 秒) を超えて
エッジが zero-Twist を維持する可能性が高い。配線確認だけなら
`edge_params.yaml` のキャッシュ閾値を緩めれば良い。

## 6. 設定リファレンス

### 6.1 エッジ — `src/raspicat_vla_edge/config/edge_params.yaml`

| Key                          | デフォルト                      | 備考                                         |
|------------------------------|---------------------------------|----------------------------------------------|
| `remote_address`             | `localhost:50051`               | launch arg `remote_address:=…` で上書き      |
| `obs_publish_rate_hz`        | `2.0`                           | リモートへ送る fps                            |
| `action_rate_hz`             | `10.0`                          | follower への path 再発行レート               |
| `image_size`                 | `[224, 224]`                    | JPEG リサイズ後のサイズ                       |
| `jpeg_quality`               | `85`                            | 1〜100                                       |
| `embedding_max_age_sec`      | `6.0`                           | これを越えると status → `DEGRADED`           |
| `embedding_hard_timeout_sec` | `15.0`                          | これを越えると status → `STALE`、safe-stop   |
| `goal_tolerance_m`           | `0.3`                           | ゴール到達判定                                |
| `image_topic`                | `/camera/image_raw`             | Sim では `/camera/color/image_raw`            |
| `goal_topic`                 | `/raspicat_vla/goal`            |                                              |
| `path_topic`                 | `/raspicat_vla/predicted_path`  | `path_follower_node` が subscribe            |
| `status_topic`               | `/raspicat_vla/status`          | `DiagnosticArray`                            |
| `embedding_debug_topic`      | `/raspicat_vla/embedding`       | `publish_embedding_debug: true` のときのみ   |
| `adapter_kind`               | `stub`                          | `stub` / `asyncvla` / `omnivla`              |
| `asyncvla_weights_path`      | `/workspace/models/AsyncVLA_release` | AsyncVLA エッジアダプタのみ              |
| `asyncvla_resume_step`       | `750000`                        | AsyncVLA エッジアダプタのみ                  |
| `asyncvla_device`            | `cpu`                           | AsyncVLA エッジアダプタのみ                  |

`edge_only.launch.py` は上書き頻度の高いキー (`remote_address`、
`adapter_kind`、`image_topic`、`with_follower`、AsyncVLA 関連 3 つ) を
launch 引数として公開する。それ以外は YAML のみ。

### 6.2 リモート — `src/raspicat_vla_remote/config/remote_params.yaml`

```yaml
server:
  host: 0.0.0.0
  port: 50051
  max_concurrent_streams: 4

dummy:
  num_tokens: 8
  embed_dim: 1024
  inference_ms: 50.0
  model_version: "dummy-v1"
```

`server_main` は同じ項目を CLI フラグで受ける (`--host`、`--port`、
`--num-tokens`、`--embed-dim`、`--inference-ms`、`--model-version`、
`--backend`、`--vla-path`、`--resume-step`、`--device`、`--log-level`)。
YAML は CLI を経由しない consumer 用で、`docker/run.sh` はすべてフラグで
渡している。

### 6.3 path follower

`path_follower_node` は `with_follower:=true` のとき `edge_only.launch.py`
から起動される。Pure-Pursuit を 20 Hz、`lookahead=0.4`、`max_v=0.4`、
`max_w=1.0` で実行。launch 引数または launch ファイル直編集で上書き可能。
受信 path の `frame_id` が `base_link` でない場合は `cmd_vel` をゼロにする。

### 6.4 環境変数による上書き

| 変数                    | 効果                                                                   |
|-------------------------|------------------------------------------------------------------------|
| `GRPC_PORT`             | `--host` のデフォルトポート (省略時は `50051`)                          |
| `HF_CACHE_DIR`          | コンテナにマウントする HF キャッシュ (デフォルト `~/.cache/huggingface`) |
| `RASPICAT_VLA_REBUILD`  | セットすると `--real` / `--sim` コンテナで `colcon build` を強制実行   |
| `ASYNCVLA_E2E`          | AsyncVLA E2E pytest スモークを有効化 (未設定時は skip)                 |
| `OMNIVLA_E2E`           | OmniVLA E2E pytest スモークを有効化 (未設定時は skip)                  |

## 7. トピックとインタフェース

### 7.1 ROS2 トピック

エッジスタックが使うトピック一覧。すべて §6 の launch 引数で remap 可能。

| Topic                            | 方向                | 型                                  | 備考                                |
|----------------------------------|---------------------|-------------------------------------|-------------------------------------|
| `/camera/image_raw`              | edge ← camera       | `sensor_msgs/Image`                 | Sim は `…/color/image_raw` で発行    |
| `/raspicat_vla/goal`             | edge ← user         | `raspicat_vla_msgs/GoalSpec`        | `POSE`/`TEXT`/`IMAGE` のいずれか    |
| `/raspicat_vla/predicted_path`   | follower ← edge     | `nav_msgs/Path`                     | `base_link` フレーム                |
| `/raspicat_vla/status`           | obs ← edge          | `diagnostic_msgs/DiagnosticArray`   | `OK`/`DEGRADED`/`WAITING_REMOTE`/`STALE` |
| `/raspicat_vla/embedding`        | obs ← edge (debug)  | `raspicat_vla_msgs/ActionEmbedding` | `publish_embedding_debug` 時のみ    |
| `/cmd_vel`                       | robot ← follower    | `geometry_msgs/Twist`               | stale またはフレーム不一致時はゼロ |

### 7.2 ライフサイクル

`vla_edge_node` は `LifecycleNode`。`edge_only.launch.py` が
`unconfigured → inactive → active` まで自動遷移させる。手動で上下させる
場合:

```bash
ros2 lifecycle set /vla_edge_node deactivate
ros2 lifecycle set /vla_edge_node activate
```

### 7.3 gRPC サービス

`proto/raspicat_vla.proto` に `raspicat_vla.v1.VLAService` を定義:

```
rpc StreamInfer(stream Observation) returns (stream ActionEmbedding);
rpc GetModelInfo(ModelInfoRequest) returns (ModelInfo);
```

`Observation` は JPEG・`GoalSpec` (pose/text/image goal)・任意の現在 pose を
持つ。`ActionEmbedding` は FP16 でパックされた embedding を返し、エッジ
アダプタがこれを `nav_msgs/Path` に展開する。

## 8. テスト

`docker/run.sh test` は `raspicat-vla-test` イメージ内で pytest を実行する。
未 build なら自動でビルドされる。

```bash
docker/run.sh test                            # フルスイート
docker/run.sh test -k checkpoint              # pytest -k フィルタ
docker/run.sh test src/raspicat_vla_edge/test # パス指定で部分実行
```

`-k`、`-x`、`--lf` のようなフラグのみ呼び出しでは、デフォルトのテストパス
リストを自動的に prepend する。これにより pytest が cwd discovery に流れて
`external/` を歩き、依存欠落でクラッシュするのを防ぐ。

AsyncVLA / OmniVLA の E2E スモークは環境変数でゲートされており、GPU 無しでも
clean に skip する。デフォルトのスイートには含まれない:

```bash
ASYNCVLA_E2E=1 docker/run.sh test -k asyncvla_e2e
OMNIVLA_E2E=1 docker/run.sh test -k omnivla_e2e
```

## 9. トラブルシューティング

**`run.sh: image XYZ not built; falling back to raspicat-vla-test`**
`real` または `sim` のフルイメージが未 build。fallback ではエッジスタックは
動くが、rt-net パッケージ・Gazebo・torch は使えない。実機やシミュレーション
が本当に必要なら以下で正式イメージを build する:

```bash
docker/run.sh build real    # または: build sim
```

**`--remote requires --cpu or --gpu`**
remote サブコマンドは明示的なデバイス指定を要求する。デフォルトは無く、
`--gpus all` か CPU のみかを意識的に選ばせる仕様。

**`--<mode> requires --host HOST[:PORT]`**
`--real` と `--sim` はリモートの所在を必要とする。`--remote` は不要 —
ローカルにバインドし、`--host` 省略時は `0.0.0.0` がデフォルト。

**エッジが `WAITING_REMOTE` から進まない**
`ActionEmbedding` の応答がエッジに届いていない。順に確認: リモートが起動
していて期待ポートで listen しているか、ホスト間で疎通するか
(`nc -z HOST PORT`)、`/raspicat_vla/goal` にゴールが publish されているか
(エッジは「最新画像」と「ゴール」の両方が揃って初めて送信を開始する)。

**エッジが `OK` → `DEGRADED` → `STALE` → safe-stop を繰り返す**
リモートは応答しているが `embedding_max_age_sec` より遅い。GPU に移すか、
`edge_params.yaml` の閾値を緩めること。

**Gazebo が `Error getting username: no matching password record` を出す**
`run.sh` はコンテナ内に UID 用の `passwd` エントリを合成している。`run.sh`
を経由せず直接 `docker run` で `sim` イメージを起動する場合は同等の処理を
自分で行う必要がある。`docker/run.sh` の `run_sim()` を参照。

**コンテナ内 colcon ビルドが毎回走る**
`run.sh` はワークスペースに `install/setup.bash` が存在すれば colcon ステップ
を skip する。ソース変更後に強制再ビルドしたいときは
`RASPICAT_VLA_REBUILD=1`。逆に再ビルドが走るべきときに走らない場合は
ホスト側の `install/` (bind mount されている) を削除する。

**ライフサイクルノードが `unconfigured` から進まない**
`edge_only.launch.py` は `OnProcessStart` で一度だけ自動 configure する。
プロセスが再起動 (例: `Ctrl+C` 後に同シェルで再起動) しても launch system が
イベントを再発行しないと configure されない。手動で:
`ros2 lifecycle set /vla_edge_node configure`。

**HuggingFace のダウンロードが固まる、または認証エラー**
リポジトリは公開設定でトークン不要。`snapshot_download` が 401 を返す場合は
HF トークンをクリア (`huggingface-cli logout`) してリトライ。期限切れ
トークンが残っていると公開リポジトリでも 401 になる。

## 10. 参考

* `docker/run.sh --help` — サブコマンドの正準リファレンス
* `proto/raspicat_vla.proto` — gRPC インタフェース定義
* `src/raspicat_vla_edge/launch/edge_only.launch.py` — エッジの launch 引数
* `src/raspicat_vla_bringup/launch/mvp_sim.launch.py` — Sim の launch 構成
* `src/raspicat_vla_edge/config/edge_params.yaml` — エッジパラメータ全件
* `src/raspicat_vla_remote/raspicat_vla_remote/server_main.py` — リモート CLI
* `scripts/download_*_checkpoints.sh` — HF モデル取得ヘルパ
* `raspicat.repos` — rt-net ソースバージョンのピン (vcstool マニフェスト)
