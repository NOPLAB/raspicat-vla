# USAGE

How to actually run `raspicat-vla` on a workstation, the real Raspberry Pi
Cat, or the Gazebo simulation. This document picks up where `README.md`
leaves off — the README explains the architecture and the colcon-based
build; this file walks through concrete operational scenarios using
`docker/run.sh` as the primary entry point.

If you only want to skim the surface, `docker/run.sh --help` prints the
authoritative subcommand reference.

## 1. Overview

The system is split across two hosts that talk over gRPC:

```
         camera/goal                         action
           ----->                             <-----
   ┌──────────────────┐   gRPC StreamInfer   ┌──────────────────┐
   │  Edge (raspicat) │ ───────────────────▶ │  Remote (workstn) │
   │  ROS2 Humble     │ ◀─────────────────── │  VLA backbone     │
   └──────────────────┘                      └──────────────────┘
```

* The **edge** runs ROS2 (`raspicat_vla_edge`), grabs camera frames, JPEG-
  encodes them, and streams `Observation` messages to the remote.
* The **remote** hosts a gRPC server (`raspicat_vla_remote`) backed by one
  of three policies: `dummy` (CI/MVP), `asyncvla`, `omnivla`.
* Everything is shipped as Docker images. `docker/run.sh` builds and runs
  them with the right mounts, networking, and entry commands.

The non-Docker colcon flow described in `README.md` is fully supported as
a development convenience — see §3.4.

## 2. Prerequisites

Hosts:

* **Workstation (remote side)** — Docker, plus NVIDIA Container Toolkit
  if you intend to run `--remote --gpu`. The `asyncvla`/`omnivla` images
  pull large models (≈15 GB for AsyncVLA), so plan disk and bandwidth.
* **Robot (edge side)** — Docker on the Pi (or any ROS2-capable host).
  The `real` image embeds rt-net's `raspicat_ros` packages.
* **Single host (loopback)** — fine for development; you can run remote
  and edge on the same machine over `localhost`.

No host-side ROS2 install is required for the Docker flow; the images
ship ROS2 Humble. Host-side ROS2 is only needed for §3.4 (colcon).

Network: the edge host must be able to reach the remote at the chosen
gRPC port (default `50051`). All `run.sh` invocations use `--network host`,
so port forwarding is unnecessary on Linux.

## 3. Initial setup

Run these once after cloning. They are independent and can be done in
any order, except that `run` subcommands need the corresponding images.

### 3.1 Build Docker images

```bash
docker/run.sh build --all              # everything
docker/run.sh build asyncvla           # remote-side AsyncVLA
docker/run.sh build omnivla            # remote-side OmniVLA
docker/run.sh build real               # edge-side full image (raspicat_ros)
docker/run.sh build sim                # edge-side + Gazebo
docker/run.sh build test               # CPU-only test image
```

The minimum useful set is `test` (lets you run pytest and a fallback
edge) plus one of `asyncvla` / `omnivla` for the remote. Build `real`
or `sim` only when you need the rt-net robot stack or Gazebo.

### 3.2 Download model checkpoints

Both remote backends load weights from `./models/`. The download scripts
use `huggingface_hub.snapshot_download` and your host's
`~/.cache/huggingface`, so re-runs are cheap.

```bash
scripts/download_asyncvla_checkpoints.sh   # → models/AsyncVLA_release/   (~15 GB)
scripts/download_omnivla_checkpoints.sh    # → models/omnivla-original/
```

The HuggingFace repos are public; no token is required. You only need
the model whose backend you actually intend to start. The `dummy`
backend needs no checkpoints.

### 3.3 (Optional) Regenerate gRPC stubs

Only needed if you edit `proto/raspicat_vla.proto`:

```bash
scripts/gen_proto.sh
```

This rewrites `src/raspicat_vla_proto/raspicat_vla_proto/raspicat_vla_pb2*.py`.
Commit the regenerated stubs alongside the proto change.

### 3.4 (Optional) Native colcon build

If you'd rather develop without Docker, follow `README.md` §Build:

```bash
source /opt/ros/humble/setup.bash
vcs import src < raspicat.repos
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

Re-run `vcs import src < raspicat.repos` whenever the manifest changes.
The Docker images perform an equivalent build internally; you do not
need to do this for the `run.sh` flow.

## 4. Models and modes

### 4.1 Backends

| Backend     | Use case            | Device          | Weights                            | Image                  |
|-------------|---------------------|-----------------|------------------------------------|------------------------|
| `dummy`     | CI / MVP / loopback | CPU only        | none                               | `raspicat-vla-test`    |
| `asyncvla`  | AsyncVLA inference  | CPU (slow), GPU | `models/AsyncVLA_release`          | `raspicat-vla-asyncvla`|
| `omnivla`   | OmniVLA inference   | CPU (slow), GPU | `models/omnivla-original`          | `raspicat-vla-omnivla` |

Resume steps are wired in `docker/run.sh`: AsyncVLA `750000`, OmniVLA
`120000`. Override by editing the `RESUME_STEP` map in the script.

### 4.2 Run modes

| Mode      | Image                | What runs in the container                                  |
|-----------|----------------------|-------------------------------------------------------------|
| `--remote`| `asyncvla`/`omnivla` | gRPC server (`raspicat_vla_remote.server_main`)            |
| `--real`  | `real`               | `edge_only.launch.py` against rt-net hardware              |
| `--sim`   | `sim`                | `mvp_sim.launch.py` (Gazebo + edge + path follower)        |
| `test`    | `test`               | pytest                                                      |

`--real` and `--sim` will silently fall back to the `test` image if
their full image isn't built (with a warning). The fallback gives you
the edge node and the path follower, but no rt-net hardware bringup,
no Gazebo, and (for AsyncVLA edge) no torch — useful only for sanity
checks.

## 5. Run cookbook

Five typical scenarios. Every command works from the repo root and uses
`--network host`; replace IPs and ports as needed.

### 5.1 Single-host loopback (dummy backend)

The fastest way to confirm the gRPC plumbing works end-to-end. Two
terminals:

```bash
# T1 — remote (dummy backend, CPU, port 50051)
docker/run.sh run omnivla --remote --cpu          # any of the run.sh remote forms
                                                  # (uses backend matching MODEL flag,
                                                  #  not 'dummy'; for plain dummy, see below)

# T2 — edge fallback against localhost
docker/run.sh run omnivla --real --host localhost
```

For a true dummy backend (no model load), bypass `run.sh` and call the
server module directly inside the test image:

```bash
docker run --rm --network host \
    -v "$PWD:/workspace" raspicat-vla-test bash -lc \
    'cd /workspace && pip install -e src/raspicat_vla_proto src/raspicat_vla_remote && \
     python3 -m raspicat_vla_remote.server_main --backend dummy --port 50051'
```

`tools/publish_fake_image.py` provides a synthetic camera + goal stream
useful when no real camera is wired up:

```bash
ros2 run raspicat_vla_edge ... # then in a separate shell:
python3 tools/publish_fake_image.py
```

### 5.2 Remote workstation, real Pi as edge

Workstation (`10.0.0.5`) hosts the GPU policy; the Pi runs the edge.

```bash
# Workstation
docker/run.sh run asyncvla --remote --gpu --host 10.0.0.5
# bind 10.0.0.5:50051; replace --gpu with --cpu if no CUDA

# Pi
docker/run.sh run asyncvla --real --host 10.0.0.5
# defaults port to 50051; appends :PORT if you need a non-default
```

Optional: pin a specific port (firewall, multi-tenant workstation):

```bash
# Workstation: bind every interface but on port 9000
docker/run.sh run asyncvla --remote --gpu --host :9000

# Pi
docker/run.sh run asyncvla --real --host 10.0.0.5:9000
```

### 5.3 Sim (Gazebo) with a remote workstation

```bash
# Workstation (remote)
docker/run.sh run omnivla --remote --gpu --host 10.0.0.5

# Sim host (anywhere with X11)
docker/run.sh run omnivla --sim --host 10.0.0.5
```

The sim entry remaps `image_topic:=/camera/color/image_raw` (raspicat's
RealSense topic) and forwards `DISPLAY` so `gzclient` renders on the
host. It also synthesizes a `/etc/passwd` entry for your UID so
Gazebo stops complaining about missing user info; nothing for you to
configure.

### 5.4 Localhost loopback (single-machine sim, big iron)

For a workstation with a GPU that also runs Gazebo:

```bash
# T1
docker/run.sh run omnivla --remote --gpu --host localhost
# T2
docker/run.sh run omnivla --sim    --host localhost
```

Both containers are on the host network namespace, so `localhost` works
across them.

### 5.5 OmniVLA on CPU (for triage)

Useful when the workstation has no GPU but you want a real backend:

```bash
docker/run.sh run omnivla --remote --cpu --host 127.0.0.1
docker/run.sh run omnivla --real   --host 127.0.0.1
```

Inference will be slow; expect `embedding_max_age_sec` (default 6 s) to
trip and the edge to hold zero-Twist. Bump the cache thresholds in
`edge_params.yaml` if you only need to confirm wiring.

## 6. Configuration reference

### 6.1 Edge — `src/raspicat_vla_edge/config/edge_params.yaml`

| Key                          | Default                       | Notes                                        |
|------------------------------|-------------------------------|----------------------------------------------|
| `remote_address`             | `localhost:50051`             | Override via launch arg `remote_address:=…`  |
| `obs_publish_rate_hz`        | `2.0`                         | Frames/s sent to remote                      |
| `action_rate_hz`             | `10.0`                        | Path republish rate to follower              |
| `image_size`                 | `[224, 224]`                  | JPEG resize target                           |
| `jpeg_quality`               | `85`                          | 1–100                                        |
| `embedding_max_age_sec`      | `6.0`                         | After this, status → `DEGRADED`              |
| `embedding_hard_timeout_sec` | `15.0`                        | After this, status → `STALE`, safe-stop      |
| `goal_tolerance_m`           | `0.3`                         | Goal-reached threshold                       |
| `image_topic`                | `/camera/image_raw`           | Sim uses `/camera/color/image_raw`           |
| `goal_topic`                 | `/raspicat_vla/goal`          |                                              |
| `path_topic`                 | `/raspicat_vla/predicted_path`| Subscribed by `path_follower_node`           |
| `status_topic`               | `/raspicat_vla/status`        | `DiagnosticArray`                            |
| `embedding_debug_topic`      | `/raspicat_vla/embedding`     | Only when `publish_embedding_debug: true`    |
| `adapter_kind`               | `stub`                        | `stub`, `asyncvla`, `omnivla`                |
| `asyncvla_weights_path`      | `/workspace/models/AsyncVLA_release` | AsyncVLA edge adapter only            |
| `asyncvla_resume_step`       | `750000`                      | AsyncVLA edge adapter only                   |
| `asyncvla_device`            | `cpu`                         | AsyncVLA edge adapter only                   |

`edge_only.launch.py` exposes the most-overridden keys as launch
arguments (`remote_address`, `adapter_kind`, `image_topic`,
`with_follower`, plus the AsyncVLA trio). The rest are YAML-only.

### 6.2 Remote — `src/raspicat_vla_remote/config/remote_params.yaml`

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

`server_main` accepts these as CLI flags (`--host`, `--port`,
`--num-tokens`, `--embed-dim`, `--inference-ms`, `--model-version`,
`--backend`, `--vla-path`, `--resume-step`, `--device`,
`--log-level`). The YAML is for non-CLI consumers; `docker/run.sh`
passes everything via flags.

### 6.3 Path follower

`path_follower_node` is launched by `edge_only.launch.py` when
`with_follower:=true`. It runs Pure-Pursuit at 20 Hz with
`lookahead=0.4`, `max_v=0.4`, `max_w=1.0`. Override via launch args
or by editing the launch file. It zeroes `cmd_vel` if the incoming
path's `frame_id` differs from `base_link`.

### 6.4 Environment overrides

| Variable               | Effect                                                                |
|------------------------|-----------------------------------------------------------------------|
| `GRPC_PORT`            | Default port for `--host` (otherwise `50051`)                         |
| `HF_CACHE_DIR`         | Mounted into containers as the HF cache (default `~/.cache/huggingface`) |
| `RASPICAT_VLA_REBUILD` | If set, forces `colcon build` inside `--real` / `--sim` containers    |
| `ASYNCVLA_E2E`         | Enables the AsyncVLA E2E pytest smoke test (otherwise skipped)        |
| `OMNIVLA_E2E`          | Enables the OmniVLA E2E pytest smoke test (otherwise skipped)         |

## 7. Topics and interfaces

### 7.1 ROS2 topics

The edge stack uses the following topics. All are remappable via the
launch arguments listed in §6.

| Topic                            | Direction          | Type                         | Notes                              |
|----------------------------------|--------------------|------------------------------|------------------------------------|
| `/camera/image_raw`              | edge ← camera      | `sensor_msgs/Image`          | Sim publishes at `…/color/image_raw` |
| `/raspicat_vla/goal`             | edge ← user        | `raspicat_vla_msgs/GoalSpec` | One of `POSE`, `TEXT`, `IMAGE`     |
| `/raspicat_vla/predicted_path`   | follower ← edge    | `nav_msgs/Path`              | Frame `base_link`                  |
| `/raspicat_vla/status`           | obs ← edge         | `diagnostic_msgs/DiagnosticArray` | `OK` / `DEGRADED` / `WAITING_REMOTE` / `STALE` |
| `/raspicat_vla/embedding`        | obs ← edge (debug) | `raspicat_vla_msgs/ActionEmbedding` | Only when `publish_embedding_debug` is true |
| `/cmd_vel`                       | robot ← follower   | `geometry_msgs/Twist`        | Zero on stale or frame mismatch    |

### 7.2 Lifecycle

`vla_edge_node` is a `LifecycleNode`. `edge_only.launch.py` auto-
transitions it `unconfigured → inactive → active`. To bring it down or
back up by hand:

```bash
ros2 lifecycle set /vla_edge_node deactivate
ros2 lifecycle set /vla_edge_node activate
```

### 7.3 gRPC service

`proto/raspicat_vla.proto` defines `raspicat_vla.v1.VLAService`:

```
rpc StreamInfer(stream Observation) returns (stream ActionEmbedding);
rpc GetModelInfo(ModelInfoRequest) returns (ModelInfo);
```

`Observation` carries a JPEG, a `GoalSpec` (pose, text, or image goal),
and an optional current pose. `ActionEmbedding` returns FP16-packed
embeddings the edge adapter decodes into a `nav_msgs/Path`.

## 8. Testing

`docker/run.sh test` runs pytest inside `raspicat-vla-test`. The image
is auto-built on first use.

```bash
docker/run.sh test                            # full suite
docker/run.sh test -k checkpoint              # pytest -k filter
docker/run.sh test src/raspicat_vla_edge/test # subset by path
```

Pure-flag invocations (`-k`, `-x`, `--lf`) automatically prepend the
default test paths so pytest doesn't fall back to cwd discovery (which
would walk `external/` and crash on missing transitive deps).

E2E smoke tests for AsyncVLA and OmniVLA are gated by environment
variables; they skip cleanly without GPU and are not part of the default
suite:

```bash
ASYNCVLA_E2E=1 docker/run.sh test -k asyncvla_e2e
OMNIVLA_E2E=1 docker/run.sh test -k omnivla_e2e
```

## 9. Troubleshooting

**`run.sh: image XYZ not built; falling back to raspicat-vla-test`**
The full `real` or `sim` image isn't built. The fallback gives you the
edge stack but no rt-net packages, no Gazebo, and no torch. Build the
proper image when you actually need hardware or simulation:

```bash
docker/run.sh build real    # or: build sim
```

**`--remote requires --cpu or --gpu`**
The remote subcommand demands an explicit device. There's no default —
it forces a conscious choice between `--gpus all` and CPU-only.

**`--<mode> requires --host HOST[:PORT]`**
`--real` and `--sim` need to know where the remote is. `--remote` does
not — it binds locally and `--host` is optional (defaults to `0.0.0.0`).

**Edge says `WAITING_REMOTE` forever**
The edge isn't getting `ActionEmbedding` replies. Check, in order: the
remote is running and listening on the expected port; the network path
between hosts (`nc -z HOST PORT`); a goal has been published on
`/raspicat_vla/goal` (the edge gates outbound traffic on having both a
fresh image AND a goal).

**Edge cycles `OK` → `DEGRADED` → `STALE` → safe-stop**
The remote is replying but slower than `embedding_max_age_sec`. Either
move the workload to GPU, or relax the thresholds in `edge_params.yaml`.

**Gazebo prints `Error getting username: no matching password record`**
`run.sh` synthesizes a `passwd` entry for your host UID inside the
container; if you bypass `run.sh` and `docker run` the `sim` image
yourself, you'll need to do the same. See `run_sim()` in `docker/run.sh`
for the recipe.

**Inside-container colcon build keeps re-running**
`run.sh` skips the colcon step when `install/setup.bash` already exists
in the workspace. Set `RASPICAT_VLA_REBUILD=1` to force a rebuild after
editing source. Conversely, if it's *not* rebuilding when you expected,
delete `install/` on the host (it's bind-mounted).

**Lifecycle node stuck in `unconfigured`**
`edge_only.launch.py` only auto-configures on `OnProcessStart`. If the
process restarted (e.g. you `Ctrl+C`'d and relaunched in the same shell)
without the launch system re-emitting the event, drive transitions by
hand: `ros2 lifecycle set /vla_edge_node configure`.

**HuggingFace download stalls or fails authentication**
The repos are public and need no token. If `snapshot_download` 401s,
clear your HF token (`huggingface-cli logout`) and retry; an expired
token causes 401 on otherwise-public repos.

## 10. References

* `docker/run.sh --help` — authoritative subcommand reference
* `proto/raspicat_vla.proto` — gRPC interface contract
* `src/raspicat_vla_edge/launch/edge_only.launch.py` — edge launch args
* `src/raspicat_vla_bringup/launch/mvp_sim.launch.py` — sim composition
* `src/raspicat_vla_edge/config/edge_params.yaml` — full edge parameter list
* `src/raspicat_vla_remote/raspicat_vla_remote/server_main.py` — remote CLI
* `scripts/download_*_checkpoints.sh` — HF model download helpers
* `raspicat.repos` — pinned rt-net source versions (vcstool manifest)
