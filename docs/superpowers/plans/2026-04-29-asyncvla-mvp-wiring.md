# AsyncVLA MVP Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the wiring (messages, gRPC protocol, edge node skeleton with lifecycle and async loops, path follower, dummy remote server) end-to-end so that a fixed dummy embedding flows from a remote server through the edge node into `/cmd_vel`. No real ML model is loaded — that comes in Plan 2.

**Architecture:** ROS2 Humble workspace with five colcon packages (msgs / proto-Python lib / remote / edge / bringup). Edge ↔ Remote uses gRPC bidi stream. Edge node is a Python `LifecycleNode` running three async tasks: observation send, embedding receive, action generation. A Pure-Pursuit `path_follower_node` converts `nav_msgs/Path` into `geometry_msgs/Twist`.

**Tech Stack:** ROS2 Humble, rclpy, lifecycle_msgs, nav_msgs, sensor_msgs, geometry_msgs, diagnostic_msgs, grpcio, grpcio-tools, protobuf, numpy, opencv-python (cv2), Pillow, pytest.

**Reference spec:** `docs/superpowers/specs/2026-04-29-asyncvla-control-node-design.md`

---

## File Structure (final state after this plan)

```
raspicat-async-vla/
├── .gitignore                     # extended
├── .gitmodules                    # NEW (AsyncVLA submodule reference, not used in MVP)
├── README.md                      # NEW (minimal)
├── docker/
│   ├── Dockerfile.sim             # existing, untouched in this plan
│   ├── Dockerfile.real            # existing, untouched in this plan
├── proto/
│   └── asyncvla.proto             # NEW
├── scripts/
│   └── gen_proto.sh               # NEW (regenerates Python stubs)
├── src/
│   ├── raspicat_async_vla_msgs/
│   │   ├── package.xml
│   │   ├── CMakeLists.txt
│   │   ├── msg/
│   │   │   ├── ActionEmbedding.msg
│   │   │   └── GoalSpec.msg
│   │   ├── srv/
│   │   │   └── SetGoal.srv
│   │   └── action/
│   │       └── NavigateAsync.action
│   ├── raspicat_async_vla_proto/  # ament_python; pure proto wrappers
│   │   ├── package.xml
│   │   ├── setup.py
│   │   ├── setup.cfg
│   │   ├── resource/raspicat_async_vla_proto
│   │   ├── raspicat_async_vla_proto/
│   │   │   ├── __init__.py
│   │   │   ├── asyncvla_pb2.py        # generated
│   │   │   ├── asyncvla_pb2_grpc.py   # generated
│   │   │   └── conversions.py         # ROS2 msg ↔ proto
│   │   └── test/test_conversions.py
│   ├── raspicat_async_vla_remote/
│   │   ├── package.xml
│   │   ├── setup.py
│   │   ├── setup.cfg
│   │   ├── resource/raspicat_async_vla_remote
│   │   ├── asyncvla_remote/
│   │   │   ├── __init__.py
│   │   │   ├── dummy_server.py
│   │   │   └── server_main.py
│   │   ├── config/remote_params.yaml
│   │   └── test/test_dummy_server.py
│   ├── raspicat_async_vla_edge/
│   │   ├── package.xml
│   │   ├── setup.py
│   │   ├── setup.cfg
│   │   ├── resource/raspicat_async_vla_edge
│   │   ├── asyncvla_edge/
│   │   │   ├── __init__.py
│   │   │   ├── preprocess.py
│   │   │   ├── embedding_cache.py
│   │   │   ├── grpc_client.py
│   │   │   ├── pure_pursuit.py
│   │   │   ├── edge_node.py             # LifecycleNode
│   │   │   └── path_follower_node.py
│   │   ├── launch/
│   │   │   ├── edge_only.launch.py
│   │   │   └── path_follower.launch.py
│   │   ├── config/edge_params.yaml
│   │   └── test/
│   │       ├── test_preprocess.py
│   │       ├── test_embedding_cache.py
│   │       ├── test_grpc_client.py
│   │       ├── test_pure_pursuit.py
│   │       └── test_edge_node_smoke.py
│   └── raspicat_async_vla_bringup/
│       ├── package.xml
│       ├── CMakeLists.txt
│       ├── launch/
│       │   └── mvp_local.launch.py
│       └── config/topic_remap.yaml
└── docs/
    └── superpowers/
        ├── specs/2026-04-29-asyncvla-control-node-design.md
        └── plans/2026-04-29-asyncvla-mvp-wiring.md     # this file
```

**Responsibilities:**

- `raspicat_async_vla_msgs`: ROS2 msg/srv/action only.
- `raspicat_async_vla_proto`: hand-edited `.proto` lives in repo root `proto/`; this package only **vendors** generated code + provides ROS2 ↔ proto conversion helpers.
- `raspicat_async_vla_remote`: gRPC server entrypoint. In Plan 1 only `DummyServer` (returns deterministic embedding). Plan 2 swaps in real model.
- `raspicat_async_vla_edge`: all edge-side logic — lifecycle node, async loops, gRPC client, embedding cache, path follower, image preprocessing, pure-pursuit.
- `raspicat_async_vla_bringup`: launch composition only.

---

## Conventions

- Tests live next to packages under `test/` and run via `colcon test --packages-select <pkg>`.
- Pure Python modules also runnable as `pytest` from package directory for quick TDD.
- Commits use `<type>(<pkg>): <summary>` — types: feat, test, chore, docs, fix.
- Always show test failing first, then make it pass (TDD red→green).
- ROS2 distro: **humble**. Source it with `source /opt/ros/humble/setup.bash` before any colcon command.
- Workspace root: `/home/nop/dev/mywork/raspicat-async-vla`. The repo root IS the colcon workspace root (no nested `ws/` dir).

---

## Task 1: Repo init — gitignore, README, gitmodules

**Files:**
- Create: `/home/nop/dev/mywork/raspicat-async-vla/.gitignore` (extends existing empty file)
- Create: `/home/nop/dev/mywork/raspicat-async-vla/README.md`
- Create: `/home/nop/dev/mywork/raspicat-async-vla/.gitmodules`

- [ ] **Step 1.1: Write `.gitignore`**

```
# Build / colcon
build/
install/
log/

# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.mypy_cache/

# Generated proto stubs are committed (they are vendored), but local debug files aren't
*.pb.cc
*.pb.h

# Editor
.vscode/
.idea/

# OS
.DS_Store
```

- [ ] **Step 1.2: Write `README.md`**

```markdown
# raspicat-async-vla

ROS2 Humble nodes for running [AsyncVLA](https://github.com/NHirose/AsyncVLA) navigation on the Raspberry Pi Cat (rt-net `raspicat`).

See `docs/superpowers/specs/2026-04-29-asyncvla-control-node-design.md` for full design.

## Workspace layout

This repository is itself a colcon workspace.

```
src/raspicat_async_vla_msgs/      # ROS2 messages, services, actions
src/raspicat_async_vla_proto/     # gRPC python stubs + conversion helpers
src/raspicat_async_vla_remote/    # gRPC server (Plan 1: dummy; Plan 2: real model)
src/raspicat_async_vla_edge/      # Edge ROS2 nodes (lifecycle, follower)
src/raspicat_async_vla_bringup/   # Launch composition
```

## Build

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## Plans / specs

- Spec: `docs/superpowers/specs/2026-04-29-asyncvla-control-node-design.md`
- Plan 1 (this MVP): `docs/superpowers/plans/2026-04-29-asyncvla-mvp-wiring.md`
```

- [ ] **Step 1.3: Write `.gitmodules` placeholder**

(The submodule itself is added in Plan 2; Plan 1 only reserves the file so the layout matches the spec.)

```
# AsyncVLA upstream is added as a submodule in Plan 2.
# Reserved for: external/AsyncVLA -> https://github.com/NHirose/AsyncVLA
```

- [ ] **Step 1.4: Commit**

```bash
cd /home/nop/dev/mywork/raspicat-async-vla
git add .gitignore README.md .gitmodules
git commit -m "chore(repo): scaffold gitignore, README, gitmodules placeholder"
```

---

## Task 2: `raspicat_async_vla_msgs` package skeleton

**Files:**
- Create: `src/raspicat_async_vla_msgs/package.xml`
- Create: `src/raspicat_async_vla_msgs/CMakeLists.txt`

- [ ] **Step 2.1: Write `package.xml`**

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>raspicat_async_vla_msgs</name>
  <version>0.1.0</version>
  <description>ROS2 messages, services and actions for AsyncVLA on raspicat.</description>
  <maintainer email="nop@example.com">nop</maintainer>
  <license>MIT</license>

  <buildtool_depend>ament_cmake</buildtool_depend>
  <buildtool_depend>rosidl_default_generators</buildtool_depend>

  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>nav_msgs</depend>

  <exec_depend>rosidl_default_runtime</exec_depend>

  <member_of_group>rosidl_interface_packages</member_of_group>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

- [ ] **Step 2.2: Write `CMakeLists.txt`**

```cmake
cmake_minimum_required(VERSION 3.8)
project(raspicat_async_vla_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/GoalSpec.msg
  msg/ActionEmbedding.msg
  srv/SetGoal.srv
  action/NavigateAsync.action
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs nav_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
```

- [ ] **Step 2.3: Commit**

```bash
git add src/raspicat_async_vla_msgs/package.xml src/raspicat_async_vla_msgs/CMakeLists.txt
git commit -m "feat(msgs): scaffold raspicat_async_vla_msgs package"
```

---

## Task 3: msg / srv / action definitions

**Files:**
- Create: `src/raspicat_async_vla_msgs/msg/GoalSpec.msg`
- Create: `src/raspicat_async_vla_msgs/msg/ActionEmbedding.msg`
- Create: `src/raspicat_async_vla_msgs/srv/SetGoal.srv`
- Create: `src/raspicat_async_vla_msgs/action/NavigateAsync.action`

- [ ] **Step 3.1: Write `msg/GoalSpec.msg`**

```
uint8 MODE_POSE  = 0
uint8 MODE_TEXT  = 1
uint8 MODE_IMAGE = 2
uint8 mode

geometry_msgs/PoseStamped pose
string                    text
sensor_msgs/CompressedImage image
```

- [ ] **Step 3.2: Write `msg/ActionEmbedding.msg`**

```
std_msgs/Header header
uint64 frame_id
uint32 num_tokens
uint32 embed_dim
float32[] embedding
float32 inference_ms
string model_version
```

- [ ] **Step 3.3: Write `srv/SetGoal.srv`**

```
raspicat_async_vla_msgs/GoalSpec goal
---
bool success
string message
```

- [ ] **Step 3.4: Write `action/NavigateAsync.action`**

```
raspicat_async_vla_msgs/GoalSpec goal
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

- [ ] **Step 3.5: Build and verify**

```bash
cd /home/nop/dev/mywork/raspicat-async-vla
source /opt/ros/humble/setup.bash
colcon build --packages-select raspicat_async_vla_msgs
```

Expected: `Summary: 1 package finished` with no errors.

```bash
source install/setup.bash
ros2 interface show raspicat_async_vla_msgs/msg/GoalSpec
```

Expected: prints the GoalSpec definition we wrote.

- [ ] **Step 3.6: Commit**

```bash
git add src/raspicat_async_vla_msgs/msg src/raspicat_async_vla_msgs/srv src/raspicat_async_vla_msgs/action
git commit -m "feat(msgs): add GoalSpec, ActionEmbedding, SetGoal, NavigateAsync"
```

---

## Task 4: gRPC proto file

**Files:**
- Create: `proto/asyncvla.proto`

- [ ] **Step 4.1: Write `proto/asyncvla.proto`**

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
  uint32 image_width     = 4;
  uint32 image_height    = 5;
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
  bytes  embedding_fp16  = 5;
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

- [ ] **Step 4.2: Commit**

```bash
git add proto/asyncvla.proto
git commit -m "feat(proto): add asyncvla.proto v1 IDL"
```

---

## Task 5: Proto-Python package + generation script

**Files:**
- Create: `scripts/gen_proto.sh`
- Create: `src/raspicat_async_vla_proto/package.xml`
- Create: `src/raspicat_async_vla_proto/setup.py`
- Create: `src/raspicat_async_vla_proto/setup.cfg`
- Create: `src/raspicat_async_vla_proto/resource/raspicat_async_vla_proto`
- Create: `src/raspicat_async_vla_proto/raspicat_async_vla_proto/__init__.py`

- [ ] **Step 5.1: Write `scripts/gen_proto.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${REPO_ROOT}/src/raspicat_async_vla_proto/raspicat_async_vla_proto"
PROTO_DIR="${REPO_ROOT}/proto"

mkdir -p "${OUT_DIR}"

python3 -m grpc_tools.protoc \
    -I "${PROTO_DIR}" \
    --python_out="${OUT_DIR}" \
    --grpc_python_out="${OUT_DIR}" \
    "${PROTO_DIR}/asyncvla.proto"

# grpc_tools generates "asyncvla_pb2.py" with `import asyncvla_pb2` -- rewrite to relative.
sed -i 's/^import asyncvla_pb2/from . import asyncvla_pb2/' "${OUT_DIR}/asyncvla_pb2_grpc.py"

echo "Generated:"
ls -1 "${OUT_DIR}"/asyncvla_pb2*.py
```

- [ ] **Step 5.2: Make script executable and run it**

```bash
cd /home/nop/dev/mywork/raspicat-async-vla
mkdir -p src/raspicat_async_vla_proto/raspicat_async_vla_proto
chmod +x scripts/gen_proto.sh
pip install --user 'grpcio-tools>=1.50,<2.0'
./scripts/gen_proto.sh
```

Expected output:
```
Generated:
asyncvla_pb2_grpc.py
asyncvla_pb2.py
```

- [ ] **Step 5.3: Write `src/raspicat_async_vla_proto/package.xml`**

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>raspicat_async_vla_proto</name>
  <version>0.1.0</version>
  <description>Generated gRPC stubs and ROS2 conversion helpers for AsyncVLA.</description>
  <maintainer email="nop@example.com">nop</maintainer>
  <license>MIT</license>

  <buildtool_depend>ament_python</buildtool_depend>

  <exec_depend>raspicat_async_vla_msgs</exec_depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

- [ ] **Step 5.4: Write `setup.py`**

```python
from setuptools import setup

package_name = 'raspicat_async_vla_proto'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'grpcio>=1.50', 'protobuf>=4.21'],
    zip_safe=True,
    maintainer='nop',
    maintainer_email='nop@example.com',
    description='Generated gRPC stubs and conversion helpers.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={'console_scripts': []},
)
```

- [ ] **Step 5.5: Write `setup.cfg`**

```
[develop]
script_dir=$base/lib/raspicat_async_vla_proto
[install]
install_scripts=$base/lib/raspicat_async_vla_proto
```

- [ ] **Step 5.6: Write resource marker and `__init__.py`**

```bash
touch /home/nop/dev/mywork/raspicat-async-vla/src/raspicat_async_vla_proto/resource/raspicat_async_vla_proto
```

`raspicat_async_vla_proto/__init__.py`:

```python
from . import asyncvla_pb2  # noqa: F401
from . import asyncvla_pb2_grpc  # noqa: F401
```

- [ ] **Step 5.7: Build and verify import**

```bash
cd /home/nop/dev/mywork/raspicat-async-vla
source /opt/ros/humble/setup.bash
colcon build --packages-select raspicat_async_vla_proto
source install/setup.bash
python3 -c "from raspicat_async_vla_proto import asyncvla_pb2, asyncvla_pb2_grpc; print(asyncvla_pb2.Observation.DESCRIPTOR.full_name)"
```

Expected: `asyncvla.v1.Observation`

- [ ] **Step 5.8: Commit**

```bash
git add scripts/gen_proto.sh src/raspicat_async_vla_proto
git commit -m "feat(proto): generate Python gRPC stubs and ament_python package"
```

---

## Task 6: ROS2 ↔ proto conversions (TDD)

**Files:**
- Create: `src/raspicat_async_vla_proto/raspicat_async_vla_proto/conversions.py`
- Create: `src/raspicat_async_vla_proto/test/test_conversions.py`

- [ ] **Step 6.1: Write the failing test**

`src/raspicat_async_vla_proto/test/test_conversions.py`:

```python
"""Tests for ROS2 <-> proto conversion helpers."""
import numpy as np
import pytest

from raspicat_async_vla_msgs.msg import ActionEmbedding as ActionEmbeddingMsg
from raspicat_async_vla_proto import asyncvla_pb2
from raspicat_async_vla_proto.conversions import (
    proto_action_embedding_to_msg,
    fp16_bytes_to_float32_list,
    float32_array_to_fp16_bytes,
)


def test_fp16_bytes_round_trip():
    arr = np.arange(8 * 1024, dtype=np.float32) / 100.0
    raw = float32_array_to_fp16_bytes(arr)
    assert isinstance(raw, bytes)
    assert len(raw) == 8 * 1024 * 2  # fp16 = 2 bytes
    back = np.array(fp16_bytes_to_float32_list(raw), dtype=np.float32)
    assert back.shape == arr.shape
    np.testing.assert_allclose(back, arr, atol=1e-2)  # fp16 quantization


def test_proto_action_embedding_to_msg_basic():
    arr = np.linspace(-1, 1, 8 * 16, dtype=np.float32)
    proto = asyncvla_pb2.ActionEmbedding(
        frame_id=42,
        server_time_ns=123,
        num_tokens=8,
        embed_dim=16,
        embedding_fp16=float32_array_to_fp16_bytes(arr),
        inference_ms=12.5,
        model_version='dummy',
    )
    msg = proto_action_embedding_to_msg(proto)
    assert isinstance(msg, ActionEmbeddingMsg)
    assert msg.frame_id == 42
    assert msg.num_tokens == 8
    assert msg.embed_dim == 16
    assert len(msg.embedding) == 8 * 16
    np.testing.assert_allclose(np.array(msg.embedding), arr, atol=1e-2)
    assert msg.inference_ms == pytest.approx(12.5)
    assert msg.model_version == 'dummy'
```

- [ ] **Step 6.2: Run the test and confirm it fails**

```bash
cd /home/nop/dev/mywork/raspicat-async-vla
source install/setup.bash
pytest src/raspicat_async_vla_proto/test/test_conversions.py -v
```

Expected: ImportError on `conversions` (module not yet created).

- [ ] **Step 6.3: Implement `conversions.py`**

```python
"""ROS2 <-> proto conversion helpers."""
from __future__ import annotations

import numpy as np

from raspicat_async_vla_msgs.msg import ActionEmbedding as ActionEmbeddingMsg

from . import asyncvla_pb2


def float32_array_to_fp16_bytes(arr: np.ndarray) -> bytes:
    """Convert a contiguous float32 array to little-endian fp16 bytes."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    fp16 = arr.astype('<f2', copy=False)
    return fp16.tobytes()


def fp16_bytes_to_float32_list(raw: bytes) -> list[float]:
    """Convert little-endian fp16 bytes to a Python list of float32 values."""
    fp16 = np.frombuffer(raw, dtype='<f2')
    return fp16.astype(np.float32).tolist()


def proto_action_embedding_to_msg(
    proto: asyncvla_pb2.ActionEmbedding,
) -> ActionEmbeddingMsg:
    """Convert a proto ActionEmbedding into the ROS2 message form."""
    msg = ActionEmbeddingMsg()
    msg.frame_id = proto.frame_id
    msg.num_tokens = proto.num_tokens
    msg.embed_dim = proto.embed_dim
    msg.embedding = fp16_bytes_to_float32_list(proto.embedding_fp16)
    msg.inference_ms = float(proto.inference_ms)
    msg.model_version = proto.model_version or ''
    return msg
```

- [ ] **Step 6.4: Run the test and confirm it passes**

```bash
pytest src/raspicat_async_vla_proto/test/test_conversions.py -v
```

Expected: 2 passed.

- [ ] **Step 6.5: Commit**

```bash
git add src/raspicat_async_vla_proto/raspicat_async_vla_proto/conversions.py \
        src/raspicat_async_vla_proto/test/test_conversions.py
git commit -m "feat(proto): add fp16 helpers and proto->ROS2 conversion"
```

---

## Task 7: Image preprocessing (TDD)

**Files:**
- Create: `src/raspicat_async_vla_edge/asyncvla_edge/preprocess.py`
- Create: `src/raspicat_async_vla_edge/test/test_preprocess.py`
- Create: `src/raspicat_async_vla_edge/package.xml`
- Create: `src/raspicat_async_vla_edge/setup.py`
- Create: `src/raspicat_async_vla_edge/setup.cfg`
- Create: `src/raspicat_async_vla_edge/resource/raspicat_async_vla_edge`
- Create: `src/raspicat_async_vla_edge/asyncvla_edge/__init__.py`

- [ ] **Step 7.1: Scaffold the `raspicat_async_vla_edge` package**

`src/raspicat_async_vla_edge/package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>raspicat_async_vla_edge</name>
  <version>0.1.0</version>
  <description>Edge ROS2 nodes for AsyncVLA on raspicat.</description>
  <maintainer email="nop@example.com">nop</maintainer>
  <license>MIT</license>

  <buildtool_depend>ament_python</buildtool_depend>

  <exec_depend>rclpy</exec_depend>
  <exec_depend>std_msgs</exec_depend>
  <exec_depend>geometry_msgs</exec_depend>
  <exec_depend>nav_msgs</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>
  <exec_depend>diagnostic_msgs</exec_depend>
  <exec_depend>lifecycle_msgs</exec_depend>
  <exec_depend>cv_bridge</exec_depend>
  <exec_depend>raspicat_async_vla_msgs</exec_depend>
  <exec_depend>raspicat_async_vla_proto</exec_depend>

  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

`src/raspicat_async_vla_edge/setup.py`:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'raspicat_async_vla_edge'

setup(
    name=package_name,
    version='0.1.0',
    packages=['asyncvla_edge'],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'Pillow',
        'grpcio>=1.50',
    ],
    zip_safe=True,
    maintainer='nop',
    maintainer_email='nop@example.com',
    description='Edge ROS2 nodes for AsyncVLA.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'asyncvla_edge_node = asyncvla_edge.edge_node:main',
            'path_follower_node = asyncvla_edge.path_follower_node:main',
        ],
    },
)
```

`src/raspicat_async_vla_edge/setup.cfg`:

```
[develop]
script_dir=$base/lib/raspicat_async_vla_edge
[install]
install_scripts=$base/lib/raspicat_async_vla_edge
```

```bash
mkdir -p /home/nop/dev/mywork/raspicat-async-vla/src/raspicat_async_vla_edge/{asyncvla_edge,resource,launch,config,test}
touch /home/nop/dev/mywork/raspicat-async-vla/src/raspicat_async_vla_edge/resource/raspicat_async_vla_edge
touch /home/nop/dev/mywork/raspicat-async-vla/src/raspicat_async_vla_edge/asyncvla_edge/__init__.py
```

- [ ] **Step 7.2: Write the failing test**

`src/raspicat_async_vla_edge/test/test_preprocess.py`:

```python
"""Tests for edge image preprocessing."""
import numpy as np
import pytest

from asyncvla_edge.preprocess import resize_and_jpeg, decode_jpeg_to_rgb


def _make_rgb(h: int, w: int) -> np.ndarray:
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def test_resize_and_jpeg_returns_bytes_and_target_size():
    img = _make_rgb(480, 640)
    raw, w, h = resize_and_jpeg(img, target=(224, 224), quality=85)
    assert isinstance(raw, (bytes, bytearray))
    assert (w, h) == (224, 224)
    # JPEG magic
    assert raw[:3] == b'\xff\xd8\xff'


def test_resize_and_jpeg_round_trip_within_jpeg_tolerance():
    img = _make_rgb(300, 400)
    raw, _, _ = resize_and_jpeg(img, target=(224, 224), quality=95)
    decoded = decode_jpeg_to_rgb(raw)
    assert decoded.shape == (224, 224, 3)
    assert decoded.dtype == np.uint8


def test_resize_and_jpeg_rejects_non_uint8():
    img = np.zeros((100, 100, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        resize_and_jpeg(img, target=(224, 224))


def test_resize_and_jpeg_rejects_wrong_channels():
    img = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError):
        resize_and_jpeg(img, target=(224, 224))
```

- [ ] **Step 7.3: Confirm test fails**

```bash
cd /home/nop/dev/mywork/raspicat-async-vla/src/raspicat_async_vla_edge
python3 -m pytest test/test_preprocess.py -v
```

Expected: ImportError (module not found).

- [ ] **Step 7.4: Implement `preprocess.py`**

```python
"""Image preprocessing for AsyncVLA edge node."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def resize_and_jpeg(
    image_rgb: np.ndarray,
    target: Tuple[int, int] = (224, 224),
    quality: int = 85,
) -> Tuple[bytes, int, int]:
    """Resize an RGB uint8 image and JPEG-encode it.

    Returns: (jpeg_bytes, width, height)
    """
    if image_rgb.dtype != np.uint8:
        raise ValueError(f'expected uint8 RGB, got dtype={image_rgb.dtype}')
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f'expected HxWx3 RGB, got shape={image_rgb.shape}')

    w, h = target
    resized = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError('cv2.imencode failed')
    return buf.tobytes(), w, h


def decode_jpeg_to_rgb(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes back to an RGB uint8 ndarray."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError('failed to decode JPEG')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
```

- [ ] **Step 7.5: Confirm tests pass**

```bash
python3 -m pytest test/test_preprocess.py -v
```

Expected: 4 passed.

- [ ] **Step 7.6: Commit**

```bash
cd /home/nop/dev/mywork/raspicat-async-vla
git add src/raspicat_async_vla_edge
git commit -m "feat(edge): scaffold package and add preprocess module with TDD"
```

---

## Task 8: EmbeddingCache (TDD)

**Files:**
- Create: `src/raspicat_async_vla_edge/asyncvla_edge/embedding_cache.py`
- Create: `src/raspicat_async_vla_edge/test/test_embedding_cache.py`

- [ ] **Step 8.1: Write the failing test**

```python
"""Tests for EmbeddingCache."""
import time

import numpy as np
import pytest

from asyncvla_edge.embedding_cache import EmbeddingCache, CachedEmbedding


def _emb(frame_id: int, value: float = 0.0) -> CachedEmbedding:
    return CachedEmbedding(
        frame_id=frame_id,
        recv_time_ns=time.monotonic_ns(),
        embedding=np.full(8 * 1024, value, dtype=np.float32),
        num_tokens=8,
        embed_dim=1024,
        inference_ms=10.0,
        model_version='dummy',
    )


def test_cache_starts_empty():
    cache = EmbeddingCache(max_age_sec=6.0, hard_timeout_sec=15.0)
    assert cache.get_fresh() is None
    assert cache.status() == 'WAITING_REMOTE'


def test_cache_stores_and_returns_latest():
    cache = EmbeddingCache(max_age_sec=6.0, hard_timeout_sec=15.0)
    cache.put(_emb(frame_id=1, value=1.0))
    cur = cache.get_fresh()
    assert cur is not None
    assert cur.frame_id == 1
    assert cache.status() == 'OK'


def test_cache_drops_older_frame_id():
    cache = EmbeddingCache(max_age_sec=6.0, hard_timeout_sec=15.0)
    cache.put(_emb(frame_id=10, value=10.0))
    cache.put(_emb(frame_id=5, value=5.0))  # older, must be dropped
    cur = cache.get_fresh()
    assert cur is not None
    assert cur.frame_id == 10
    assert cur.embedding[0] == pytest.approx(10.0)


def test_cache_returns_none_when_stale_past_max_age():
    cache = EmbeddingCache(max_age_sec=0.001, hard_timeout_sec=0.01)
    e = _emb(frame_id=1)
    cache.put(e)
    time.sleep(0.005)
    assert cache.get_fresh() is None
    # but raw still readable for diagnostics
    assert cache.get_latest_raw() is not None
    assert cache.status() == 'DEGRADED'


def test_cache_status_stale_after_hard_timeout():
    cache = EmbeddingCache(max_age_sec=0.001, hard_timeout_sec=0.005)
    cache.put(_emb(frame_id=1))
    time.sleep(0.020)
    assert cache.status() == 'STALE'


def test_cache_invalidate_clears_state():
    cache = EmbeddingCache(max_age_sec=6.0, hard_timeout_sec=15.0)
    cache.put(_emb(frame_id=1))
    cache.invalidate()
    assert cache.get_fresh() is None
    assert cache.status() == 'WAITING_REMOTE'
```

- [ ] **Step 8.2: Confirm test fails**

```bash
cd /home/nop/dev/mywork/raspicat-async-vla/src/raspicat_async_vla_edge
python3 -m pytest test/test_embedding_cache.py -v
```

Expected: ImportError.

- [ ] **Step 8.3: Implement `embedding_cache.py`**

```python
"""Thread-safe cache for the latest action embedding from the remote VLA."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CachedEmbedding:
    frame_id: int
    recv_time_ns: int           # monotonic ns at time of insertion
    embedding: np.ndarray       # shape (num_tokens * embed_dim,), dtype float32
    num_tokens: int
    embed_dim: int
    inference_ms: float
    model_version: str


class EmbeddingCache:
    """Holds the single latest embedding. Frame-id monotonic, age-aware."""

    STATUS_WAITING = 'WAITING_REMOTE'
    STATUS_OK = 'OK'
    STATUS_DEGRADED = 'DEGRADED'
    STATUS_STALE = 'STALE'

    def __init__(self, *, max_age_sec: float, hard_timeout_sec: float) -> None:
        if hard_timeout_sec < max_age_sec:
            raise ValueError('hard_timeout_sec must be >= max_age_sec')
        self._max_age_ns = int(max_age_sec * 1e9)
        self._hard_ns = int(hard_timeout_sec * 1e9)
        self._lock = threading.Lock()
        self._latest: Optional[CachedEmbedding] = None

    def put(self, emb: CachedEmbedding) -> None:
        with self._lock:
            if self._latest is None or emb.frame_id > self._latest.frame_id:
                self._latest = emb

    def invalidate(self) -> None:
        with self._lock:
            self._latest = None

    def get_latest_raw(self) -> Optional[CachedEmbedding]:
        with self._lock:
            return self._latest

    def _age_ns_locked(self) -> Optional[int]:
        if self._latest is None:
            return None
        return time.monotonic_ns() - self._latest.recv_time_ns

    def get_fresh(self) -> Optional[CachedEmbedding]:
        with self._lock:
            age = self._age_ns_locked()
            if age is None or age >= self._max_age_ns:
                return None
            return self._latest

    def status(self) -> str:
        with self._lock:
            age = self._age_ns_locked()
            if age is None:
                return self.STATUS_WAITING
            if age >= self._hard_ns:
                return self.STATUS_STALE
            if age >= self._max_age_ns:
                return self.STATUS_DEGRADED
            return self.STATUS_OK
```

- [ ] **Step 8.4: Confirm tests pass**

```bash
python3 -m pytest test/test_embedding_cache.py -v
```

Expected: 6 passed.

- [ ] **Step 8.5: Commit**

```bash
cd /home/nop/dev/mywork/raspicat-async-vla
git add src/raspicat_async_vla_edge/asyncvla_edge/embedding_cache.py \
        src/raspicat_async_vla_edge/test/test_embedding_cache.py
git commit -m "feat(edge): add thread-safe EmbeddingCache with TDD"
```

---

## Task 9: Pure Pursuit follower (TDD)

**Files:**
- Create: `src/raspicat_async_vla_edge/asyncvla_edge/pure_pursuit.py`
- Create: `src/raspicat_async_vla_edge/test/test_pure_pursuit.py`

- [ ] **Step 9.1: Write the failing test**

```python
"""Tests for Pure Pursuit controller."""
import math

import pytest

from asyncvla_edge.pure_pursuit import PurePursuit, Pose2D, Waypoint


def test_straight_path_outputs_forward_velocity():
    pp = PurePursuit(lookahead=0.5, max_v=0.4, max_w=1.0)
    path = [Waypoint(x=0.0, y=0.0), Waypoint(x=1.0, y=0.0), Waypoint(x=2.0, y=0.0)]
    cmd = pp.compute(robot=Pose2D(0.0, 0.0, 0.0), path=path)
    assert cmd.linear > 0.0
    assert abs(cmd.angular) < 1e-3
    assert cmd.linear <= 0.4


def test_target_to_left_turns_left():
    pp = PurePursuit(lookahead=0.5, max_v=0.4, max_w=1.0)
    path = [Waypoint(x=0.0, y=0.0), Waypoint(x=0.5, y=0.5)]
    cmd = pp.compute(robot=Pose2D(0.0, 0.0, 0.0), path=path)
    assert cmd.angular > 0.0


def test_target_behind_emits_no_backward_motion():
    pp = PurePursuit(lookahead=0.5, max_v=0.4, max_w=1.0, no_backward=True)
    path = [Waypoint(x=-1.0, y=0.0)]
    cmd = pp.compute(robot=Pose2D(0.0, 0.0, 0.0), path=path)
    assert cmd.linear == pytest.approx(0.0)
    assert abs(cmd.angular) > 0.0  # rotates in place


def test_empty_path_emits_zero_command():
    pp = PurePursuit(lookahead=0.5, max_v=0.4, max_w=1.0)
    cmd = pp.compute(robot=Pose2D(0.0, 0.0, 0.0), path=[])
    assert cmd.linear == 0.0
    assert cmd.angular == 0.0


def test_command_clipped_to_limits():
    pp = PurePursuit(lookahead=0.05, max_v=0.4, max_w=1.0)  # tiny lookahead -> high curvature
    path = [Waypoint(x=0.05, y=0.05)]
    cmd = pp.compute(robot=Pose2D(0.0, 0.0, 0.0), path=path)
    assert cmd.linear <= 0.4
    assert abs(cmd.angular) <= 1.0
```

- [ ] **Step 9.2: Confirm test fails**

```bash
cd /home/nop/dev/mywork/raspicat-async-vla/src/raspicat_async_vla_edge
python3 -m pytest test/test_pure_pursuit.py -v
```

Expected: ImportError.

- [ ] **Step 9.3: Implement `pure_pursuit.py`**

```python
"""Pure Pursuit path follower."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass
class Pose2D:
    x: float
    y: float
    theta: float  # radians


@dataclass
class Waypoint:
    x: float
    y: float


@dataclass
class TwistCmd:
    linear: float   # m/s
    angular: float  # rad/s


class PurePursuit:
    """Minimal Pure Pursuit controller.

    Picks the first waypoint farther than `lookahead` along the path,
    or the last waypoint if none qualify. If the picked target is behind
    the robot and `no_backward` is true, command zero linear velocity
    and rotate toward it.
    """

    def __init__(
        self,
        *,
        lookahead: float,
        max_v: float,
        max_w: float,
        no_backward: bool = True,
        kw: float = 1.5,
    ) -> None:
        self.lookahead = lookahead
        self.max_v = max_v
        self.max_w = max_w
        self.no_backward = no_backward
        self.kw = kw

    def compute(
        self, *, robot: Pose2D, path: Sequence[Waypoint],
    ) -> TwistCmd:
        if not path:
            return TwistCmd(0.0, 0.0)

        # Pick lookahead target in robot frame
        target = path[-1]
        for wp in path:
            if math.hypot(wp.x - robot.x, wp.y - robot.y) >= self.lookahead:
                target = wp
                break

        dx = target.x - robot.x
        dy = target.y - robot.y
        cos_t = math.cos(robot.theta)
        sin_t = math.sin(robot.theta)
        x_local = cos_t * dx + sin_t * dy
        y_local = -sin_t * dx + cos_t * dy

        if x_local <= 0.0 and self.no_backward:
            # Target is behind: rotate toward it without moving forward.
            heading_err = math.atan2(y_local, x_local)
            angular = max(-self.max_w, min(self.max_w, self.kw * heading_err))
            return TwistCmd(linear=0.0, angular=angular)

        l_sq = x_local * x_local + y_local * y_local
        if l_sq < 1e-9:
            return TwistCmd(0.0, 0.0)
        curvature = 2.0 * y_local / l_sq

        linear = min(self.max_v, max(0.0, x_local))
        angular = linear * curvature

        if abs(angular) > self.max_w:
            angular = math.copysign(self.max_w, angular)

        return TwistCmd(linear=linear, angular=angular)
```

- [ ] **Step 9.4: Confirm tests pass**

```bash
python3 -m pytest test/test_pure_pursuit.py -v
```

Expected: 5 passed.

- [ ] **Step 9.5: Commit**

```bash
cd /home/nop/dev/mywork/raspicat-async-vla
git add src/raspicat_async_vla_edge/asyncvla_edge/pure_pursuit.py \
        src/raspicat_async_vla_edge/test/test_pure_pursuit.py
git commit -m "feat(edge): add Pure Pursuit controller with TDD"
```

---

## Task 10: Dummy Remote gRPC server (TDD)

**Files:**
- Create: `src/raspicat_async_vla_remote/package.xml`
- Create: `src/raspicat_async_vla_remote/setup.py`
- Create: `src/raspicat_async_vla_remote/setup.cfg`
- Create: `src/raspicat_async_vla_remote/resource/raspicat_async_vla_remote`
- Create: `src/raspicat_async_vla_remote/asyncvla_remote/__init__.py`
- Create: `src/raspicat_async_vla_remote/asyncvla_remote/dummy_server.py`
- Create: `src/raspicat_async_vla_remote/asyncvla_remote/server_main.py`
- Create: `src/raspicat_async_vla_remote/config/remote_params.yaml`
- Create: `src/raspicat_async_vla_remote/test/test_dummy_server.py`

- [ ] **Step 10.1: Scaffold the package**

`src/raspicat_async_vla_remote/package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>raspicat_async_vla_remote</name>
  <version>0.1.0</version>
  <description>gRPC server for AsyncVLA (dummy in Plan 1, real model in Plan 2).</description>
  <maintainer email="nop@example.com">nop</maintainer>
  <license>MIT</license>

  <buildtool_depend>ament_python</buildtool_depend>

  <exec_depend>raspicat_async_vla_proto</exec_depend>

  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

`src/raspicat_async_vla_remote/setup.py`:

```python
from setuptools import setup
import os
from glob import glob

package_name = 'raspicat_async_vla_remote'

setup(
    name=package_name,
    version='0.1.0',
    packages=['asyncvla_remote'],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy', 'grpcio>=1.50'],
    zip_safe=True,
    maintainer='nop',
    maintainer_email='nop@example.com',
    description='AsyncVLA remote gRPC server.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'asyncvla_dummy_server = asyncvla_remote.server_main:main',
        ],
    },
)
```

`src/raspicat_async_vla_remote/setup.cfg`:

```
[develop]
script_dir=$base/lib/raspicat_async_vla_remote
[install]
install_scripts=$base/lib/raspicat_async_vla_remote
```

```bash
cd /home/nop/dev/mywork/raspicat-async-vla
mkdir -p src/raspicat_async_vla_remote/{asyncvla_remote,resource,config,test}
touch src/raspicat_async_vla_remote/resource/raspicat_async_vla_remote
touch src/raspicat_async_vla_remote/asyncvla_remote/__init__.py
```

`src/raspicat_async_vla_remote/config/remote_params.yaml`:

```yaml
server:
  host: 0.0.0.0
  port: 50051
  max_concurrent_streams: 4

dummy:
  num_tokens: 8
  embed_dim: 1024
  inference_ms: 50.0      # simulated latency added before reply
  model_version: "dummy-v1"
```

- [ ] **Step 10.2: Write the failing test**

`src/raspicat_async_vla_remote/test/test_dummy_server.py`:

```python
"""Integration test: the DummyServer answers a StreamInfer with deterministic embedding."""
import threading
import time

import grpc
import numpy as np
import pytest

from raspicat_async_vla_proto import asyncvla_pb2, asyncvla_pb2_grpc
from raspicat_async_vla_proto.conversions import fp16_bytes_to_float32_list
from asyncvla_remote.dummy_server import DummyServer


@pytest.fixture
def server_address():
    server = DummyServer(
        host='localhost',
        port=0,                # let OS pick a port
        num_tokens=8,
        embed_dim=16,
        inference_ms=5.0,
        model_version='dummy-test',
    )
    actual_port = server.start()
    addr = f'localhost:{actual_port}'
    yield addr
    server.stop(grace_sec=0.5)


def test_get_model_info_reports_ready(server_address):
    with grpc.insecure_channel(server_address) as ch:
        stub = asyncvla_pb2_grpc.AsyncVLAServiceStub(ch)
        info = stub.GetModelInfo(asyncvla_pb2.ModelInfoRequest(), timeout=2.0)
        assert info.ready is True
        assert info.num_tokens == 8
        assert info.embed_dim == 16


def test_stream_infer_round_trip(server_address):
    with grpc.insecure_channel(server_address) as ch:
        stub = asyncvla_pb2_grpc.AsyncVLAServiceStub(ch)

        def gen():
            for i in range(3):
                yield asyncvla_pb2.Observation(
                    frame_id=i,
                    capture_time_ns=time.monotonic_ns(),
                    image_jpeg=b'\xff\xd8\xff' + b'\x00' * 32,
                    image_width=224,
                    image_height=224,
                    goal=asyncvla_pb2.GoalSpec(
                        mode=asyncvla_pb2.GoalSpec.POSE,
                        pose=asyncvla_pb2.Pose2D(x=1.0, y=0.0, theta=0.0),
                        frame_id='base_link',
                    ),
                )

        replies = list(stub.StreamInfer(gen(), timeout=5.0))
        assert len(replies) == 3
        assert {r.frame_id for r in replies} == {0, 1, 2}
        for r in replies:
            assert r.num_tokens == 8
            assert r.embed_dim == 16
            arr = np.array(fp16_bytes_to_float32_list(r.embedding_fp16), dtype=np.float32)
            assert arr.shape == (8 * 16,)
```

- [ ] **Step 10.3: Confirm test fails**

```bash
cd /home/nop/dev/mywork/raspicat-async-vla
source /opt/ros/humble/setup.bash
colcon build --packages-select raspicat_async_vla_remote
source install/setup.bash
pytest src/raspicat_async_vla_remote/test/test_dummy_server.py -v
```

Expected: ImportError on `DummyServer` (or build failure if package not yet registered).

- [ ] **Step 10.4: Implement `dummy_server.py`**

```python
"""Dummy gRPC server returning deterministic embeddings (no ML model)."""
from __future__ import annotations

import logging
import threading
import time
from concurrent import futures
from typing import Iterator, Optional

import grpc
import numpy as np

from raspicat_async_vla_proto import asyncvla_pb2, asyncvla_pb2_grpc
from raspicat_async_vla_proto.conversions import float32_array_to_fp16_bytes


_LOG = logging.getLogger(__name__)


class _Servicer(asyncvla_pb2_grpc.AsyncVLAServiceServicer):
    def __init__(
        self,
        *,
        num_tokens: int,
        embed_dim: int,
        inference_ms: float,
        model_version: str,
    ) -> None:
        self._num_tokens = num_tokens
        self._embed_dim = embed_dim
        self._inference_ms = inference_ms
        self._model_version = model_version

    def _embedding_for(self, frame_id: int) -> bytes:
        # Deterministic: every element = sin(frame_id * pi / 17) so it varies but is reproducible.
        seed = float(np.sin(frame_id * np.pi / 17))
        arr = np.full(self._num_tokens * self._embed_dim, seed, dtype=np.float32)
        return float32_array_to_fp16_bytes(arr)

    def GetModelInfo(self, request, context):
        return asyncvla_pb2.ModelInfo(
            model_name='dummy',
            model_version=self._model_version,
            num_tokens=self._num_tokens,
            embed_dim=self._embed_dim,
            device='cpu',
            ready=True,
        )

    def StreamInfer(
        self,
        request_iterator: Iterator[asyncvla_pb2.Observation],
        context,
    ) -> Iterator[asyncvla_pb2.ActionEmbedding]:
        for obs in request_iterator:
            if self._inference_ms > 0:
                time.sleep(self._inference_ms / 1000.0)
            yield asyncvla_pb2.ActionEmbedding(
                frame_id=obs.frame_id,
                server_time_ns=time.monotonic_ns(),
                num_tokens=self._num_tokens,
                embed_dim=self._embed_dim,
                embedding_fp16=self._embedding_for(obs.frame_id),
                inference_ms=self._inference_ms,
                model_version=self._model_version,
            )


class DummyServer:
    """Process-local dummy server. Useful for tests and Plan 1 integration."""

    def __init__(
        self,
        *,
        host: str = '0.0.0.0',
        port: int = 50051,
        num_tokens: int = 8,
        embed_dim: int = 1024,
        inference_ms: float = 50.0,
        model_version: str = 'dummy-v1',
        max_workers: int = 4,
    ) -> None:
        self._host = host
        self._port = port
        self._max_workers = max_workers
        self._servicer = _Servicer(
            num_tokens=num_tokens,
            embed_dim=embed_dim,
            inference_ms=inference_ms,
            model_version=model_version,
        )
        self._server: Optional[grpc.Server] = None
        self._actual_port: Optional[int] = None

    def start(self) -> int:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=self._max_workers))
        asyncvla_pb2_grpc.add_AsyncVLAServiceServicer_to_server(self._servicer, server)
        bind = f'{self._host}:{self._port}'
        self._actual_port = server.add_insecure_port(bind)
        server.start()
        self._server = server
        _LOG.info('DummyServer listening on %s:%d', self._host, self._actual_port)
        return self._actual_port

    def wait_for_termination(self) -> None:
        if self._server is None:
            return
        self._server.wait_for_termination()

    def stop(self, grace_sec: float = 1.0) -> None:
        if self._server is not None:
            self._server.stop(grace_sec)
            self._server = None
```

- [ ] **Step 10.5: Implement `server_main.py` entrypoint**

```python
"""Entry point for `asyncvla_dummy_server` console script."""
from __future__ import annotations

import argparse
import logging
import signal

from .dummy_server import DummyServer


def main() -> None:
    parser = argparse.ArgumentParser(description='AsyncVLA dummy gRPC server')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=50051)
    parser.add_argument('--num-tokens', type=int, default=8)
    parser.add_argument('--embed-dim', type=int, default=1024)
    parser.add_argument('--inference-ms', type=float, default=50.0)
    parser.add_argument('--model-version', default='dummy-v1')
    parser.add_argument('--log-level', default='INFO')
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    server = DummyServer(
        host=args.host,
        port=args.port,
        num_tokens=args.num_tokens,
        embed_dim=args.embed_dim,
        inference_ms=args.inference_ms,
        model_version=args.model_version,
    )
    port = server.start()
    logging.info('listening on %s:%d', args.host, port)

    def _sigterm(signum, frame):  # noqa: ARG001
        logging.info('SIGTERM received, stopping...')
        server.stop(grace_sec=1.0)

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)
    server.wait_for_termination()
```

- [ ] **Step 10.6: Build, source, run tests**

```bash
cd /home/nop/dev/mywork/raspicat-async-vla
source /opt/ros/humble/setup.bash
colcon build --packages-select raspicat_async_vla_remote
source install/setup.bash
pytest src/raspicat_async_vla_remote/test/test_dummy_server.py -v
```

Expected: 2 passed.

- [ ] **Step 10.7: Commit**

```bash
git add src/raspicat_async_vla_remote
git commit -m "feat(remote): add dummy gRPC server with TDD"
```

---

## Task 11: gRPC client wrapper (TDD against DummyServer)

**Files:**
- Create: `src/raspicat_async_vla_edge/asyncvla_edge/grpc_client.py`
- Create: `src/raspicat_async_vla_edge/test/test_grpc_client.py`

- [ ] **Step 11.1: Write the failing test**

```python
"""Tests for the AsyncVLA gRPC bidi-stream client wrapper.

The client must:
  - send observations from a thread-safe input
  - deliver embeddings via a callback
  - be startable / stoppable cleanly
  - tolerate the server going away (reconnect attempt)
"""
import threading
import time

import numpy as np
import pytest

from asyncvla_remote.dummy_server import DummyServer
from asyncvla_edge.grpc_client import AsyncVLAClient
from raspicat_async_vla_proto import asyncvla_pb2


@pytest.fixture
def server():
    s = DummyServer(host='localhost', port=0, num_tokens=4, embed_dim=8, inference_ms=1.0)
    port = s.start()
    yield port
    s.stop(grace_sec=0.5)


def _make_obs(frame_id: int) -> asyncvla_pb2.Observation:
    return asyncvla_pb2.Observation(
        frame_id=frame_id,
        capture_time_ns=time.monotonic_ns(),
        image_jpeg=b'\xff\xd8\xff' + b'\x00' * 16,
        image_width=224,
        image_height=224,
        goal=asyncvla_pb2.GoalSpec(
            mode=asyncvla_pb2.GoalSpec.POSE,
            pose=asyncvla_pb2.Pose2D(x=0.0, y=0.0, theta=0.0),
            frame_id='base_link',
        ),
    )


def test_client_round_trips_via_dummy_server(server):
    received = []
    cond = threading.Condition()

    def on_emb(emb):
        with cond:
            received.append(emb)
            cond.notify_all()

    client = AsyncVLAClient(address=f'localhost:{server}', on_embedding=on_emb)
    client.start()
    try:
        for i in range(5):
            client.send(_make_obs(i))
        with cond:
            cond.wait_for(lambda: len(received) >= 5, timeout=5.0)
        assert len(received) == 5
        assert {r.frame_id for r in received} == {0, 1, 2, 3, 4}
    finally:
        client.stop()
```

- [ ] **Step 11.2: Confirm test fails**

```bash
cd /home/nop/dev/mywork/raspicat-async-vla/src/raspicat_async_vla_edge
python3 -m pytest test/test_grpc_client.py -v
```

Expected: ImportError.

- [ ] **Step 11.3: Implement `grpc_client.py`**

```python
"""Threaded gRPC bidi-stream client for AsyncVLA."""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Optional

import grpc

from raspicat_async_vla_proto import asyncvla_pb2, asyncvla_pb2_grpc


_LOG = logging.getLogger(__name__)

EmbeddingCallback = Callable[[asyncvla_pb2.ActionEmbedding], None]


class AsyncVLAClient:
    """Threaded bidirectional gRPC client.

    Two threads run between start() and stop():
      - sender: pulls Observation from the queue and yields them to gRPC
      - receiver: iterates the gRPC reply stream and calls on_embedding(...)

    On stream failure the client reconnects with exponential backoff up to
    `max_backoff_sec`. While disconnected, queued observations are dropped
    if the queue exceeds `queue_max`.
    """

    def __init__(
        self,
        *,
        address: str,
        on_embedding: EmbeddingCallback,
        queue_max: int = 32,
        initial_backoff_sec: float = 0.5,
        max_backoff_sec: float = 5.0,
    ) -> None:
        self._address = address
        self._on_embedding = on_embedding
        self._queue: 'queue.Queue[asyncvla_pb2.Observation]' = queue.Queue(maxsize=queue_max)
        self._sentinel = object()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._initial_backoff = initial_backoff_sec
        self._max_backoff = max_backoff_sec

    # ------------------------------------------------------------------ public

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name='AsyncVLAClient', daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(self._sentinel)  # type: ignore[arg-type]
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def send(self, obs: asyncvla_pb2.Observation) -> bool:
        try:
            self._queue.put_nowait(obs)
            return True
        except queue.Full:
            _LOG.warning('observation queue full; dropping frame_id=%s', obs.frame_id)
            return False

    # ------------------------------------------------------------------ internal

    def _run(self) -> None:
        backoff = self._initial_backoff
        while not self._stop_event.is_set():
            try:
                channel = grpc.insecure_channel(self._address)
                stub = asyncvla_pb2_grpc.AsyncVLAServiceStub(channel)
                _LOG.info('connecting to %s', self._address)
                self._run_stream(stub)
                channel.close()
                backoff = self._initial_backoff
            except grpc.RpcError as exc:
                _LOG.warning('gRPC error: %s; backing off %.2fs', exc, backoff)
                self._stop_event.wait(timeout=backoff)
                backoff = min(backoff * 2.0, self._max_backoff)
            except Exception as exc:  # noqa: BLE001
                _LOG.exception('unexpected client error: %s', exc)
                self._stop_event.wait(timeout=backoff)
                backoff = min(backoff * 2.0, self._max_backoff)

    def _request_iter(self):
        while not self._stop_event.is_set():
            item = self._queue.get()
            if item is self._sentinel:
                return
            yield item

    def _run_stream(self, stub: asyncvla_pb2_grpc.AsyncVLAServiceStub) -> None:
        replies = stub.StreamInfer(self._request_iter())
        for reply in replies:
            try:
                self._on_embedding(reply)
            except Exception:  # noqa: BLE001
                _LOG.exception('on_embedding callback raised')
            if self._stop_event.is_set():
                return
```

- [ ] **Step 11.4: Confirm test passes**

```bash
cd /home/nop/dev/mywork/raspicat-async-vla
source install/setup.bash
pytest src/raspicat_async_vla_edge/test/test_grpc_client.py -v
```

Expected: 1 passed (may take ~1s).

- [ ] **Step 11.5: Commit**

```bash
git add src/raspicat_async_vla_edge/asyncvla_edge/grpc_client.py \
        src/raspicat_async_vla_edge/test/test_grpc_client.py
git commit -m "feat(edge): add threaded gRPC bidi client with TDD"
```

---

## Task 12: Lifecycle edge node skeleton (with stub adapter producing fixed straight-ahead Path)

**Goal:** Get a `LifecycleNode` that subscribes to camera + goal, runs the three async loops, talks to the dummy server, and publishes a Path. The "edge adapter" here is a stub that ignores the embedding and emits a fixed forward path — Plan 2 will replace it with the real adapter.

**Files:**
- Create: `src/raspicat_async_vla_edge/asyncvla_edge/edge_node.py`
- Create: `src/raspicat_async_vla_edge/launch/edge_only.launch.py`
- Create: `src/raspicat_async_vla_edge/config/edge_params.yaml`
- Create: `src/raspicat_async_vla_edge/test/test_edge_node_smoke.py`

- [ ] **Step 12.1: Write `config/edge_params.yaml`**

```yaml
asyncvla_edge_node:
  ros__parameters:
    remote_address: "localhost:50051"
    obs_publish_rate_hz: 2.0
    action_rate_hz: 10.0
    image_size: [224, 224]
    jpeg_quality: 85
    embedding_max_age_sec: 6.0
    embedding_hard_timeout_sec: 15.0
    goal_tolerance_m: 0.3
    image_topic: "/camera/image_raw"
    goal_topic: "/asyncvla/goal"
    path_topic: "/asyncvla/predicted_path"
    status_topic: "/asyncvla/status"
    embedding_debug_topic: "/asyncvla/embedding"
    publish_embedding_debug: true
```

- [ ] **Step 12.2: Implement `edge_node.py` with stub adapter**

```python
"""AsyncVLA edge LifecycleNode — Plan 1 skeleton.

In Plan 1 the "edge adapter" is a stub that emits a fixed straight-ahead
path of length 1.0 m sampled at 0.1 m. Plan 2 replaces this stub with the
real Edge Adapter PyTorch model.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State
from sensor_msgs.msg import Image

from raspicat_async_vla_msgs.msg import (
    ActionEmbedding as ActionEmbeddingMsg,
    GoalSpec as GoalSpecMsg,
)
from raspicat_async_vla_proto import asyncvla_pb2
from raspicat_async_vla_proto.conversions import (
    fp16_bytes_to_float32_list,
    float32_array_to_fp16_bytes,
)

from .preprocess import resize_and_jpeg
from .embedding_cache import EmbeddingCache, CachedEmbedding
from .grpc_client import AsyncVLAClient


def _ros_goal_to_proto(goal: GoalSpecMsg) -> asyncvla_pb2.GoalSpec:
    if goal.mode == GoalSpecMsg.MODE_POSE:
        return asyncvla_pb2.GoalSpec(
            mode=asyncvla_pb2.GoalSpec.POSE,
            pose=asyncvla_pb2.Pose2D(
                x=goal.pose.pose.position.x,
                y=goal.pose.pose.position.y,
                theta=0.0,  # extracting yaw is done in Plan 2 with tf
            ),
            frame_id=goal.pose.header.frame_id or 'odom',
        )
    if goal.mode == GoalSpecMsg.MODE_TEXT:
        return asyncvla_pb2.GoalSpec(
            mode=asyncvla_pb2.GoalSpec.TEXT, text=goal.text, frame_id='',
        )
    if goal.mode == GoalSpecMsg.MODE_IMAGE:
        return asyncvla_pb2.GoalSpec(
            mode=asyncvla_pb2.GoalSpec.IMAGE,
            image_jpeg=bytes(goal.image.data),
            frame_id='',
        )
    raise ValueError(f'unknown goal mode {goal.mode}')


def _stub_adapter_to_path(
    embedding: Optional[CachedEmbedding],
    *,
    n_pts: int = 10,
    step_m: float = 0.1,
    frame: str = 'base_link',
) -> Path:
    """Plan 1 stub: ignore embedding contents, emit straight-ahead path."""
    path = Path()
    path.header.frame_id = frame
    for i in range(1, n_pts + 1):
        ps = PoseStamped()
        ps.header.frame_id = frame
        ps.pose.position.x = i * step_m
        ps.pose.position.y = 0.0
        ps.pose.orientation.w = 1.0
        path.poses.append(ps)
    return path


class AsyncVLAEdgeNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__('asyncvla_edge_node')
        self._declare_parameters()
        self._bridge = CvBridge()
        self._latest_image: Optional[np.ndarray] = None
        self._latest_image_lock = threading.Lock()
        self._latest_goal: Optional[GoalSpecMsg] = None
        self._latest_goal_lock = threading.Lock()
        self._cache: Optional[EmbeddingCache] = None
        self._client: Optional[AsyncVLAClient] = None
        self._frame_counter = 0
        self._send_timer = None
        self._action_timer = None
        self._status_timer = None
        self._image_sub = None
        self._goal_sub = None
        self._path_pub = None
        self._embedding_pub = None
        self._status_pub = None

    # ----------------------------------------------------------------- params

    def _declare_parameters(self) -> None:
        self.declare_parameter('remote_address', 'localhost:50051')
        self.declare_parameter('obs_publish_rate_hz', 2.0)
        self.declare_parameter('action_rate_hz', 10.0)
        self.declare_parameter('image_size', [224, 224])
        self.declare_parameter('jpeg_quality', 85)
        self.declare_parameter('embedding_max_age_sec', 6.0)
        self.declare_parameter('embedding_hard_timeout_sec', 15.0)
        self.declare_parameter('goal_tolerance_m', 0.3)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('goal_topic', '/asyncvla/goal')
        self.declare_parameter('path_topic', '/asyncvla/predicted_path')
        self.declare_parameter('status_topic', '/asyncvla/status')
        self.declare_parameter('embedding_debug_topic', '/asyncvla/embedding')
        self.declare_parameter('publish_embedding_debug', True)

    # ------------------------------------------------------------- lifecycle

    def on_configure(self, state: State) -> TransitionCallbackReturn:  # noqa: ARG002
        self.get_logger().info('on_configure')
        addr = self.get_parameter('remote_address').get_parameter_value().string_value
        max_age = self.get_parameter('embedding_max_age_sec').value
        hard = self.get_parameter('embedding_hard_timeout_sec').value
        self._cache = EmbeddingCache(max_age_sec=float(max_age), hard_timeout_sec=float(hard))
        self._client = AsyncVLAClient(address=addr, on_embedding=self._on_embedding_received)

        image_topic = self.get_parameter('image_topic').value
        goal_topic = self.get_parameter('goal_topic').value
        path_topic = self.get_parameter('path_topic').value
        status_topic = self.get_parameter('status_topic').value
        emb_topic = self.get_parameter('embedding_debug_topic').value

        self._image_sub = self.create_subscription(
            Image, image_topic, self._on_image, 10,
        )
        self._goal_sub = self.create_subscription(
            GoalSpecMsg, goal_topic, self._on_goal, 1,
        )
        self._path_pub = self.create_publisher(Path, path_topic, 10)
        self._status_pub = self.create_publisher(DiagnosticArray, status_topic, 10)
        if self.get_parameter('publish_embedding_debug').value:
            self._embedding_pub = self.create_publisher(ActionEmbeddingMsg, emb_topic, 10)

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:  # noqa: ARG002
        self.get_logger().info('on_activate')
        assert self._client is not None
        self._client.start()
        obs_rate = float(self.get_parameter('obs_publish_rate_hz').value)
        act_rate = float(self.get_parameter('action_rate_hz').value)
        self._send_timer = self.create_timer(1.0 / obs_rate, self._send_observation_tick)
        self._action_timer = self.create_timer(1.0 / act_rate, self._action_tick)
        self._status_timer = self.create_timer(1.0, self._publish_status)
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:  # noqa: ARG002
        self.get_logger().info('on_deactivate')
        for t in (self._send_timer, self._action_timer, self._status_timer):
            if t is not None:
                t.cancel()
        self._send_timer = self._action_timer = self._status_timer = None
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:  # noqa: ARG002
        self.get_logger().info('on_cleanup')
        if self._client is not None:
            self._client.stop()
        self._client = None
        self._cache = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:  # noqa: ARG002
        self.get_logger().info('on_shutdown')
        if self._client is not None:
            self._client.stop()
        return TransitionCallbackReturn.SUCCESS

    # ----------------------------------------------------------- subscribers

    def _on_image(self, msg: Image) -> None:
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'cv_bridge failed: {exc}')
            return
        with self._latest_image_lock:
            self._latest_image = cv_img

    def _on_goal(self, msg: GoalSpecMsg) -> None:
        self.get_logger().info(f'received goal mode={msg.mode}')
        with self._latest_goal_lock:
            self._latest_goal = msg
        if self._cache is not None:
            self._cache.invalidate()

    # ------------------------------------------------------------ tick: send

    def _send_observation_tick(self) -> None:
        if self._client is None:
            return
        with self._latest_image_lock:
            img = None if self._latest_image is None else self._latest_image.copy()
        with self._latest_goal_lock:
            goal = self._latest_goal
        if img is None or goal is None:
            return
        size = self.get_parameter('image_size').value
        quality = int(self.get_parameter('jpeg_quality').value)
        try:
            jpeg, w, h = resize_and_jpeg(img, target=(int(size[0]), int(size[1])), quality=quality)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'preprocess failed: {exc}')
            return
        try:
            proto_goal = _ros_goal_to_proto(goal)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'goal conversion failed: {exc}')
            return

        self._frame_counter += 1
        obs = asyncvla_pb2.Observation(
            frame_id=self._frame_counter,
            capture_time_ns=time.monotonic_ns(),
            image_jpeg=jpeg,
            image_width=w,
            image_height=h,
            goal=proto_goal,
        )
        self._client.send(obs)

    # -------------------------------------------------- callback: embeddings

    def _on_embedding_received(self, proto_emb: asyncvla_pb2.ActionEmbedding) -> None:
        if self._cache is None:
            return
        arr = np.array(fp16_bytes_to_float32_list(proto_emb.embedding_fp16), dtype=np.float32)
        cached = CachedEmbedding(
            frame_id=proto_emb.frame_id,
            recv_time_ns=time.monotonic_ns(),
            embedding=arr,
            num_tokens=proto_emb.num_tokens,
            embed_dim=proto_emb.embed_dim,
            inference_ms=float(proto_emb.inference_ms),
            model_version=proto_emb.model_version,
        )
        self._cache.put(cached)
        if self._embedding_pub is not None:
            from raspicat_async_vla_proto.conversions import proto_action_embedding_to_msg
            ros_msg = proto_action_embedding_to_msg(proto_emb)
            ros_msg.header.stamp = self.get_clock().now().to_msg()
            self._embedding_pub.publish(ros_msg)

    # ---------------------------------------------------------- tick: action

    def _action_tick(self) -> None:
        """Publish a Path. Plan 1 stub: straight-ahead path of 1.0 m.

        Status-aware: WAITING_REMOTE / STALE → publish empty path so the
        follower outputs zero Twist (safe-stop). DEGRADED is treated as
        usable but logged.
        """
        if self._cache is None or self._path_pub is None:
            return
        status = self._cache.status()
        path = Path()
        path.header.frame_id = 'base_link'
        path.header.stamp = self.get_clock().now().to_msg()

        if status in (EmbeddingCache.STATUS_WAITING, EmbeddingCache.STATUS_STALE):
            # Empty path → follower emits zero Twist (safe-stop).
            self._path_pub.publish(path)
            return

        if status == EmbeddingCache.STATUS_DEGRADED:
            self.get_logger().warn('embedding age over max_age; running degraded')

        emb = self._cache.get_latest_raw()  # OK or DEGRADED
        path = _stub_adapter_to_path(emb)
        path.header.stamp = self.get_clock().now().to_msg()
        self._path_pub.publish(path)

    # ----------------------------------------------------------- tick: status

    def _publish_status(self) -> None:
        if self._cache is None or self._status_pub is None:
            return
        status_str = self._cache.status()
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        ds = DiagnosticStatus()
        ds.name = 'asyncvla_edge'
        ds.message = status_str
        ds.level = (
            DiagnosticStatus.OK
            if status_str == 'OK'
            else DiagnosticStatus.WARN
            if status_str in ('DEGRADED', 'WAITING_REMOTE')
            else DiagnosticStatus.ERROR
        )
        ds.values.append(KeyValue(key='frame_counter', value=str(self._frame_counter)))
        msg.status.append(ds)
        self._status_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = AsyncVLAEdgeNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

- [ ] **Step 12.3: Write `launch/edge_only.launch.py`**

```python
"""Launch the AsyncVLA edge lifecycle node (auto-transitions to active)."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.events import matches_action
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('raspicat_async_vla_edge'),
        'config', 'edge_params.yaml',
    )
    edge = LifecycleNode(
        package='raspicat_async_vla_edge',
        executable='asyncvla_edge_node',
        name='asyncvla_edge_node',
        namespace='',
        output='screen',
        parameters=[config],
    )
    configure = EmitEvent(event=ChangeState(
        lifecycle_node_matcher=matches_action(edge),
        transition_id=Transition.TRANSITION_CONFIGURE,
    ))
    activate = EmitEvent(event=ChangeState(
        lifecycle_node_matcher=matches_action(edge),
        transition_id=Transition.TRANSITION_ACTIVATE,
    ))
    on_started = RegisterEventHandler(
        OnProcessStart(target_action=edge, on_start=[configure]),
    )
    on_inactive = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=edge,
            goal_state='inactive',
            entities=[activate],
        ),
    )
    return LaunchDescription([edge, on_started, on_inactive])
```

- [ ] **Step 12.4: Smoke test (without launch — direct node spin)**

`src/raspicat_async_vla_edge/test/test_edge_node_smoke.py`:

```python
"""Smoke test: bring up the edge node connected to a DummyServer and verify
that a Path is published within a short timeout."""
import threading
import time

import pytest
import rclpy
from nav_msgs.msg import Path
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
import numpy as np

from asyncvla_remote.dummy_server import DummyServer
from asyncvla_edge.edge_node import AsyncVLAEdgeNode
from raspicat_async_vla_msgs.msg import GoalSpec as GoalSpecMsg
from geometry_msgs.msg import PoseStamped


@pytest.fixture(scope='module')
def ros_runtime():
    rclpy.init()
    yield
    rclpy.shutdown()


def _make_dummy_image_msg() -> Image:
    msg = Image()
    msg.height = 240
    msg.width = 320
    msg.encoding = 'rgb8'
    msg.is_bigendian = 0
    msg.step = 320 * 3
    msg.data = (np.zeros((240, 320, 3), dtype=np.uint8)).tobytes()
    return msg


def test_edge_node_publishes_path(ros_runtime):
    server = DummyServer(host='localhost', port=0, num_tokens=4, embed_dim=8, inference_ms=1.0)
    port = server.start()
    try:
        node = AsyncVLAEdgeNode()
        node.set_parameters([
            rclpy.parameter.Parameter('remote_address', value=f'localhost:{port}'),
            rclpy.parameter.Parameter('obs_publish_rate_hz', value=10.0),
            rclpy.parameter.Parameter('action_rate_hz', value=20.0),
            rclpy.parameter.Parameter('embedding_max_age_sec', value=6.0),
            rclpy.parameter.Parameter('embedding_hard_timeout_sec', value=15.0),
        ])
        # configure -> activate
        node.trigger_configure()
        node.trigger_activate()

        # External publisher to push goal + image
        pub_node = rclpy.create_node('test_pub')
        goal_pub = pub_node.create_publisher(GoalSpecMsg, '/asyncvla/goal', 1)
        img_pub = pub_node.create_publisher(Image, '/camera/image_raw', 1)

        received_paths = []
        path_node = rclpy.create_node('test_sub')
        path_node.create_subscription(Path, '/asyncvla/predicted_path',
                                      lambda m: received_paths.append(m), 10)

        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.add_node(pub_node)
        executor.add_node(path_node)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        # Send goal once and image periodically
        goal = GoalSpecMsg()
        goal.mode = GoalSpecMsg.MODE_POSE
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'odom'
        goal.pose.pose.position.x = 1.0
        goal_pub.publish(goal)

        deadline = time.time() + 5.0
        while time.time() < deadline and not received_paths:
            img_pub.publish(_make_dummy_image_msg())
            time.sleep(0.05)

        executor.shutdown(timeout_sec=1.0)
        node.trigger_deactivate()
        node.trigger_cleanup()
        node.destroy_node()
        pub_node.destroy_node()
        path_node.destroy_node()

        assert received_paths, 'no Path was published within 5s'
    finally:
        server.stop(grace_sec=0.5)
```

- [ ] **Step 12.5: Build and run smoke test**

```bash
cd /home/nop/dev/mywork/raspicat-async-vla
source /opt/ros/humble/setup.bash
colcon build --packages-select raspicat_async_vla_edge
source install/setup.bash
pytest src/raspicat_async_vla_edge/test/test_edge_node_smoke.py -v
```

Expected: 1 passed (may take ~3–5s).

- [ ] **Step 12.6: Commit**

```bash
git add src/raspicat_async_vla_edge/asyncvla_edge/edge_node.py \
        src/raspicat_async_vla_edge/launch/edge_only.launch.py \
        src/raspicat_async_vla_edge/config/edge_params.yaml \
        src/raspicat_async_vla_edge/test/test_edge_node_smoke.py
git commit -m "feat(edge): add lifecycle edge node with stub adapter and smoke test"
```

---

## Task 13: Path follower node (Pure Pursuit ROS2 wrapper)

**Files:**
- Create: `src/raspicat_async_vla_edge/asyncvla_edge/path_follower_node.py`
- Create: `src/raspicat_async_vla_edge/launch/path_follower.launch.py`

(Path-follower **logic** is already TDD-tested in Task 9; this task is a thin ROS2 wrapper, so we test by inspection in Task 14's local-integration script.)

- [ ] **Step 13.1: Implement `path_follower_node.py`**

```python
"""ROS2 wrapper around PurePursuit: subscribe to Path, publish Twist."""
from __future__ import annotations

import math
from typing import List

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from rclpy.node import Node

from .pure_pursuit import PurePursuit, Pose2D, Waypoint


class PathFollowerNode(Node):

    def __init__(self) -> None:
        super().__init__('path_follower_node')
        self.declare_parameter('lookahead', 0.4)
        self.declare_parameter('max_v', 0.4)
        self.declare_parameter('max_w', 1.0)
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('path_topic', '/asyncvla/predicted_path')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('path_in_robot_frame', True)

        self._pp = PurePursuit(
            lookahead=float(self.get_parameter('lookahead').value),
            max_v=float(self.get_parameter('max_v').value),
            max_w=float(self.get_parameter('max_w').value),
            no_backward=True,
        )
        self._latest: List[Waypoint] = []
        self._sub = self.create_subscription(
            Path,
            self.get_parameter('path_topic').value,
            self._on_path, 10,
        )
        self._pub = self.create_publisher(
            Twist, self.get_parameter('cmd_vel_topic').value, 10,
        )
        rate = float(self.get_parameter('rate_hz').value)
        self._timer = self.create_timer(1.0 / rate, self._tick)

    def _on_path(self, msg: Path) -> None:
        wps: List[Waypoint] = []
        for ps in msg.poses:
            wps.append(Waypoint(x=ps.pose.position.x, y=ps.pose.position.y))
        self._latest = wps

    def _tick(self) -> None:
        # Plan 1 simplification: assume path is in robot frame so the robot pose is origin.
        cmd = self._pp.compute(robot=Pose2D(0.0, 0.0, 0.0), path=self._latest)
        twist = Twist()
        twist.linear.x = float(cmd.linear)
        twist.angular.z = float(cmd.angular)
        self._pub.publish(twist)


def main() -> None:
    rclpy.init()
    node = PathFollowerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

- [ ] **Step 13.2: Write `launch/path_follower.launch.py`**

```python
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='raspicat_async_vla_edge',
            executable='path_follower_node',
            name='path_follower_node',
            output='screen',
            parameters=[{
                'lookahead': 0.4,
                'max_v': 0.4,
                'max_w': 1.0,
                'rate_hz': 20.0,
                'path_topic': '/asyncvla/predicted_path',
                'cmd_vel_topic': '/cmd_vel',
            }],
        ),
    ])
```

- [ ] **Step 13.3: Build and verify the executable launches**

```bash
cd /home/nop/dev/mywork/raspicat-async-vla
source /opt/ros/humble/setup.bash
colcon build --packages-select raspicat_async_vla_edge
source install/setup.bash
ros2 run raspicat_async_vla_edge path_follower_node --ros-args -p rate_hz:=20.0 &
PID=$!
sleep 2
ros2 topic info /cmd_vel
kill $PID
```

Expected: `Type: geometry_msgs/msg/Twist` with one publisher.

- [ ] **Step 13.4: Commit**

```bash
git add src/raspicat_async_vla_edge/asyncvla_edge/path_follower_node.py \
        src/raspicat_async_vla_edge/launch/path_follower.launch.py
git commit -m "feat(edge): add path_follower_node ROS2 wrapper around Pure Pursuit"
```

---

## Task 14: Bringup composition for local MVP

**Files:**
- Create: `src/raspicat_async_vla_bringup/package.xml`
- Create: `src/raspicat_async_vla_bringup/CMakeLists.txt`
- Create: `src/raspicat_async_vla_bringup/launch/mvp_local.launch.py`
- Create: `src/raspicat_async_vla_bringup/config/topic_remap.yaml`

- [ ] **Step 14.1: Scaffold the package**

`package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>raspicat_async_vla_bringup</name>
  <version>0.1.0</version>
  <description>Launch composition for AsyncVLA on raspicat.</description>
  <maintainer email="nop@example.com">nop</maintainer>
  <license>MIT</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <exec_depend>raspicat_async_vla_edge</exec_depend>
  <exec_depend>raspicat_async_vla_remote</exec_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.8)
project(raspicat_async_vla_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
```

- [ ] **Step 14.2: Write `launch/mvp_local.launch.py`**

```python
"""Launch the full Plan-1 MVP locally:
 - asyncvla_dummy_server  (gRPC, deterministic embeddings)
 - asyncvla_edge_node     (lifecycle, configured + activated)
 - path_follower_node     (Path -> /cmd_vel)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, EmitEvent, ExecuteProcess, RegisterEventHandler,
)
from launch.event_handlers import OnProcessStart
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


def generate_launch_description():
    grpc_port = LaunchConfiguration('grpc_port')
    inference_ms = LaunchConfiguration('inference_ms')

    edge_config = os.path.join(
        get_package_share_directory('raspicat_async_vla_edge'),
        'config', 'edge_params.yaml',
    )

    dummy_server = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'raspicat_async_vla_remote', 'asyncvla_dummy_server',
            '--port', grpc_port,
            '--inference-ms', inference_ms,
            '--num-tokens', '8',
            '--embed-dim', '1024',
        ],
        output='screen',
    )

    edge = LifecycleNode(
        package='raspicat_async_vla_edge',
        executable='asyncvla_edge_node',
        name='asyncvla_edge_node',
        namespace='',
        output='screen',
        parameters=[edge_config, {
            'remote_address': ['localhost:', grpc_port],
        }],
    )
    configure = EmitEvent(event=ChangeState(
        lifecycle_node_matcher=matches_action(edge),
        transition_id=Transition.TRANSITION_CONFIGURE,
    ))
    activate = EmitEvent(event=ChangeState(
        lifecycle_node_matcher=matches_action(edge),
        transition_id=Transition.TRANSITION_ACTIVATE,
    ))

    follower = Node(
        package='raspicat_async_vla_edge',
        executable='path_follower_node',
        name='path_follower_node',
        output='screen',
        parameters=[{
            'lookahead': 0.4, 'max_v': 0.4, 'max_w': 1.0, 'rate_hz': 20.0,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('grpc_port', default_value='50051'),
        DeclareLaunchArgument('inference_ms', default_value='50.0'),
        dummy_server,
        edge,
        RegisterEventHandler(OnProcessStart(target_action=edge, on_start=[configure])),
        RegisterEventHandler(OnStateTransition(
            target_lifecycle_node=edge, goal_state='inactive', entities=[activate],
        )),
        follower,
    ])
```

- [ ] **Step 14.3: Write `config/topic_remap.yaml` placeholder**

```yaml
# Topic remappings for sim vs real raspicat (filled in Plan 2 sim_full.launch.py).
sim:
  image: "/camera/image_raw"
  cmd_vel: "/cmd_vel"
real:
  image: "/raspicat/camera/image_raw"
  cmd_vel: "/cmd_vel"
```

- [ ] **Step 14.4: Build all packages**

```bash
cd /home/nop/dev/mywork/raspicat-async-vla
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

Expected: `Summary: 5 packages finished` with no errors.

- [ ] **Step 14.5: Commit**

```bash
git add src/raspicat_async_vla_bringup
git commit -m "feat(bringup): add mvp_local.launch.py composition"
```

---

## Task 15: End-to-end manual verification of Plan 1 MVP

**Goal:** Confirm that `mvp_local.launch.py` end-to-end produces a non-zero `/cmd_vel` linear velocity given a goal and a synthetic image stream.

**Files:**
- Create: `tools/publish_fake_image.py`

- [ ] **Step 15.1: Write `tools/publish_fake_image.py`**

```python
"""Publish a constant black image at 5 Hz on /camera/image_raw, plus a fixed goal."""
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image

from raspicat_async_vla_msgs.msg import GoalSpec as GoalSpecMsg


class FakePub(Node):
    def __init__(self) -> None:
        super().__init__('fake_pub')
        self._img_pub = self.create_publisher(Image, '/camera/image_raw', 1)
        self._goal_pub = self.create_publisher(GoalSpecMsg, '/asyncvla/goal', 1)
        self._timer = self.create_timer(0.2, self._tick)
        self._goal_sent = False

    def _tick(self) -> None:
        msg = Image()
        msg.height = 240
        msg.width = 320
        msg.encoding = 'rgb8'
        msg.is_bigendian = 0
        msg.step = 320 * 3
        msg.data = np.zeros((240, 320, 3), dtype=np.uint8).tobytes()
        self._img_pub.publish(msg)
        if not self._goal_sent:
            g = GoalSpecMsg()
            g.mode = GoalSpecMsg.MODE_POSE
            g.pose = PoseStamped()
            g.pose.header.frame_id = 'odom'
            g.pose.pose.position.x = 1.0
            g.pose.pose.orientation.w = 1.0
            self._goal_pub.publish(g)
            self._goal_sent = True


def main() -> None:
    rclpy.init()
    node = FakePub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 15.2: Run the launch + fake publisher and observe `/cmd_vel`**

In one terminal:

```bash
cd /home/nop/dev/mywork/raspicat-async-vla
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch raspicat_async_vla_bringup mvp_local.launch.py
```

In a second terminal:

```bash
cd /home/nop/dev/mywork/raspicat-async-vla
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 tools/publish_fake_image.py
```

In a third terminal:

```bash
source /opt/ros/humble/setup.bash
source /home/nop/dev/mywork/raspicat-async-vla/install/setup.bash
ros2 topic echo /cmd_vel --once
```

Expected output (values):
- `linear.x` is `> 0` and `<= 0.4`
- `angular.z` is small (≈ 0 because the stub Path is straight ahead)

Also verify status:

```bash
ros2 topic echo /asyncvla/status --once
```

Expected: `message: "OK"` (or `"WAITING_REMOTE"` for the very first second).

Also verify Path:

```bash
ros2 topic echo /asyncvla/predicted_path --once
```

Expected: a Path with 10 PoseStamped points along x = 0.1 .. 1.0, y = 0.

- [ ] **Step 15.3: Commit verification helper**

```bash
git add tools/publish_fake_image.py
git commit -m "chore(tools): add publish_fake_image helper for MVP verification"
```

- [ ] **Step 15.4: Run full test suite once more**

```bash
cd /home/nop/dev/mywork/raspicat-async-vla
source /opt/ros/humble/setup.bash
source install/setup.bash
colcon test --event-handlers console_direct+
colcon test-result --verbose
```

Expected: 0 failures across all packages.

---

## Done condition (Plan 1 acceptance)

When all of the following are true, Plan 1 is complete:

- [ ] `colcon build` with all 5 packages succeeds.
- [ ] `colcon test` reports 0 failures.
- [ ] `ros2 launch raspicat_async_vla_bringup mvp_local.launch.py` brings up dummy server, edge node, follower without crash.
- [ ] With `tools/publish_fake_image.py` running, `/cmd_vel` shows `linear.x > 0`.
- [ ] `/asyncvla/status` reports `OK` (or transient `WAITING_REMOTE` only).
- [ ] All commits land in `main` (or the active branch) — clean tree.

When done, **Plan 2** picks up:

1. Add `external/AsyncVLA` git submodule and use the real Edge Adapter PyTorch module to replace `_stub_adapter_to_path`.
2. Replace `dummy_server.py` with a real OmniVLA-backed `inference.py` (Token Projector → 1024-dim) on GPU, loaded from HF `NHirose/AsyncVLA_release`.
3. Add `Dockerfile.remote` and `sim_full.launch.py` (Gazebo + raspicat_sim + edge + remote).
4. Add `tools/benchmark.py` and run latency-injection benchmarks.
5. Tune Pure Pursuit params on real raspicat.
