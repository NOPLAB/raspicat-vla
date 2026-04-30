# OmniVLA Real Model Integration (Plan 2B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Plan 1's `DummyServer` (or, alternatively, run alongside it and Plan 2A's AsyncVLA backend) with an `OmniVLAServer` that loads OmniVLA's `vla + pose_projector + action_head` on the GPU side, and an `OmniVLA_edge` adapter on the edge side, so real images and goals flow end-to-end through the OmniVLA-edge pipeline (`external/OmniVLA/inference/run_omnivla_edge.py`) and the edge node emits a real predicted Path.

**Architecture:** Reuses the gRPC bidi-stream wiring and the model-agnostic packages produced by the `refactor: rename packages to model-agnostic raspicat_vla_*` commit. The Remote-side server picks a backend (`dummy` / `asyncvla` / `omnivla`) at startup; the Edge-side picks a matching adapter (`stub` / `asyncvla` / `omnivla`) via parameters. The gRPC contract (`raspicat_vla.v1.VLAService`) and the `nav_msgs/Path` topic do not change.

**Why OmniVLA-edge (not OmniVLA-original):** OmniVLA ships in two flavors — `run_omnivla.py` (monolithic, full pipeline on GPU; outputs control directly) and `run_omnivla_edge.py` (cloud + edge split with `OmniVLA_edge` from `inference/model_omnivla_edge.py`). Only the edge variant maps cleanly onto our `Observation → ActionEmbedding → Path` pipeline. OmniVLA-original could be added later via a separate "all-on-remote" launch that publishes `nav_msgs/Path` directly and bypasses the edge adapter; that is **out of scope** for this plan.

**Tech Stack:** PyTorch 2.2 + CUDA bf16 + transformers + prismatic (vendored at `external/OmniVLA`) + huggingface_hub for `NHirose/omnivla-original` (cloud backbone) and `NHirose/omnivla-edge` (edge head). The edge-adapter dependencies (`efficientnet_pytorch`, `torchvision`) are shared with Plan 2A. ROS2 Humble unchanged.

**Reference spec:** `docs/superpowers/specs/2026-04-29-asyncvla-control-node-design.md` §6 (Edge), §7 (Remote) — OmniVLA's contract is identical at the proto layer; only the backend differs.
**Companion plan:** `docs/superpowers/plans/2026-04-29-asyncvla-real-model-integration.md` (Plan 2A, AsyncVLA). Plan 2A and Plan 2B can land in either order; whichever lands first introduces the multi-backend abstraction (Task 3 below). The other plan then plugs into it.
**Predecessor commit:** `refactor: rename packages to model-agnostic raspicat_vla_*` (rename to `raspicat_vla_*`).

**Branch suggestion:** `feat/omnivla-real-model` off `main`.

---

## Pre-flight Assumptions

These need to hold; bail out and escalate if any of them fail at runtime:

1. **GPU available** with CUDA 11.8+ and ≥ 24 GB VRAM. The OmniVLA backbone is OpenVLA-OFT-class (~7.5B params in bf16 ≈ 15 GB) plus activations.
2. **`external/OmniVLA` git submodule populated** (already true: see `.gitmodules`).
3. **Network access** to HuggingFace and Docker Hub from the build/run host.
4. **Disk space** ≥ 35 GB for HF cache (omnivla-original ~15 GB + omnivla-edge ~ a few GB) plus image build.
5. **`prismatic` framework from OmniVLA** has all imports (`from prismatic.models.action_heads import L1RegressionActionHead_idcat`, `from prismatic.models.projectors import ProprioProjector`, `from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1`) resolvable when `external/OmniVLA` is on PYTHONPATH. **Open question:** OmniVLA's `prismatic/` and AsyncVLA's `prismatic/` are sibling vendored copies that may or may not be byte-identical; if both Plan 2A and Plan 2B run in the same container, this needs a tiebreak. Default for Plan 2B: install **only** OmniVLA's prismatic; AsyncVLA's container is separate.

---

## Architectural decisions

### D1. Multi-backend abstraction in `raspicat_vla_remote/`

Plan 1's `dummy_server.py` directly implements `VLAServiceServicer`. To accommodate AsyncVLA + OmniVLA + Dummy without three forks of the same servicer, introduce a thin abstraction:

```python
# raspicat_vla_remote/backends/base.py
class VLABackend(ABC):
    @abstractmethod
    def warmup(self, num_iters: int = 1) -> None: ...
    @abstractmethod
    def infer(self, *, current_image, past_image, lang_instruction,
              goal_image, goal_pose_xy_theta) -> tuple[np.ndarray, dict]:
        """Returns (projected_actions: np.ndarray (num_tokens, embed_dim) float32,
                    metrics: {'inference_ms': float, 'modality_id': int, ...})."""
    @abstractmethod
    def model_info(self) -> ModelInfoDict: ...
```

The single `VLAServer` (gRPC servicer) takes a `VLABackend` instance and is fully model-agnostic. Backends:

| Backend       | File                                          | Status after Plan 2B |
| ------------- | --------------------------------------------- | -------------------- |
| `DummyBackend`   | `backends/dummy.py` (port from `dummy_server`) | done                 |
| `AsyncVLABackend`| `backends/asyncvla.py` (Plan 2A)              | done iff Plan 2A done|
| `OmniVLABackend` | `backends/omnivla.py` (this plan)             | done                 |

`server_main.py` adds `--backend {dummy,asyncvla,omnivla}` and dispatches to the right backend's constructor.

### D2. Multi-adapter abstraction in `raspicat_vla_edge/`

Symmetric on the edge side. Today `_stub_adapter_to_path` lives inline in `edge_node.py`. Promote it to:

```python
# raspicat_vla_edge/adapters/base.py
class EdgeAdapter(ABC):
    @abstractmethod
    def predict_path(self, *, embedding, embedding_shape,
                     cur_image_rgb, past_image_rgb,
                     frame_id: str = 'base_link') -> Path: ...
```

| Adapter          | File                                       | Status after Plan 2B |
| ---------------- | ------------------------------------------ | -------------------- |
| `StubAdapter`    | `adapters/stub.py` (port from inline stub) | done                 |
| `AsyncVLAAdapter`| `adapters/asyncvla.py` (Plan 2A)           | done iff Plan 2A done|
| `OmniVLAAdapter` | `adapters/omnivla.py` (this plan)          | done                 |

`edge_node.py` reads a parameter `adapter_kind: stub|asyncvla|omnivla`, instantiates the matching adapter in `on_configure`, and calls `predict_path(...)` in `_action_tick`.

### D3. Constraints on goal modalities for v1

OmniVLA-edge's full input set includes `obs_images`, `goal_pose`, `map_images`, `goal_image`, `modality_id`, `feat_text_lan`, `cur_large_img`. The current proto (`raspicat_vla.v1.Observation`) carries only the first JPEG, the `GoalSpec` (POSE/TEXT/IMAGE), and `current_pose`. **Plan 2B v1 supports POSE-goal and TEXT-goal modalities only.** `map_images`, `satellite`, and `cur_large_img` are stubbed (set to a blank tensor of the right shape) and flagged in `ModelInfo.metadata`. Adding satellite / map / large-image support is a follow-up plan that extends the proto.

### D4. Checkpoints

OmniVLA ships separate HF repos:
- `NHirose/omnivla-original` — cloud backbone (`vla`, `pose_projector`, `action_head` checkpoints at step 120000).
- `NHirose/omnivla-edge` — edge model checkpoint(s).

Plan 2B uses `omnivla-original` for the remote and `omnivla-edge` for the edge. Both are downloaded by `scripts/download_omnivla_checkpoints.sh` (Task 2).

---

## File Structure (final state after this plan, assuming Plan 2A also lands)

```
raspicat-vla/
├── docker/
│   ├── Dockerfile.test               # Plan 1 (unchanged)
│   ├── Dockerfile.remote             # Plan 2A (AsyncVLA prismatic)
│   └── Dockerfile.omnivla            # NEW (Plan 2B): CUDA + torch + OmniVLA prismatic
├── scripts/
│   ├── gen_proto.sh                  # Plan 1
│   ├── download_checkpoints.sh       # Plan 2A (NHirose/AsyncVLA_release)
│   └── download_omnivla_checkpoints.sh  # NEW (Plan 2B): NHirose/omnivla-original + omnivla-edge
├── external/
│   ├── AsyncVLA/                     # Plan 2A submodule
│   ├── OmniVLA/                      # already submodule'd
│   └── ...
├── src/
│   ├── raspicat_vla_remote/
│   │   ├── raspicat_vla_remote/
│   │   │   ├── __init__.py
│   │   │   ├── server_main.py            # MODIFIED: --backend {dummy,asyncvla,omnivla}
│   │   │   ├── server.py                 # NEW (or refactor of dummy_server): generic VLAServer servicer
│   │   │   ├── dummy_server.py           # KEPT for back-compat (re-exports DummyBackend)
│   │   │   └── backends/
│   │   │       ├── __init__.py
│   │   │       ├── base.py               # NEW: VLABackend ABC + ModelInfoDict
│   │   │       ├── dummy.py              # NEW: DummyBackend (port of dummy_server logic)
│   │   │       ├── asyncvla.py           # Plan 2A
│   │   │       ├── omnivla.py            # NEW (Plan 2B): OmniVLABackend
│   │   │       ├── omnivla_data_transform.py  # NEW: build_inference_batch for OmniVLA-edge
│   │   │       └── omnivla_model_loader.py    # NEW (or shared util): load_checkpoint helper
│   │   └── test/
│   │       ├── test_dummy_server.py      # Plan 1 — adjusted to use DummyBackend
│   │       ├── test_omnivla_data_transform.py # NEW
│   │       └── test_omnivla_engine_smoke.py   # NEW: opt-in via OMNIVLA_E2E=1
│   ├── raspicat_vla_edge/
│   │   ├── raspicat_vla_edge/
│   │   │   ├── ...                       # Plan 1 modules unchanged
│   │   │   ├── edge_node.py              # MODIFIED: dispatch via EdgeAdapter selected by adapter_kind
│   │   │   └── adapters/
│   │   │       ├── __init__.py
│   │   │       ├── base.py               # NEW: EdgeAdapter ABC
│   │   │       ├── stub.py               # NEW: port of _stub_adapter_to_path
│   │   │       ├── asyncvla.py           # Plan 2A
│   │   │       ├── omnivla.py            # NEW: OmniVLAAdapter (load + predict_path)
│   │   │       └── omnivla_inference.py  # NEW: forward + waypoints_to_path
│   │   ├── config/
│   │   │   └── edge_params.yaml          # MODIFIED: + adapter_kind, + omnivla_edge_path / step / device
│   │   └── test/
│   │       ├── ...                       # Plan 1 tests unchanged
│   │       ├── test_omnivla_adapter.py   # NEW (CPU OK with synthetic checkpoint)
│   │       └── test_omnivla_inference.py # NEW (CPU OK)
│   └── raspicat_vla_bringup/
│       └── launch/
│           ├── mvp_local.launch.py       # Plan 1
│           ├── mvp_real.launch.py        # Plan 2A
│           └── mvp_omnivla.launch.py     # NEW (Plan 2B)
└── docs/
    └── superpowers/
        └── plans/
            └── 2026-04-30-omnivla-real-model-integration.md  # this file
```

---

## Conventions

- All host-side commands assume the user runs them from `/home/nop/dev/mywork/raspicat-vla`.
- All GPU-required steps run inside `docker/Dockerfile.omnivla` (pull `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`).
- All ROS2 / pytest steps that don't need GPU run inside `docker/Dockerfile.test` from Plan 1.
- Tests follow Plan 1's pattern: write failing test → run → make pass → commit.
- Commit format: `<type>(<scope>): <summary>`. Scopes: `remote`, `edge`, `infra`, `bringup`, `docs`.
- **Verbatim policy:** when porting code from `external/OmniVLA/inference/run_omnivla_edge.py` and `external/OmniVLA/inference/model_omnivla_edge.py`, **read** that file and **adapt**. The originals are monolithic and use globals; we want clean module-level functions.

---

## Task 0: Investigate OmniVLA dependency closure

**Why this task exists:** OmniVLA's `run_omnivla_edge.py` references modules under `prismatic.*` that may overlap with — but not be identical to — AsyncVLA's `prismatic`. We need to confirm exactly which submodules are required at inference (vs training-only) and what the OmniVLA-edge model's forward signature actually consumes.

**Files to investigate (read only, no changes):**
- `/home/nop/dev/mywork/raspicat-vla/external/OmniVLA/inference/run_omnivla_edge.py`
- `/home/nop/dev/mywork/raspicat-vla/external/OmniVLA/inference/run_omnivla.py`
- `/home/nop/dev/mywork/raspicat-vla/external/OmniVLA/inference/model_omnivla_edge.py`
- `/home/nop/dev/mywork/raspicat-vla/external/OmniVLA/prismatic/models/projectors.py`
- `/home/nop/dev/mywork/raspicat-vla/external/OmniVLA/prismatic/models/action_heads.py`
- `/home/nop/dev/mywork/raspicat-vla/external/OmniVLA/prismatic/extern/hf/modeling_prismatic.py`
- `/home/nop/dev/mywork/raspicat-vla/external/OmniVLA/pyproject.toml`

**Deliverable:** a short markdown report at `docs/superpowers/notes/2026-04-30-omnivla-deps.md` that lists:
- The minimal set of `prismatic.*` modules `OmniVLABackend` needs.
- Whether `lerobot` / `Learning-to-Drive-Anywhere-with-MBRA` / `flash-attn` are required at inference or only at training.
- The exact `*--{step}_checkpoint.pt` filenames present in `NHirose/omnivla-original` (`vla` shards, `pose_projector`, `action_head`, …) and `NHirose/omnivla-edge`.
- The exact forward signature of `OmniVLA_edge.__call__` (positional args, types, expected shapes).
- A list of inputs to `OmniVLA_edge` that we currently *cannot* produce from the existing proto (e.g. `map_images`, `cur_large_img`) — and the proposed stub for each.

- [ ] **Step 0.1: Inventory checkpoints needed**

```bash
cd /home/nop/dev/mywork/raspicat-vla
grep -nE "load_checkpoint\(|init_module\(" external/OmniVLA/inference/run_omnivla_edge.py
grep -nE "load_checkpoint\(|init_module\(" external/OmniVLA/inference/run_omnivla.py
```

- [ ] **Step 0.2: Inventory prismatic imports**

```bash
grep -hE "^from prismatic|^import prismatic" \
  external/OmniVLA/inference/run_omnivla_edge.py \
  external/OmniVLA/inference/run_omnivla.py \
  external/OmniVLA/inference/model_omnivla_edge.py \
  external/OmniVLA/prismatic/models/projectors.py \
  external/OmniVLA/prismatic/models/action_heads.py | sort -u
```

- [ ] **Step 0.3: Check HF release contents**

```bash
docker run --rm python:3.10 bash -c "
  pip install -q huggingface_hub
  python3 -c \"
from huggingface_hub import HfApi
api = HfApi()
for repo in ('NHirose/omnivla-original', 'NHirose/omnivla-edge'):
    print('==', repo)
    for f in api.list_repo_files(repo):
        print(' ', f)
\"
"
```

- [ ] **Step 0.4: Document `OmniVLA_edge.forward` signature**

Read `external/OmniVLA/inference/model_omnivla_edge.py` lines around `class OmniVLA_edge(BaseModel)` and `class FiLMNetwork`. Capture: positional args, expected dtypes, and what each input represents. Cross-reference `external/OmniVLA/inference/run_omnivla_edge.py` Inference.run_forward_pass for the actual call sites.

- [ ] **Step 0.5: Write the report**

- [ ] **Step 0.6: Commit**

```bash
git add docs/superpowers/notes/2026-04-30-omnivla-deps.md
git commit -m "docs(plan-2b): record OmniVLA dependency closure for Plan 2B"
```

---

## Task 1: `Dockerfile.omnivla` for GPU inference

**Files:**
- Create: `docker/Dockerfile.omnivla`

- [ ] **Step 1.1: Write `Dockerfile.omnivla`**

```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && ln -sf /usr/bin/python3 /usr/bin/python

RUN pip3 install --no-cache-dir \
    'torch==2.2.0+cu121' 'torchvision==0.17.0+cu121' 'torchaudio==2.2.0+cu121' \
    --extra-index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir \
    'numpy==1.26.4' \
    'transformers>=4.40' \
    'huggingface_hub>=0.23' \
    'accelerate>=0.30' \
    'safetensors' \
    'Pillow' \
    'efficientnet_pytorch>=0.7.1' \
    'einops' \
    'utm' \
    'PyYAML' \
    'grpcio>=1.50' \
    'grpcio-tools>=1.50' \
    'protobuf>=4.21' \
    'typing_extensions>=4.5' \
    'opencv-python-headless'

# Install OmniVLA's prismatic from the vendored submodule.
COPY external/OmniVLA /opt/OmniVLA
RUN pip3 install --no-cache-dir -e /opt/OmniVLA

WORKDIR /workspace
```

- [ ] **Step 1.2: Build the image**

```bash
DOCKER_CONFIG=/tmp/dckr-noauth docker build -f docker/Dockerfile.omnivla -t raspicat-vla-omnivla .
```

- [ ] **Step 1.3: Smoke checks**

```bash
DOCKER_CONFIG=/tmp/dckr-noauth docker run --rm --gpus all raspicat-vla-omnivla python3 -c "
import torch
print('cuda:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
"

DOCKER_CONFIG=/tmp/dckr-noauth docker run --rm raspicat-vla-omnivla python3 -c "
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.models.action_heads import L1RegressionActionHead_idcat
from prismatic.models.projectors import ProprioProjector
print('omnivla prismatic imports: ok')
"
```

- [ ] **Step 1.4: Commit**

```bash
git add docker/Dockerfile.omnivla
git commit -m "feat(infra): add Dockerfile.omnivla for OmniVLA GPU serving"
```

---

## Task 2: Checkpoint download script

**Files:**
- Create: `scripts/download_omnivla_checkpoints.sh`

- [ ] **Step 2.1: Write the script**

```bash
#!/usr/bin/env bash
# Download OmniVLA checkpoints used by the cloud backbone (omnivla-original)
# and the edge head (omnivla-edge) into ./omnivla-original/ and ./omnivla-edge/.
#
# Uses the host's ~/.cache/huggingface so repeat runs are instant.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORIGINAL_DIR="${REPO_ROOT}/omnivla-original"
EDGE_DIR="${REPO_ROOT}/omnivla-edge"

mkdir -p "${ORIGINAL_DIR}" "${EDGE_DIR}"

python3 - <<PY
import os
from huggingface_hub import snapshot_download

for repo, out in [
    ("NHirose/omnivla-original", "${ORIGINAL_DIR}"),
    ("NHirose/omnivla-edge",     "${EDGE_DIR}"),
]:
    p = snapshot_download(repo_id=repo, local_dir=out, local_dir_use_symlinks=False)
    print(f"== {repo} -> {p}")
    for root, _, files in os.walk(p):
        for f in files:
            full = os.path.join(root, f)
            size_mb = os.path.getsize(full) / (1024 * 1024)
            print(f"   {os.path.relpath(full, p)}  ({size_mb:.1f} MB)")
PY
```

- [ ] **Step 2.2: Make executable + ignore artifacts**

```bash
chmod +x scripts/download_omnivla_checkpoints.sh
printf '\n# OmniVLA model weights (large; do not commit)\nomnivla-original/\nomnivla-edge/\n' >> .gitignore
```

- [ ] **Step 2.3: Run the download once**

```bash
DOCKER_CONFIG=/tmp/dckr-noauth docker run --rm \
  -v /home/nop/dev/mywork/raspicat-vla:/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  raspicat-vla-omnivla bash -lc "cd /workspace && ./scripts/download_omnivla_checkpoints.sh"
```

Capture the file lists. Cross-check against Task 0.3's expected listing.

- [ ] **Step 2.4: Commit**

```bash
git add scripts/download_omnivla_checkpoints.sh .gitignore
git commit -m "feat(infra): add OmniVLA HF checkpoint download script"
```

---

## Task 3: Backend / adapter abstraction (refactor existing code)

**Why this task:** Plan 1's dummy server is the only backend today. To plug OmniVLA in cleanly without forking the gRPC servicer, define `VLABackend` and `EdgeAdapter` ABCs and port the existing dummy / stub logic onto them. **This task is a no-op behavior change; all Plan 1 tests must still pass.**

**Note:** if Plan 2A lands first, it does this refactor; this Task 3 then becomes a check-in only ("verify the abstraction is in place"). If Plan 2B lands first, this Task 3 introduces it.

**Files:**
- Create: `src/raspicat_vla_remote/raspicat_vla_remote/backends/__init__.py`
- Create: `src/raspicat_vla_remote/raspicat_vla_remote/backends/base.py`
- Create: `src/raspicat_vla_remote/raspicat_vla_remote/backends/dummy.py`
- Create: `src/raspicat_vla_remote/raspicat_vla_remote/server.py` (generic `VLAServer`)
- Modify: `src/raspicat_vla_remote/raspicat_vla_remote/dummy_server.py` (re-export `DummyBackend` + thin wrapper for back-compat)
- Modify: `src/raspicat_vla_remote/raspicat_vla_remote/server_main.py` (add `--backend` arg, default `dummy`)
- Create: `src/raspicat_vla_edge/raspicat_vla_edge/adapters/__init__.py`
- Create: `src/raspicat_vla_edge/raspicat_vla_edge/adapters/base.py`
- Create: `src/raspicat_vla_edge/raspicat_vla_edge/adapters/stub.py`
- Modify: `src/raspicat_vla_edge/raspicat_vla_edge/edge_node.py` (instantiate adapter via `adapter_kind`)
- Modify: `src/raspicat_vla_edge/config/edge_params.yaml` (`adapter_kind: "stub"` default)

- [ ] **Step 3.1: `backends/base.py`**

```python
"""VLABackend ABC. Backends (dummy / asyncvla / omnivla) implement this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import PIL.Image


@dataclass
class ModelInfoDict:
    model_name: str
    model_version: str
    num_tokens: int
    embed_dim: int
    device: str
    ready: bool


class VLABackend(ABC):
    @abstractmethod
    def warmup(self, num_iters: int = 1) -> None: ...

    @abstractmethod
    def infer(
        self,
        *,
        current_image: PIL.Image.Image,
        past_image: Optional[PIL.Image.Image],
        lang_instruction: str,
        goal_image: Optional[PIL.Image.Image],
        goal_pose_xy_theta: Optional[Tuple[float, float, float]],
    ) -> Tuple[np.ndarray, dict]: ...

    @abstractmethod
    def model_info(self) -> ModelInfoDict: ...
```

- [ ] **Step 3.2: `backends/dummy.py`** — port the deterministic `_embedding_for` from `dummy_server.py` into a `DummyBackend(VLABackend)` whose `infer()` returns the same `(arr, metrics)` shape. Keep the same RNG seed so `test_dummy_server.py` still passes.

- [ ] **Step 3.3: Generic `server.py`**

```python
"""Generic gRPC servicer wrapping a VLABackend."""
from __future__ import annotations

import io, logging, threading, time
from concurrent import futures
from typing import Iterator, Optional

import grpc
import numpy as np
import PIL.Image

from raspicat_vla_proto import raspicat_vla_pb2, raspicat_vla_pb2_grpc
from raspicat_vla_proto.conversions import float32_array_to_fp16_bytes

from .backends.base import VLABackend


_LOG = logging.getLogger(__name__)


def _proto_goal_to_python(goal):
    if goal.mode == raspicat_vla_pb2.GoalSpec.POSE:
        return ('pose', (goal.pose.x, goal.pose.y, goal.pose.theta), '', None)
    if goal.mode == raspicat_vla_pb2.GoalSpec.TEXT:
        return ('text', None, goal.text, None)
    if goal.mode == raspicat_vla_pb2.GoalSpec.IMAGE:
        img = PIL.Image.open(io.BytesIO(goal.image_jpeg)).convert('RGB')
        return ('image', None, '', img)
    raise ValueError(f'unknown goal mode {goal.mode}')


class _Servicer(raspicat_vla_pb2_grpc.VLAServiceServicer):
    def __init__(self, *, backend: VLABackend) -> None:
        self._backend = backend
        self._past_image_per_client = {}
        self._past_image_lock = threading.Lock()

    def GetModelInfo(self, request, context):
        info = self._backend.model_info()
        return raspicat_vla_pb2.ModelInfo(
            model_name=info.model_name,
            model_version=info.model_version,
            num_tokens=info.num_tokens,
            embed_dim=info.embed_dim,
            device=info.device,
            ready=info.ready,
        )

    def StreamInfer(self, request_iterator, context):
        peer = context.peer()
        for obs in request_iterator:
            cur_img = PIL.Image.open(io.BytesIO(obs.image_jpeg)).convert('RGB')
            with self._past_image_lock:
                past = self._past_image_per_client.get(peer, cur_img)
                self._past_image_per_client[peer] = cur_img
            mode, pose, text, goal_img = _proto_goal_to_python(obs.goal)
            try:
                proj, metrics = self._backend.infer(
                    current_image=cur_img, past_image=past,
                    lang_instruction=text, goal_image=goal_img,
                    goal_pose_xy_theta=pose,
                )
            except Exception as exc:  # noqa: BLE001
                _LOG.exception('inference failed for frame_id=%s: %s', obs.frame_id, exc)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(exc))
                return
            num_tokens, embed_dim = proj.shape
            yield raspicat_vla_pb2.ActionEmbedding(
                frame_id=obs.frame_id,
                server_time_ns=time.monotonic_ns(),
                num_tokens=int(num_tokens),
                embed_dim=int(embed_dim),
                embedding_fp16=float32_array_to_fp16_bytes(proj.astype(np.float32)),
                inference_ms=float(metrics['inference_ms']),
                model_version=self._backend.model_info().model_version,
            )


class VLAServer:
    def __init__(self, *, backend: VLABackend, host='0.0.0.0', port=50051, max_workers=4) -> None:
        self._backend = backend
        self._host = host
        self._port = port
        self._max_workers = max_workers
        self._servicer = _Servicer(backend=backend)
        self._server: Optional[grpc.Server] = None

    def start(self) -> int:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=self._max_workers))
        raspicat_vla_pb2_grpc.add_VLAServiceServicer_to_server(self._servicer, server)
        port = server.add_insecure_port(f'{self._host}:{self._port}')
        server.start()
        self._server = server
        return port

    def stop(self, grace_sec=1.0) -> None:
        if self._server is not None:
            self._server.stop(grace_sec)
            self._server = None

    def wait_for_termination(self) -> None:
        if self._server is not None:
            self._server.wait_for_termination()
```

- [ ] **Step 3.4: Update `dummy_server.py`** — keep `DummyServer` class as a thin adapter over `VLAServer(backend=DummyBackend(...))` so existing `test_dummy_server.py` still passes unchanged.

- [ ] **Step 3.5: Update `server_main.py`** — add `--backend {dummy,asyncvla,omnivla}` (default `dummy`). For now `asyncvla` and `omnivla` raise `NotImplementedError` (filled in by their respective tasks/plans).

- [ ] **Step 3.6: Edge `adapters/base.py` + `adapters/stub.py`**

```python
# adapters/base.py
class EdgeAdapter(ABC):
    @abstractmethod
    def predict_path(
        self,
        *,
        embedding: np.ndarray,
        embedding_shape: tuple[int, int, int],   # (B, num_tokens, embed_dim)
        cur_image_rgb: np.ndarray,
        past_image_rgb: np.ndarray,
        frame_id: str = 'base_link',
    ) -> Path: ...
```

`adapters/stub.py` is the existing `_stub_adapter_to_path` extracted into a class.

- [ ] **Step 3.7: Modify `edge_node.py`** — declare param `adapter_kind` (default `stub`); in `on_configure`, pick adapter by name; replace the inline `_stub_adapter_to_path` call with `self._adapter.predict_path(...)`.

- [ ] **Step 3.8: All Plan 1 tests must still pass**

```bash
DOCKER_CONFIG=/tmp/dckr-noauth docker run --rm --user $(id -u):$(id -g) \
  -v /home/nop/dev/mywork/raspicat-vla:/workspace -e HOME=/tmp \
  raspicat-vla-test bash -c "
    source /opt/ros/humble/setup.bash; cd /workspace; source install/setup.bash
    pytest src/raspicat_vla_proto/test src/raspicat_vla_remote/test src/raspicat_vla_edge/test -v
  "
```

Expected: green across the board.

- [ ] **Step 3.9: Commit**

```bash
git add src/raspicat_vla_remote/raspicat_vla_remote/backends \
        src/raspicat_vla_remote/raspicat_vla_remote/server.py \
        src/raspicat_vla_remote/raspicat_vla_remote/dummy_server.py \
        src/raspicat_vla_remote/raspicat_vla_remote/server_main.py \
        src/raspicat_vla_edge/raspicat_vla_edge/adapters \
        src/raspicat_vla_edge/raspicat_vla_edge/edge_node.py \
        src/raspicat_vla_edge/config/edge_params.yaml
git commit -m "refactor(remote+edge): introduce VLABackend / EdgeAdapter abstractions"
```

---

## Task 4: Port `load_checkpoint` and helpers (TDD)

**Files:**
- Create: `src/raspicat_vla_remote/raspicat_vla_remote/backends/_checkpoints.py`
- Create: `src/raspicat_vla_remote/test/test_checkpoints.py`

If Plan 2A already shipped this file under another name (e.g. `model_loader.py`), reuse it instead of duplicating. Otherwise port `remove_ddp_in_checkpoint` and `load_checkpoint` from `external/OmniVLA/inference/run_omnivla_edge.py:37–55` — they are identical between OmniVLA and AsyncVLA modulo error messages.

- [ ] **Step 4.1: Failing test** (mirror Plan 2A Task 3.1 with `omnivla-edge` style filenames)

```python
def test_load_checkpoint_pose_projector_falls_back_to_proprio_projector(tmp_path):
    """OmniVLA quirk: the on-disk filename for pose_projector is actually
    proprio_projector--{step}_checkpoint.pt; load_checkpoint must transparently
    fall back."""
    sd = {'foo.weight': torch.zeros(1)}
    torch.save(sd, tmp_path / 'proprio_projector--120000_checkpoint.pt')
    loaded = load_checkpoint('pose_projector', str(tmp_path), step=120000)
    assert 'foo.weight' in loaded
```

- [ ] **Step 4.2: Implement** — straight port. Match the OmniVLA quirk where `pose_projector` falls back to `proprio_projector`.

- [ ] **Step 4.3: Tests pass**

- [ ] **Step 4.4: Commit**

```bash
git commit -m "feat(remote): add checkpoint loader helpers (shared by AsyncVLA/OmniVLA)"
```

---

## Task 5: `omnivla_data_transform.build_inference_batch` (TDD)

**Files:**
- Create: `src/raspicat_vla_remote/raspicat_vla_remote/backends/omnivla_data_transform.py`
- Create: `src/raspicat_vla_remote/test/test_omnivla_data_transform.py`

The original `data_transformer_omnivla` is in `external/OmniVLA/inference/run_omnivla_edge.py` Inference class. We extract it into a free function:

```python
def build_inference_batch(
    *,
    current_image: PIL.Image.Image,
    past_image: PIL.Image.Image | None,
    lang_instruction: str,
    goal_image: PIL.Image.Image | None,
    goal_pose_xy_theta: tuple[float, float, float] | None,
    action_tokenizer,
    processor,
    num_images_in_input: int = 2,
) -> dict[str, torch.Tensor]:
    """Returns a batch dict with input_ids, attention_mask, pixel_values,
    labels, modality_id, goal_pose, num_patches.
    Mirrors what run_omnivla_edge feeds into vla(...)."""
```

**Implementer note:** read `external/OmniVLA/inference/run_omnivla_edge.py:155–200` for the original `data_transformer_omnivla` body. Adapt — don't copy line-for-line.

- [ ] **Step 5.1: Failing test (with stub processor / tokenizer)** — same shape as Plan 2A Task 4.2, asserting required keys (`input_ids`, `attention_mask`, `pixel_values`, `labels`, `modality_id`, `goal_pose`, `num_patches`).

- [ ] **Step 5.2: Implement.** Key differences from AsyncVLA's variant:
  - No `action_proj` step (OmniVLA doesn't have `Proj_Actiontokens`); the batch is consumed directly by `vla` + `action_head`.
  - `goal_pose` shape per spec is `(B, 1, POSE_DIM)`; populate from `goal_pose_xy_theta` or zero if absent.
  - `modality_id` is set per the priority order in `run_omnivla_edge.py` (`pose > image_goal > satellite > lang`).

- [ ] **Step 5.3: Tests pass**

- [ ] **Step 5.4: Commit**

```bash
git add src/raspicat_vla_remote/raspicat_vla_remote/backends/omnivla_data_transform.py \
        src/raspicat_vla_remote/test/test_omnivla_data_transform.py
git commit -m "feat(remote): add omnivla_data_transform.build_inference_batch (TDD)"
```

---

## Task 6: `OmniVLABackend` — full Remote forward pass

**Files:**
- Create: `src/raspicat_vla_remote/raspicat_vla_remote/backends/omnivla.py`
- Create: `src/raspicat_vla_remote/test/test_omnivla_engine_smoke.py`

**Implementer reference:** `external/OmniVLA/inference/run_omnivla_edge.py:411–478` (`run_forward_pass`) + `external/OmniVLA/inference/run_omnivla.py:411–478` (the non-edge version is also useful as a self-contained full forward).

- [ ] **Step 6.1: Sketch the public API**

```python
class OmniVLABackend(VLABackend):
    def __init__(
        self,
        *,
        vla_path: str,                  # e.g. './omnivla-original'
        resume_step: int = 120000,
        device: str = 'cuda:0',
        dtype: torch.dtype = torch.bfloat16,
        num_images_in_input: int = 2,
        use_l1_regression: bool = True,
    ): ...

    def warmup(self, num_iters: int = 1) -> None: ...
    def infer(self, *, current_image, past_image, lang_instruction,
              goal_image, goal_pose_xy_theta) -> tuple[np.ndarray, dict]: ...
    def model_info(self) -> ModelInfoDict: ...
```

- [ ] **Step 6.2: Implement `omnivla.py`**

Key porting steps from `run_forward_pass`:
1. Build batch via `omnivla_data_transform.build_inference_batch(...)`.
2. Forward through `vla` with `torch.autocast('cuda', dtype=bfloat16)` + `torch.no_grad()`. Pass `proprio=batch['goal_pose']`, `proprio_projector=self.pose_projector`, `output_hidden_states=True`, `use_film=False`.
3. Slice `actions_hidden_states` from `last_hidden_states` using `get_current_action_mask` + `get_next_actions_mask` from `prismatic.training.train_utils`.
4. Reshape to `(1, NUM_ACTIONS_CHUNK * ACTION_DIM, hidden_dim)`. **OmniVLA-original returns this directly as the cloud → edge payload** (no separate `action_proj`).
5. Cast to float32 numpy and return `(projected_actions, metrics)`.

The actual constructor arguments for `ProprioProjector(llm_dim=vla.llm_dim, proprio_dim=POSE_DIM)` and `L1RegressionActionHead_idcat(input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=ACTION_DIM)` come straight from `run_omnivla_edge.py:393–410` — copy those values.

`model_info()` returns:
- `model_name='NHirose/omnivla-original'`
- `model_version=f'omnivla-orig-step{resume_step}'`
- `num_tokens=NUM_ACTIONS_CHUNK * ACTION_DIM` (typically 8 × 4 = 32)
- `embed_dim=vla.llm_dim` (typically 4096; verify via Task 0)
- `device=str(self.device)`
- `ready=True`

- [ ] **Step 6.3: Smoke test (slow, GPU-only, opt-in via `OMNIVLA_E2E=1`)**

```python
"""Slow GPU-only smoke test. Skipped unless OMNIVLA_E2E=1."""
import os, pytest
if os.environ.get('OMNIVLA_E2E') != '1':
    pytest.skip('set OMNIVLA_E2E=1', allow_module_level=True)


def test_omnivla_backend_returns_correct_shape():
    import PIL.Image
    from raspicat_vla_remote.backends.omnivla import OmniVLABackend
    b = OmniVLABackend(vla_path='/workspace/omnivla-original',
                       resume_step=120000, device='cuda:0')
    b.warmup(1)
    img = PIL.Image.new('RGB', (224, 224), (128, 128, 128))
    proj, metrics = b.infer(current_image=img, past_image=img,
                            lang_instruction='go forward', goal_image=None,
                            goal_pose_xy_theta=(1.0, 0.0, 0.0))
    assert proj.ndim == 2
    print(f'projected shape={proj.shape} inf_ms={metrics["inference_ms"]:.1f}')
```

- [ ] **Step 6.4: Run smoke test on a GPU host**

```bash
DOCKER_CONFIG=/tmp/dckr-noauth docker run --rm --gpus all \
  -v /home/nop/dev/mywork/raspicat-vla:/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e OMNIVLA_E2E=1 \
  raspicat-vla-omnivla bash -lc "
    cd /workspace
    pip install -e src/raspicat_vla_proto src/raspicat_vla_remote >/dev/null 2>&1 || true
    cd src/raspicat_vla_remote
    python3 -m pytest test/test_omnivla_engine_smoke.py -v -s
  "
```

- [ ] **Step 6.5: Wire into `server_main.py`** — the `--backend omnivla` branch instantiates `OmniVLABackend(...)` and starts `VLAServer(backend=...)`. Add `--vla-path`, `--resume-step` args.

- [ ] **Step 6.6: Commit**

```bash
git add src/raspicat_vla_remote/raspicat_vla_remote/backends/omnivla.py \
        src/raspicat_vla_remote/test/test_omnivla_engine_smoke.py \
        src/raspicat_vla_remote/raspicat_vla_remote/server_main.py
git commit -m "feat(remote): add OmniVLABackend wired into VLAServer"
```

---

## Task 7: Edge — load `OmniVLA_edge` model (TDD)

**Files:**
- Create: `src/raspicat_vla_edge/raspicat_vla_edge/adapters/omnivla.py`
- Create: `src/raspicat_vla_edge/test/test_omnivla_adapter.py`

The `OmniVLA_edge` class lives at `external/OmniVLA/inference/model_omnivla_edge.py:84+`. It is **not** part of `prismatic`; it is its own file. Plan 2B's adapter wraps it.

- [ ] **Step 7.1: Failing test (synthetic checkpoint)**

```python
"""Tests for OmniVLAAdapter using a synthetic state_dict."""
import torch, pytest
from inference.model_omnivla_edge import OmniVLA_edge  # path adjusted by sys.path tweak in conftest
from raspicat_vla_edge.adapters.omnivla import load_omnivla_edge


def _save_random_checkpoint(tmp_path, step=750000):
    model = OmniVLA_edge(...)  # constructor args TBD by Task 0.4
    cp = tmp_path / f'edge_model--{step}_checkpoint.pt'
    torch.save(model.state_dict(), cp)
    return str(tmp_path)


def test_load_omnivla_edge_returns_eval_module(tmp_path):
    path = _save_random_checkpoint(tmp_path)
    adapter = load_omnivla_edge(path=path, step=750000, device='cpu')
    assert isinstance(adapter, OmniVLA_edge)
    assert not adapter.training
```

- [ ] **Step 7.2: Implement `adapters/omnivla.py`**

Key shape: `OmniVLA_edge.forward(obs_images, goal_pose, map_images, goal_image, modality_id_select, feat_text_lan, cur_large_img)` returns `(predicted_actions, distances, mask_number)`. Wrap into `OmniVLAAdapter(EdgeAdapter)` whose `predict_path(...)` consumes the embedding from gRPC, builds the missing inputs from cur/past images + zeros for `map_images`/`cur_large_img`, runs `forward`, and converts `predicted_actions` to `nav_msgs/Path`.

- [ ] **Step 7.3: Tests pass**

- [ ] **Step 7.4: Commit**

```bash
git add src/raspicat_vla_edge/raspicat_vla_edge/adapters/omnivla.py \
        src/raspicat_vla_edge/test/test_omnivla_adapter.py
git commit -m "feat(edge): add OmniVLA_edge loader (TDD)"
```

---

## Task 8: OmniVLA edge inference — embedding + (cur, past) → Path

**Files:**
- Create: `src/raspicat_vla_edge/raspicat_vla_edge/adapters/omnivla_inference.py`
- Create: `src/raspicat_vla_edge/test/test_omnivla_inference.py`

OmniVLA_edge already produces `predicted_actions` shape `(B, T, ACTION_DIM)` directly (delta-poses or absolute waypoints — to be confirmed in Task 0.4). Convert to `nav_msgs/Path`.

- [ ] **Step 8.1: Test** — stub adapter returns `predicted_actions = torch.zeros(1, 8, 4)`; assert path shape and frame_id.

- [ ] **Step 8.2: Implement** — port `delta_to_pose` from `external/OmniVLA/inference/run_omnivla_edge.py:92–135` (same math as AsyncVLA's `delta_to_pose`).

- [ ] **Step 8.3: Wire into `OmniVLAAdapter.predict_path`** — call `forward` then `delta_to_pose` then build `Path`.

- [ ] **Step 8.4: Tests pass**

- [ ] **Step 8.5: Commit**

```bash
git commit -m "feat(edge): add omnivla_inference (forward + delta_to_pose -> Path)"
```

---

## Task 9: Wire `edge_node` to use `OmniVLAAdapter` via `adapter_kind: omnivla`

**Files:**
- Modify: `src/raspicat_vla_edge/raspicat_vla_edge/edge_node.py`
- Modify: `src/raspicat_vla_edge/config/edge_params.yaml`

- [ ] **Step 9.1: Add params**

```yaml
    adapter_kind: "stub"        # stub|asyncvla|omnivla
    omnivla_edge_path: "/workspace/omnivla-edge"
    omnivla_edge_step: 750000
    edge_device: "cpu"
```

- [ ] **Step 9.2: Modify `edge_node.py`** — extend the `adapter_kind` dispatch added in Task 3 with a branch for `omnivla` that constructs `OmniVLAAdapter(...)` from the params.

- [ ] **Step 9.3: Plan 1 smoke test must still pass with `adapter_kind=stub` (default)**

- [ ] **Step 9.4: Commit**

```bash
git commit -m "feat(edge): wire edge_node to support adapter_kind=omnivla"
```

---

## Task 10: `mvp_omnivla.launch.py` — full real-stack OmniVLA bringup

**Files:**
- Create: `src/raspicat_vla_bringup/launch/mvp_omnivla.launch.py`

Mirrors `mvp_local.launch.py` but uses `--backend omnivla` and `adapter_kind:=omnivla`.

- [ ] **Step 10.1: Implement** — same skeleton as Plan 2A's `mvp_real.launch.py`, swap:
  - server cmd: `python3 -m raspicat_vla_remote.server_main --backend omnivla --vla-path /workspace/omnivla-original --port {grpc_port}`
  - edge params: `adapter_kind: 'omnivla'`, `omnivla_edge_path: /workspace/omnivla-edge`

- [ ] **Step 10.2: Commit**

```bash
git add src/raspicat_vla_bringup/launch/mvp_omnivla.launch.py
git commit -m "feat(bringup): add mvp_omnivla.launch.py"
```

---

## Task 11: Real-stack manual smoke

**Goal:** run the full OmniVLA stack (GPU host) with `tools/publish_fake_image.py` and confirm `/cmd_vel` is non-zero — proving OmniVLA produces a usable Path.

This is a manual verification, not a pytest. Deliverable: a markdown note recording the run output.

- [ ] **Step 11.1: Set up host** — Docker + NVIDIA Container Toolkit + `raspicat-vla-omnivla` image (Task 1) + `omnivla-original/` and `omnivla-edge/` populated (Task 2).

- [ ] **Step 11.2: Run remote server in one terminal**

```bash
DOCKER_CONFIG=/tmp/dckr-noauth docker run --rm --gpus all \
  -v /home/nop/dev/mywork/raspicat-vla:/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  --network host \
  raspicat-vla-omnivla bash -lc "
    cd /workspace
    pip install -e src/raspicat_vla_proto src/raspicat_vla_remote >/dev/null 2>&1 || true
    python3 -m raspicat_vla_remote.server_main --backend omnivla \
      --vla-path /workspace/omnivla-original --port 50051
  "
```

- [ ] **Step 11.3: Run edge + follower in a second terminal** (using Plan 1's `Dockerfile.test`, with the edge OmniVLA model loaded on CPU).

```bash
DOCKER_CONFIG=/tmp/dckr-noauth docker run --rm --user $(id -u):$(id -g) \
  -v /home/nop/dev/mywork/raspicat-vla:/workspace -e HOME=/tmp \
  --network host \
  raspicat-vla-test bash -lc "
    source /opt/ros/humble/setup.bash; cd /workspace; source install/setup.bash
    ros2 launch raspicat_vla_bringup mvp_omnivla.launch.py
  "
```

- [ ] **Step 11.4: Publish fake images, observe topics**

```bash
ros2 topic echo /cmd_vel --once
ros2 topic echo /raspicat_vla/status --once
```

Expected: `linear.x` non-zero; `/raspicat_vla/status` reports `OK` after first inference completes.

- [ ] **Step 11.5: Record results** in `docs/superpowers/notes/2026-04-30-omnivla-stack-smoke.md`:
- Hardware (GPU model)
- Inference latency (from `inference_ms` reported via `/raspicat_vla/embedding`)
- Cmd values observed
- Any errors / fallbacks (`STALE` / `WAITING_REMOTE` / `DEGRADED` durations)

- [ ] **Step 11.6: Commit**

```bash
git add docs/superpowers/notes/2026-04-30-omnivla-stack-smoke.md
git commit -m "docs(plan-2b): record OmniVLA real-stack smoke results"
```

---

## Done condition (Plan 2B acceptance)

When all of the following are true, Plan 2B is complete:

- [ ] `Dockerfile.omnivla` builds successfully on a GPU host.
- [ ] `scripts/download_omnivla_checkpoints.sh` populates `omnivla-original/` and `omnivla-edge/`.
- [ ] `colcon build` succeeds for all 5 packages.
- [ ] All Plan 1 + Plan 2A pytest tests still pass (no regression).
- [ ] New unit tests (`test_omnivla_data_transform.py`, `test_omnivla_adapter.py`, `test_omnivla_inference.py`) pass.
- [ ] GPU smoke test (`test_omnivla_engine_smoke.py` with `OMNIVLA_E2E=1`) passes.
- [ ] `mvp_omnivla.launch.py` brings up server + edge + follower without crash.
- [ ] With `tools/publish_fake_image.py` running, `/cmd_vel` shows non-zero values driven by `OmniVLAAdapter` (not the stub).
- [ ] `/raspicat_vla/status` reports `OK` once the first embedding arrives.
- [ ] `--backend dummy|asyncvla|omnivla` and `adapter_kind=stub|asyncvla|omnivla` all work; switching between them requires only config changes.

---

## Open questions

These should be resolved during Task 0 investigation:

1. **Prismatic version conflict.** OmniVLA's `external/OmniVLA/prismatic` and AsyncVLA's `external/AsyncVLA/prismatic` are sibling vendored copies. Are they byte-identical? If not, can a single Docker image install both, or do we permanently keep two images (`raspicat-vla-asyncvla` + `raspicat-vla-omnivla`)?
2. **`OmniVLA_edge` constructor signature.** Exact `__init__` args (channels, num_layers, etc.) — extract from `external/OmniVLA/inference/model_omnivla_edge.py` and the YAML config inside `omnivla-edge/`.
3. **`OmniVLA_edge.forward` output semantics.** Is `predicted_actions` deltas (like AsyncVLA) or absolute waypoints? Drives whether `delta_to_pose` is needed.
4. **`map_images` / `cur_large_img` / `satellite`.** Confirm the model accepts blank tensors of the right shape when those modalities aren't being used. If not, the proto needs extension before Plan 2B v1 can land.
5. **Edge OmniVLA model size.** AsyncVLA's `Edge_adapter` is ~5M params; OmniVLA-edge is heavier (FiLM + multi-stage feature extractors). Does it fit on the raspicat (Pi 4 / Jetson Nano level) at acceptable latency? Benchmark in Task 11 — may need to push compute back to the cloud and revisit Plan 2B's edge/cloud split.
6. **Checkpoint step number for `omnivla-edge`.** Task 0.3 will tell us. Default placeholder: `750000`.
7. **`POSE_DIM` mismatch.** OmniVLA uses `POSE_DIM` from `prismatic.vla.constants`. Must match the value AsyncVLA uses if both backends share the same `pose_projector` interface.

---

## After Plan 2B

When Plan 2A and Plan 2B are both done:

1. **Plan 3** — sim integration (`sim_full.launch.py` with Gazebo + raspicat_sim) for both backends.
2. **Plan 4** — closed-loop benchmarking (`tools/benchmark.py`): RTT, throughput, latency injection, AsyncVLA vs OmniVLA comparison on the same scenarios.
3. **Plan 5** — real raspicat hardware launch + safety stop field tests.
4. **Possible follow-up** — OmniVLA-original (monolithic, no edge split) as an additional `--backend omnivla-mono` that publishes `nav_msgs/Path` directly via a ROS2 server-side node, bypassing the gRPC stream entirely. Useful when the edge has zero compute headroom.
