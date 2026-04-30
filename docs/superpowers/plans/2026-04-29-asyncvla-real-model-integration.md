# AsyncVLA Real Model Integration (Plan 2A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Plan 1's `DummyServer` with a real `RealServer` running the OmniVLA backbone (~7.5B params, GPU) and replace the `_stub_adapter_to_path` in the edge node with the real `Edge_adapter` (~5M params, CPU OK), so that real images and goals flow end-to-end through the actual AsyncVLA model and the edge node emits a real predicted Path.

**Architecture:** Reuses the gRPC bidi-stream wiring from Plan 1. The Remote-side server now loads OmniVLA + action projector + pose projector via the upstream `prismatic` library and produces a real `projected_actions` tensor (the same payload shape Plan 1's dummy used). The Edge-side node now feeds that tensor — together with the latest two RGB camera frames at 96×96 — to `Edge_adapter` and converts the predicted delta-poses into a `nav_msgs/Path`. The Plan 1 `EmbeddingCache`, `AsyncVLAClient`, lifecycle, and Pure Pursuit follower remain unchanged.

**Tech Stack:** PyTorch 2.2 + CUDA bf16 + transformers + prismatic (vendored at `external/AsyncVLA`) + huggingface_hub for `NHirose/AsyncVLA_release` checkpoint download. `EfficientNet-b0` (via `efficientnet-pytorch`) for `Edge_adapter`. ROS2 Humble unchanged.

**Reference spec:** `docs/superpowers/specs/2026-04-29-asyncvla-control-node-design.md` §7 (Remote), §6 (Edge).
**Predecessor plan:** `docs/superpowers/plans/2026-04-29-asyncvla-mvp-wiring.md` (Plan 1, completed).

**Branch suggestion:** `feat/asyncvla-real-model` off `main` (after Plan 1 merges) or directly off `feat/asyncvla-mvp-wiring`.

---

## Pre-flight Assumptions

These need to hold; bail out and escalate if any of them fail at runtime:

1. **GPU available** with CUDA 11.8+ and ≥ 24 GB VRAM (RTX 3090 / A6000 / H100). The OmniVLA backbone is ~7.5B params in bf16 ≈ 15 GB, plus activations.
2. **`external/AsyncVLA` git submodule populated** (already true after Plan 1).
3. **Network access** to HuggingFace and Docker Hub from the build/run host.
4. **Disk space** ≥ 30 GB for HF cache + image build.
5. **`prismatic` framework** has all imports (`from prismatic.models.action_heads import L1RegressionActionHead_idcat`, etc.) resolvable when `external/AsyncVLA` is on PYTHONPATH. Some imports in `inference/run_asyncvla.py` reference `../Learning-to-Drive-Anywhere-with-MBRA/train/` and `../lerobot` — those paths are NOT in our submodules and need investigation in Task 0.

---

## File Structure (final state after this plan)

```
raspicat-async-vla/
├── docker/
│   ├── Dockerfile.test               # Plan 1 (unchanged)
│   └── Dockerfile.remote             # NEW: CUDA + torch 2.2 + prismatic + transformers
├── scripts/
│   ├── gen_proto.sh                  # Plan 1
│   └── download_checkpoints.sh       # NEW: huggingface_hub + sanity check
├── external/
│   └── AsyncVLA/                     # already submodule'd
│       └── ...                       # plus possibly Learning-to-Drive-Anywhere-with-MBRA submodule (Task 0)
├── src/
│   ├── raspicat_async_vla_remote/
│   │   ├── asyncvla_remote/
│   │   │   ├── dummy_server.py       # Plan 1 (kept for tests)
│   │   │   ├── server_main.py        # MODIFIED: --backend=dummy|real switch
│   │   │   ├── model_loader.py       # NEW: load_vla, load_helpers, load_checkpoint
│   │   │   ├── data_transform.py     # NEW: image+goal -> batch (port of data_transformer_asyncvla)
│   │   │   ├── inference_engine.py   # NEW: forward pass -> projected_actions
│   │   │   ├── real_server.py        # NEW: gRPC servicer using inference_engine
│   │   │   └── __init__.py
│   │   └── test/
│   │       ├── test_dummy_server.py  # Plan 1
│   │       ├── test_data_transform.py# NEW (image preprocess + goal encode)
│   │       └── test_inference_engine_smoke.py # NEW (slow, GPU-only, opt-in)
│   ├── raspicat_async_vla_edge/
│   │   ├── asyncvla_edge/
│   │   │   ├── ...                   # Plan 1 modules unchanged
│   │   │   ├── edge_adapter.py       # NEW: load Edge_adapter + checkpoint
│   │   │   ├── edge_inference.py     # NEW: forward + delta_to_pose -> Path
│   │   │   └── edge_node.py          # MODIFIED: replace _stub_adapter_to_path
│   │   └── test/
│   │       ├── ...                   # Plan 1 tests unchanged
│   │       ├── test_edge_adapter.py  # NEW (CPU OK with fake checkpoint)
│   │       └── test_edge_inference.py# NEW (CPU OK)
│   └── raspicat_async_vla_bringup/
│       └── launch/
│           ├── mvp_local.launch.py   # Plan 1
│           └── mvp_real.launch.py    # NEW: real_server + edge_node + follower
└── docs/
    └── superpowers/
        └── plans/
            └── 2026-04-29-asyncvla-real-model-integration.md  # this file
```

---

## Conventions

- All host-side commands assume the user runs them from `/home/nop/dev/mywork/raspicat-async-vla`.
- All GPU-required steps run inside `docker/Dockerfile.remote` (pull `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` or similar).
- All ROS2 / pytest steps that don't need GPU run inside `docker/Dockerfile.test` from Plan 1 (we'll extend it slightly).
- Tests follow Plan 1's pattern: write failing test → run → make pass → commit.
- Commit format: `<type>(<scope>): <summary>`. Scopes: `remote`, `edge`, `infra`, `bringup`, `docs`.
- **Verbatim policy: when porting code from `external/AsyncVLA/inference/run_asyncvla.py`, the implementer reads that file and adapts. Don't blindly copy — the original is monolithic and uses globals; we want clean module-level functions.**

---

## Task 0: Investigate prismatic dependency closure

**Why this task exists:** Plan 1 was almost a verbatim copy of my plan text. Plan 2A's first job is to find out which `prismatic.*` imports actually need to resolve at runtime (vs being import-only for unused training paths). We also need to confirm whether the missing `Learning-to-Drive-Anywhere-with-MBRA` and `lerobot` paths matter for inference.

**Files to investigate (read only, no changes):**
- `/home/nop/dev/mywork/raspicat-async-vla/external/AsyncVLA/inference/run_asyncvla.py` (730 lines)
- `/home/nop/dev/mywork/raspicat-async-vla/external/AsyncVLA/prismatic/models/small_head.py`
- `/home/nop/dev/mywork/raspicat-async-vla/external/AsyncVLA/prismatic/models/projectors.py`
- `/home/nop/dev/mywork/raspicat-async-vla/external/AsyncVLA/prismatic/models/action_heads.py`
- `/home/nop/dev/mywork/raspicat-async-vla/external/AsyncVLA/prismatic/extern/hf/modeling_prismatic.py`
- `/home/nop/dev/mywork/raspicat-async-vla/external/AsyncVLA/prismatic/vla/constants.py`
- `/home/nop/dev/mywork/raspicat-async-vla/external/AsyncVLA/pyproject.toml` (pip deps)

**Deliverable:** a short markdown report at `docs/superpowers/notes/2026-04-30-prismatic-deps.md` that lists:
- The minimal set of prismatic modules `RealServer` needs to import for inference (NOT training).
- Whether `lerobot` / `Learning-to-Drive-Anywhere-with-MBRA` are required for inference. If so, which submodules.
- Whether `flash-attn` is required for inference (it's listed for training; check that `OpenVLAForActionPrediction_MMNv1` works without it).
- The exact list of `*--{step}_checkpoint.pt` files referenced (e.g. `edge_adapter--750000_checkpoint.pt`, `pose_projector--750000_checkpoint.pt`, etc.) and whether the HF release at `NHirose/AsyncVLA_release` contains them or only the OmniVLA backbone.

- [ ] **Step 0.1: Inventory checkpoints needed**

```bash
cd /home/nop/dev/mywork/raspicat-async-vla
grep -nE "load_checkpoint\(" external/AsyncVLA/inference/run_asyncvla.py
grep -nE "init_module\(" external/AsyncVLA/inference/run_asyncvla.py
grep -nE "module_name\s*=" external/AsyncVLA/inference/run_asyncvla.py
```

Capture the exact module names. Common candidates (from prior reading): `edge_adapter`, `pose_projector`, `proprio_projector`, `action_head`, `noisy_action_projector`, `action_proj` (Proj_Actiontokens).

- [ ] **Step 0.2: Inventory prismatic imports**

```bash
grep -hE "^from prismatic|^import prismatic" external/AsyncVLA/inference/run_asyncvla.py external/AsyncVLA/prismatic/models/small_head.py external/AsyncVLA/prismatic/models/projectors.py external/AsyncVLA/prismatic/models/action_heads.py | sort -u
```

- [ ] **Step 0.3: Check HF release contents**

```bash
docker run --rm -v /home/nop/dev/mywork/raspicat-async-vla:/workspace python:3.10 bash -c "
  pip install -q huggingface_hub
  python3 -c \"
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files('NHirose/AsyncVLA_release')
for f in files: print(f)
  \"
"
```

The output tells us whether the HF release is just the OmniVLA backbone or includes the smaller checkpoints.

- [ ] **Step 0.4: Check pyproject + missing repos**

Read `external/AsyncVLA/pyproject.toml` for declared dependencies. List anything not standard PyPI (e.g. `git+https://...`).

Search `run_asyncvla.py` for `sys.path.extend(...)`. We saw it adds `../Learning-to-Drive-Anywhere-with-MBRA/train/` and `../lerobot`. Identify which symbols from those paths are actually used at inference time (vs only at training).

- [ ] **Step 0.5: Write the report**

Save to `/home/nop/dev/mywork/raspicat-async-vla/docs/superpowers/notes/2026-04-30-prismatic-deps.md` with sections:
- "Minimum required prismatic modules"
- "External dependencies status" (lerobot, MBRA, flash-attn — required or not)
- "Checkpoint file list" (with HF availability flagged)
- "Open questions" (anything that wasn't conclusively answerable from a code read)

- [ ] **Step 0.6: Commit**

```bash
git add docs/superpowers/notes/2026-04-30-prismatic-deps.md
git commit -m "docs(plan-2a): record prismatic dependency closure for Plan 2A"
```

---

## Task 1: `Dockerfile.remote` for GPU inference

**Files:**
- Create: `docker/Dockerfile.remote`

The Dockerfile must produce an image that can: import `prismatic`, load the HF model, run a CUDA forward pass, and serve gRPC.

- [ ] **Step 1.1: Write `Dockerfile.remote`**

```dockerfile
# CUDA 12.1 + cuDNN runtime so PyTorch 2.2.0+cu121 wheels match the host driver.
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

# Install torch first to pin the CUDA build.
RUN pip3 install --no-cache-dir \
    'torch==2.2.0+cu121' 'torchvision==0.17.0+cu121' 'torchaudio==2.2.0+cu121' \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Common deps. Pin numpy<2 to avoid prismatic surprises.
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

# OPTIONAL: flash-attn (skip if Task 0 confirms inference doesn't need it).
# Build is slow and host-toolchain-sensitive; comment out if not strictly required.
# RUN pip3 install --no-cache-dir packaging ninja && \
#     pip3 install --no-cache-dir 'flash-attn==2.5.5' --no-build-isolation

# Install prismatic from the vendored submodule.
COPY external/AsyncVLA /opt/AsyncVLA
RUN pip3 install --no-cache-dir -e /opt/AsyncVLA

WORKDIR /workspace
```

- [ ] **Step 1.2: Build the image**

```bash
cd /home/nop/dev/mywork/raspicat-async-vla
DOCKER_CONFIG=/tmp/dckr-noauth docker build -f docker/Dockerfile.remote -t raspicat-asyncvla-remote .
```

Expected: image builds with no errors. May take 10–20 minutes (mostly torch + cuda).

- [ ] **Step 1.3: Smoke check — torch.cuda.is_available()**

```bash
DOCKER_CONFIG=/tmp/dckr-noauth docker run --rm --gpus all raspicat-asyncvla-remote python3 -c "
import torch
print('cuda:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
print('torch:', torch.__version__)
"
```

Expected: `cuda: True` and a GPU name. **If False, escalate** — the host driver / NVIDIA container toolkit is misconfigured.

- [ ] **Step 1.4: Smoke check — `import prismatic` works**

```bash
DOCKER_CONFIG=/tmp/dckr-noauth docker run --rm raspicat-asyncvla-remote python3 -c "
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.models.small_head import Edge_adapter
print('prismatic imports: ok')
"
```

Expected: `prismatic imports: ok`. **If ImportError**, fall back to Task 0's report — likely a missing transitive dep.

- [ ] **Step 1.5: Commit**

```bash
git add docker/Dockerfile.remote
git commit -m "feat(infra): add Dockerfile.remote for GPU model serving"
```

---

## Task 2: Checkpoint download script

**Files:**
- Create: `scripts/download_checkpoints.sh`

- [ ] **Step 2.1: Write `download_checkpoints.sh`**

```bash
#!/usr/bin/env bash
# Download AsyncVLA checkpoints to ./AsyncVLA_release/ (the path expected by
# external/AsyncVLA/inference/run_asyncvla.py).
#
# Uses the host's ~/.cache/huggingface so repeat runs are instant.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${REPO_ROOT}/AsyncVLA_release"
HF_REPO="${HF_REPO:-NHirose/AsyncVLA_release}"

mkdir -p "${OUT_DIR}"

python3 - <<PY
import os, sys
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="${HF_REPO}",
    local_dir="${OUT_DIR}",
    local_dir_use_symlinks=False,
)
print(f"Downloaded to: {path}")
print("Files:")
for root, _, files in os.walk(path):
    for f in files:
        full = os.path.join(root, f)
        size_mb = os.path.getsize(full) / (1024 * 1024)
        print(f"  {os.path.relpath(full, path)}  ({size_mb:.1f} MB)")
PY
```

- [ ] **Step 2.2: Make executable**

```bash
chmod +x scripts/download_checkpoints.sh
```

- [ ] **Step 2.3: Run the download once**

```bash
DOCKER_CONFIG=/tmp/dckr-noauth docker run --rm \
  -v /home/nop/dev/mywork/raspicat-async-vla:/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  raspicat-asyncvla-remote bash -lc "cd /workspace && ./scripts/download_checkpoints.sh"
```

Capture the file list. If the HF release contains only the OmniVLA backbone (Task 0 will tell us), this script needs additional `snapshot_download(...)` calls for the supplementary checkpoints — extend the script accordingly.

Add `AsyncVLA_release/` to `.gitignore` (large model weights, shouldn't be committed):

```bash
echo "AsyncVLA_release/" >> .gitignore
```

- [ ] **Step 2.4: Commit**

```bash
git add scripts/download_checkpoints.sh .gitignore
git commit -m "feat(infra): add HF checkpoint download script"
```

---

## Task 3: Port `load_checkpoint` and `init_module` helpers (TDD)

**Files:**
- Create: `src/raspicat_async_vla_remote/asyncvla_remote/model_loader.py`
- Create: `src/raspicat_async_vla_remote/test/test_model_loader.py`

The original `load_checkpoint` is a thin wrapper around `torch.load`. We want unit-testable helpers that we can verify with a fake checkpoint, without needing the real 7.5B-param weights.

- [ ] **Step 3.1: Write the failing test**

`src/raspicat_async_vla_remote/test/test_model_loader.py`:

```python
"""Tests for asyncvla_remote.model_loader helpers (don't load real weights)."""
import os
import tempfile

import pytest
import torch
import torch.nn as nn

from asyncvla_remote.model_loader import load_checkpoint, remove_ddp_prefix


def test_remove_ddp_prefix_strips_module():
    sd = {'module.layer.weight': torch.zeros(1), 'fc.bias': torch.zeros(1)}
    out = remove_ddp_prefix(sd)
    assert set(out.keys()) == {'layer.weight', 'fc.bias'}


def test_load_checkpoint_path_resolution(tmp_path):
    sd = {'foo.weight': torch.zeros(2, 3)}
    cp = tmp_path / 'edge_adapter--42_checkpoint.pt'
    torch.save({'module.foo.weight': torch.zeros(2, 3)}, cp)
    loaded = load_checkpoint('edge_adapter', str(tmp_path), step=42)
    assert 'foo.weight' in loaded
    assert loaded['foo.weight'].shape == (2, 3)


def test_load_checkpoint_pose_projector_fallback(tmp_path):
    """Spec quirk: pose_projector falls back to proprio_projector if missing."""
    sd = {'foo.weight': torch.zeros(1)}
    cp = tmp_path / 'proprio_projector--42_checkpoint.pt'
    torch.save(sd, cp)
    loaded = load_checkpoint('pose_projector', str(tmp_path), step=42)
    assert 'foo.weight' in loaded


def test_load_checkpoint_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint('nonexistent', str(tmp_path), step=42)
```

- [ ] **Step 3.2: Confirm failing**

```bash
DOCKER_CONFIG=/tmp/dckr-noauth docker run --rm --user $(id -u):$(id -g) \
  -v /home/nop/dev/mywork/raspicat-async-vla:/workspace -e HOME=/tmp \
  raspicat-asyncvla-test bash -c "
    cd /workspace/src/raspicat_async_vla_remote && python3 -m pytest test/test_model_loader.py -v
  "
```

Expected: ImportError on `model_loader`.

- [ ] **Step 3.3: Implement `model_loader.py`**

```python
"""Helpers for loading AsyncVLA checkpoints saved by the prismatic training loop."""
from __future__ import annotations

import os
from typing import Dict

import torch


def remove_ddp_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip the 'module.' prefix that DDP adds to checkpoint keys."""
    return {
        (k[len('module.'):] if k.startswith('module.') else k): v
        for k, v in state_dict.items()
    }


def _resolve_path(module_name: str, path: str, step: int) -> str:
    """Match the original run_asyncvla.py behaviour, including the
    pose_projector → proprio_projector fallback."""
    primary = os.path.join(path, f'{module_name}--{step}_checkpoint.pt')
    if os.path.exists(primary):
        return primary
    if module_name == 'pose_projector':
        fallback = os.path.join(path, f'proprio_projector--{step}_checkpoint.pt')
        if os.path.exists(fallback):
            return fallback
    raise FileNotFoundError(
        f'no checkpoint for module={module_name!r} step={step} under {path!r}'
    )


def load_checkpoint(module_name: str, path: str, step: int,
                    device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """Load a `<module>--{step}_checkpoint.pt` and strip DDP wrapping."""
    full = _resolve_path(module_name, path, step)
    state_dict = torch.load(full, map_location=device)
    return remove_ddp_prefix(state_dict)
```

- [ ] **Step 3.4: Confirm tests pass**

```bash
DOCKER_CONFIG=/tmp/dckr-noauth docker run --rm --user $(id -u):$(id -g) \
  -v /home/nop/dev/mywork/raspicat-async-vla:/workspace -e HOME=/tmp \
  raspicat-asyncvla-test bash -c "
    cd /workspace/src/raspicat_async_vla_remote && python3 -m pytest test/test_model_loader.py -v
  "
```

Expected: 4 passed.

- [ ] **Step 3.5: Commit**

```bash
git add src/raspicat_async_vla_remote/asyncvla_remote/model_loader.py \
        src/raspicat_async_vla_remote/test/test_model_loader.py
git commit -m "feat(remote): add model_loader checkpoint helpers with TDD"
```

---

## Task 4: Port `data_transformer_asyncvla` (image + goal → batch)

**Files:**
- Create: `src/raspicat_async_vla_remote/asyncvla_remote/data_transform.py`
- Create: `src/raspicat_async_vla_remote/test/test_data_transform.py`

The original `data_transformer_asyncvla` (in `Inference` class) builds the batch from PIL images + lang_inst + goal_image_PIL + goal_pose_loc_norm. We extract this into a free function `build_inference_batch(...)` we can test.

**Implementer note:** open `external/AsyncVLA/inference/run_asyncvla.py` lines ~407–428 to read the original. Adapt — don't copy line-for-line — into a function with this signature:

```python
def build_inference_batch(
    *,
    current_image: PIL.Image.Image,
    past_image: PIL.Image.Image | None,        # may be None at first frame
    lang_instruction: str,                      # '' if no language goal
    goal_image: PIL.Image.Image | None,
    goal_pose_xy_theta: tuple[float, float, float] | None,  # in odom or as-given
    action_tokenizer,                           # from prismatic.vla.action_tokenizer
    processor,                                  # PrismaticProcessor
    num_images_in_input: int = 2,
) -> dict[str, torch.Tensor]:
    """Returns a batch dict with input_ids, attention_mask, pixel_values,
    labels, goal_pose. Mirrors the format that run_asyncvla.run_forward_pass
    feeds into vla(...)."""
```

- [ ] **Step 4.1: Read the original implementation**

```bash
sed -n '355,470p' external/AsyncVLA/inference/run_asyncvla.py
```

- [ ] **Step 4.2: Write the failing test**

`test_data_transform.py` (using a stub processor / tokenizer to avoid loading the real OmniVLA):

```python
"""Tests for data_transform.build_inference_batch shape/key contract.

We can't easily run the real PrismaticProcessor without a 7.5B-param model
loaded, so we use stubs that record what they were called with and return
shape-compatible tensors. The test asserts the *output contract* of our
batch-builder, not the correctness of the full prismatic pipeline.
"""
from unittest.mock import MagicMock

import torch
from PIL import Image

from asyncvla_remote.data_transform import build_inference_batch


def _make_pil(size=(224, 224)):
    return Image.new('RGB', size, color=(127, 127, 127))


def _stub_processor():
    p = MagicMock()
    p.tokenizer.pad_token_id = 0
    p.tokenizer.return_value = {
        'input_ids': torch.zeros(1, 16, dtype=torch.long),
        'attention_mask': torch.ones(1, 16, dtype=torch.long),
    }
    p.image_processor.return_value = {
        'pixel_values': torch.zeros(1, 6, 224, 224, dtype=torch.float32),
    }
    p.return_value = {
        'input_ids': torch.zeros(1, 16, dtype=torch.long),
        'attention_mask': torch.ones(1, 16, dtype=torch.long),
        'pixel_values': torch.zeros(1, 6, 224, 224, dtype=torch.float32),
        'labels': torch.zeros(1, 16, dtype=torch.long),
    }
    return p


def _stub_tokenizer():
    t = MagicMock()
    t.encode_actions = MagicMock(return_value=[1] * 7)
    return t


def test_build_inference_batch_returns_required_keys():
    batch = build_inference_batch(
        current_image=_make_pil(),
        past_image=_make_pil(),
        lang_instruction='go to the door',
        goal_image=None,
        goal_pose_xy_theta=(1.0, 0.0, 0.0),
        action_tokenizer=_stub_tokenizer(),
        processor=_stub_processor(),
        num_images_in_input=2,
    )
    for k in ('input_ids', 'attention_mask', 'pixel_values', 'labels', 'goal_pose'):
        assert k in batch, f'missing key: {k}'
    assert batch['goal_pose'].shape[-1] >= 2  # at least (x, y) — actual layout per spec


def test_build_inference_batch_handles_missing_past_image():
    batch = build_inference_batch(
        current_image=_make_pil(),
        past_image=None,            # first frame — uses current_image as past
        lang_instruction='',
        goal_image=None,
        goal_pose_xy_theta=(1.0, 0.0, 0.0),
        action_tokenizer=_stub_tokenizer(),
        processor=_stub_processor(),
    )
    assert 'pixel_values' in batch
```

- [ ] **Step 4.3: Run, see fail**

```bash
DOCKER_CONFIG=/tmp/dckr-noauth docker run --rm --user $(id -u):$(id -g) \
  -v /home/nop/dev/mywork/raspicat-async-vla:/workspace -e HOME=/tmp \
  raspicat-asyncvla-test bash -c "
    cd /workspace/src/raspicat_async_vla_remote && python3 -m pytest test/test_data_transform.py -v
  "
```

Expected: ImportError on `data_transform`.

- [ ] **Step 4.4: Implement `data_transform.py`**

This is non-mechanical — read `external/AsyncVLA/inference/run_asyncvla.py:407–428` (`data_transformer_asyncvla`) and port to a free function. Key adaptations:
- Drop the `self.` prefixes; pass `action_tokenizer` and `processor` as parameters.
- Make `past_image` default to `current_image` if None (first-frame case).
- Make `goal_image` and `goal_pose_xy_theta` optional and set the modality flags accordingly.
- Return a dict whose keys match what the test expects AND what the real `vla(...)` call needs.

If, after reading the original, the modality-flag logic is too entangled, factor it out into a helper:

```python
def determine_modality_id(*, has_lang: bool, has_pose_goal: bool, has_image_goal: bool) -> torch.Tensor:
    """Port the if/elif tree from run_forward_pass:485–516."""
    ...
```

- [ ] **Step 4.5: Run tests**

Expected: 2 passed (or, if the implementer found edge cases, both still passing).

- [ ] **Step 4.6: Commit**

```bash
git add src/raspicat_async_vla_remote/asyncvla_remote/data_transform.py \
        src/raspicat_async_vla_remote/test/test_data_transform.py
git commit -m "feat(remote): add data_transform.build_inference_batch with TDD"
```

---

## Task 5: `inference_engine.py` — full Remote forward pass

**Files:**
- Create: `src/raspicat_async_vla_remote/asyncvla_remote/inference_engine.py`
- Create: `src/raspicat_async_vla_remote/test/test_inference_engine_smoke.py`

This is the heart of Plan 2A. The engine loads the OmniVLA backbone + action_proj + pose_projector once, then per-call runs the forward pass and returns a `projected_actions` tensor matching the proto contract.

**Implementer reference:** `external/AsyncVLA/inference/run_asyncvla.py:481–600` (`run_forward_pass`).

- [ ] **Step 5.1: Sketch the public API**

```python
class InferenceEngine:
    def __init__(
        self,
        *,
        vla_path: str,                  # e.g. './AsyncVLA_release'
        resume_step: int = 750000,
        device: str = 'cuda:0',
        dtype: torch.dtype = torch.bfloat16,
        num_images_in_input: int = 2,
    ):
        ...

    def warmup(self, num_iters: int = 1) -> None: ...

    def infer(
        self,
        *,
        current_image: PIL.Image.Image,
        past_image: PIL.Image.Image | None,
        lang_instruction: str,
        goal_image: PIL.Image.Image | None,
        goal_pose_xy_theta: tuple[float, float, float] | None,
    ) -> tuple[np.ndarray, dict]:
        """Returns (projected_actions, metrics).

        projected_actions: shape (num_tokens, embed_dim) float32, where
          num_tokens = NUM_ACTIONS_CHUNK * ACTION_DIM (typically 8 × 4 = 32)
          embed_dim  = depends on action_proj output (1024 in spec)
        metrics: {'inference_ms': float, 'modality_id': int}
        """
```

- [ ] **Step 5.2: Implement `inference_engine.py`**

Key porting from run_forward_pass:
1. Build batch: call `data_transform.build_inference_batch(...)` from Task 4.
2. Forward through `vla` with `torch.autocast('cuda', dtype=bfloat16)` and `torch.no_grad()`.
3. Pull `last_hidden_states` and slice out `actions_hidden_states` using the masking trick from `prismatic.training.train_utils.get_current_action_mask` / `get_next_actions_mask`.
4. Call `action_proj.predict_action(actions_hidden_states, modality_id)` — this returns `projected_actions`.
5. Detach, move to CPU, cast to float32, convert to numpy.

```python
"""Real OmniVLA inference engine. Returns the projected_actions tensor that
Plan 1's gRPC payload was carrying as a dummy."""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import PIL.Image
import torch
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

# Late imports inside functions to keep the test image start-up fast and
# isolate the heavy framework bring-up to the inference path.

from .data_transform import build_inference_batch
from .model_loader import load_checkpoint


class InferenceEngine:

    def __init__(
        self,
        *,
        vla_path: str,
        resume_step: int = 750000,
        device: str = 'cuda:0',
        dtype: torch.dtype = torch.bfloat16,
        num_images_in_input: int = 2,
    ) -> None:
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        from prismatic.vla.action_tokenizer import ActionTokenizer
        from prismatic.models.projectors import ProprioProjector
        from prismatic.models.action_heads import L1RegressionActionHead_idcat
        from prismatic.models.small_head import Proj_Actiontokens

        # 1. Register custom types with HF AutoConfig so AutoModel can find them.
        AutoConfig.register('openvla', OpenVLAConfig, exist_ok=True)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, PrismaticImageProcessor, exist_ok=True)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1, exist_ok=True)

        self.device = torch.device(device)
        self.dtype = dtype
        self.num_images_in_input = num_images_in_input

        # 2. Load processor + main VLA backbone.
        self.processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            vla_path, torch_dtype=dtype, trust_remote_code=True,
        ).to(self.device).eval()

        # 3. Load auxiliary heads from raw checkpoints.
        # The exact init args for L1RegressionActionHead_idcat / ProprioProjector /
        # Proj_Actiontokens depend on the trained config. Read prismatic source to
        # confirm; values below are placeholders the implementer must verify.
        # (Hint: `external/AsyncVLA/experiments/.../config.yaml` typically records them.)
        self.action_head = L1RegressionActionHead_idcat(...).to(self.device).to(dtype).eval()
        self.action_head.load_state_dict(load_checkpoint('action_head', vla_path, resume_step, device=str(self.device)))

        self.action_proj = Proj_Actiontokens(...).to(self.device).to(dtype).eval()
        self.action_proj.load_state_dict(load_checkpoint('action_proj', vla_path, resume_step, device=str(self.device)))

        self.pose_projector = ProprioProjector(...).to(self.device).to(dtype).eval()
        self.pose_projector.load_state_dict(load_checkpoint('pose_projector', vla_path, resume_step, device=str(self.device)))

        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)

    @torch.no_grad()
    def warmup(self, num_iters: int = 1) -> None:
        dummy = PIL.Image.new('RGB', (224, 224))
        for _ in range(num_iters):
            self.infer(
                current_image=dummy, past_image=dummy,
                lang_instruction='warmup', goal_image=None,
                goal_pose_xy_theta=(0.0, 0.0, 0.0),
            )

    @torch.no_grad()
    def infer(
        self,
        *,
        current_image: PIL.Image.Image,
        past_image: Optional[PIL.Image.Image],
        lang_instruction: str,
        goal_image: Optional[PIL.Image.Image],
        goal_pose_xy_theta: Optional[Tuple[float, float, float]],
    ) -> Tuple[np.ndarray, dict]:
        from prismatic.training.train_utils import (
            get_current_action_mask, get_next_actions_mask,
        )
        from prismatic.vla.constants import NUM_ACTIONS_CHUNK, ACTION_DIM

        t0 = time.monotonic()
        batch = build_inference_batch(
            current_image=current_image,
            past_image=past_image or current_image,
            lang_instruction=lang_instruction,
            goal_image=goal_image,
            goal_pose_xy_theta=goal_pose_xy_theta,
            action_tokenizer=self.action_tokenizer,
            processor=self.processor,
            num_images_in_input=self.num_images_in_input,
        )

        with torch.autocast('cuda', dtype=self.dtype):
            output = self.vla(
                input_ids=batch['input_ids'].to(self.device),
                attention_mask=batch['attention_mask'].to(self.device),
                pixel_values=batch['pixel_values'].to(self.dtype).to(self.device),
                modality_id=batch['modality_id'].to(self.dtype).to(self.device),
                labels=batch['labels'].to(self.device),
                output_hidden_states=True,
                proprio=batch['goal_pose'].to(self.dtype).to(self.device),
                proprio_projector=self.pose_projector,
                use_film=False,
            )

        last_hidden = output.hidden_states[-1]
        gt_token_ids = batch['labels'][:, 1:].to(self.device)
        cur_mask = get_current_action_mask(gt_token_ids)
        next_mask = get_next_actions_mask(gt_token_ids)

        num_patches = batch['num_patches']  # build_inference_batch must surface this
        text_hidden = last_hidden[:, num_patches:-1]
        actions_hidden = (
            text_hidden[cur_mask | next_mask]
            .reshape(1, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(self.dtype)
        )

        projected = self.action_proj.predict_action(
            actions_hidden.detach(), batch['modality_id'].to(self.dtype).to(self.device),
        )
        projected_np = projected.detach().cpu().to(torch.float32).numpy()
        projected_np = projected_np.reshape(projected_np.shape[-2], projected_np.shape[-1])

        return projected_np, {
            'inference_ms': (time.monotonic() - t0) * 1000.0,
            'modality_id': int(batch['modality_id'].item()),
        }
```

**Important:** the placeholder `...` values for `L1RegressionActionHead_idcat(...)`, `ProprioProjector(...)`, `Proj_Actiontokens(...)` constructor arguments must be filled in by reading `prismatic` source files — this is exactly the kind of thing the implementer must investigate, not guess. If the original `define_model` (run_asyncvla.py:603+) gives concrete values, copy them exactly.

- [ ] **Step 5.3: Smoke test (slow, GPU-only, opt-in)**

`test_inference_engine_smoke.py`:

```python
"""Slow GPU-only smoke test. Skipped unless ASYNCVLA_E2E=1.

Run inside Dockerfile.remote with --gpus all and the AsyncVLA_release/
directory mounted."""
import os
import pytest

# Skip the entire module unless the opt-in env var is set.
if os.environ.get('ASYNCVLA_E2E') != '1':
    pytest.skip('set ASYNCVLA_E2E=1 to run', allow_module_level=True)


def test_inference_engine_returns_correct_shape():
    import PIL.Image
    from asyncvla_remote.inference_engine import InferenceEngine

    eng = InferenceEngine(
        vla_path='/workspace/AsyncVLA_release',
        resume_step=750000,
        device='cuda:0',
    )
    eng.warmup(num_iters=1)

    img = PIL.Image.new('RGB', (224, 224), (128, 128, 128))
    proj, metrics = eng.infer(
        current_image=img, past_image=img,
        lang_instruction='go forward',
        goal_image=None, goal_pose_xy_theta=(1.0, 0.0, 0.0),
    )
    assert proj.ndim == 2
    assert proj.dtype.name in ('float32',)
    assert metrics['inference_ms'] > 0
    print(f'projected_actions shape={proj.shape} inf_ms={metrics["inference_ms"]:.1f}')
```

- [ ] **Step 5.4: Run smoke test on a GPU host**

```bash
DOCKER_CONFIG=/tmp/dckr-noauth docker run --rm --gpus all \
  -v /home/nop/dev/mywork/raspicat-async-vla:/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e ASYNCVLA_E2E=1 \
  raspicat-asyncvla-remote bash -lc "
    cd /workspace
    pip install -e src/raspicat_async_vla_proto src/raspicat_async_vla_remote >/dev/null 2>&1 || true
    cd src/raspicat_async_vla_remote
    python3 -m pytest test/test_inference_engine_smoke.py -v -s
  "
```

Expected: PASSED with `projected_actions shape=(N, D)` printed. **If shape doesn't match `(num_tokens, embed_dim)`, fix the reshape in `infer()`** before continuing.

- [ ] **Step 5.5: Commit**

```bash
git add src/raspicat_async_vla_remote/asyncvla_remote/inference_engine.py \
        src/raspicat_async_vla_remote/test/test_inference_engine_smoke.py
git commit -m "feat(remote): add real InferenceEngine wrapping OmniVLA + Token Projector"
```

---

## Task 6: `RealServer` — gRPC servicer using `InferenceEngine`

**Files:**
- Create: `src/raspicat_async_vla_remote/asyncvla_remote/real_server.py`
- Modify: `src/raspicat_async_vla_remote/asyncvla_remote/server_main.py` (add `--backend dummy|real` switch)

- [ ] **Step 6.1: Implement `real_server.py`**

```python
"""gRPC servicer that wraps InferenceEngine (real OmniVLA model)."""
from __future__ import annotations

import io
import logging
import threading
import time
from concurrent import futures
from typing import Iterator, Optional

import grpc
import numpy as np
import PIL.Image

from raspicat_async_vla_proto import asyncvla_pb2, asyncvla_pb2_grpc
from raspicat_async_vla_proto.conversions import float32_array_to_fp16_bytes

from .inference_engine import InferenceEngine


_LOG = logging.getLogger(__name__)


def _proto_goal_to_python(goal: asyncvla_pb2.GoalSpec):
    if goal.mode == asyncvla_pb2.GoalSpec.POSE:
        return ('pose', (goal.pose.x, goal.pose.y, goal.pose.theta), '', None)
    if goal.mode == asyncvla_pb2.GoalSpec.TEXT:
        return ('text', None, goal.text, None)
    if goal.mode == asyncvla_pb2.GoalSpec.IMAGE:
        img = PIL.Image.open(io.BytesIO(goal.image_jpeg)).convert('RGB')
        return ('image', None, '', img)
    raise ValueError(f'unknown goal mode {goal.mode}')


class _Servicer(asyncvla_pb2_grpc.AsyncVLAServiceServicer):

    def __init__(self, *, engine: InferenceEngine, model_version: str = 'asyncvla-real') -> None:
        self._engine = engine
        self._model_version = model_version
        self._past_image_per_client = {}
        self._past_image_lock = threading.Lock()

    def _get_set_past(self, peer: str, current: PIL.Image.Image) -> PIL.Image.Image:
        with self._past_image_lock:
            past = self._past_image_per_client.get(peer, current)
            self._past_image_per_client[peer] = current
            return past

    def GetModelInfo(self, request, context):
        # NOTE: num_tokens / embed_dim should match what InferenceEngine produces.
        # Plumb these through engine config in a follow-up if they're not constant.
        return asyncvla_pb2.ModelInfo(
            model_name='NHirose/AsyncVLA_release',
            model_version=self._model_version,
            num_tokens=8,        # to be confirmed against engine output
            embed_dim=1024,      # to be confirmed against engine output
            device=str(self._engine.device),
            ready=True,
        )

    def StreamInfer(
        self,
        request_iterator: Iterator[asyncvla_pb2.Observation],
        context,
    ) -> Iterator[asyncvla_pb2.ActionEmbedding]:
        peer = context.peer()
        for obs in request_iterator:
            cur_img = PIL.Image.open(io.BytesIO(obs.image_jpeg)).convert('RGB').resize(
                (obs.image_width or 224, obs.image_height or 224), PIL.Image.BILINEAR,
            )
            past_img = self._get_set_past(peer, cur_img)
            mode, pose, text, goal_img = _proto_goal_to_python(obs.goal)

            try:
                proj, metrics = self._engine.infer(
                    current_image=cur_img,
                    past_image=past_img,
                    lang_instruction=text,
                    goal_image=goal_img,
                    goal_pose_xy_theta=pose,
                )
            except Exception as exc:  # noqa: BLE001
                _LOG.exception('inference failed for frame_id=%s: %s', obs.frame_id, exc)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(exc))
                return

            num_tokens, embed_dim = proj.shape
            yield asyncvla_pb2.ActionEmbedding(
                frame_id=obs.frame_id,
                server_time_ns=time.monotonic_ns(),
                num_tokens=int(num_tokens),
                embed_dim=int(embed_dim),
                embedding_fp16=float32_array_to_fp16_bytes(proj.astype(np.float32)),
                inference_ms=float(metrics['inference_ms']),
                model_version=self._model_version,
            )


class RealServer:
    def __init__(
        self,
        *,
        engine: InferenceEngine,
        host: str = '0.0.0.0',
        port: int = 50051,
        max_workers: int = 4,
        model_version: str = 'asyncvla-real',
    ) -> None:
        self._engine = engine
        self._host = host
        self._port = port
        self._max_workers = max_workers
        self._servicer = _Servicer(engine=engine, model_version=model_version)
        self._server: Optional[grpc.Server] = None
        self._actual_port: Optional[int] = None

    def start(self) -> int:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=self._max_workers))
        asyncvla_pb2_grpc.add_AsyncVLAServiceServicer_to_server(self._servicer, server)
        self._actual_port = server.add_insecure_port(f'{self._host}:{self._port}')
        server.start()
        self._server = server
        return self._actual_port

    def stop(self, grace_sec: float = 1.0) -> None:
        if self._server is not None:
            self._server.stop(grace_sec)
            self._server = None

    def wait_for_termination(self) -> None:
        if self._server is not None:
            self._server.wait_for_termination()
```

- [ ] **Step 6.2: Modify `server_main.py` to switch backend**

Read current `server_main.py` then add a `--backend` flag and a `--vla-path`/`--resume-step` group used only when `--backend=real`. Default remains `dummy` so existing tests don't break.

- [ ] **Step 6.3: Manual smoke test (GPU host)**

```bash
DOCKER_CONFIG=/tmp/dckr-noauth docker run --rm --gpus all \
  -v /home/nop/dev/mywork/raspicat-async-vla:/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  raspicat-asyncvla-remote bash -lc "
    cd /workspace
    pip install -e src/raspicat_async_vla_proto src/raspicat_async_vla_remote >/dev/null 2>&1 || true
    python3 -m asyncvla_remote.server_main --backend real --vla-path /workspace/AsyncVLA_release --port 50051 &
    SERVER_PID=\$!
    sleep 30   # model load is slow
    python3 -c '
import grpc, time
from raspicat_async_vla_proto import asyncvla_pb2, asyncvla_pb2_grpc
ch = grpc.insecure_channel(\"localhost:50051\")
stub = asyncvla_pb2_grpc.AsyncVLAServiceStub(ch)
info = stub.GetModelInfo(asyncvla_pb2.ModelInfoRequest(), timeout=60)
print(\"ready:\", info.ready, \"device:\", info.device)
'
    kill \$SERVER_PID
  "
```

Expected: `ready: True`, GPU device name printed.

- [ ] **Step 6.4: Commit**

```bash
git add src/raspicat_async_vla_remote/asyncvla_remote/real_server.py \
        src/raspicat_async_vla_remote/asyncvla_remote/server_main.py
git commit -m "feat(remote): add RealServer wired to InferenceEngine"
```

---

## Task 7: Edge — load `Edge_adapter` (TDD against fake checkpoint)

**Files:**
- Create: `src/raspicat_async_vla_edge/asyncvla_edge/edge_adapter.py`
- Create: `src/raspicat_async_vla_edge/test/test_edge_adapter.py`

The `Edge_adapter` class lives in `prismatic.models.small_head` (~5M params). Plan 2A's edge module wraps it for clean loading on the Pi.

- [ ] **Step 7.1: Write failing test (with synthetic checkpoint)**

```python
"""Tests for edge_adapter.load_edge_adapter using a synthetic state_dict."""
import os
import torch
import pytest

from prismatic.models.small_head import Edge_adapter
from asyncvla_edge.edge_adapter import load_edge_adapter


def _save_random_checkpoint(tmp_path, step: int = 750000):
    model = Edge_adapter()
    cp = tmp_path / f'edge_adapter--{step}_checkpoint.pt'
    torch.save(model.state_dict(), cp)
    return str(tmp_path)


def test_load_edge_adapter_returns_eval_module(tmp_path):
    path = _save_random_checkpoint(tmp_path)
    adapter = load_edge_adapter(path=path, step=750000, device='cpu')
    assert isinstance(adapter, Edge_adapter)
    assert not adapter.training


def test_load_edge_adapter_runs_forward_with_dummy_inputs(tmp_path):
    path = _save_random_checkpoint(tmp_path)
    adapter = load_edge_adapter(path=path, step=750000, device='cpu')
    obs = torch.randn(1, 3, 96, 96)
    past = torch.randn(1, 3, 96, 96)
    # vla_feature shape per Edge_adapter contract: (B, num_action_tokens, D).
    # Both num and D depend on training config; pick small smoke values.
    vla_feat = torch.randn(1, 32, 1024)
    out = adapter(obs, past, vla_feat)
    assert out.ndim >= 2  # delta-pose chunk
```

- [ ] **Step 7.2: Run, see fail**

```bash
DOCKER_CONFIG=/tmp/dckr-noauth docker run --rm --user $(id -u):$(id -g) \
  -v /home/nop/dev/mywork/raspicat-async-vla:/workspace -e HOME=/tmp \
  raspicat-asyncvla-test bash -c "
    pip install -e /workspace/external/AsyncVLA --no-deps > /dev/null 2>&1 || true
    cd /workspace/src/raspicat_async_vla_edge && python3 -m pytest test/test_edge_adapter.py -v
  "
```

(Adjust the test image to include prismatic; if too heavy, push edge_adapter tests to run inside `Dockerfile.remote` instead. Implementer judgement.)

- [ ] **Step 7.3: Implement `edge_adapter.py`**

```python
"""Wrapper for prismatic.models.small_head.Edge_adapter."""
from __future__ import annotations

from typing import Optional

import torch

from prismatic.models.small_head import Edge_adapter


def load_edge_adapter(
    *,
    path: str,
    step: int,
    device: str = 'cpu',
    dtype: Optional[torch.dtype] = None,
) -> Edge_adapter:
    """Construct an Edge_adapter, load its checkpoint, return it in eval mode."""
    import os
    candidate = os.path.join(path, f'edge_adapter--{step}_checkpoint.pt')
    if not os.path.exists(candidate):
        raise FileNotFoundError(f'no checkpoint at {candidate}')
    adapter = Edge_adapter()
    raw = torch.load(candidate, map_location=device)
    cleaned = {(k[7:] if k.startswith('module.') else k): v for k, v in raw.items()}
    adapter.load_state_dict(cleaned)
    adapter = adapter.to(device).eval()
    if dtype is not None:
        adapter = adapter.to(dtype)
    return adapter
```

- [ ] **Step 7.4: Tests pass**

Expected: 2 passed.

- [ ] **Step 7.5: Commit**

```bash
git add src/raspicat_async_vla_edge/asyncvla_edge/edge_adapter.py \
        src/raspicat_async_vla_edge/test/test_edge_adapter.py
git commit -m "feat(edge): add Edge_adapter loader with TDD"
```

---

## Task 8: Edge inference — embedding + (cur, past) images → Path

**Files:**
- Create: `src/raspicat_async_vla_edge/asyncvla_edge/edge_inference.py`
- Create: `src/raspicat_async_vla_edge/test/test_edge_inference.py`

The original `delta_to_pose` function (`run_asyncvla.py:92–135`) converts the adapter's delta output to world-frame poses; we port it.

- [ ] **Step 8.1: Test (with random Edge_adapter and synthetic embedding)**

```python
"""Tests for asyncvla_edge.edge_inference."""
import torch
import numpy as np

from asyncvla_edge.edge_inference import (
    delta_to_pose,
    embedding_to_path,
    preprocess_for_edge_adapter,
)


def test_delta_to_pose_zero_input_zero_pose():
    delta = torch.zeros(1, 8, 4)
    delta[..., 2] = 1.0  # cos(theta=0)
    pose = delta_to_pose(delta)
    assert pose.shape == (1, 8, 4)
    assert torch.allclose(pose[..., :2], torch.zeros_like(pose[..., :2]), atol=1e-5)


def test_preprocess_for_edge_adapter_returns_normalized_96():
    rng = np.random.default_rng(seed=0)
    img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    t = preprocess_for_edge_adapter(img, dtype=torch.float32)
    assert t.shape == (1, 3, 96, 96)
    # ImageNet-normalized means ~[-2.1, 2.6] is plausible
    assert -3.0 < t.min().item() < 0.0 < t.max().item() < 3.0


def test_embedding_to_path_with_stub_adapter_shape():
    """We stub the adapter so this runs on CPU without checkpoints."""
    import torch.nn as nn

    class StubAdapter(nn.Module):
        def forward(self, obs, past, vla):
            return torch.zeros(obs.shape[0], 8, 4)  # delta = identity

    adapter = StubAdapter()
    rng = np.random.default_rng(seed=0)
    cur = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    past = cur.copy()
    embedding = np.zeros((32, 1024), dtype=np.float32)

    path = embedding_to_path(
        adapter=adapter,
        cur_image_rgb=cur, past_image_rgb=past,
        embedding=embedding,
        embedding_shape=(1, 32, 1024),
        device='cpu', dtype=torch.float32,
        frame_id='base_link',
    )
    # 8 action steps + 0 origin → 8 PoseStamped
    assert len(path.poses) == 8
    assert path.header.frame_id == 'base_link'
```

- [ ] **Step 8.2: Implement `edge_inference.py`**

Read `run_asyncvla.py:92–135` for `delta_to_pose`. Port carefully — note the loop over `t = 1..T-1` accumulates poses iteratively via cos/sin; copy the math, not the imports.

```python
"""Edge-side inference: embedding + images -> nav_msgs/Path."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


_IMAGENET_NORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
)


def preprocess_for_edge_adapter(
    image_rgb: np.ndarray,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """RGB uint8 HxWx3 → (1, 3, 96, 96), ImageNet-normalized in `dtype`."""
    img = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    t = _IMAGENET_NORM(t).unsqueeze(0).to(dtype)
    return t


def delta_to_pose(delta: torch.Tensor) -> torch.Tensor:
    """Port of run_asyncvla.delta_to_pose. delta: (N, T, 4) -> (N, T, 4)
    where last dim is (x, y, cos(theta), sin(theta))."""
    dx, dy = delta[..., 0], delta[..., 1]
    dtheta = torch.atan2(delta[..., 3], delta[..., 2])
    N, T = dx.shape

    poses: list[torch.Tensor] = []
    x, y, theta = dx[:, 0], dy[:, 0], dtheta[:, 0]
    poses.append(torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1))
    for t in range(1, T):
        ct, st = torch.cos(theta), torch.sin(theta)
        dx_w = ct * dx[:, t] - st * dy[:, t]
        dy_w = st * dx[:, t] + ct * dy[:, t]
        x = x + dx_w
        y = y + dy_w
        theta = theta + dtheta[:, t]
        poses.append(torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1))
    return torch.stack(poses, dim=1)


def embedding_to_path(
    *,
    adapter: nn.Module,
    cur_image_rgb: np.ndarray,
    past_image_rgb: np.ndarray,
    embedding: np.ndarray,
    embedding_shape: Tuple[int, int, int],   # (B, num_tokens, embed_dim)
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
    frame_id: str = 'base_link',
) -> Path:
    """Run Edge_adapter, return nav_msgs/Path in `frame_id`."""
    cur = preprocess_for_edge_adapter(cur_image_rgb, dtype=dtype).to(device)
    past = preprocess_for_edge_adapter(past_image_rgb, dtype=dtype).to(device)

    B, num, D = embedding_shape
    feat = torch.from_numpy(embedding).reshape(B, num, D).to(dtype).to(device)

    with torch.no_grad():
        delta = adapter(cur, past, feat)  # shape: (B, T, 4)
    poses = delta_to_pose(delta).squeeze(0).cpu().numpy()  # (T, 4)

    path = Path()
    path.header.frame_id = frame_id
    for x, y, c, s in poses:
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.orientation.z = float(s)  # placeholder; full quat math optional
        ps.pose.orientation.w = float(c)
        path.poses.append(ps)
    return path
```

- [ ] **Step 8.3: Test runs (CPU OK)**

Expected: 3 passed.

- [ ] **Step 8.4: Commit**

```bash
git add src/raspicat_async_vla_edge/asyncvla_edge/edge_inference.py \
        src/raspicat_async_vla_edge/test/test_edge_inference.py
git commit -m "feat(edge): add edge_inference (Edge_adapter forward + delta_to_pose)"
```

---

## Task 9: Wire edge_node to use real adapter

**Files:**
- Modify: `src/raspicat_async_vla_edge/asyncvla_edge/edge_node.py`
- Modify: `src/raspicat_async_vla_edge/config/edge_params.yaml` (add `edge_adapter_path`, `edge_adapter_step`, `use_real_adapter`)

- [ ] **Step 9.1: Add params**

Append to `config/edge_params.yaml`:

```yaml
    use_real_adapter: false
    edge_adapter_path: "/workspace/AsyncVLA_release"
    edge_adapter_step: 750000
    edge_device: "cpu"
```

- [ ] **Step 9.2: Modify `edge_node.py`**

Add to `__init__` parameter list:

```python
self.declare_parameter('use_real_adapter', False)
self.declare_parameter('edge_adapter_path', '/workspace/AsyncVLA_release')
self.declare_parameter('edge_adapter_step', 750000)
self.declare_parameter('edge_device', 'cpu')
```

In `on_configure`, conditionally load the real adapter:

```python
self._real_adapter = None
self._past_image_rgb = None
if self.get_parameter('use_real_adapter').value:
    from .edge_adapter import load_edge_adapter
    self._real_adapter = load_edge_adapter(
        path=str(self.get_parameter('edge_adapter_path').value),
        step=int(self.get_parameter('edge_adapter_step').value),
        device=str(self.get_parameter('edge_device').value),
    )
```

In `_action_tick`, branch on `self._real_adapter`:

```python
if self._real_adapter is not None and emb is not None:
    from .edge_inference import embedding_to_path
    cur = self._latest_image  # already RGB ndarray
    past = self._past_image_rgb if self._past_image_rgb is not None else cur
    path = embedding_to_path(
        adapter=self._real_adapter,
        cur_image_rgb=cur,
        past_image_rgb=past,
        embedding=emb.embedding,
        embedding_shape=(1, emb.num_tokens, emb.embed_dim),
        device=str(self.get_parameter('edge_device').value),
        dtype=torch.float32,
        frame_id='base_link',
    )
    path.header.stamp = self.get_clock().now().to_msg()
    self._past_image_rgb = cur.copy()
else:
    path = _stub_adapter_to_path(emb)
    path.header.stamp = self.get_clock().now().to_msg()
self._path_pub.publish(path)
```

Make sure `import torch` is at the module top (currently isn't).

In `on_cleanup`, drop `self._real_adapter = None` and `self._past_image_rgb = None` to release GPU memory (even on CPU device, free the tensors).

- [ ] **Step 9.3: Smoke test — Plan 1's existing `test_edge_node_smoke.py` must still pass**

```bash
# Same Docker run as Plan 1 verification
DOCKER_CONFIG=/tmp/dckr-noauth docker run --rm --user $(id -u):$(id -g) \
  -v /home/nop/dev/mywork/raspicat-async-vla:/workspace -e HOME=/tmp \
  raspicat-asyncvla-test bash -c "
    source /opt/ros/humble/setup.bash; cd /workspace; source install/setup.bash
    cd src/raspicat_async_vla_edge && python3 -m pytest test/test_edge_node_smoke.py -v
  "
```

Expected: 1 passed (use_real_adapter defaults to False, so behaviour matches Plan 1).

- [ ] **Step 9.4: Commit**

```bash
git add src/raspicat_async_vla_edge/asyncvla_edge/edge_node.py \
        src/raspicat_async_vla_edge/config/edge_params.yaml
git commit -m "feat(edge): wire edge_node to optionally use real Edge_adapter"
```

---

## Task 10: `mvp_real.launch.py` — full real-stack bringup

**Files:**
- Create: `src/raspicat_async_vla_bringup/launch/mvp_real.launch.py`

Mirrors `mvp_local.launch.py` but uses `--backend real` and sets `use_real_adapter:=true`.

- [ ] **Step 10.1: Implement `mvp_real.launch.py`**

```python
"""Launch the real-model AsyncVLA stack (GPU on remote, CPU on edge)."""
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
    vla_path = LaunchConfiguration('vla_path')
    edge_device = LaunchConfiguration('edge_device')

    edge_config = os.path.join(
        get_package_share_directory('raspicat_async_vla_edge'),
        'config', 'edge_params.yaml',
    )

    real_server = ExecuteProcess(
        cmd=[
            'python3', '-m', 'asyncvla_remote.server_main',
            '--backend', 'real',
            '--port', grpc_port,
            '--vla-path', vla_path,
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
            'use_real_adapter': True,
            'edge_adapter_path': vla_path,
            'edge_device': edge_device,
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
        DeclareLaunchArgument('vla_path', default_value='/workspace/AsyncVLA_release'),
        DeclareLaunchArgument('edge_device', default_value='cpu'),
        real_server,
        edge,
        RegisterEventHandler(OnProcessStart(target_action=edge, on_start=[configure])),
        RegisterEventHandler(OnStateTransition(
            target_lifecycle_node=edge, goal_state='inactive', entities=[activate],
        )),
        follower,
    ])
```

- [ ] **Step 10.2: Commit**

```bash
git add src/raspicat_async_vla_bringup/launch/mvp_real.launch.py
git commit -m "feat(bringup): add mvp_real.launch.py with real model + adapter"
```

---

## Task 11: Real-stack manual smoke

**Goal:** run the full stack (GPU host) with `tools/publish_fake_image.py` and confirm `/cmd_vel` is non-zero — proving the real model produces a usable Path.

This is a manual verification, not a pytest. The deliverable is a markdown note recording the run output.

- [ ] **Step 11.1: Set up host requirements**

The GPU host must have:
- Docker + NVIDIA Container Toolkit
- The `raspicat-asyncvla-remote` image built (Task 1)
- `AsyncVLA_release/` populated (Task 2)

ROS2 Humble can run inside `Dockerfile.test` for the edge half.

- [ ] **Step 11.2: Run real_server in one terminal**

```bash
DOCKER_CONFIG=/tmp/dckr-noauth docker run --rm --gpus all \
  -v /home/nop/dev/mywork/raspicat-async-vla:/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  --network host \
  raspicat-asyncvla-remote bash -lc "
    cd /workspace
    pip install -e src/raspicat_async_vla_proto src/raspicat_async_vla_remote >/dev/null 2>&1 || true
    python3 -m asyncvla_remote.server_main --backend real --vla-path /workspace/AsyncVLA_release --port 50051
  "
```

Wait for "listening on 0.0.0.0:50051" and the model warmup messages.

- [ ] **Step 11.3: Run edge + follower in a second terminal**

```bash
DOCKER_CONFIG=/tmp/dckr-noauth docker run --rm --user $(id -u):$(id -g) \
  -v /home/nop/dev/mywork/raspicat-async-vla:/workspace -e HOME=/tmp \
  --network host \
  raspicat-asyncvla-test bash -lc "
    source /opt/ros/humble/setup.bash
    cd /workspace
    source install/setup.bash
    pip install --no-deps efficientnet_pytorch torch torchvision >/dev/null 2>&1 || true  # if needed
    ros2 launch raspicat_async_vla_bringup mvp_real.launch.py
  "
```

In a third terminal, publish fake images:

```bash
docker exec -it <edge_container> bash -c "
  source /opt/ros/humble/setup.bash
  source /workspace/install/setup.bash
  python3 /workspace/tools/publish_fake_image.py
"
```

Or simpler: include `publish_fake_image.py` in the same Docker run via `&` and `wait`.

- [ ] **Step 11.4: Observe `/cmd_vel`**

```bash
ros2 topic echo /cmd_vel --once
```

Expected: `linear.x` non-zero (sign / magnitude depend on model output for the all-grey image; doesn't have to make physical sense, only has to be non-default-zero, proving the real adapter is producing output).

Also check `/asyncvla/status`:

```bash
ros2 topic echo /asyncvla/status --once
```

Expected: `OK` after first inference completes.

- [ ] **Step 11.5: Record results**

Save to `docs/superpowers/notes/2026-04-30-real-stack-smoke.md`:
- Hardware (GPU model)
- Inference latency (from `/asyncvla/embedding inference_ms`)
- Cmd values observed
- Any errors

- [ ] **Step 11.6: Commit**

```bash
git add docs/superpowers/notes/2026-04-30-real-stack-smoke.md
git commit -m "docs(plan-2a): record real-stack smoke results"
```

---

## Done condition (Plan 2A acceptance)

When all of the following are true, Plan 2A is complete:

- [ ] `Dockerfile.remote` builds successfully on a GPU host.
- [ ] `scripts/download_checkpoints.sh` populates `AsyncVLA_release/` with all required checkpoints.
- [ ] `colcon build` succeeds for all 5 Plan 1 packages plus any modifications from Plan 2A.
- [ ] All Plan 1 pytest tests still pass (no regression).
- [ ] New unit tests (model_loader, data_transform, edge_adapter, edge_inference) pass.
- [ ] GPU smoke test (`test_inference_engine_smoke.py` with `ASYNCVLA_E2E=1`) passes.
- [ ] `mvp_real.launch.py` brings up real_server + edge + follower without crash.
- [ ] With `tools/publish_fake_image.py` running, `/cmd_vel` shows non-zero values driven by the real Edge_adapter (not the stub).
- [ ] `/asyncvla/status` reports `OK` once the first embedding arrives.

When done, **Plan 2B** picks up:

1. Wire `mvp_real.launch.py` into `sim_full.launch.py` with Gazebo + raspicat_sim.
2. Run goal-driven 5 m navigation scenarios (pose / language / image goals).
3. Confirm closed-loop control (the robot actually goes toward the goal).
4. Tune Pure Pursuit lookahead/max_v on raspicat_sim.

And **Plan 3** picks up:

1. Real raspicat hardware launch.
2. `tools/benchmark.py` for RTT / throughput / latency injection.
3. Safety stop field tests.

---

## Open questions

These should be resolved during Task 0 investigation:

1. Does `NHirose/AsyncVLA_release` ship the `edge_adapter--*_checkpoint.pt`, `action_proj--*_checkpoint.pt`, `pose_projector--*_checkpoint.pt` files, or only the OmniVLA backbone safetensors? If only the backbone, where do the other checkpoints come from?
2. Does `OpenVLAForActionPrediction_MMNv1` actually require flash-attn at inference, or only at training?
3. What are the exact constructor args for `L1RegressionActionHead_idcat`, `ProprioProjector`, `Proj_Actiontokens`? (Likely in a config.yaml shipped with the checkpoints.)
4. What is the actual `embed_dim` of `action_proj.predict_action` output? Spec says 1024; verify against engine output.
5. Is `Learning-to-Drive-Anywhere-with-MBRA` ever called at inference time? If yes, add as another git submodule before Task 0.6 commits the dep report.
