#!/usr/bin/env bash
# Download the AsyncVLA cloud + edge release (NHirose/AsyncVLA_release) into
# ./models/AsyncVLA_release/. Contains:
#   - 4-shard OpenVLA-OFT backbone (model-{1..4}-of-4.safetensors, ~15 GB)
#   - action_head--750000_checkpoint.pt
#   - action_proj--750000_checkpoint.pt        (AsyncVLA-only; cloud projector)
#   - pose_projector--750000_checkpoint.pt
#   - shead--750000_checkpoint.pt              (Edge_adapter weights)
#   - LoRA adapter, processor + tokenizer files
#
# Uses the host's ~/.cache/huggingface so repeat runs are instant.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${REPO_ROOT}/models/AsyncVLA_release"

mkdir -p "${OUT_DIR}"

python3 - <<PY
import os
from huggingface_hub import snapshot_download

p = snapshot_download(
    repo_id="NHirose/AsyncVLA_release",
    local_dir="${OUT_DIR}",
    local_dir_use_symlinks=False,
)
print(f"== NHirose/AsyncVLA_release -> {p}")
for root, _, files in os.walk(p):
    for f in files:
        full = os.path.join(root, f)
        size_mb = os.path.getsize(full) / (1024 * 1024)
        print(f"   {os.path.relpath(full, p)}  ({size_mb:.1f} MB)")
PY
