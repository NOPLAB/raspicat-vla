#!/usr/bin/env bash
# Download the OmniVLA cloud backbone (NHirose/omnivla-original) into
# ./models/omnivla-original/. Plan 2B Path 1 only needs this one repo;
# omnivla-edge is reserved for a future Path-2 plan.
#
# Uses the host's ~/.cache/huggingface so repeat runs are instant.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${REPO_ROOT}/models/omnivla-original"

mkdir -p "${OUT_DIR}"

python3 - <<PY
import os
from huggingface_hub import snapshot_download

p = snapshot_download(
    repo_id="NHirose/omnivla-original",
    local_dir="${OUT_DIR}",
    local_dir_use_symlinks=False,
)
print(f"== NHirose/omnivla-original -> {p}")
for root, _, files in os.walk(p):
    for f in files:
        full = os.path.join(root, f)
        size_mb = os.path.getsize(full) / (1024 * 1024)
        print(f"   {os.path.relpath(full, p)}  ({size_mb:.1f} MB)")
PY
