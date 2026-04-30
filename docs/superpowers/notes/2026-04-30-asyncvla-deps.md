# AsyncVLA Dependency Closure Report (Plan 2A / Task 0)

Date: 2026-04-30
Submodule: `external/AsyncVLA` (commit at the time of writing)

This report answers Plan 2A Task 0: what does AsyncVLA actually need at inference, what is the cloud/edge module split, and what extra git/Python machinery do we need beyond what Plan 2B already set up?

**Headline:** AsyncVLA fits the cloud→embedding→edge architecture cleanly. The cloud emits a `(NUM_ACTIONS_CHUNK=8, 1024)` tensor via `Proj_Actiontokens`; the edge runs `Edge_adapter` (~5 M params, efficientnet-b0 + transformer decoder) over `(cur, past, vla_feature)` to produce the (8, 4) waypoints. **One non-trivial extra dep: `vint_train` from `Learning-to-Drive-Anywhere-with-MBRA` is required at import time of `prismatic.models.small_head` on both cloud and edge.**

---

## 1. Minimum required `prismatic` modules

From `external/AsyncVLA/inference/run_asyncvla.py` (inference-only imports):

```
prismatic.extern.hf.configuration_prismatic.OpenVLAConfig
prismatic.extern.hf.modeling_prismatic.OpenVLAForActionPrediction_MMNv1
prismatic.extern.hf.processing_prismatic.PrismaticImageProcessor
prismatic.extern.hf.processing_prismatic.PrismaticProcessor
prismatic.models.action_heads.L1RegressionActionHead_idcat
prismatic.models.action_heads.L1RegressionDistHead    # AsyncVLA-only (extra distance head)
prismatic.models.backbones.llm.prompting.PurePromptBuilder
prismatic.models.projectors.ProprioProjector
prismatic.models.small_head.Edge_adapter              # AsyncVLA-only (edge model)
prismatic.models.small_head.Proj_Actiontokens         # AsyncVLA-only (cloud action projector)
prismatic.training.train_utils.{get_current_action_mask, get_next_actions_mask}
prismatic.vla.action_tokenizer.ActionTokenizer
prismatic.vla.constants.{ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE}
```

`Proj_Actiontokens` (`small_head.py:221+`) is the projection head AsyncVLA uses to compress the language-model hidden states into a fixed `(8, 1024)` tensor. OmniVLA-original does **not** have this layer — that's the architectural difference that makes AsyncVLA naturally cloud/edge-splittable.

`Edge_adapter` (`small_head.py:9-78`) is the on-edge model: efficientnet-b0 over the current frame + a 6-channel `cat(cur, past)` encoder, fused via `MultiLayerDecoder_trans`, regressing to `(NUM_ACTIONS_CHUNK, 4)` (i.e. (x, y, cos, sin)).

## 2. The `vint_train` dependency

`prismatic/models/small_head.py:6` imports

```python
from vint_train.models.vint.self_attention import MultiLayerDecoder, MultiLayerDecoder_idcat, MultiLayerDecoder_trans
```

That package is **not** part of `external/AsyncVLA`. It lives in `Learning-to-Drive-Anywhere-with-MBRA/train/`, which `run_asyncvla.py:12-14` adds to `sys.path` at runtime:

```python
sys.path.extend([
    "../Learning-to-Drive-Anywhere-with-MBRA/train/", '../lerobot'
])
```

**Implication.** Importing **any** class from `prismatic.models.small_head` (i.e. `Edge_adapter` *or* `Proj_Actiontokens`) requires `vint_train` to be on `PYTHONPATH`. Both the cloud `AsyncVLABackend` (which uses `Proj_Actiontokens`) and the edge `AsyncVLAEdgeAdapter` (which uses `Edge_adapter`) need this.

`lerobot` is **not** required at inference — it's only used by the Frodobots dataset loader during training. Skip.

### Recommendation

Add `Learning-to-Drive-Anywhere-with-MBRA` as a git submodule under `external/MBRA/`, then expose `external/MBRA/train/` on PYTHONPATH in both Docker images:

```bash
git submodule add https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA external/MBRA
```

In Dockerfile.asyncvla:
```dockerfile
COPY external/MBRA /opt/MBRA
ENV PYTHONPATH=/opt/MBRA/train:$PYTHONPATH
```

Edge container (Dockerfile.test in this repo) needs the same `train/` dir on PYTHONPATH for `Edge_adapter` to import. Since edge runs in CPU-only `Dockerfile.test`, copy MBRA there too — or vendor just the single file `vint_train/models/vint/self_attention.py` if its size becomes a concern. Vendoring is also acceptable; the file is small (single-purpose transformer decoder) and unlikely to drift.

## 3. External (non-PyPI) dependencies

| Package                                | AsyncVLA inference required? |
| -------------------------------------- | ---------------------------- |
| `transformers @ git+...moojink/transformers-openvla-oft.git` | **Yes** (bidirectional attn) |
| `efficientnet_pytorch>=0.7.1`          | **Yes** (Edge_adapter uses it) |
| `vint_train` (from MBRA repo)          | **Yes** (small_head import-time dep) |
| `lerobot`                              | No (training-only)            |
| `flash-attn`                           | No at inference (SDPA fallback OK) |
| `clip` (OpenAI)                        | No                            |
| `utm`                                  | Yes (`run_asyncvla.py:6` etc.; can be stubbed) |

## 4. Checkpoint file inventory

`huggingface_hub.HfApi.list_repo_files('NHirose/AsyncVLA_release')`:

```
.gitattributes
README.md
action_head--750000_checkpoint.pt        <- L1RegressionActionHead_idcat
action_proj--750000_checkpoint.pt        <- Proj_Actiontokens (AsyncVLA-only)
added_tokens.json
config.json
configuration_prismatic.py
generation_config.json
lora_adapter/...                          (fine-tuning support; not used in v1)
model-00001..00004-of-00004.safetensors  <- OpenVLA-OFT backbone shards
model.safetensors.index.json
modeling_prismatic.py
pose_projector--750000_checkpoint.pt      <- ProprioProjector (filename matches code; no fallback needed)
preprocessor_config.json
processing_prismatic.py
processor_config.json
shead--750000_checkpoint.pt              <- Edge_adapter (a.k.a. "shead" / small head)
special_tokens_map.json
tokenizer.json
tokenizer.model
tokenizer_config.json
```

Step number: **750000** (different from OmniVLA's 120000).

Notable differences vs OmniVLA-original:
- `pose_projector` filename is correct on disk (no `proprio_projector` fallback needed); the shared `_checkpoints.py:_resolve_path` still works because the primary lookup hits.
- `dist_head` is **not** in this release (OmniVLA-original ships one but AsyncVLA does not).
- `action_proj` is **only** in this release (OmniVLA-original does not ship one; that's the architectural split).
- `shead` is the Edge_adapter checkpoint — referenced as "shead" in code (`shead--{step}_checkpoint.pt`) but the class is `Edge_adapter`. Don't confuse with `action_head`.

## 5. AsyncVLA cloud/edge contract

| Step                          | Where     | Module                  | Output shape            |
| ----------------------------- | --------- | ----------------------- | ----------------------- |
| Backbone forward              | Cloud GPU | OpenVLA-OFT vla         | hidden states            |
| Action token slicing          | Cloud GPU | (mask via train_utils)  | `(1, 8*4, 4096)`         |
| Cloud projection              | Cloud GPU | `Proj_Actiontokens`     | `(1, 8, 1024)`           |
| **gRPC ActionEmbedding**      |  →        |                         | **`(8, 1024)`** float32 |
| Edge model                    | Edge CPU  | `Edge_adapter`          | `(1, 8, 4)`              |
| `delta_to_pose` accumulation  | Edge CPU  | (math)                  | `(1, 8, 4)` world-frame |
| Path build                    | Edge CPU  | (ROS msg)               | `nav_msgs/Path`          |

So the `ActionEmbedding` carries `num_tokens=8`, `embed_dim=1024` — matching Plan 1's dummy contract exactly. **No proto change needed.**

## 6. `delta_to_pose` semantics

From `run_asyncvla.py:92-135` (`delta_to_pose`): given `delta` shape `(N, T, 4)` packed as `(dx, dy, cos(dtheta), sin(dtheta))`, accumulate iteratively in world frame:

```
x_t = x_{t-1} + cos(theta_{t-1}) * dx_t - sin(theta_{t-1}) * dy_t
y_t = y_{t-1} + sin(theta_{t-1}) * dx_t + cos(theta_{t-1}) * dy_t
theta_t = theta_{t-1} + dtheta_t
```

This is identical between AsyncVLA and OmniVLA-edge variants. Since Plan 2B Path 1 puts the cumsum on the cloud (OmniVLA-original applies it inside `OmniVLA_edge`), the OmniVLA edge adapter does no math; AsyncVLA's edge adapter must apply `delta_to_pose` to the `Edge_adapter` output.

## 7. Open questions / risks

1. **MBRA submodule add.** Confirmed required. Adding it before Task 1 is a prerequisite.
2. **`L1RegressionDistHead`.** AsyncVLA imports it but `define_model` does not instantiate it — distance prediction is implicit / inline. Plan 2A v1 ignores it.
3. **`shead` checkpoint key compatibility.** The upstream `define_model` does `strict=False` when loading shead because of unexpected keys. Our loader should match (or filter to expected keys upfront).
4. **`utm` use at inference.** `run_asyncvla.py` imports `utm` for goal lat/lon → meters conversion in its `__main__`. The `Inference` class itself doesn't use `utm` once the goal_pose arrives in metric coords, so we can skip installing `utm` in the cloud image. Re-check after writing AsyncVLABackend.
5. **`config_nav/dataset_config.yaml`.** `define_model` loads this YAML to get `obs_encoding_size`, `mha_num_attention_heads`, etc. for `Edge_adapter`. We need to either copy the values inline (preferred — single source of truth) or ship the YAML. Recommended: extract the four values into AsyncVLAEdgeAdapter's constructor defaults.
6. **Edge inference latency on raspicat.** Edge_adapter is ~5 M params (efficientnet-b0 + small transformer + MLP). Should be ~100–200 ms on a Pi 4. Confirm in Task 8 manual smoke; if too slow, options are: (a) push edge inference to a CPU container on the dev workstation, (b) quantize Edge_adapter, (c) skip the edge model and treat AsyncVLA like OmniVLA Path 1 (waypoints from cloud).

## 8. Plan 2A revision deltas

The existing Plan 2A doc (`docs/superpowers/plans/2026-04-29-asyncvla-real-model-integration.md`) was written before the rename and Plan 2B's abstraction. The post-Task 0 deltas are:

- All paths `raspicat_async_vla_*` → `raspicat_vla_*`.
- All Python module names `asyncvla_remote` / `asyncvla_edge` → `raspicat_vla_remote` / `raspicat_vla_edge`.
- Model loader (Plan 2A's Task 3) → reuse the shared `_checkpoints.py` (already shipped in Plan 2B Task 4 / commit 17b4670).
- Backend goes under `raspicat_vla_remote/backends/asyncvla.py` (matches the abstraction from Plan 2B Task 3 / commit b31a6fc).
- Edge adapter goes under `raspicat_vla_edge/adapters/asyncvla.py`.
- `RealServer` is replaced by the generic `VLAServer` (Plan 2B Task 3) — `server_main.py --backend asyncvla` already routes to `backends.asyncvla.AsyncVLABackend`.
- New prerequisite Task 0.5: add MBRA submodule + plumb its `train/` onto PYTHONPATH in `Dockerfile.asyncvla`.

The functional task list shrinks accordingly:

| Task | Status                                                           |
| ---- | ---------------------------------------------------------------- |
| 0    | This report.                                                     |
| 0.5  | Add `external/MBRA` submodule.                                   |
| 1    | `Dockerfile.asyncvla` (similar to `Dockerfile.omnivla` plus efficientnet_pytorch + MBRA path). |
| 2    | `scripts/download_asyncvla_checkpoints.sh` (NHirose/AsyncVLA_release → `./AsyncVLA_release/`). |
| 3    | `omnivla_data_transform` is largely identical to AsyncVLA's; add `asyncvla_data_transform` with the small differences (e.g. action_proj feed shape) — or generalize the shared one and parametrize. |
| 4    | `AsyncVLABackend` in `raspicat_vla_remote/backends/asyncvla.py`. |
| 5    | `AsyncVLAEdgeAdapter` in `raspicat_vla_edge/adapters/asyncvla.py` — loads `Edge_adapter` from prismatic, runs `forward(cur, past, vla_feature)`, applies `delta_to_pose`, builds `Path`. |
| 6    | `mvp_asyncvla.launch.py` (mirrors `mvp_omnivla.launch.py`). |
| 7    | Real-stack manual smoke (GPU host).                              |

The original Plan 2A's Task 7-9 (edge_adapter / edge_inference / edge_node wiring) collapse into the single Task 5 above because `_build_adapter` dispatch already exists (Plan 2B commit b31a6fc).
