# OmniVLA Dependency Closure Report (Plan 2B / Task 0)

Date: 2026-04-30
Submodule: `external/OmniVLA` (commit at the time of writing)

This report answers Plan 2B Task 0: what does OmniVLA actually need at inference, and does it fit our existing cloud→embedding→edge gRPC architecture?

**Headline finding — architecture mismatch:** OmniVLA ships in two non-overlapping flavours, neither of which fits the AsyncVLA-style "cloud emits an embedding consumed by a small edge adapter" pipeline. Plan 2B as originally drafted implicitly assumed an OmniVLA-edge variant that sits behind a cloud backbone. The upstream code does not implement that. **Plan 2B v1 must pick one of three revised paths** (see "Architecture decision required" below). Recommended: **Path 1**.

---

## 1. Minimum required `prismatic` modules

OmniVLA-original (`external/OmniVLA/inference/run_omnivla.py`) — used at inference:

```
prismatic.vla.action_tokenizer.ActionTokenizer
prismatic.models.projectors.ProprioProjector
prismatic.models.action_heads.L1RegressionActionHead_idcat
prismatic.models.action_heads.L1RegressionDistHead
prismatic.extern.hf.modeling_prismatic.OpenVLAForActionPrediction_MMNv1
prismatic.extern.hf.configuration_prismatic.OpenVLAConfig
prismatic.extern.hf.processing_prismatic.PrismaticImageProcessor
prismatic.extern.hf.processing_prismatic.PrismaticProcessor
prismatic.models.backbones.llm.prompting.PurePromptBuilder
prismatic.training.train_utils.get_current_action_mask
prismatic.training.train_utils.get_next_actions_mask
prismatic.vla.constants.{ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE}
```

OmniVLA-edge (`external/OmniVLA/inference/run_omnivla_edge.py` + `model_omnivla_edge.py`) — used at inference:

- **None.** The edge model is pure PyTorch + `efficientnet_pytorch` + `clip` (the OpenAI CLIP package). It does not import from `prismatic` at all.

So OmniVLA-edge has zero prismatic dependency. OmniVLA-original needs the same prismatic surface as AsyncVLA Plan 2A.

## 2. External (non-PyPI) dependencies

| Package                                | OmniVLA-original | OmniVLA-edge | Required at inference |
| -------------------------------------- | ---------------- | ------------ | --------------------- |
| `flash-attn`                           | yes              | no           | **No** at inference; only declared for training (commented in pyproject) |
| `lerobot`                              | no               | no           | No                    |
| `Learning-to-Drive-Anywhere-with-MBRA` | no (training)    | no           | No                    |
| `clip` (OpenAI CLIP)                   | no               | **yes**      | **Yes** for OmniVLA-edge — encodes language goal |
| `efficientnet_pytorch`                 | no               | yes          | Yes                   |
| `utm`                                  | yes              | yes          | Yes                   |

`clip` import is `import clip; clip.tokenize(...); text_encoder, _ = clip.load("ViT-B/32")`. PyPI package: `git+https://github.com/openai/CLIP.git` (the OpenAI CLIP repo; `pip install ftfy regex tqdm` then install CLIP from git). Note this is NOT `transformers.CLIPModel` — it's the original OpenAI clip package, which expects a particular torchvision version.

## 3. Checkpoint file inventory

`huggingface_hub.HfApi.list_repo_files` output:

### `NHirose/omnivla-original` (cloud backbone)
```
.gitattributes
README.md
action_head--120000_checkpoint.pt
added_tokens.json
config.json
configuration_prismatic.py
dist_head--120000_checkpoint.pt
generation_config.json
lora_adapter/README.md
lora_adapter/adapter_config.json
lora_adapter/adapter_model.safetensors
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
model.safetensors.index.json
modeling_prismatic.py
modeling_prismatic_____.py
preprocessor_config.json
processing_prismatic.py
processor_config.json
proprio_projector--120000_checkpoint.pt
special_tokens_map.json
tokenizer.json
tokenizer.model
tokenizer_config.json
```

Key takeaways:
- Backbone is sharded across 4 safetensors files (totals ~15 GB at bf16).
- The auxiliary heads we need are `action_head--120000_checkpoint.pt` and `proprio_projector--120000_checkpoint.pt`. (`pose_projector--*.pt` doesn't exist on disk; `load_checkpoint` in upstream falls back to `proprio_projector--*.pt` — same quirk as AsyncVLA Plan 2A's report calls out.)
- `dist_head--120000_checkpoint.pt` is OmniVLA-original-specific (distance prediction head). Not in AsyncVLA.
- The `lora_adapter/` directory is for fine-tuning support; not strictly needed at base inference.

### `NHirose/omnivla-edge` (edge model)
```
.gitattributes
README.md
omnivla-edge.pth
```

Single monolithic `.pth` file containing the full `OmniVLA_edge.state_dict()`. No step number convention — not in the same `<module>--<step>_checkpoint.pt` format as omnivla-original.

## 4. `OmniVLA_edge.forward` signature

From `external/OmniVLA/inference/model_omnivla_edge.py:217`:

```python
def forward(
    self,
    obs_img: torch.Tensor,        # (B, 3*(context_size+1), 96, 96) — concatenated history
    goal_pose: torch.Tensor,      # (B, 4) — local goal pose (x, y, cos, sin)
    map_images: torch.Tensor,     # (B, 9, 96, 96) — concat(satellite_cur, satellite_goal, obs_cur)
    goal_img: torch.Tensor,       # (B, 3, 96, 96) — egocentric goal image (or zeros)
    goal_mask: torch.Tensor,      # (B,) long — modality selector index (0..8)
    feat_text: torch.Tensor,      # (B, 512) — CLIP ViT-B/32 text feature
    current_img: torch.Tensor,    # (B, 3, 224, 224) — large current image (FiLM input)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Returns:
    #   action_pred:  (B, len_traj_pred=8, num_action_params=4) — cumsum trajectory waypoints
    #                                                             last 2 dims = normalized cos/sin
    #   dist_pred:    (B, 1) — distance estimate
    #   no_goal_mask: (B,) — pass-through of goal_mask
```

### What we can produce from the current proto

| Input         | Source                                         | Status |
| ------------- | ---------------------------------------------- | ------ |
| `obs_img`     | History of recent images (we have only 1)      | **MISSING** — need ring buffer of 6 frames in edge node |
| `goal_pose`   | `Observation.goal.pose` when mode=POSE         | OK    |
| `map_images`  | Satellite tiles (we don't have)                | **MISSING** — fill with zeros, modality flag will route around it |
| `goal_img`    | `Observation.goal.image_jpeg` when mode=IMAGE  | OK    |
| `goal_mask`   | Derived from `Observation.goal.mode`           | OK (compute on edge) |
| `feat_text`   | CLIP-encoded `Observation.goal.text`           | **NEEDS CLIP ON EDGE OR SENT FROM CLOUD** |
| `current_img` | Resize current frame to 224×224                | OK    |

So OmniVLA-edge needs: 6-frame history buffer on edge (we don't keep one), CLIP text encoder on edge (or via gRPC), and zero-fill map_images. The model accepts the missing modalities by setting `goal_mask` such that those tokens are masked out.

## 5. Architecture decision required

The original Plan 2B assumed OmniVLA-edge would slot in as an `EdgeAdapter` consuming an embedding from a cloud server. The actual upstream code makes that impossible without significant departure from the released model:

| Variant                | Where it runs               | Cloud→edge data flow it expects |
| ---------------------- | --------------------------- | ------------------------------- |
| OmniVLA-original       | Cloud only (one device)     | None — single-pass on GPU      |
| OmniVLA-edge           | Edge only (one device)      | None — single-pass on edge     |
| AsyncVLA (Plan 2A)     | Cloud (~7.5 B) + edge (~5 M)| `projected_actions` tensor      |

**Three revised paths for Plan 2B:**

### Path 1 (RECOMMENDED): OmniVLA-original on cloud, trivial edge waypoint conversion

- Cloud server (`OmniVLABackend`) runs OmniVLA-original: `vla + pose_projector + action_head` → predicted waypoints `(B, NUM_ACTIONS_CHUNK, ACTION_DIM)`.
- Reshape predicted waypoints into our existing `ActionEmbedding` payload (`num_tokens × embed_dim`) — i.e. carry the *waypoints themselves* as the cloud→edge tensor, not a hidden state.
- Edge adapter for OmniVLA does the same `delta_to_pose` accumulation + `Path` build as Plan 2A's edge inference, but **without a learned edge model** — it's just math.
- **Pros:** Minimum new infra; matches AsyncVLA-style cloud-heavy/edge-light split; preserves the gRPC contract; uses real OmniVLA weights.
- **Cons:** Does not exercise `omnivla-edge.pth`; OmniVLA-edge's modality flexibility (image goals, multi-modal masking) is not consumed.

### Path 2: OmniVLA-edge fully on edge (no cloud)

- Add `adapter_kind=omnivla_edge_local` that loads `OmniVLA_edge` + CLIP and runs the whole pipeline on the device. Maintain a 6-frame ring buffer in the edge node; compute CLIP features locally; zero-fill `map_images`.
- The remote gRPC server is not started (or is left as dummy).
- **Pros:** Uses the OmniVLA-edge model as designed; no GPU dependency for the cloud side.
- **Cons:** Departs from the AsyncVLA cloud/edge split (which the user explicitly asked us to mirror); raspicat compute may be insufficient (`OmniVLA_edge` ≈ 5× `efficientnet-b0` ≈ tens of M params plus FiLM transformer); CLIP-B/32 is also non-trivial on a Pi.

### Path 3 (REJECT): Synthesize a cloud/edge split for OmniVLA-edge

- Run CLIP and a partial featurizer on the cloud, send those features over gRPC, run the rest of `OmniVLA_edge` on the edge.
- This requires inventing layer boundaries that don't exist upstream and is fragile to release-time weight changes.
- **Reject.** Keep the upstream model black-box.

### Recommendation

Adopt **Path 1**. It best satisfies the user's stated requirement ("OmniVLAをAsyncVLAと同じような形にしたい") because:
- The cloud/edge split mirrors AsyncVLA's deployment topology.
- The edge stays light (no learned model on the raspicat).
- The gRPC interface is unchanged.
- We use the canonical OmniVLA-original release weights.

**Plan 2B should be revised** to:
- Drop Task 7 / Task 8 (`OmniVLA_edge` loader + edge inference) as written. Replace with a thin `OmniVLAEdgeAdapter` whose `predict_path(...)` only does `delta_to_pose` and `Path` construction.
- Drop Task 2's `omnivla-edge` download (only `omnivla-original` is needed).
- Treat OmniVLA-edge integration (Path 2) as a separate follow-up plan if/when raspicat compute headroom is confirmed sufficient.

## 6. Open questions for the implementer

1. **Embed-dim / num_tokens contract for the waypoint payload.** Path 1 carries shape `(NUM_ACTIONS_CHUNK, ACTION_DIM)` = (8, 4). That's `num_tokens=8`, `embed_dim=4`. Confirm the proto's `embedding_fp16` field (and the edge adapter cache) tolerate `embed_dim=4`. Plan 1's dummy and AsyncVLA Plan 2A use `embed_dim=1024`, so this is a shape change — verify nothing has hard-coded `embed_dim≥256` somewhere.
2. **`L1RegressionDistHead` use.** OmniVLA-original ships a `dist_head--120000_checkpoint.pt` for distance prediction. Plan 2B v1 ignores it (we only need the action chunk). Worth surfacing later as a `/raspicat_vla/distance_to_goal` topic.
3. **Output normalization.** OmniVLA-original normalizes actions per `ACTION_PROPRIO_NORMALIZATION_TYPE`. Verify that the edge `delta_to_pose` interprets the un-normalized waypoints correctly (or de-normalize on the cloud before serializing).
4. **CLIP fallback (if Path 2 is later adopted).** OpenAI's `clip` package is finicky about torchvision versions; if Path 2 is revisited, evaluate `transformers.CLIPModel` as a drop-in instead.
