"""Build a single-observation inference batch for OmniVLA-original.

Adapts ``Inference.data_transformer_omnivla`` and ``Inference.transform_datatype``
from ``external/OmniVLA/inference/run_omnivla.py`` into a free function that
takes a PIL image + goal + processor and returns the dict consumed by
``vla(...)`` in :class:`raspicat_vla_remote.backends.omnivla.OmniVLABackend`.

Modality flags (lang / pose / image) are passed in by the backend; this
function does not pick them. The collator is inlined (single-instance only,
no padding to a longer sequence).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import PIL.Image
import torch
from torch.nn.utils.rnn import pad_sequence


_IGNORE_INDEX = -100


def determine_modality_id(
    *,
    has_lang: bool,
    has_pose: bool,
    has_image_goal: bool,
    has_satellite: bool = False,
) -> int:
    """Map the active goal modalities to OmniVLA's modality_id (0..8).

    Mirrors run_omnivla.py:417-435. Plan 2B Path 1 v1 only ever sees
    {pose, lang, pose+lang} (modality_id in {4, 7, 8}); other combinations are
    accepted in case a future proto extension turns them on.
    """
    if has_satellite and not has_lang and not has_pose and not has_image_goal:
        return 0
    if has_satellite and not has_lang and has_pose and not has_image_goal:
        return 1
    if has_satellite and not has_lang and not has_pose and has_image_goal:
        return 2
    if has_satellite and not has_lang and has_pose and has_image_goal:
        return 3
    if not has_satellite and not has_lang and has_pose and not has_image_goal:
        return 4
    if not has_satellite and not has_lang and has_pose and has_image_goal:
        return 5
    if not has_satellite and not has_lang and not has_pose and has_image_goal:
        return 6
    if not has_satellite and has_lang and not has_pose and not has_image_goal:
        return 7
    if not has_satellite and has_lang and has_pose and not has_image_goal:
        return 8
    # Fallback: language-only when nothing else matches (empty observation).
    return 7


def _goal_pose_cos_sin(
    goal_pose_xy_theta: Optional[Tuple[float, float, float]],
    *,
    pose_dim: int,
) -> np.ndarray:
    """Pack (x, y, theta) into the (x, y, cos, sin) layout OmniVLA expects."""
    if goal_pose_xy_theta is None:
        return np.zeros(pose_dim, dtype=np.float32)
    x, y, theta = goal_pose_xy_theta
    if pose_dim == 4:
        return np.array([x, y, math.cos(theta), math.sin(theta)], dtype=np.float32)
    if pose_dim == 2:
        return np.array([x, y], dtype=np.float32)
    out = np.zeros(pose_dim, dtype=np.float32)
    out[: min(2, pose_dim)] = (x, y)[: min(2, pose_dim)]
    return out


def _build_conversation(lang_instruction: str, action_chunk_string: str) -> list:
    """Mirror run_omnivla.py:343-353. Empty / 'xxxx' lang -> placeholder turn."""
    if not lang_instruction or lang_instruction == 'xxxx':
        return [
            {'from': 'human', 'value': 'No language instruction'},
            {'from': 'gpt', 'value': action_chunk_string},
        ]
    return [
        {'from': 'human', 'value': f'What action should the robot take to {lang_instruction}?'},
        {'from': 'gpt', 'value': action_chunk_string},
    ]


def build_inference_batch(
    *,
    current_image: PIL.Image.Image,
    goal_image: Optional[PIL.Image.Image],
    lang_instruction: str,
    goal_pose_xy_theta: Optional[Tuple[float, float, float]],
    action_tokenizer: Any,
    processor: Any,
    prompt_builder_cls: Any,
    pose_dim: int = 4,
    num_actions_chunk: int = 8,
    action_dim: int = 4,
) -> Dict[str, torch.Tensor]:
    """Single-observation batch matching run_omnivla.collator_custom output.

    Returned keys:
        input_ids, attention_mask, pixel_values, labels, goal_pose

    `pixel_values` packs ``cat(current, goal, dim=channels)`` if a goal_image
    is present (matches OmniVLA-original training); otherwise it's just the
    current image's pixels.
    """
    # 1. Dummy action chunk -> action chunk string. Used only for tokenizer
    #    placement; actual values are ignored at inference (`labels` are masked
    #    so the model doesn't condition on them).
    actions = np.zeros((num_actions_chunk, action_dim), dtype=np.float32)
    current_action = actions[0]
    future_actions = actions[1:]
    action_chunk_string = action_tokenizer(current_action) + ''.join(action_tokenizer(future_actions))
    action_chunk_len = len(action_chunk_string)

    # 2. Build conversation + prompt + tokenize.
    conversation = _build_conversation(lang_instruction, action_chunk_string)
    builder = prompt_builder_cls('openvla')
    for turn in conversation:
        builder.add_turn(turn['from'], turn['value'])
    base_tokenizer = processor.tokenizer
    input_ids = torch.tensor(
        base_tokenizer(builder.get_prompt(), add_special_tokens=True).input_ids,
        dtype=torch.long,
    )

    # 3. Build labels: mask everything except the trailing action chunk.
    labels = input_ids.clone()
    labels[: -(action_chunk_len + 1)] = _IGNORE_INDEX

    # 4. Image transform.
    pixel_values_current = processor.image_processor.apply_transform(current_image)
    pixel_values_goal = (
        processor.image_processor.apply_transform(goal_image)
        if goal_image is not None
        else None
    )

    # 5. Single-instance collation (mirrors collator_custom).
    pad_id = base_tokenizer.pad_token_id
    input_ids_b = pad_sequence([input_ids], batch_first=True, padding_value=pad_id)
    labels_b = pad_sequence([labels], batch_first=True, padding_value=_IGNORE_INDEX)
    model_max = base_tokenizer.model_max_length
    input_ids_b = input_ids_b[:, :model_max]
    labels_b = labels_b[:, :model_max]
    attention_mask = input_ids_b.ne(pad_id)

    if pixel_values_goal is not None:
        # OmniVLA-original concatenates along channel dim (matches collator_custom).
        pixel_values = torch.cat(
            (torch.stack([pixel_values_current]), torch.stack([pixel_values_goal])),
            dim=1,
        )
    else:
        pixel_values = torch.stack([pixel_values_current])

    goal_pose = torch.from_numpy(
        _goal_pose_cos_sin(goal_pose_xy_theta, pose_dim=pose_dim)
    ).unsqueeze(0)

    return {
        'input_ids': input_ids_b,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'labels': labels_b,
        'goal_pose': goal_pose,
    }
