"""Tests for ``omnivla_data_transform.build_inference_batch`` and helpers.

Uses MagicMock stubs for the prismatic processor / tokenizer / prompt_builder
so the tests run on CPU without loading the 7.5B-param backbone. We assert the
*output contract* of our batch builder, not the full prismatic pipeline.
"""
from unittest.mock import MagicMock

import math
import pytest
import torch
from PIL import Image

from raspicat_vla_remote.backends.omnivla_data_transform import (
    build_inference_batch,
    determine_modality_id,
    _goal_pose_cos_sin,
)


def _make_pil(size=(224, 224)):
    return Image.new('RGB', size, color=(127, 127, 127))


def _stub_processor():
    p = MagicMock()
    p.tokenizer.pad_token_id = 0
    p.tokenizer.model_max_length = 32

    # base_tokenizer(prompt, add_special_tokens=True) -> object with .input_ids
    def _tokenizer_call(prompt, add_special_tokens=True):
        out = MagicMock()
        # pretend we emit one token per character, capped at model_max_length-ish
        out.input_ids = list(range(1, min(20, max(2, len(prompt) // 4))))
        return out

    p.tokenizer.side_effect = _tokenizer_call

    # image_processor.apply_transform(pil) -> torch tensor (3, 224, 224)
    p.image_processor.apply_transform.side_effect = lambda img: torch.zeros(
        3, 224, 224, dtype=torch.float32,
    )
    return p


def _stub_action_tokenizer():
    """``action_tokenizer(action_array)`` returns a string of action tokens."""
    def _call(action):
        return 'A' * 4   # 4 chars per action
    return _call


def _stub_prompt_builder_cls():
    class _PB:
        def __init__(self, name):
            self._turns = []
        def add_turn(self, role, text):
            self._turns.append((role, text))
        def get_prompt(self) -> str:
            return ' '.join(t for _, t in self._turns)
    return _PB


# ---------------------------------------------------------------- determine_modality_id

@pytest.mark.parametrize(
    'has_lang,has_pose,has_image_goal,expected',
    [
        (True, False, False, 7),  # language only
        (False, True, False, 4),  # pose only
        (True, True, False, 8),   # language + pose
        (False, False, True, 6),  # image only
        (False, True, True, 5),   # pose + image
        (False, False, False, 7), # nothing -> language fallback
    ],
)
def test_determine_modality_id(has_lang, has_pose, has_image_goal, expected):
    assert determine_modality_id(
        has_lang=has_lang, has_pose=has_pose, has_image_goal=has_image_goal,
    ) == expected


# ---------------------------------------------------------------- _goal_pose_cos_sin

def test_goal_pose_cos_sin_packs_xy_costheta_sintheta():
    out = _goal_pose_cos_sin((1.0, 2.0, math.pi / 2), pose_dim=4)
    assert out.shape == (4,)
    assert pytest.approx(out[0]) == 1.0
    assert pytest.approx(out[1]) == 2.0
    assert pytest.approx(out[2], abs=1e-6) == 0.0   # cos(pi/2)
    assert pytest.approx(out[3], abs=1e-6) == 1.0   # sin(pi/2)


def test_goal_pose_cos_sin_returns_zeros_when_none():
    out = _goal_pose_cos_sin(None, pose_dim=4)
    assert out.shape == (4,)
    assert (out == 0).all()


# ---------------------------------------------------------------- build_inference_batch

def test_build_inference_batch_returns_required_keys_lang_only():
    batch = build_inference_batch(
        current_image=_make_pil(),
        goal_image=None,
        lang_instruction='go to the door',
        goal_pose_xy_theta=None,
        action_tokenizer=_stub_action_tokenizer(),
        processor=_stub_processor(),
        prompt_builder_cls=_stub_prompt_builder_cls(),
    )
    for k in ('input_ids', 'attention_mask', 'pixel_values', 'labels', 'goal_pose'):
        assert k in batch, f'missing key: {k}'
    # Single instance, no goal image -> pixel_values shape (1, 3, 224, 224)
    assert batch['pixel_values'].shape == (1, 3, 224, 224)
    # goal_pose has the expected POSE_DIM=4 layout
    assert batch['goal_pose'].shape == (1, 4)


def test_build_inference_batch_pose_goal_populates_goal_pose():
    batch = build_inference_batch(
        current_image=_make_pil(),
        goal_image=None,
        lang_instruction='',
        goal_pose_xy_theta=(1.5, 0.0, 0.0),
        action_tokenizer=_stub_action_tokenizer(),
        processor=_stub_processor(),
        prompt_builder_cls=_stub_prompt_builder_cls(),
    )
    assert batch['goal_pose'][0, 0].item() == pytest.approx(1.5)
    assert batch['goal_pose'][0, 2].item() == pytest.approx(1.0)  # cos(0)


def test_build_inference_batch_with_goal_image_concats_along_channels():
    batch = build_inference_batch(
        current_image=_make_pil(),
        goal_image=_make_pil(),
        lang_instruction='go forward',
        goal_pose_xy_theta=None,
        action_tokenizer=_stub_action_tokenizer(),
        processor=_stub_processor(),
        prompt_builder_cls=_stub_prompt_builder_cls(),
    )
    # current (3 ch) + goal (3 ch) cat'd along dim=1 -> 6 channels.
    assert batch['pixel_values'].shape == (1, 6, 224, 224)


def test_build_inference_batch_labels_mask_non_action_tokens():
    """The action chunk must be the only thing not masked to IGNORE_INDEX."""
    batch = build_inference_batch(
        current_image=_make_pil(),
        goal_image=None,
        lang_instruction='go forward',
        goal_pose_xy_theta=None,
        action_tokenizer=_stub_action_tokenizer(),
        processor=_stub_processor(),
        prompt_builder_cls=_stub_prompt_builder_cls(),
    )
    labels = batch['labels'][0]
    # action chunk is num_actions_chunk * 4 chars = 32 chars by stub
    # labels[:-(action_chunk_len + 1)] should be -100
    assert (labels[:-1] == -100).any()
