"""Slow GPU-only smoke test for OmniVLABackend.

Skipped unless OMNIVLA_E2E=1. Run inside Dockerfile.omnivla with --gpus all
and the omnivla-original/ directory mounted at /workspace/omnivla-original.
"""
import os

import pytest

if os.environ.get('OMNIVLA_E2E') != '1':
    pytest.skip('set OMNIVLA_E2E=1 to run', allow_module_level=True)


def test_omnivla_backend_returns_action_chunk_shape():
    import PIL.Image

    from raspicat_vla_remote.backends.omnivla import OmniVLABackend

    backend = OmniVLABackend(
        vla_path=os.environ.get('OMNIVLA_VLA_PATH', '/workspace/omnivla-original'),
        resume_step=int(os.environ.get('OMNIVLA_RESUME_STEP', '120000')),
        device='cuda:0',
    )
    backend.warmup(num_iters=1)

    img = PIL.Image.new('RGB', (224, 224), (128, 128, 128))
    arr, metrics = backend.infer(
        current_image=img,
        past_image=img,
        lang_instruction='go forward',
        goal_image=None,
        goal_pose_xy_theta=(1.0, 0.0, 0.0),
    )
    info = backend.model_info()
    assert arr.ndim == 2
    assert arr.shape == (info.num_tokens, info.embed_dim)
    assert arr.dtype.name == 'float32'
    assert metrics['inference_ms'] > 0
    print(
        f'projected shape={arr.shape} '
        f'inf_ms={metrics["inference_ms"]:.1f} '
        f'modality_id={metrics["modality_id"]}'
    )


def test_omnivla_backend_pose_only_modality_is_4():
    import PIL.Image

    from raspicat_vla_remote.backends.omnivla import OmniVLABackend

    backend = OmniVLABackend(
        vla_path=os.environ.get('OMNIVLA_VLA_PATH', '/workspace/omnivla-original'),
        resume_step=int(os.environ.get('OMNIVLA_RESUME_STEP', '120000')),
        device='cuda:0',
    )

    img = PIL.Image.new('RGB', (224, 224), (128, 128, 128))
    _, metrics = backend.infer(
        current_image=img, past_image=img,
        lang_instruction='',                       # no lang
        goal_image=None,                            # no image goal
        goal_pose_xy_theta=(1.0, 0.0, 0.0),         # pose goal
    )
    assert metrics['modality_id'] == 4
