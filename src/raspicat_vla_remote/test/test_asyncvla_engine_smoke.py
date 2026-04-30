"""Slow GPU-only smoke test for AsyncVLABackend.

Skipped unless ASYNCVLA_E2E=1. Run inside Dockerfile.asyncvla with --gpus all
and the AsyncVLA_release/ directory mounted at /workspace/AsyncVLA_release.
Pre-requisite: external/MBRA submodule must be checked out so that
prismatic.models.small_head.Proj_Actiontokens can import vint_train.
"""
import os

import pytest

if os.environ.get('ASYNCVLA_E2E') != '1':
    pytest.skip('set ASYNCVLA_E2E=1 to run', allow_module_level=True)


def test_asyncvla_backend_returns_projection_shape():
    import PIL.Image

    from raspicat_vla_remote.backends.asyncvla import AsyncVLABackend

    backend = AsyncVLABackend(
        vla_path=os.environ.get('ASYNCVLA_VLA_PATH', '/workspace/AsyncVLA_release'),
        resume_step=int(os.environ.get('ASYNCVLA_RESUME_STEP', '750000')),
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
    assert info.embed_dim == 1024     # AsyncVLA cloud_action_dim
    assert info.num_tokens == 8       # NUM_ACTIONS_CHUNK
    assert arr.dtype.name == 'float32'
    assert metrics['inference_ms'] > 0
    print(
        f'projected shape={arr.shape} '
        f'inf_ms={metrics["inference_ms"]:.1f} '
        f'modality_id={metrics["modality_id"]}'
    )
