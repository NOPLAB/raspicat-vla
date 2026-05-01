"""Slow GPU-only smoke test for AsyncVLABackend.

Skipped unless ASYNCVLA_E2E=1. Run inside Dockerfile.asyncvla with --gpus all
and models/AsyncVLA_release/ mounted at /workspace/models/AsyncVLA_release.
Pre-requisite: external/MBRA submodule must be checked out so that
prismatic.models.small_head.Proj_Actiontokens can import vint_train.
"""
import os

import pytest

# Skip the whole module unless ASYNCVLA_E2E=1. Use a pytestmark so pytest's
# collection finishes cleanly (a top-level pytest.skip(allow_module_level=True)
# trips a known pytest 6.2 bug that drops every other test in the same
# session: https://github.com/pytest-dev/pytest/issues/4946).
pytestmark = pytest.mark.skipif(
    os.environ.get('ASYNCVLA_E2E') != '1',
    reason='set ASYNCVLA_E2E=1 to run',
)


def test_asyncvla_backend_returns_projection_shape():
    import PIL.Image

    from raspicat_vla_remote.backends.asyncvla import AsyncVLABackend

    backend = AsyncVLABackend(
        vla_path=os.environ.get('ASYNCVLA_VLA_PATH', '/workspace/models/AsyncVLA_release'),
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
