"""Deterministic dummy backend used by Plan 1 / CI / smoke tests."""
from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import numpy as np

from .base import ModelInfoDict, VLABackend


class DummyBackend(VLABackend):
    """Returns sin-seeded constants. Ignores all observation contents."""

    def __init__(
        self,
        *,
        num_tokens: int = 8,
        embed_dim: int = 1024,
        inference_ms: float = 50.0,
        model_version: str = 'dummy-v1',
    ) -> None:
        self._num_tokens = num_tokens
        self._embed_dim = embed_dim
        self._inference_ms = inference_ms
        self._model_version = model_version
        self._counter = 0
        self._lock = threading.Lock()

    def warmup(self, num_iters: int = 1) -> None:  # noqa: ARG002
        return

    def infer(
        self,
        *,
        current_image=None,
        past_image=None,
        lang_instruction: str = '',
        goal_image=None,
        goal_pose_xy_theta=None,
    ) -> Tuple[np.ndarray, dict]:
        with self._lock:
            self._counter += 1
            cid = self._counter
        if self._inference_ms > 0:
            time.sleep(self._inference_ms / 1000.0)
        seed = float(np.sin(cid * np.pi / 17))
        arr = np.full((self._num_tokens, self._embed_dim), seed, dtype=np.float32)
        return arr, {'inference_ms': self._inference_ms}

    def model_info(self) -> ModelInfoDict:
        return ModelInfoDict(
            model_name='dummy',
            model_version=self._model_version,
            num_tokens=self._num_tokens,
            embed_dim=self._embed_dim,
            device='cpu',
            ready=True,
        )
