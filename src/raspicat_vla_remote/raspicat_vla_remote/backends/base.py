"""VLABackend ABC. Concrete backends (dummy / asyncvla / omnivla) implement this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import PIL.Image


@dataclass
class ModelInfoDict:
    model_name: str
    model_version: str
    num_tokens: int
    embed_dim: int
    device: str
    ready: bool


class VLABackend(ABC):
    """Pure-python interface to a VLA model that the generic VLAServer hosts.

    Concrete implementations adapt different families (dummy passthrough,
    AsyncVLA, OmniVLA-original) onto a single shape contract:
    a `(num_tokens, embed_dim)` float32 numpy array per observation.
    """

    @abstractmethod
    def warmup(self, num_iters: int = 1) -> None:
        """Optional: pre-compile graphs / page weights in. No-op by default."""

    @abstractmethod
    def infer(
        self,
        *,
        current_image: PIL.Image.Image,
        past_image: Optional[PIL.Image.Image],
        lang_instruction: str,
        goal_image: Optional[PIL.Image.Image],
        goal_pose_xy_theta: Optional[Tuple[float, float, float]],
    ) -> Tuple[np.ndarray, dict]:
        """Run one observation through the backend.

        Returns:
            embedding: (num_tokens, embed_dim) float32 ndarray.
            metrics:   {'inference_ms': float, ...} (backend-specific extras OK).
        """

    @abstractmethod
    def model_info(self) -> ModelInfoDict:
        """Static metadata for ``GetModelInfo`` and the response header."""
