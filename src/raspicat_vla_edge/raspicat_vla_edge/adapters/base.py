"""EdgeAdapter ABC. Concrete adapters (stub / asyncvla / omnivla) implement this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from nav_msgs.msg import Path


class EdgeAdapter(ABC):
    """Convert a `(B=1, num_tokens, embed_dim)` cloud embedding into a Path.

    Concrete adapters are responsible for producing a `nav_msgs/Path` in the
    robot frame. They may consult the latest RGB images (e.g. AsyncVLA's
    Edge_adapter) or use the embedding alone (Plan 2B Path 1's OmniVLA).
    """

    @abstractmethod
    def predict_path(
        self,
        *,
        embedding: np.ndarray,
        embedding_shape: Tuple[int, int, int],
        cur_image_rgb: Optional[np.ndarray] = None,
        past_image_rgb: Optional[np.ndarray] = None,
        frame_id: str = 'base_link',
    ) -> Path:
        ...
