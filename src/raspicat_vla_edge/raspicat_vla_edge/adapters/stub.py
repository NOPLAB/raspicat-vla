"""Plan-1 stub adapter: ignore embedding contents, emit a straight-ahead path."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from .base import EdgeAdapter


class StubAdapter(EdgeAdapter):
    """Returns ``n_pts`` waypoints along the +x axis of ``frame_id``."""

    def __init__(self, *, n_pts: int = 10, step_m: float = 0.1) -> None:
        self._n_pts = n_pts
        self._step_m = step_m

    def predict_path(
        self,
        *,
        embedding: Optional[np.ndarray] = None,
        embedding_shape: Optional[Tuple[int, int, int]] = None,
        cur_image_rgb: Optional[np.ndarray] = None,
        past_image_rgb: Optional[np.ndarray] = None,
        frame_id: str = 'base_link',
    ) -> Path:
        path = Path()
        path.header.frame_id = frame_id
        for i in range(1, self._n_pts + 1):
            ps = PoseStamped()
            ps.header.frame_id = frame_id
            ps.pose.position.x = i * self._step_m
            ps.pose.position.y = 0.0
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        return path
