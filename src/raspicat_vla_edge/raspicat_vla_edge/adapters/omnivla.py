"""OmniVLAEdgeAdapter: action chunk -> nav_msgs/Path. Pure math, no model.

Plan 2B Path 1: the cloud already ran the full OmniVLA-original pipeline and
serialized the predicted absolute waypoints with shape
``(NUM_ACTIONS_CHUNK, ACTION_DIM)`` = typically ``(8, 4)`` where the last
dim packs ``(x, y, cos(theta), sin(theta))``. The edge just builds the Path.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from .base import EdgeAdapter


class OmniVLAEdgeAdapter(EdgeAdapter):
    """Path-only adapter: read (x, y, cos, sin) tuples from the embedding."""

    def predict_path(
        self,
        *,
        embedding: np.ndarray,
        embedding_shape: Tuple[int, int, int],
        cur_image_rgb: Optional[np.ndarray] = None,    # noqa: ARG002 (unused)
        past_image_rgb: Optional[np.ndarray] = None,   # noqa: ARG002 (unused)
        frame_id: str = 'base_link',
    ) -> Path:
        wp = np.asarray(embedding, dtype=np.float32).reshape(embedding_shape[1:])
        if wp.ndim != 2 or wp.shape[-1] < 4:
            raise ValueError(
                f'OmniVLA expects (num_tokens, ACTION_DIM>=4) packed as '
                f'(x, y, cos, sin); got shape={wp.shape}'
            )
        path = Path()
        path.header.frame_id = frame_id
        for x, y, c, s in wp[:, :4]:
            ps = PoseStamped()
            ps.header.frame_id = frame_id
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.z = float(s)
            ps.pose.orientation.w = float(c)
            path.poses.append(ps)
        return path
