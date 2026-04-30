"""AsyncVLA edge adapter (Plan 2A).

Loads ``Edge_adapter`` from ``prismatic.models.small_head`` (~5 M params,
efficientnet-b0 + transformer decoder) and the corresponding
``shead--{step}_checkpoint.pt`` from NHirose/AsyncVLA_release. On every
``predict_path`` call:

1. Resize / ImageNet-normalize cur + past frames to (3, 96, 96).
2. Reshape the gRPC ActionEmbedding to ``(1, 8, 1024)``.
3. Run ``Edge_adapter(cur, past, vla_feature)`` -> ``(1, 8, 4)`` deltas.
4. Apply ``delta_to_pose`` to accumulate into world-frame waypoints.
5. Build ``nav_msgs/Path`` (8 PoseStamped, base_link).

Pre-requisite: ``vint_train`` from MBRA must be on PYTHONPATH at load
time (small_head imports MultiLayerDecoder_trans). See
``Dockerfile.asyncvla``.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from .base import EdgeAdapter


_LOG = logging.getLogger(__name__)


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess_for_edge_adapter(image_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 HxWx3 -> (1, 3, 96, 96) ImageNet-normalized float32 ndarray.

    Mirrors run_asyncvla.py:1058-1062 (TF.resize + transform).
    """
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f'expected uint8 HxWx3 RGB, got dtype={image_rgb.dtype} shape={image_rgb.shape}')
    img = cv2.resize(image_rgb, (96, 96), interpolation=cv2.INTER_AREA)
    arr = img.astype(np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.transpose(arr, (2, 0, 1))[None, ...]


def _delta_to_pose_np(delta: np.ndarray) -> np.ndarray:
    """Port of run_asyncvla.delta_to_pose for numpy.

    delta: (N, T, 4) packed as (dx, dy, cos(dtheta), sin(dtheta)).
    Returns: (N, T, 4) where last dim packs (x, y, cos(theta), sin(theta))
    in world frame.
    """
    dx = delta[..., 0]
    dy = delta[..., 1]
    dtheta = np.arctan2(delta[..., 3], delta[..., 2])
    N, T = dx.shape
    poses = np.zeros((N, T, 4), dtype=np.float32)

    x = dx[:, 0].copy()
    y = dy[:, 0].copy()
    theta = dtheta[:, 0].copy()
    poses[:, 0, 0] = x
    poses[:, 0, 1] = y
    poses[:, 0, 2] = np.cos(theta)
    poses[:, 0, 3] = np.sin(theta)

    for t in range(1, T):
        ct = np.cos(theta)
        st = np.sin(theta)
        dx_w = ct * dx[:, t] - st * dy[:, t]
        dy_w = st * dx[:, t] + ct * dy[:, t]
        x = x + dx_w
        y = y + dy_w
        theta = theta + dtheta[:, t]
        poses[:, t, 0] = x
        poses[:, t, 1] = y
        poses[:, t, 2] = np.cos(theta)
        poses[:, t, 3] = np.sin(theta)

    return poses


class AsyncVLAEdgeAdapter(EdgeAdapter):
    """Wraps ``prismatic.models.small_head.Edge_adapter`` for the edge node."""

    def __init__(
        self,
        *,
        weights_path: str,
        resume_step: int = 750000,
        device: str = 'cpu',
        # Edge_adapter constructor knobs (from AsyncVLA's config_nav/dataset_config.yaml).
        # obs_encoding_size MUST match the cloud's Proj_Actiontokens output dim
        # (action_dim=1024 in run_asyncvla.py:660); otherwise the decoder cat fails.
        obs_encoding_size: int = 1024,
        mha_num_attention_heads: int = 4,
        mha_num_attention_layers: int = 4,
        mha_ff_dim_factor: int = 4,
    ) -> None:
        # Imports are deferred to keep edge-node startup fast when adapter_kind
        # is not asyncvla -- prismatic + efficientnet_pytorch + vint_train are
        # heavy and fail noisily if MBRA isn't on PYTHONPATH.
        import torch
        from prismatic.models.small_head import Edge_adapter

        self._device = torch.device(device)
        self._dtype = torch.float32  # edge runs CPU fp32 by default

        cp_path = os.path.join(weights_path, f'shead--{resume_step}_checkpoint.pt')
        if not os.path.exists(cp_path):
            raise FileNotFoundError(f'Edge_adapter checkpoint not found at {cp_path}')

        _LOG.info('loading Edge_adapter from %s', cp_path)
        self._model = Edge_adapter(
            obs_encoding_size=obs_encoding_size,
            mha_num_attention_heads=mha_num_attention_heads,
            mha_num_attention_layers=mha_num_attention_layers,
            mha_ff_dim_factor=mha_ff_dim_factor,
        )
        raw = torch.load(cp_path, map_location=device)
        cleaned = {(k[len('module.'):] if k.startswith('module.') else k): v
                   for k, v in raw.items()}
        missing, unexpected = self._model.load_state_dict(cleaned, strict=False)
        if missing:
            _LOG.warning('Edge_adapter missing keys: %s', missing[:5])
        if unexpected:
            _LOG.warning('Edge_adapter unexpected keys: %s', unexpected[:5])
        self._model = self._model.to(self._device).to(self._dtype).eval()

    def predict_path(
        self,
        *,
        embedding: np.ndarray,
        embedding_shape: Tuple[int, int, int],
        cur_image_rgb: Optional[np.ndarray] = None,
        past_image_rgb: Optional[np.ndarray] = None,
        frame_id: str = 'base_link',
    ) -> Path:
        if cur_image_rgb is None:
            raise ValueError('AsyncVLAEdgeAdapter requires cur_image_rgb')
        # First-frame fallback: reuse current image if no past available.
        if past_image_rgb is None:
            past_image_rgb = cur_image_rgb

        import torch

        cur = torch.from_numpy(_preprocess_for_edge_adapter(cur_image_rgb)).to(self._device).to(self._dtype)
        past = torch.from_numpy(_preprocess_for_edge_adapter(past_image_rgb)).to(self._device).to(self._dtype)

        B, num_tokens, embed_dim = embedding_shape
        feat = (
            torch.from_numpy(np.asarray(embedding, dtype=np.float32))
            .reshape(B, num_tokens, embed_dim)
            .to(self._device)
            .to(self._dtype)
        )

        with torch.no_grad():
            delta = self._model(cur, past, feat)        # (1, 8, 4)
        poses = _delta_to_pose_np(delta.cpu().numpy()) # (1, 8, 4)
        wp = poses[0]                                   # (8, 4)

        path = Path()
        path.header.frame_id = frame_id
        for x, y, c, s in wp:
            ps = PoseStamped()
            ps.header.frame_id = frame_id
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.z = float(s)
            ps.pose.orientation.w = float(c)
            path.poses.append(ps)
        return path
