"""Tests for OmniVLAEdgeAdapter -- pure shape + path-building math, no model."""
import math

import numpy as np
import pytest

from raspicat_vla_edge.adapters.omnivla import OmniVLAEdgeAdapter


def test_predict_path_zero_waypoints_yields_origin_path():
    adapter = OmniVLAEdgeAdapter()
    emb = np.zeros((8, 4), dtype=np.float32)
    emb[..., 2] = 1.0  # cos(theta=0)
    path = adapter.predict_path(
        embedding=emb, embedding_shape=(1, 8, 4),
        frame_id='base_link',
    )
    assert path.header.frame_id == 'base_link'
    assert len(path.poses) == 8
    for ps in path.poses:
        assert ps.pose.position.x == 0.0
        assert ps.pose.position.y == 0.0
        assert ps.pose.orientation.w == 1.0  # cos(0)
        assert ps.pose.orientation.z == 0.0  # sin(0)
        assert ps.header.frame_id == 'base_link'


def test_predict_path_propagates_xy():
    adapter = OmniVLAEdgeAdapter()
    emb = np.zeros((4, 4), dtype=np.float32)
    emb[:, 0] = [0.1, 0.2, 0.3, 0.4]   # x at each step
    emb[:, 2] = 1.0                     # cos(0)
    path = adapter.predict_path(
        embedding=emb, embedding_shape=(1, 4, 4),
        frame_id='base_link',
    )
    xs = [ps.pose.position.x for ps in path.poses]
    assert xs == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3), pytest.approx(0.4)]


def test_predict_path_preserves_orientation_cos_sin():
    """The (cos, sin) packed into the last two dims map to (z, w) of the yaw quaternion."""
    adapter = OmniVLAEdgeAdapter()
    emb = np.zeros((1, 4), dtype=np.float32)
    emb[0, :] = [1.0, 0.0, math.cos(math.pi / 4), math.sin(math.pi / 4)]
    path = adapter.predict_path(
        embedding=emb, embedding_shape=(1, 1, 4),
        frame_id='base_link',
    )
    ps = path.poses[0]
    assert ps.pose.position.x == pytest.approx(1.0)
    assert ps.pose.position.y == pytest.approx(0.0)
    assert ps.pose.orientation.w == pytest.approx(math.cos(math.pi / 4), abs=1e-6)
    assert ps.pose.orientation.z == pytest.approx(math.sin(math.pi / 4), abs=1e-6)


def test_predict_path_accepts_flat_1d_embedding_with_shape():
    """The edge_node passes embedding as a flat 1D ndarray; reshape via embedding_shape."""
    adapter = OmniVLAEdgeAdapter()
    flat = np.zeros(8 * 4, dtype=np.float32)
    flat[2::4] = 1.0  # set every cos to 1
    path = adapter.predict_path(
        embedding=flat, embedding_shape=(1, 8, 4),
        frame_id='odom',
    )
    assert path.header.frame_id == 'odom'
    assert len(path.poses) == 8


def test_predict_path_rejects_too_few_action_dims():
    adapter = OmniVLAEdgeAdapter()
    emb = np.zeros((8, 3), dtype=np.float32)
    with pytest.raises(ValueError, match='ACTION_DIM>=4'):
        adapter.predict_path(
            embedding=emb, embedding_shape=(1, 8, 3),
            frame_id='base_link',
        )


def test_predict_path_uses_default_frame_id_base_link():
    adapter = OmniVLAEdgeAdapter()
    emb = np.zeros((1, 4), dtype=np.float32)
    emb[0, 2] = 1.0
    path = adapter.predict_path(embedding=emb, embedding_shape=(1, 1, 4))
    assert path.header.frame_id == 'base_link'
