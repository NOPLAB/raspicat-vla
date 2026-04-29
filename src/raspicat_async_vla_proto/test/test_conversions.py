"""Tests for ROS2 <-> proto conversion helpers."""
import numpy as np
import pytest

from raspicat_async_vla_msgs.msg import ActionEmbedding as ActionEmbeddingMsg
from raspicat_async_vla_proto import asyncvla_pb2
from raspicat_async_vla_proto.conversions import (
    proto_action_embedding_to_msg,
    fp16_bytes_to_float32_list,
    float32_array_to_fp16_bytes,
)


def test_fp16_bytes_round_trip():
    arr = np.arange(8 * 1024, dtype=np.float32) / 100.0
    raw = float32_array_to_fp16_bytes(arr)
    assert isinstance(raw, bytes)
    assert len(raw) == 8 * 1024 * 2  # fp16 = 2 bytes
    back = np.array(fp16_bytes_to_float32_list(raw), dtype=np.float32)
    assert back.shape == arr.shape
    np.testing.assert_allclose(back, arr, atol=1e-2)  # fp16 quantization


def test_proto_action_embedding_to_msg_basic():
    arr = np.linspace(-1, 1, 8 * 16, dtype=np.float32)
    proto = asyncvla_pb2.ActionEmbedding(
        frame_id=42,
        server_time_ns=123,
        num_tokens=8,
        embed_dim=16,
        embedding_fp16=float32_array_to_fp16_bytes(arr),
        inference_ms=12.5,
        model_version='dummy',
    )
    msg = proto_action_embedding_to_msg(proto)
    assert isinstance(msg, ActionEmbeddingMsg)
    assert msg.frame_id == 42
    assert msg.num_tokens == 8
    assert msg.embed_dim == 16
    assert len(msg.embedding) == 8 * 16
    np.testing.assert_allclose(np.array(msg.embedding), arr, atol=1e-2)
    assert msg.inference_ms == pytest.approx(12.5)
    assert msg.model_version == 'dummy'
