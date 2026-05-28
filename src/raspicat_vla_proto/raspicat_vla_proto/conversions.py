"""ROS2 <-> proto conversion helpers."""
from __future__ import annotations

import numpy as np

from . import raspicat_vla_pb2


def float32_array_to_fp16_bytes(arr: np.ndarray) -> bytes:
    """Convert a contiguous float32 array to little-endian fp16 bytes."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    fp16 = arr.astype('<f2', copy=False)
    return fp16.tobytes()


def fp16_bytes_to_float32_list(raw: bytes) -> list[float]:
    """Convert little-endian fp16 bytes to a Python list of float32 values."""
    fp16 = np.frombuffer(raw, dtype='<f2')
    return fp16.astype(np.float32).tolist()


def proto_action_embedding_to_msg(
    proto: raspicat_vla_pb2.ActionEmbedding,
):
    """Convert a proto ActionEmbedding into the ROS2 message form.

    The ``raspicat_vla_msgs`` import is lazy so this module stays importable
    on cloud-only hosts (no ROS2 install).
    """
    from raspicat_vla_msgs.msg import ActionEmbedding as ActionEmbeddingMsg

    msg = ActionEmbeddingMsg()
    msg.frame_id = proto.frame_id
    msg.num_tokens = proto.num_tokens
    msg.embed_dim = proto.embed_dim
    msg.embedding = fp16_bytes_to_float32_list(proto.embedding_fp16)
    msg.inference_ms = float(proto.inference_ms)
    msg.model_version = proto.model_version or ''
    return msg
