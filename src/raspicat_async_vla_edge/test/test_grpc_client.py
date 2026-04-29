"""Tests for the AsyncVLA gRPC bidi-stream client wrapper.

The client must:
  - send observations from a thread-safe input
  - deliver embeddings via a callback
  - be startable / stoppable cleanly
  - tolerate the server going away (reconnect attempt)
"""
import threading
import time

import numpy as np
import pytest

from asyncvla_remote.dummy_server import DummyServer
from asyncvla_edge.grpc_client import AsyncVLAClient
from raspicat_async_vla_proto import asyncvla_pb2


@pytest.fixture
def server():
    s = DummyServer(host='localhost', port=0, num_tokens=4, embed_dim=8, inference_ms=1.0)
    port = s.start()
    yield port
    s.stop(grace_sec=0.5)


def _make_obs(frame_id: int) -> asyncvla_pb2.Observation:
    return asyncvla_pb2.Observation(
        frame_id=frame_id,
        capture_time_ns=time.monotonic_ns(),
        image_jpeg=b'\xff\xd8\xff' + b'\x00' * 16,
        image_width=224,
        image_height=224,
        goal=asyncvla_pb2.GoalSpec(
            mode=asyncvla_pb2.GoalSpec.POSE,
            pose=asyncvla_pb2.Pose2D(x=0.0, y=0.0, theta=0.0),
            frame_id='base_link',
        ),
    )


def test_client_round_trips_via_dummy_server(server):
    received = []
    cond = threading.Condition()

    def on_emb(emb):
        with cond:
            received.append(emb)
            cond.notify_all()

    client = AsyncVLAClient(address=f'localhost:{server}', on_embedding=on_emb)
    client.start()
    try:
        for i in range(5):
            client.send(_make_obs(i))
        with cond:
            cond.wait_for(lambda: len(received) >= 5, timeout=5.0)
        assert len(received) == 5
        assert {r.frame_id for r in received} == {0, 1, 2, 3, 4}
    finally:
        client.stop()
