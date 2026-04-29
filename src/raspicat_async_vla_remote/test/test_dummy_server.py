"""Integration test: the DummyServer answers a StreamInfer with deterministic embedding."""
import threading
import time

import grpc
import numpy as np
import pytest

from raspicat_async_vla_proto import asyncvla_pb2, asyncvla_pb2_grpc
from raspicat_async_vla_proto.conversions import fp16_bytes_to_float32_list
from asyncvla_remote.dummy_server import DummyServer


@pytest.fixture
def server_address():
    server = DummyServer(
        host='localhost',
        port=0,                # let OS pick a port
        num_tokens=8,
        embed_dim=16,
        inference_ms=5.0,
        model_version='dummy-test',
    )
    actual_port = server.start()
    addr = f'localhost:{actual_port}'
    yield addr
    server.stop(grace_sec=0.5)


def test_get_model_info_reports_ready(server_address):
    with grpc.insecure_channel(server_address) as ch:
        stub = asyncvla_pb2_grpc.AsyncVLAServiceStub(ch)
        info = stub.GetModelInfo(asyncvla_pb2.ModelInfoRequest(), timeout=2.0)
        assert info.ready is True
        assert info.num_tokens == 8
        assert info.embed_dim == 16


def test_stream_infer_round_trip(server_address):
    with grpc.insecure_channel(server_address) as ch:
        stub = asyncvla_pb2_grpc.AsyncVLAServiceStub(ch)

        def gen():
            for i in range(3):
                yield asyncvla_pb2.Observation(
                    frame_id=i,
                    capture_time_ns=time.monotonic_ns(),
                    image_jpeg=b'\xff\xd8\xff' + b'\x00' * 32,
                    image_width=224,
                    image_height=224,
                    goal=asyncvla_pb2.GoalSpec(
                        mode=asyncvla_pb2.GoalSpec.POSE,
                        pose=asyncvla_pb2.Pose2D(x=1.0, y=0.0, theta=0.0),
                        frame_id='base_link',
                    ),
                )

        replies = list(stub.StreamInfer(gen(), timeout=5.0))
        assert len(replies) == 3
        assert {r.frame_id for r in replies} == {0, 1, 2}
        for r in replies:
            assert r.num_tokens == 8
            assert r.embed_dim == 16
            arr = np.array(fp16_bytes_to_float32_list(r.embedding_fp16), dtype=np.float32)
            assert arr.shape == (8 * 16,)
