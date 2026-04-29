"""Dummy gRPC server returning deterministic embeddings (no ML model)."""
from __future__ import annotations

import logging
import threading
import time
from concurrent import futures
from typing import Iterator, Optional

import grpc
import numpy as np

from raspicat_async_vla_proto import asyncvla_pb2, asyncvla_pb2_grpc
from raspicat_async_vla_proto.conversions import float32_array_to_fp16_bytes


_LOG = logging.getLogger(__name__)


class _Servicer(asyncvla_pb2_grpc.AsyncVLAServiceServicer):
    def __init__(
        self,
        *,
        num_tokens: int,
        embed_dim: int,
        inference_ms: float,
        model_version: str,
    ) -> None:
        self._num_tokens = num_tokens
        self._embed_dim = embed_dim
        self._inference_ms = inference_ms
        self._model_version = model_version

    def _embedding_for(self, frame_id: int) -> bytes:
        # Deterministic: every element = sin(frame_id * pi / 17) so it varies but is reproducible.
        seed = float(np.sin(frame_id * np.pi / 17))
        arr = np.full(self._num_tokens * self._embed_dim, seed, dtype=np.float32)
        return float32_array_to_fp16_bytes(arr)

    def GetModelInfo(self, request, context):
        return asyncvla_pb2.ModelInfo(
            model_name='dummy',
            model_version=self._model_version,
            num_tokens=self._num_tokens,
            embed_dim=self._embed_dim,
            device='cpu',
            ready=True,
        )

    def StreamInfer(
        self,
        request_iterator: Iterator[asyncvla_pb2.Observation],
        context,
    ) -> Iterator[asyncvla_pb2.ActionEmbedding]:
        for obs in request_iterator:
            if self._inference_ms > 0:
                time.sleep(self._inference_ms / 1000.0)
            yield asyncvla_pb2.ActionEmbedding(
                frame_id=obs.frame_id,
                server_time_ns=time.monotonic_ns(),
                num_tokens=self._num_tokens,
                embed_dim=self._embed_dim,
                embedding_fp16=self._embedding_for(obs.frame_id),
                inference_ms=self._inference_ms,
                model_version=self._model_version,
            )


class DummyServer:
    """Process-local dummy server. Useful for tests and Plan 1 integration."""

    def __init__(
        self,
        *,
        host: str = '0.0.0.0',
        port: int = 50051,
        num_tokens: int = 8,
        embed_dim: int = 1024,
        inference_ms: float = 50.0,
        model_version: str = 'dummy-v1',
        max_workers: int = 4,
    ) -> None:
        self._host = host
        self._port = port
        self._max_workers = max_workers
        self._servicer = _Servicer(
            num_tokens=num_tokens,
            embed_dim=embed_dim,
            inference_ms=inference_ms,
            model_version=model_version,
        )
        self._server: Optional[grpc.Server] = None
        self._actual_port: Optional[int] = None

    def start(self) -> int:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=self._max_workers))
        asyncvla_pb2_grpc.add_AsyncVLAServiceServicer_to_server(self._servicer, server)
        bind = f'{self._host}:{self._port}'
        self._actual_port = server.add_insecure_port(bind)
        server.start()
        self._server = server
        _LOG.info('DummyServer listening on %s:%d', self._host, self._actual_port)
        return self._actual_port

    def wait_for_termination(self) -> None:
        if self._server is None:
            return
        self._server.wait_for_termination()

    def stop(self, grace_sec: float = 1.0) -> None:
        if self._server is not None:
            self._server.stop(grace_sec)
            self._server = None
