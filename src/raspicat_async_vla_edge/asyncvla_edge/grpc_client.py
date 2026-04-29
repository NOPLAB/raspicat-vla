"""Threaded gRPC bidi-stream client for AsyncVLA."""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Optional

import grpc

from raspicat_async_vla_proto import asyncvla_pb2, asyncvla_pb2_grpc


_LOG = logging.getLogger(__name__)

EmbeddingCallback = Callable[[asyncvla_pb2.ActionEmbedding], None]


class AsyncVLAClient:
    """Threaded bidirectional gRPC client.

    Two threads run between start() and stop():
      - sender: pulls Observation from the queue and yields them to gRPC
      - receiver: iterates the gRPC reply stream and calls on_embedding(...)

    On stream failure the client reconnects with exponential backoff up to
    `max_backoff_sec`. While disconnected, queued observations are dropped
    if the queue exceeds `queue_max`.
    """

    def __init__(
        self,
        *,
        address: str,
        on_embedding: EmbeddingCallback,
        queue_max: int = 32,
        initial_backoff_sec: float = 0.5,
        max_backoff_sec: float = 5.0,
    ) -> None:
        self._address = address
        self._on_embedding = on_embedding
        self._queue: 'queue.Queue[asyncvla_pb2.Observation]' = queue.Queue(maxsize=queue_max)
        self._sentinel = object()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._initial_backoff = initial_backoff_sec
        self._max_backoff = max_backoff_sec

    # ------------------------------------------------------------------ public

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name='AsyncVLAClient', daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(self._sentinel)  # type: ignore[arg-type]
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def send(self, obs: asyncvla_pb2.Observation) -> bool:
        try:
            self._queue.put_nowait(obs)
            return True
        except queue.Full:
            _LOG.warning('observation queue full; dropping frame_id=%s', obs.frame_id)
            return False

    # ------------------------------------------------------------------ internal

    def _run(self) -> None:
        backoff = self._initial_backoff
        while not self._stop_event.is_set():
            try:
                channel = grpc.insecure_channel(self._address)
                stub = asyncvla_pb2_grpc.AsyncVLAServiceStub(channel)
                _LOG.info('connecting to %s', self._address)
                self._run_stream(stub)
                channel.close()
                backoff = self._initial_backoff
            except grpc.RpcError as exc:
                _LOG.warning('gRPC error: %s; backing off %.2fs', exc, backoff)
                self._stop_event.wait(timeout=backoff)
                backoff = min(backoff * 2.0, self._max_backoff)
            except Exception as exc:  # noqa: BLE001
                _LOG.exception('unexpected client error: %s', exc)
                self._stop_event.wait(timeout=backoff)
                backoff = min(backoff * 2.0, self._max_backoff)

    def _request_iter(self):
        while not self._stop_event.is_set():
            item = self._queue.get()
            if item is self._sentinel:
                return
            yield item

    def _run_stream(self, stub: asyncvla_pb2_grpc.AsyncVLAServiceStub) -> None:
        replies = stub.StreamInfer(self._request_iter())
        for reply in replies:
            try:
                self._on_embedding(reply)
            except Exception:  # noqa: BLE001
                _LOG.exception('on_embedding callback raised')
            if self._stop_event.is_set():
                return
