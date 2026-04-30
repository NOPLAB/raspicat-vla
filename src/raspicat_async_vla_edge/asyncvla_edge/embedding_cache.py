"""Thread-safe cache for the latest action embedding from the remote VLA."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CachedEmbedding:
    frame_id: int
    recv_time_ns: int           # monotonic ns at time of insertion
    embedding: np.ndarray       # shape (num_tokens * embed_dim,), dtype float32
    num_tokens: int
    embed_dim: int
    inference_ms: float
    model_version: str


class EmbeddingCache:
    """Holds the single latest embedding. Frame-id monotonic, age-aware."""

    STATUS_WAITING = 'WAITING_REMOTE'
    STATUS_OK = 'OK'
    STATUS_DEGRADED = 'DEGRADED'
    STATUS_STALE = 'STALE'

    def __init__(self, *, max_age_sec: float, hard_timeout_sec: float) -> None:
        if hard_timeout_sec < max_age_sec:
            raise ValueError('hard_timeout_sec must be >= max_age_sec')
        self._max_age_ns = int(max_age_sec * 1e9)
        self._hard_ns = int(hard_timeout_sec * 1e9)
        self._lock = threading.Lock()
        self._latest: Optional[CachedEmbedding] = None
        # Frame-id floor: any put() with frame_id <= _floor is rejected.
        # Set by invalidate(floor=...) when the goal changes so embeddings
        # derived from in-flight observations of the OLD goal don't land in
        # the cache. None means "no floor". The floor is sticky — it stays in
        # place even after a put() succeeds; the existing monotonicity rule
        # (frame_id > latest.frame_id) then keeps things consistent.
        self._floor: Optional[int] = None

    def put(self, emb: CachedEmbedding) -> None:
        with self._lock:
            if self._floor is not None and emb.frame_id <= self._floor:
                return
            if self._latest is None or emb.frame_id > self._latest.frame_id:
                self._latest = emb

    def invalidate(self, *, floor: Optional[int] = None) -> None:
        """Drop the cached embedding.

        If ``floor`` is given, also reject any future put() whose frame_id is
        <= floor. Callers use this when the goal changes so embeddings from
        observations sent under the previous goal can't sneak back in.
        """
        with self._lock:
            self._latest = None
            if floor is not None:
                self._floor = floor

    def get_latest_raw(self) -> Optional[CachedEmbedding]:
        with self._lock:
            return self._latest

    def _age_ns_locked(self) -> Optional[int]:
        if self._latest is None:
            return None
        return time.monotonic_ns() - self._latest.recv_time_ns

    def get_fresh(self) -> Optional[CachedEmbedding]:
        with self._lock:
            age = self._age_ns_locked()
            if age is None or age >= self._max_age_ns:
                return None
            return self._latest

    def status(self) -> str:
        with self._lock:
            age = self._age_ns_locked()
            if age is None:
                return self.STATUS_WAITING
            if age >= self._hard_ns:
                return self.STATUS_STALE
            if age >= self._max_age_ns:
                return self.STATUS_DEGRADED
            return self.STATUS_OK
