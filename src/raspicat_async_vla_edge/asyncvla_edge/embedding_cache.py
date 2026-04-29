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

    def put(self, emb: CachedEmbedding) -> None:
        with self._lock:
            if self._latest is None or emb.frame_id > self._latest.frame_id:
                self._latest = emb

    def invalidate(self) -> None:
        with self._lock:
            self._latest = None

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
