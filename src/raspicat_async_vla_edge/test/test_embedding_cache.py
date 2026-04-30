"""Tests for EmbeddingCache."""
import time

import numpy as np
import pytest

from asyncvla_edge.embedding_cache import EmbeddingCache, CachedEmbedding


def _emb(frame_id: int, value: float = 0.0) -> CachedEmbedding:
    return CachedEmbedding(
        frame_id=frame_id,
        recv_time_ns=time.monotonic_ns(),
        embedding=np.full(8 * 1024, value, dtype=np.float32),
        num_tokens=8,
        embed_dim=1024,
        inference_ms=10.0,
        model_version='dummy',
    )


def test_cache_starts_empty():
    cache = EmbeddingCache(max_age_sec=6.0, hard_timeout_sec=15.0)
    assert cache.get_fresh() is None
    assert cache.status() == 'WAITING_REMOTE'


def test_cache_stores_and_returns_latest():
    cache = EmbeddingCache(max_age_sec=6.0, hard_timeout_sec=15.0)
    cache.put(_emb(frame_id=1, value=1.0))
    cur = cache.get_fresh()
    assert cur is not None
    assert cur.frame_id == 1
    assert cache.status() == 'OK'


def test_cache_drops_older_frame_id():
    cache = EmbeddingCache(max_age_sec=6.0, hard_timeout_sec=15.0)
    cache.put(_emb(frame_id=10, value=10.0))
    cache.put(_emb(frame_id=5, value=5.0))  # older, must be dropped
    cur = cache.get_fresh()
    assert cur is not None
    assert cur.frame_id == 10
    assert cur.embedding[0] == pytest.approx(10.0)


def test_cache_returns_none_when_stale_past_max_age():
    cache = EmbeddingCache(max_age_sec=0.001, hard_timeout_sec=0.01)
    e = _emb(frame_id=1)
    cache.put(e)
    time.sleep(0.005)
    assert cache.get_fresh() is None
    # but raw still readable for diagnostics
    assert cache.get_latest_raw() is not None
    assert cache.status() == 'DEGRADED'


def test_cache_status_stale_after_hard_timeout():
    cache = EmbeddingCache(max_age_sec=0.001, hard_timeout_sec=0.005)
    cache.put(_emb(frame_id=1))
    time.sleep(0.020)
    assert cache.status() == 'STALE'


def test_cache_invalidate_clears_state():
    cache = EmbeddingCache(max_age_sec=6.0, hard_timeout_sec=15.0)
    cache.put(_emb(frame_id=1))
    cache.invalidate()
    assert cache.get_fresh() is None
    assert cache.status() == 'WAITING_REMOTE'


def test_cache_invalidate_with_floor_rejects_stale_frames():
    """invalidate(floor=N) must reject all subsequent puts with frame_id <= N.

    This closes a race: when the edge node receives a new goal it invalidates
    the cache, but observations for the OLD goal may already be in flight. Their
    embeddings will arrive shortly after invalidate() and, without a floor,
    would be re-cached and steer the robot toward the stale goal for one tick.
    """
    cache = EmbeddingCache(max_age_sec=6.0, hard_timeout_sec=15.0)
    cache.put(_emb(frame_id=5, value=5.0))
    assert cache.get_latest_raw().frame_id == 5

    cache.invalidate(floor=5)
    # invalidate must also drop the current latest
    assert cache.get_latest_raw() is None
    assert cache.status() == 'WAITING_REMOTE'

    # frame_id == floor: rejected
    cache.put(_emb(frame_id=5, value=5.0))
    assert cache.get_latest_raw() is None

    # frame_id < floor: rejected
    cache.put(_emb(frame_id=4, value=4.0))
    assert cache.get_latest_raw() is None

    # frame_id > floor: accepted
    cache.put(_emb(frame_id=6, value=6.0))
    cur = cache.get_latest_raw()
    assert cur is not None
    assert cur.frame_id == 6
    assert cur.embedding[0] == pytest.approx(6.0)


def test_cache_invalidate_floor_sticky_until_exceeded():
    """After a floor-bypassing put lands, monotonicity (frame_id > latest.frame_id)
    keeps older frames out — documenting the intended composition with
    invalidate(floor=...).
    """
    cache = EmbeddingCache(max_age_sec=6.0, hard_timeout_sec=15.0)
    cache.invalidate(floor=10)
    cache.put(_emb(frame_id=11, value=11.0))
    assert cache.get_latest_raw().frame_id == 11

    # Older frame still rejected (existing monotonicity rule).
    cache.put(_emb(frame_id=10, value=10.0))
    assert cache.get_latest_raw().frame_id == 11

    # Frame at the floor itself also rejected (older than current latest=11).
    cache.put(_emb(frame_id=8, value=8.0))
    assert cache.get_latest_raw().frame_id == 11


def test_cache_invalidate_no_floor_default_unchanged():
    """invalidate() with no floor must behave exactly as before."""
    cache = EmbeddingCache(max_age_sec=6.0, hard_timeout_sec=15.0)
    cache.put(_emb(frame_id=10))
    cache.invalidate()
    assert cache.get_latest_raw() is None
    # No floor → frame_id=1 should be accepted (it's > latest which is None).
    cache.put(_emb(frame_id=1, value=1.0))
    cur = cache.get_latest_raw()
    assert cur is not None
    assert cur.frame_id == 1
