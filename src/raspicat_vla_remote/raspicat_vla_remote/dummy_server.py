"""Plan-1 back-compat shim: ``DummyServer`` constructed from ``VLAServer + DummyBackend``.

The original Plan-1 implementation lived entirely in this file. The Task 3
refactor moved the gRPC servicing into :mod:`server` and the model side into
:mod:`backends.dummy`; this module is preserved so existing imports keep
working (``from raspicat_vla_remote.dummy_server import DummyServer``).
"""
from __future__ import annotations

from .backends.dummy import DummyBackend
from .server import VLAServer


class DummyServer:
    """Process-local dummy gRPC server. Useful for tests and Plan 1 integration."""

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
        self._backend = DummyBackend(
            num_tokens=num_tokens,
            embed_dim=embed_dim,
            inference_ms=inference_ms,
            model_version=model_version,
        )
        self._inner = VLAServer(
            backend=self._backend,
            host=host,
            port=port,
            max_workers=max_workers,
        )

    def start(self) -> int:
        return self._inner.start()

    def stop(self, grace_sec: float = 1.0) -> None:
        self._inner.stop(grace_sec)

    def wait_for_termination(self) -> None:
        self._inner.wait_for_termination()
