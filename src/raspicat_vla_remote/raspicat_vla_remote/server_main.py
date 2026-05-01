"""Entry point for the `vla_dummy_server` console script.

Selects a backend (``--backend {dummy,asyncvla,omnivla}``) and hosts it via
the generic :class:`VLAServer`. Plan 2A / 2B fill in the asyncvla / omnivla
branches as those backends land.
"""
from __future__ import annotations

import argparse
import logging
import signal

from .backends.dummy import DummyBackend
from .server import VLAServer


_LOG = logging.getLogger(__name__)


def _build_backend(args: argparse.Namespace):
    if args.backend == 'dummy':
        return DummyBackend(
            num_tokens=args.num_tokens,
            embed_dim=args.embed_dim,
            inference_ms=args.inference_ms,
            model_version=args.model_version,
        )
    if args.backend == 'omnivla':
        # Plan 2B Task 6 plugs OmniVLABackend in here.
        try:
            from .backends.omnivla import OmniVLABackend
        except ImportError as exc:
            raise SystemExit(
                f'--backend omnivla not yet available ({exc}); '
                'implement Plan 2B Task 6 first',
            )
        return OmniVLABackend(
            vla_path=args.vla_path,
            resume_step=args.resume_step,
            device=args.device,
        )
    if args.backend == 'asyncvla':
        try:
            from .backends.asyncvla import AsyncVLABackend
        except ImportError as exc:
            raise SystemExit(
                f'--backend asyncvla not yet available ({exc}); '
                'implement Plan 2A first',
            )
        return AsyncVLABackend(
            vla_path=args.vla_path,
            resume_step=args.resume_step,
            device=args.device,
        )
    raise SystemExit(f'unknown --backend {args.backend!r}')


def main() -> None:
    parser = argparse.ArgumentParser(description='VLA gRPC server (dummy/asyncvla/omnivla)')
    parser.add_argument('--backend', default='dummy', choices=['dummy', 'asyncvla', 'omnivla'])
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=50051)
    parser.add_argument('--log-level', default='INFO')

    # Dummy-only knobs.
    parser.add_argument('--num-tokens', type=int, default=8)
    parser.add_argument('--embed-dim', type=int, default=1024)
    parser.add_argument('--inference-ms', type=float, default=50.0)
    parser.add_argument('--model-version', default='dummy-v1')

    # Real-model (asyncvla/omnivla) knobs.
    parser.add_argument('--vla-path', default='/workspace/models/omnivla-original',
                        help='checkpoint dir (omnivla: ./models/omnivla-original; asyncvla: ./models/AsyncVLA_release)')
    parser.add_argument('--resume-step', type=int, default=120000)
    parser.add_argument('--device', default='cuda:0')

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    backend = _build_backend(args)
    server = VLAServer(backend=backend, host=args.host, port=args.port)
    port = server.start()
    _LOG.info('backend=%s listening on %s:%d', args.backend, args.host, port)

    def _sigterm(signum, frame):  # noqa: ARG001
        _LOG.info('SIGTERM received, stopping...')
        server.stop(grace_sec=1.0)

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)
    server.wait_for_termination()
