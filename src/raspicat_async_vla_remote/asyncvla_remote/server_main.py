"""Entry point for `asyncvla_dummy_server` console script."""
from __future__ import annotations

import argparse
import logging
import signal

from .dummy_server import DummyServer


def main() -> None:
    parser = argparse.ArgumentParser(description='AsyncVLA dummy gRPC server')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=50051)
    parser.add_argument('--num-tokens', type=int, default=8)
    parser.add_argument('--embed-dim', type=int, default=1024)
    parser.add_argument('--inference-ms', type=float, default=50.0)
    parser.add_argument('--model-version', default='dummy-v1')
    parser.add_argument('--log-level', default='INFO')
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    server = DummyServer(
        host=args.host,
        port=args.port,
        num_tokens=args.num_tokens,
        embed_dim=args.embed_dim,
        inference_ms=args.inference_ms,
        model_version=args.model_version,
    )
    port = server.start()
    logging.info('listening on %s:%d', args.host, port)

    def _sigterm(signum, frame):  # noqa: ARG001
        logging.info('SIGTERM received, stopping...')
        server.stop(grace_sec=1.0)

    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)
    server.wait_for_termination()
