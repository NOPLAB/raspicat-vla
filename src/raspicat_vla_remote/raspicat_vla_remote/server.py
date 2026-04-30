"""Generic gRPC servicer wrapping any VLABackend."""
from __future__ import annotations

import io
import logging
import threading
import time
from concurrent import futures
from typing import Iterator, Optional

import grpc
import numpy as np
import PIL.Image

from raspicat_vla_proto import raspicat_vla_pb2, raspicat_vla_pb2_grpc
from raspicat_vla_proto.conversions import float32_array_to_fp16_bytes

from .backends.base import VLABackend


_LOG = logging.getLogger(__name__)


def _proto_goal_to_python(goal):
    if goal.mode == raspicat_vla_pb2.GoalSpec.POSE:
        return (goal.pose.x, goal.pose.y, goal.pose.theta), '', None
    if goal.mode == raspicat_vla_pb2.GoalSpec.TEXT:
        return None, goal.text, None
    if goal.mode == raspicat_vla_pb2.GoalSpec.IMAGE:
        try:
            img = PIL.Image.open(io.BytesIO(bytes(goal.image_jpeg))).convert('RGB')
        except Exception as exc:  # noqa: BLE001
            _LOG.warning('goal image decode failed: %s', exc)
            img = None
        return None, '', img
    raise ValueError(f'unknown goal mode {goal.mode}')


def _decode_jpeg_to_pil(jpeg_bytes: bytes) -> PIL.Image.Image:
    """Decode a JPEG bytestring or return a 1x1 placeholder on failure.

    The dummy backend ignores image contents, and Plan 1 tests pass JPEG-shaped
    garbage. Real backends consume the result via a HF processor that will
    fail noisily if the image is unusable, so silent fallback here is safe.
    """
    if not jpeg_bytes:
        return PIL.Image.new('RGB', (1, 1))
    try:
        return PIL.Image.open(io.BytesIO(jpeg_bytes)).convert('RGB')
    except Exception as exc:  # noqa: BLE001
        _LOG.warning('image decode failed: %s; using 1x1 placeholder', exc)
        return PIL.Image.new('RGB', (1, 1))


class _Servicer(raspicat_vla_pb2_grpc.VLAServiceServicer):
    def __init__(self, *, backend: VLABackend) -> None:
        self._backend = backend
        self._past_image_per_client: dict = {}
        self._past_image_lock = threading.Lock()

    def GetModelInfo(self, request, context):  # noqa: ARG002
        info = self._backend.model_info()
        return raspicat_vla_pb2.ModelInfo(
            model_name=info.model_name,
            model_version=info.model_version,
            num_tokens=info.num_tokens,
            embed_dim=info.embed_dim,
            device=info.device,
            ready=info.ready,
        )

    def StreamInfer(
        self,
        request_iterator: Iterator[raspicat_vla_pb2.Observation],
        context,
    ) -> Iterator[raspicat_vla_pb2.ActionEmbedding]:
        peer = context.peer()
        info = self._backend.model_info()
        for obs in request_iterator:
            cur_img = _decode_jpeg_to_pil(bytes(obs.image_jpeg))
            with self._past_image_lock:
                past = self._past_image_per_client.get(peer, cur_img)
                self._past_image_per_client[peer] = cur_img
            try:
                pose, text, goal_img = _proto_goal_to_python(obs.goal)
            except ValueError as exc:
                _LOG.warning('bad goal mode for frame_id=%s: %s', obs.frame_id, exc)
                pose, text, goal_img = None, '', None
            try:
                proj, metrics = self._backend.infer(
                    current_image=cur_img,
                    past_image=past,
                    lang_instruction=text,
                    goal_image=goal_img,
                    goal_pose_xy_theta=pose,
                )
            except Exception as exc:  # noqa: BLE001
                _LOG.exception('inference failed for frame_id=%s: %s', obs.frame_id, exc)
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(exc))
                return
            arr = np.asarray(proj, dtype=np.float32)
            num_tokens, embed_dim = arr.shape[-2], arr.shape[-1]
            yield raspicat_vla_pb2.ActionEmbedding(
                frame_id=obs.frame_id,
                server_time_ns=time.monotonic_ns(),
                num_tokens=int(num_tokens),
                embed_dim=int(embed_dim),
                embedding_fp16=float32_array_to_fp16_bytes(arr.reshape(-1)),
                inference_ms=float(metrics.get('inference_ms', 0.0)),
                model_version=info.model_version,
            )


class VLAServer:
    """Generic gRPC server wrapping a VLABackend.

    Replaces Plan 1's `DummyServer._Servicer` with a backend-dispatched servicer
    so dummy / AsyncVLA / OmniVLA (Plan 2A / 2B) all share the same plumbing.
    """

    def __init__(
        self,
        *,
        backend: VLABackend,
        host: str = '0.0.0.0',
        port: int = 50051,
        max_workers: int = 4,
    ) -> None:
        self._backend = backend
        self._host = host
        self._port = port
        self._max_workers = max_workers
        self._servicer = _Servicer(backend=backend)
        self._server: Optional[grpc.Server] = None
        self._actual_port: Optional[int] = None

    def start(self) -> int:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=self._max_workers))
        raspicat_vla_pb2_grpc.add_VLAServiceServicer_to_server(self._servicer, server)
        self._actual_port = server.add_insecure_port(f'{self._host}:{self._port}')
        server.start()
        self._server = server
        _LOG.info('VLAServer listening on %s:%d', self._host, self._actual_port)
        return self._actual_port

    def stop(self, grace_sec: float = 1.0) -> None:
        if self._server is not None:
            self._server.stop(grace_sec)
            self._server = None

    def wait_for_termination(self) -> None:
        if self._server is not None:
            self._server.wait_for_termination()
