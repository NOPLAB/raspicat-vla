"""VLA edge LifecycleNode — Plan 1 skeleton.

In Plan 1 the "edge adapter" is a stub that emits a fixed straight-ahead
path of length 1.0 m sampled at 0.1 m. Plan 2 replaces this stub with the
real Edge Adapter PyTorch model (model-specific, e.g. AsyncVLA / OmniVLA).
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State
from sensor_msgs.msg import Image

from raspicat_vla_msgs.msg import (
    ActionEmbedding as ActionEmbeddingMsg,
    GoalSpec as GoalSpecMsg,
)
from raspicat_vla_proto import raspicat_vla_pb2
from raspicat_vla_proto.conversions import (
    fp16_bytes_to_float32_list,
    proto_action_embedding_to_msg,
)

from .preprocess import resize_and_jpeg
from .embedding_cache import EmbeddingCache, CachedEmbedding
from .grpc_client import VLAClient
from .adapters.base import EdgeAdapter


def _build_adapter(kind: str, *, params: dict) -> EdgeAdapter:
    """Construct the EdgeAdapter selected by the ``adapter_kind`` parameter.

    ``params`` is a dict of node parameters used by adapters that need extra
    config (e.g. AsyncVLA's Edge_adapter weights path).
    """
    if kind == 'stub':
        from .adapters.stub import StubAdapter
        return StubAdapter()
    if kind == 'omnivla':
        from .adapters.omnivla import OmniVLAEdgeAdapter
        return OmniVLAEdgeAdapter()
    if kind == 'asyncvla':
        from .adapters.asyncvla import AsyncVLAEdgeAdapter
        return AsyncVLAEdgeAdapter(
            weights_path=str(params.get('asyncvla_weights_path', '/workspace/models/AsyncVLA_release')),
            resume_step=int(params.get('asyncvla_resume_step', 750000)),
            device=str(params.get('asyncvla_device', 'cpu')),
        )
    raise ValueError(f'unknown adapter_kind: {kind!r} (choices: stub|asyncvla|omnivla)')


def _ros_goal_to_proto(goal: GoalSpecMsg) -> raspicat_vla_pb2.GoalSpec:
    if goal.mode == GoalSpecMsg.MODE_POSE:
        return raspicat_vla_pb2.GoalSpec(
            mode=raspicat_vla_pb2.GoalSpec.POSE,
            pose=raspicat_vla_pb2.Pose2D(
                x=goal.pose.pose.position.x,
                y=goal.pose.pose.position.y,
                theta=0.0,  # extracting yaw is done in Plan 2 with tf
            ),
            frame_id=goal.pose.header.frame_id or 'odom',
        )
    if goal.mode == GoalSpecMsg.MODE_TEXT:
        return raspicat_vla_pb2.GoalSpec(
            mode=raspicat_vla_pb2.GoalSpec.TEXT, text=goal.text, frame_id='',
        )
    if goal.mode == GoalSpecMsg.MODE_IMAGE:
        return raspicat_vla_pb2.GoalSpec(
            mode=raspicat_vla_pb2.GoalSpec.IMAGE,
            image_jpeg=bytes(goal.image.data),
            frame_id='',
        )
    raise ValueError(f'unknown goal mode {goal.mode}')


class VLAEdgeNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__('vla_edge_node')
        self._declare_parameters()
        self._bridge = CvBridge()
        self._latest_image: Optional[np.ndarray] = None
        self._latest_image_lock = threading.Lock()
        self._latest_goal: Optional[GoalSpecMsg] = None
        self._latest_goal_lock = threading.Lock()
        self._cache: Optional[EmbeddingCache] = None
        self._client: Optional[VLAClient] = None
        self._adapter: Optional[EdgeAdapter] = None
        self._frame_counter = 0
        self._send_timer = None
        self._action_timer = None
        self._status_timer = None
        self._image_sub = None
        self._goal_sub = None
        self._path_pub = None
        self._embedding_pub = None
        self._status_pub = None

    # ----------------------------------------------------------------- params

    def _declare_parameters(self) -> None:
        self.declare_parameter('remote_address', 'localhost:50051')
        self.declare_parameter('obs_publish_rate_hz', 2.0)
        self.declare_parameter('action_rate_hz', 10.0)
        self.declare_parameter('image_size', [224, 224])
        self.declare_parameter('jpeg_quality', 85)
        self.declare_parameter('embedding_max_age_sec', 6.0)
        self.declare_parameter('embedding_hard_timeout_sec', 15.0)
        self.declare_parameter('goal_tolerance_m', 0.3)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('goal_topic', '/raspicat_vla/goal')
        self.declare_parameter('path_topic', '/raspicat_vla/predicted_path')
        self.declare_parameter('status_topic', '/raspicat_vla/status')
        self.declare_parameter('embedding_debug_topic', '/raspicat_vla/embedding')
        self.declare_parameter('publish_embedding_debug', True)
        self.declare_parameter('adapter_kind', 'stub')  # stub|asyncvla|omnivla
        # AsyncVLA edge knobs (only used when adapter_kind='asyncvla').
        self.declare_parameter('asyncvla_weights_path', '/workspace/models/AsyncVLA_release')
        self.declare_parameter('asyncvla_resume_step', 750000)
        self.declare_parameter('asyncvla_device', 'cpu')

    # ------------------------------------------------------------- lifecycle

    def on_configure(self, state: State) -> TransitionCallbackReturn:  # noqa: ARG002
        self.get_logger().info('on_configure')
        addr = self.get_parameter('remote_address').get_parameter_value().string_value
        max_age = self.get_parameter('embedding_max_age_sec').value
        hard = self.get_parameter('embedding_hard_timeout_sec').value
        self._cache = EmbeddingCache(max_age_sec=float(max_age), hard_timeout_sec=float(hard))
        self._client = VLAClient(address=addr, on_embedding=self._on_embedding_received)
        adapter_kind = str(self.get_parameter('adapter_kind').value)
        adapter_params = {
            'asyncvla_weights_path': self.get_parameter('asyncvla_weights_path').value,
            'asyncvla_resume_step': self.get_parameter('asyncvla_resume_step').value,
            'asyncvla_device': self.get_parameter('asyncvla_device').value,
        }
        self._adapter = _build_adapter(adapter_kind, params=adapter_params)
        self.get_logger().info(f'edge adapter_kind={adapter_kind!r}')

        image_topic = self.get_parameter('image_topic').value
        goal_topic = self.get_parameter('goal_topic').value
        path_topic = self.get_parameter('path_topic').value
        status_topic = self.get_parameter('status_topic').value
        emb_topic = self.get_parameter('embedding_debug_topic').value

        self._image_sub = self.create_subscription(
            Image, image_topic, self._on_image, 10,
        )
        self._goal_sub = self.create_subscription(
            GoalSpecMsg, goal_topic, self._on_goal, 1,
        )
        self._path_pub = self.create_publisher(Path, path_topic, 10)
        self._status_pub = self.create_publisher(DiagnosticArray, status_topic, 10)
        if self.get_parameter('publish_embedding_debug').value:
            self._embedding_pub = self.create_publisher(ActionEmbeddingMsg, emb_topic, 10)

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:  # noqa: ARG002
        self.get_logger().info('on_activate')
        assert self._client is not None
        self._client.start()
        obs_rate = float(self.get_parameter('obs_publish_rate_hz').value)
        act_rate = float(self.get_parameter('action_rate_hz').value)
        self._send_timer = self.create_timer(1.0 / obs_rate, self._send_observation_tick)
        self._action_timer = self.create_timer(1.0 / act_rate, self._action_tick)
        self._status_timer = self.create_timer(1.0, self._publish_status)
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:  # noqa: ARG002
        self.get_logger().info('on_deactivate')
        for t in (self._send_timer, self._action_timer, self._status_timer):
            if t is not None:
                self.destroy_timer(t)
        self._send_timer = self._action_timer = self._status_timer = None
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:  # noqa: ARG002
        self.get_logger().info('on_cleanup')
        if self._client is not None:
            self._client.stop()
        self._client = None
        self._cache = None
        self._adapter = None
        # Destroy subscriptions and publishers so a subsequent configure
        # doesn't leak duplicates (lifecycle expects on_cleanup to invert
        # on_configure).
        if self._image_sub is not None:
            self.destroy_subscription(self._image_sub)
            self._image_sub = None
        if self._goal_sub is not None:
            self.destroy_subscription(self._goal_sub)
            self._goal_sub = None
        for pub_attr in ('_path_pub', '_status_pub', '_embedding_pub'):
            pub = getattr(self, pub_attr)
            if pub is not None:
                self.destroy_publisher(pub)
                setattr(self, pub_attr, None)
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:  # noqa: ARG002
        self.get_logger().info('on_shutdown')
        if self._client is not None:
            self._client.stop()
        return TransitionCallbackReturn.SUCCESS

    # ----------------------------------------------------------- subscribers

    def _on_image(self, msg: Image) -> None:
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'cv_bridge failed: {exc}')
            return
        with self._latest_image_lock:
            self._latest_image = cv_img

    def _on_goal(self, msg: GoalSpecMsg) -> None:
        self.get_logger().info(f'received goal mode={msg.mode}')
        with self._latest_goal_lock:
            self._latest_goal = msg
        if self._cache is not None:
            # Read of _frame_counter is unlocked: _send_observation_tick (on
            # another executor thread) increments without a lock. Worst-case
            # we read one too low/high — both are tolerable: too low keeps a
            # stale embedding for at most one tick, too high rejects one fresh
            # embedding. Locking here would just trade a one-tick delay for a
            # rare lock contention, so we accept the race for the MVP.
            floor = self._frame_counter
            self._cache.invalidate(floor=floor)

    # ------------------------------------------------------------ tick: send

    def _send_observation_tick(self) -> None:
        if self._client is None:
            return
        with self._latest_image_lock:
            img = None if self._latest_image is None else self._latest_image.copy()
        with self._latest_goal_lock:
            goal = self._latest_goal
        if img is None or goal is None:
            return
        size = self.get_parameter('image_size').value
        quality = int(self.get_parameter('jpeg_quality').value)
        try:
            jpeg, w, h = resize_and_jpeg(img, target=(int(size[0]), int(size[1])), quality=quality)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'preprocess failed: {exc}')
            return
        try:
            proto_goal = _ros_goal_to_proto(goal)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'goal conversion failed: {exc}')
            return

        self._frame_counter += 1
        obs = raspicat_vla_pb2.Observation(
            frame_id=self._frame_counter,
            capture_time_ns=time.monotonic_ns(),
            image_jpeg=jpeg,
            image_width=w,
            image_height=h,
            goal=proto_goal,
        )
        self._client.send(obs)

    # -------------------------------------------------- callback: embeddings

    def _on_embedding_received(self, proto_emb: raspicat_vla_pb2.ActionEmbedding) -> None:
        if self._cache is None:
            return
        arr = np.array(fp16_bytes_to_float32_list(proto_emb.embedding_fp16), dtype=np.float32)
        cached = CachedEmbedding(
            frame_id=proto_emb.frame_id,
            recv_time_ns=time.monotonic_ns(),
            embedding=arr,
            num_tokens=proto_emb.num_tokens,
            embed_dim=proto_emb.embed_dim,
            inference_ms=float(proto_emb.inference_ms),
            model_version=proto_emb.model_version,
        )
        self._cache.put(cached)
        if self._embedding_pub is not None:
            ros_msg = proto_action_embedding_to_msg(proto_emb)
            ros_msg.header.stamp = self.get_clock().now().to_msg()
            self._embedding_pub.publish(ros_msg)

    # ---------------------------------------------------------- tick: action

    def _action_tick(self) -> None:
        """Publish a Path. Plan 1 stub: straight-ahead path of 1.0 m.

        Status-aware: WAITING_REMOTE / STALE → publish empty path so the
        follower outputs zero Twist (safe-stop). DEGRADED is treated as
        usable but logged.
        """
        if self._cache is None or self._path_pub is None or self._adapter is None:
            return
        status = self._cache.status()
        path = Path()
        path.header.frame_id = 'base_link'
        path.header.stamp = self.get_clock().now().to_msg()

        if status in (EmbeddingCache.STATUS_WAITING, EmbeddingCache.STATUS_STALE):
            # Empty path → follower emits zero Twist (safe-stop).
            self._path_pub.publish(path)
            return

        if status == EmbeddingCache.STATUS_DEGRADED:
            self.get_logger().warn('embedding age over max_age; running degraded')

        emb = self._cache.get_latest_raw()  # OK or DEGRADED
        with self._latest_image_lock:
            cur = None if self._latest_image is None else self._latest_image.copy()
        try:
            path = self._adapter.predict_path(
                embedding=np.asarray(emb.embedding, dtype=np.float32),
                embedding_shape=(1, int(emb.num_tokens), int(emb.embed_dim)),
                cur_image_rgb=cur,
                past_image_rgb=cur,
                frame_id='base_link',
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'adapter.predict_path failed: {exc}; safe-stopping')
            path = Path()
            path.header.frame_id = 'base_link'
        path.header.stamp = self.get_clock().now().to_msg()
        self._path_pub.publish(path)

    # ----------------------------------------------------------- tick: status

    def _publish_status(self) -> None:
        if self._cache is None or self._status_pub is None:
            return
        status_str = self._cache.status()
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        ds = DiagnosticStatus()
        ds.name = 'vla_edge'
        ds.message = status_str
        ds.level = (
            DiagnosticStatus.OK
            if status_str == 'OK'
            else DiagnosticStatus.WARN
            if status_str in ('DEGRADED', 'WAITING_REMOTE')
            else DiagnosticStatus.ERROR
        )
        ds.values.append(KeyValue(key='frame_counter', value=str(self._frame_counter)))
        msg.status.append(ds)
        self._status_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = VLAEdgeNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        # If we exited spin() while the node was still active (no cleanup
        # transition issued), the gRPC client thread is still alive. Stop it
        # explicitly so the bidi stream is closed gracefully — otherwise the
        # daemon thread is killed at interpreter exit and the server logs an
        # aborted RPC. Mirrors what on_shutdown does and reaches into _client
        # the same way (no public accessor exists).
        if node._client is not None:
            node._client.stop()
        node.destroy_node()
        rclpy.shutdown()
