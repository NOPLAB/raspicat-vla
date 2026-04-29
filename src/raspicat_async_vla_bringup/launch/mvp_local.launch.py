"""Launch the full Plan-1 MVP locally:
 - asyncvla_dummy_server  (gRPC, deterministic embeddings)
 - asyncvla_edge_node     (lifecycle, configured + activated)
 - path_follower_node     (Path -> /cmd_vel)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, EmitEvent, ExecuteProcess, RegisterEventHandler,
)
from launch.event_handlers import OnProcessStart
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


def generate_launch_description():
    grpc_port = LaunchConfiguration('grpc_port')
    inference_ms = LaunchConfiguration('inference_ms')

    edge_config = os.path.join(
        get_package_share_directory('raspicat_async_vla_edge'),
        'config', 'edge_params.yaml',
    )

    dummy_server = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'raspicat_async_vla_remote', 'asyncvla_dummy_server',
            '--port', grpc_port,
            '--inference-ms', inference_ms,
            '--num-tokens', '8',
            '--embed-dim', '1024',
        ],
        output='screen',
    )

    edge = LifecycleNode(
        package='raspicat_async_vla_edge',
        executable='asyncvla_edge_node',
        name='asyncvla_edge_node',
        namespace='',
        output='screen',
        parameters=[edge_config, {
            'remote_address': ['localhost:', grpc_port],
        }],
    )
    configure = EmitEvent(event=ChangeState(
        lifecycle_node_matcher=matches_action(edge),
        transition_id=Transition.TRANSITION_CONFIGURE,
    ))
    activate = EmitEvent(event=ChangeState(
        lifecycle_node_matcher=matches_action(edge),
        transition_id=Transition.TRANSITION_ACTIVATE,
    ))

    follower = Node(
        package='raspicat_async_vla_edge',
        executable='path_follower_node',
        name='path_follower_node',
        output='screen',
        parameters=[{
            'lookahead': 0.4, 'max_v': 0.4, 'max_w': 1.0, 'rate_hz': 20.0,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('grpc_port', default_value='50051'),
        DeclareLaunchArgument('inference_ms', default_value='50.0'),
        dummy_server,
        edge,
        RegisterEventHandler(OnProcessStart(target_action=edge, on_start=[configure])),
        RegisterEventHandler(OnStateTransition(
            target_lifecycle_node=edge, goal_state='inactive', entities=[activate],
        )),
        follower,
    ])
