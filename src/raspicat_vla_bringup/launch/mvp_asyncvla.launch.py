"""Launch the AsyncVLA real-stack MVP (Plan 2A):
 - AsyncVLA cloud server  (--backend asyncvla, GPU; loads AsyncVLA_release)
 - vla_edge_node          (lifecycle; adapter_kind=asyncvla, runs Edge_adapter)
 - path_follower_node     (Path -> /cmd_vel)

The cloud runs the heavy backbone (~7.5 B params) on GPU and emits a
(8, 1024) projected_actions tensor; the edge runs a small ~5 M-param
Edge_adapter (efficientnet-b0 + transformer decoder) over (cur, past, vla_feature),
applies delta_to_pose, and publishes a nav_msgs/Path.

For split-host deployment, run the cloud in Dockerfile.asyncvla on a GPU
box and point the edge's remote_address at it. Both hosts need
external/MBRA on PYTHONPATH (Edge_adapter's transitive dep).
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
    vla_path = LaunchConfiguration('vla_path')
    resume_step = LaunchConfiguration('resume_step')
    device = LaunchConfiguration('device')
    edge_device = LaunchConfiguration('edge_device')

    edge_config = os.path.join(
        get_package_share_directory('raspicat_vla_edge'),
        'config', 'edge_params.yaml',
    )

    asyncvla_server = ExecuteProcess(
        cmd=[
            'python3', '-m', 'raspicat_vla_remote.server_main',
            '--backend', 'asyncvla',
            '--port', grpc_port,
            '--vla-path', vla_path,
            '--resume-step', resume_step,
            '--device', device,
        ],
        output='screen',
    )

    edge = LifecycleNode(
        package='raspicat_vla_edge',
        executable='vla_edge_node',
        name='vla_edge_node',
        namespace='',
        output='screen',
        parameters=[edge_config, {
            'remote_address': ['localhost:', grpc_port],
            'adapter_kind': 'asyncvla',
            'asyncvla_weights_path': vla_path,
            'asyncvla_resume_step': resume_step,
            'asyncvla_device': edge_device,
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
        package='raspicat_vla_edge',
        executable='path_follower_node',
        name='path_follower_node',
        output='screen',
        parameters=[{
            'lookahead': 0.4, 'max_v': 0.4, 'max_w': 1.0, 'rate_hz': 20.0,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('grpc_port', default_value='50051'),
        DeclareLaunchArgument('vla_path', default_value='/workspace/models/AsyncVLA_release'),
        DeclareLaunchArgument('resume_step', default_value='750000'),
        DeclareLaunchArgument('device', default_value='cuda:0'),
        DeclareLaunchArgument('edge_device', default_value='cpu'),
        asyncvla_server,
        edge,
        RegisterEventHandler(OnProcessStart(target_action=edge, on_start=[configure])),
        RegisterEventHandler(OnStateTransition(
            target_lifecycle_node=edge, goal_state='inactive', entities=[activate],
        )),
        follower,
    ])
