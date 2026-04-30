"""Launch the OmniVLA real-stack MVP (Plan 2B Path 1):
 - OmniVLA cloud server  (--backend omnivla, GPU; loads omnivla-original)
 - vla_edge_node         (lifecycle; adapter_kind=omnivla)
 - path_follower_node    (Path -> /cmd_vel)

Cloud and edge can run on different hosts; this launch file assumes both
are on localhost. For split-host deployment, run the OmniVLA server in
Dockerfile.omnivla on the GPU box and bring up only the edge + follower
on the raspicat (use ``edge_only.launch.py`` and point ``remote_address``
at the cloud).
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

    edge_config = os.path.join(
        get_package_share_directory('raspicat_vla_edge'),
        'config', 'edge_params.yaml',
    )

    omnivla_server = ExecuteProcess(
        cmd=[
            'python3', '-m', 'raspicat_vla_remote.server_main',
            '--backend', 'omnivla',
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
            'adapter_kind': 'omnivla',
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
        DeclareLaunchArgument('vla_path', default_value='/workspace/omnivla-original'),
        DeclareLaunchArgument('resume_step', default_value='120000'),
        DeclareLaunchArgument('device', default_value='cuda:0'),
        omnivla_server,
        edge,
        RegisterEventHandler(OnProcessStart(target_action=edge, on_start=[configure])),
        RegisterEventHandler(OnStateTransition(
            target_lifecycle_node=edge, goal_state='inactive', entities=[activate],
        )),
        follower,
    ])
