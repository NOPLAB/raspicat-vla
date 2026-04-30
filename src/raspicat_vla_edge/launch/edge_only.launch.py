"""Launch the VLA edge lifecycle node (auto-transitions to active).

Optional launch args (override edge_params.yaml):
  remote_address  - gRPC server address (default: from yaml, typically localhost:50051)
  adapter_kind    - stub|asyncvla|omnivla
  with_follower   - true|false (also bring up path_follower_node)
  asyncvla_weights_path / asyncvla_resume_step / asyncvla_device

Use cases:
  ros2 launch raspicat_vla_bringup edge_only.launch.py            # yaml defaults
  ros2 launch raspicat_vla_bringup edge_only.launch.py \\
      remote_address:=192.168.1.2:50051 adapter_kind:=asyncvla with_follower:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


def generate_launch_description():
    remote_address = LaunchConfiguration('remote_address')
    adapter_kind = LaunchConfiguration('adapter_kind')
    with_follower = LaunchConfiguration('with_follower')
    asyncvla_weights_path = LaunchConfiguration('asyncvla_weights_path')
    asyncvla_resume_step = LaunchConfiguration('asyncvla_resume_step')
    asyncvla_device = LaunchConfiguration('asyncvla_device')

    config = os.path.join(
        get_package_share_directory('raspicat_vla_edge'),
        'config', 'edge_params.yaml',
    )

    # Per-launch parameter overrides; only emit the ones that were set explicitly
    # (default '' means "leave the YAML value alone").
    overrides = {
        'remote_address': remote_address,
        'adapter_kind': adapter_kind,
        'asyncvla_weights_path': asyncvla_weights_path,
        'asyncvla_resume_step': asyncvla_resume_step,
        'asyncvla_device': asyncvla_device,
    }

    edge = LifecycleNode(
        package='raspicat_vla_edge',
        executable='vla_edge_node',
        name='vla_edge_node',
        namespace='',
        output='screen',
        parameters=[config, overrides],
    )
    configure = EmitEvent(event=ChangeState(
        lifecycle_node_matcher=matches_action(edge),
        transition_id=Transition.TRANSITION_CONFIGURE,
    ))
    activate = EmitEvent(event=ChangeState(
        lifecycle_node_matcher=matches_action(edge),
        transition_id=Transition.TRANSITION_ACTIVATE,
    ))
    on_started = RegisterEventHandler(
        OnProcessStart(target_action=edge, on_start=[configure]),
    )
    on_inactive = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=edge,
            goal_state='inactive',
            entities=[activate],
        ),
    )

    follower = Node(
        package='raspicat_vla_edge',
        executable='path_follower_node',
        name='path_follower_node',
        output='screen',
        parameters=[{
            'lookahead': 0.4, 'max_v': 0.4, 'max_w': 1.0, 'rate_hz': 20.0,
        }],
        condition=IfCondition(with_follower),
    )

    return LaunchDescription([
        DeclareLaunchArgument('remote_address', default_value='localhost:50051'),
        DeclareLaunchArgument('adapter_kind', default_value='stub'),
        DeclareLaunchArgument('with_follower', default_value='false'),
        DeclareLaunchArgument('asyncvla_weights_path', default_value='/workspace/AsyncVLA_release'),
        DeclareLaunchArgument('asyncvla_resume_step', default_value='750000'),
        DeclareLaunchArgument('asyncvla_device', default_value='cpu'),
        edge,
        on_started,
        on_inactive,
        follower,
    ])
