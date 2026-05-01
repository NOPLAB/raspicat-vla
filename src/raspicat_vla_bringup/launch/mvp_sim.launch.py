"""Launch raspicat_sim (Gazebo) + the VLA edge stack.

Used by ``docker/run.sh run {asyncvla,omnivla} --sim --host HOST[:PORT]`` to
bring up Gazebo with raspicat in an empty world and our edge node + path
follower pointed at a remote cloud server.

Launch args:
  remote_address  gRPC cloud (default localhost:50051)
  adapter_kind    stub | asyncvla | omnivla       (default omnivla)
  world           gazebo .world path              (raspicat_gazebo/empty.world default)
  rviz            true|false                       (default false; sim is mostly headless)
  asyncvla_weights_path / asyncvla_resume_step / asyncvla_device

The raspicat sim publishes its camera at ``/camera/color/image_raw``; we
remap our edge node accordingly.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    remote_address = LaunchConfiguration('remote_address')
    adapter_kind = LaunchConfiguration('adapter_kind')
    world = LaunchConfiguration('world')
    rviz = LaunchConfiguration('rviz')
    asyncvla_weights_path = LaunchConfiguration('asyncvla_weights_path')
    asyncvla_resume_step = LaunchConfiguration('asyncvla_resume_step')
    asyncvla_device = LaunchConfiguration('asyncvla_device')

    raspicat_gazebo_share = get_package_share_directory('raspicat_gazebo')
    sim_launch_path = os.path.join(
        raspicat_gazebo_share, 'launch', 'raspicat_with_emptyworld.launch.py',
    )

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch_path),
        launch_arguments={
            'world': world,
            'rviz': rviz,
        }.items(),
    )

    edge_launch_path = os.path.join(
        get_package_share_directory('raspicat_vla_edge'),
        'launch', 'edge_only.launch.py',
    )
    edge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(edge_launch_path),
        launch_arguments={
            'remote_address': remote_address,
            'adapter_kind': adapter_kind,
            'image_topic': '/camera/color/image_raw',   # raspicat_sim RealSense topic
            'with_follower': 'true',
            'asyncvla_weights_path': asyncvla_weights_path,
            'asyncvla_resume_step': asyncvla_resume_step,
            'asyncvla_device': asyncvla_device,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('remote_address', default_value='localhost:50051'),
        DeclareLaunchArgument('adapter_kind', default_value='omnivla'),
        DeclareLaunchArgument(
            'world',
            default_value=os.path.join(raspicat_gazebo_share, 'worlds', 'empty.world'),
        ),
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('asyncvla_weights_path',
                              default_value='/workspace/models/AsyncVLA_release'),
        DeclareLaunchArgument('asyncvla_resume_step', default_value='750000'),
        DeclareLaunchArgument('asyncvla_device', default_value='cpu'),
        sim,
        edge,
    ])
