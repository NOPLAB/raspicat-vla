# raspicat-async-vla

ROS2 Humble nodes for running [AsyncVLA](https://github.com/NHirose/AsyncVLA) navigation on the Raspberry Pi Cat (rt-net `raspicat`).

See `docs/superpowers/specs/2026-04-29-asyncvla-control-node-design.md` for full design.

## Workspace layout

This repository is itself a colcon workspace.

```
src/raspicat_async_vla_msgs/      # ROS2 messages, services, actions
src/raspicat_async_vla_proto/     # gRPC python stubs + conversion helpers
src/raspicat_async_vla_remote/    # gRPC server (Plan 1: dummy; Plan 2: real model)
src/raspicat_async_vla_edge/      # Edge ROS2 nodes (lifecycle, follower)
src/raspicat_async_vla_bringup/   # Launch composition
```

## Build

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## Plans / specs

- Spec: `docs/superpowers/specs/2026-04-29-asyncvla-control-node-design.md`
- Plan 1 (this MVP): `docs/superpowers/plans/2026-04-29-asyncvla-mvp-wiring.md`
