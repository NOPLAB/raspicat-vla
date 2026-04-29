"""Pure Pursuit path follower."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass
class Pose2D:
    x: float
    y: float
    theta: float  # radians


@dataclass
class Waypoint:
    x: float
    y: float


@dataclass
class TwistCmd:
    linear: float   # m/s
    angular: float  # rad/s


class PurePursuit:
    """Minimal Pure Pursuit controller.

    Picks the first waypoint farther than `lookahead` along the path,
    or the last waypoint if none qualify. If the picked target is behind
    the robot and `no_backward` is true, command zero linear velocity
    and rotate toward it.
    """

    def __init__(
        self,
        *,
        lookahead: float,
        max_v: float,
        max_w: float,
        no_backward: bool = True,
        kw: float = 1.5,
    ) -> None:
        self.lookahead = lookahead
        self.max_v = max_v
        self.max_w = max_w
        self.no_backward = no_backward
        self.kw = kw

    def compute(
        self, *, robot: Pose2D, path: Sequence[Waypoint],
    ) -> TwistCmd:
        if not path:
            return TwistCmd(0.0, 0.0)

        # Pick lookahead target in robot frame
        target = path[-1]
        for wp in path:
            if math.hypot(wp.x - robot.x, wp.y - robot.y) >= self.lookahead:
                target = wp
                break

        dx = target.x - robot.x
        dy = target.y - robot.y
        cos_t = math.cos(robot.theta)
        sin_t = math.sin(robot.theta)
        x_local = cos_t * dx + sin_t * dy
        y_local = -sin_t * dx + cos_t * dy

        if x_local <= 0.0 and self.no_backward:
            # Target is behind: rotate toward it without moving forward.
            heading_err = math.atan2(y_local, x_local)
            angular = max(-self.max_w, min(self.max_w, self.kw * heading_err))
            return TwistCmd(linear=0.0, angular=angular)

        l_sq = x_local * x_local + y_local * y_local
        if l_sq < 1e-9:
            return TwistCmd(0.0, 0.0)
        curvature = 2.0 * y_local / l_sq

        linear = min(self.max_v, max(0.0, x_local))
        angular = linear * curvature

        if abs(angular) > self.max_w:
            angular = math.copysign(self.max_w, angular)

        return TwistCmd(linear=linear, angular=angular)
