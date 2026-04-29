"""Tests for Pure Pursuit controller."""
import math

import pytest

from asyncvla_edge.pure_pursuit import PurePursuit, Pose2D, Waypoint


def test_straight_path_outputs_forward_velocity():
    pp = PurePursuit(lookahead=0.5, max_v=0.4, max_w=1.0)
    path = [Waypoint(x=0.0, y=0.0), Waypoint(x=1.0, y=0.0), Waypoint(x=2.0, y=0.0)]
    cmd = pp.compute(robot=Pose2D(0.0, 0.0, 0.0), path=path)
    assert cmd.linear > 0.0
    assert abs(cmd.angular) < 1e-3
    assert cmd.linear <= 0.4


def test_target_to_left_turns_left():
    pp = PurePursuit(lookahead=0.5, max_v=0.4, max_w=1.0)
    path = [Waypoint(x=0.0, y=0.0), Waypoint(x=0.5, y=0.5)]
    cmd = pp.compute(robot=Pose2D(0.0, 0.0, 0.0), path=path)
    assert cmd.angular > 0.0


def test_target_behind_emits_no_backward_motion():
    pp = PurePursuit(lookahead=0.5, max_v=0.4, max_w=1.0, no_backward=True)
    path = [Waypoint(x=-1.0, y=0.0)]
    cmd = pp.compute(robot=Pose2D(0.0, 0.0, 0.0), path=path)
    assert cmd.linear == pytest.approx(0.0)
    assert abs(cmd.angular) > 0.0  # rotates in place


def test_empty_path_emits_zero_command():
    pp = PurePursuit(lookahead=0.5, max_v=0.4, max_w=1.0)
    cmd = pp.compute(robot=Pose2D(0.0, 0.0, 0.0), path=[])
    assert cmd.linear == 0.0
    assert cmd.angular == 0.0


def test_command_clipped_to_limits():
    pp = PurePursuit(lookahead=0.05, max_v=0.4, max_w=1.0)  # tiny lookahead -> high curvature
    path = [Waypoint(x=0.05, y=0.05)]
    cmd = pp.compute(robot=Pose2D(0.0, 0.0, 0.0), path=path)
    assert cmd.linear <= 0.4
    assert abs(cmd.angular) <= 1.0
