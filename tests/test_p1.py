"""Finite geometry, infrastructure failure and cross-layout replay regressions."""

import numpy as np
import pytest

from feedingrobot.sim.beans import ellipse_radial_extent, spawn_clearance
from feedingrobot.sim.task import FeedingTask, InvalidInitialState
from feedingrobot.scripts.validate_p1 import check_physics, check_head_driver, check_restore


@pytest.fixture
def task():
    return FeedingTask()


def test_rotated_ellipse_extent_against_surface_points():
    # Independent dense boundary samples detect underestimation and degenerate axes.
    angles = np.linspace(-np.pi, np.pi, 100001)
    circle = np.array([np.cos(angles), np.sin(angles)])
    for centre, angle, axes in [([0., 0.], 0., [.007, .004]),
                                ([.025, -.005], .7, [.006, .004]),
                                ([0., .02], 0., [.007, .004]),
                                ([.02, 0.], 0., [.005, .005])]:
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        transform = rotation @ np.diag(axes)
        points = np.array(centre)[:, None] + transform @ circle
        sampled = np.linalg.norm(points, axis=0).max()
        computed = ellipse_radial_extent(np.array(centre), transform @ transform.T)
        assert computed >= sampled-1e-12
        assert computed-sampled < 1e-9


def test_clearance_alone_does_not_accept_outside_bean(task):
    with pytest.raises(InvalidInitialState, match="bean_envelope_outside_bottom"):
        task.reset(layout=dict(bowl_offset_m=[0., 0.], bean_offset_m=[.02, .02]))
    assert spawn_clearance(task) > task.bean_acceptance['spawn_gap_m']
    assert task.rejections[0]['reason'] == 'bean_envelope_outside_bottom'


def test_robot_fault_is_not_rejection_sample(task, monkeypatch):
    def fail_step():
        task._terminate('joint_speed_limit')
    monkeypatch.setattr(task, 'step_physics', fail_step)
    with pytest.raises(RuntimeError, match='Robot reset failed'):
        task.reset(split='calibration', episode_id=10)
    assert task.candidate_index == 0 and task.rejections == []


def test_physics_and_complete_fixed_head_pd_match_reference(task):
    check_physics(task)
    check_head_driver(task)


def test_restore_updates_collision_geometry_and_all_execution_memory(task):
    check_restore(task)


def test_snapshot_does_not_alias_live_arrays(task):
    snapshot = task.get_state()
    physics = snapshot['physics'].copy()
    task.step_physics()
    np.testing.assert_array_equal(snapshot['physics'], physics)
    snapshot['task']['servo_reference_q'][0] += .01
    assert snapshot['task']['servo_reference_q'][0] != task.servo_reference_q[0]
