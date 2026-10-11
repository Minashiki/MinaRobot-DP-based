"""Delta target, IK feasibility, safe termination and replay contracts."""

import copy

import mink
import numpy as np
import pytest

from feedingrobot.sim.task import FeedingTask
from feedingrobot.scripts.validate_p3 import (check_axes, check_pending_replay,
                                            check_rotated_base, check_safety_timing)


@pytest.fixture
def task():
    return FeedingTask(cameras=False)


def test_six_axes_once_for_50_substeps_and_post_physics_residuals(task):
    check_axes(task)


def test_base_frame_coordinates_and_rotation_left_multiplication():
    check_rotated_base()


def test_partial_action_cross_layout_restore_continues_identically(task):
    check_pending_replay(task)


def test_holding_does_not_run_ik_and_pending_action_is_not_replaced(task, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Holding must not solve IK")
    monkeypatch.setattr(mink, "solve_ik", unexpected)
    reference = task.servo_reference_q.copy()
    task.step_physics()
    np.testing.assert_array_equal(task.servo_reference_q, reference)
    task.adapter.set_delta(np.zeros(6))
    original = task.adapter.get_state()
    with pytest.raises(RuntimeError, match="pending"):
        task.step_delta(np.ones(6))
    np.testing.assert_array_equal(task.adapter.last_action["action"], original["last_action"]["action"])


def test_no_normalized_action_workspace_or_reference_deviation_rejection(task):
    delta = np.array([1.2, -1.3, 1.4, .3, -.4, .5])
    position = task.data.site_xpos[task.index.tcp].copy()
    task.adapter.set_delta(delta)
    np.testing.assert_array_equal(task.adapter.last_action["action"], delta)
    np.testing.assert_allclose(task.adapter.target.translation(), position + delta[:3], atol=1e-12)
    assert task.adapter.fault is None and task.adapter.remaining_substeps == 50


def test_ik_floating_point_tolerance_does_not_clip_velocity(task, monkeypatch):
    task.adapter.set_delta(np.zeros(6))
    before = task.adapter.reference.q[task.index.qpos].copy()
    velocity = np.zeros(task.model.nv)
    velocity[task.index.dofs[0]] = .8 + 1e-10
    monkeypatch.setattr(mink, "solve_ik", lambda *args, **kwargs: velocity.copy())
    reference = task.adapter.update(task.dt)
    assert task.adapter.fault is None
    np.testing.assert_array_equal(task.adapter.last_ik_velocity, velocity)
    assert reference[0] == before[0] + velocity[task.index.dofs[0]] * task.dt


@pytest.mark.parametrize("bound", ["lower", "upper"])
@pytest.mark.parametrize("overrun,accepted", [(5e-8, True), (2e-7, False)])
def test_reference_boundary_tolerance_matches_restore_without_clipping(task, monkeypatch,
                                                                     bound, overrun, accepted):
    bounds = task.model.actuator_ctrlrange[task.index.actuators[0]]
    value = bounds[0] - overrun if bound == "lower" else bounds[1] + overrun
    task.adapter.set_delta(np.zeros(6))
    payload = task.adapter.get_state()
    payload["reference_q"][task.index.qpos[0]] = value
    if accepted:
        task.adapter.set_state(payload)
        assert task.adapter.reference.q[task.index.qpos[0]] == value
    else:
        with pytest.raises(ValueError, match="Invalid restored joint reference"):
            task.adapter.set_state(payload)
    # Drive the same candidate through the runtime check with zero IK motion;
    # changing the candidate here is deliberate fault injection, not a restore.
    task.adapter.reference.update(payload["reference_q"])
    monkeypatch.setattr(mink, "solve_ik", lambda *args, **kwargs: np.zeros(task.model.nv))
    reference = task.adapter.update(task.dt)
    if accepted:
        assert task.adapter.fault is None
        assert reference[0] == value
        assert task.adapter.reference.q[task.index.qpos[0]] == value
        assert task.adapter.remaining_substeps == 49
    else:
        assert task.adapter.fault == "reference_out_of_range"
        assert task.adapter.remaining_substeps == 0


@pytest.mark.parametrize("fault", ["solver", "velocity", "frozen", "nonfinite", "reference"])
def test_ik_fault_stops_before_physics_and_preserves_request(task, monkeypatch, fault):
    delta = np.array([.001, 0., 0., 0., .01, 0.])
    task.adapter.set_delta(delta)
    target = task.adapter.target.wxyz_xyz.copy()
    before_tick, before_time = task.tick, task.data.time
    velocity = np.zeros(task.model.nv)
    if fault == "reference":
        q = task.adapter.reference.q.copy()
        q[task.index.qpos[0]] = task.model.actuator_ctrlrange[task.index.actuators[0], 1] - 1e-5
        task.adapter.reference.update(q)
        velocity[task.index.dofs[0]] = .8
    elif fault == "velocity":
        velocity[task.index.dofs[0]] = .8 + 2e-7
    elif fault == "frozen":
        velocity[task.adapter.frozen_dofs[0]] = 2e-7
    elif fault == "nonfinite":
        velocity[task.index.dofs[0]] = np.nan
    def solve(*args, **kwargs):
        if fault == "solver":
            raise mink.exceptions.NoSolutionFound("daqp")
        return velocity.copy()
    monkeypatch.setattr(mink, "solve_ik", solve)
    task.step_physics()
    assert task.terminated and task.tick == before_tick and task.data.time == before_time
    assert task.adapter.remaining_substeps == 0
    assert task.failure_reason == ("reference_out_of_range" if fault == "reference" else "ik_failure")
    np.testing.assert_array_equal(task.adapter.last_action["action"], delta)
    np.testing.assert_array_equal(task.adapter.target.wxyz_xyz, target)
    np.testing.assert_array_equal(task.servo_reference_q, task.data.qpos[task.index.qpos])
    with pytest.raises(RuntimeError, match="reset required"):
        task.step_delta(delta)


def test_force_protection_interrupts_delta_and_keeps_original_target(task):
    delta = np.array([.001, 0., 0., 0., 0., 0.])
    task.external_wrench = np.array([[20., 0., 0.], [0., 0., 0.],
                                    task.data.site_xpos[task.index.tcp]])
    result = task.step_delta(delta)
    assert result["terminated"] and result["failure_reason"] == "contact_limit"
    assert 0 < task.adapter.executed_substeps < 50 and task.adapter.remaining_substeps == 0
    np.testing.assert_array_equal(task.adapter.last_action["action"], delta)
    np.testing.assert_array_equal(task.adapter.last_result["target_world_wxyz_xyz"],
                                  task.adapter.last_action["target_world_wxyz_xyz"])
    assert task.adapter.last_result["fault"] == "contact_limit"


def test_penetration_timing_counts_substeps_once_and_restores():
    check_safety_timing()


@pytest.mark.parametrize("kind,steps", [("generic", 5), ("bean", 1), ("force", 1)])
def test_contact_safety_terminates_at_triggering_substep(task, monkeypatch, kind, steps):
    row = dict(group1="spoon", group2="bowl", force_n=0., distance=-.0031,
               bean1_id=None, bean2_id=None)
    if kind == "bean":
        row.update(distance=-.00041, bean1_id="bean_000")
    elif kind == "force":
        row.update(distance=0., force_n=5.1)
    monkeypatch.setattr("feedingrobot.sim.task.read_contacts", lambda *args: [row.copy()])
    delta = np.array([.001, 0., 0., 0., 0., 0.])
    before = task.tick
    result = task.step_delta(delta)
    assert result["terminated"] and task.tick - before == steps
    assert result["failure_reason"] == ("contact_limit" if kind == "force" else "model_penetration")
    np.testing.assert_array_equal(task.adapter.last_action["action"], delta)
    assert task.adapter.remaining_substeps == 0


@pytest.mark.parametrize("action", [np.zeros(5), np.zeros(7), [0., 0., 0., np.nan, 0., 0.]])
def test_invalid_delta_never_advances_physics(task, action):
    before = task.tick, task.data.time
    with pytest.raises(ValueError, match="finite six-dimensional"):
        task.step_delta(action)
    assert (task.tick, task.data.time) == before
    assert task.adapter.fault == "invalid_command"


@pytest.mark.parametrize("damage", ["reference_shape", "reference_nonfinite", "reference_range", "progress", "servo"])
def test_invalid_controller_snapshot_is_rejected(task, damage):
    state = copy.deepcopy(task.get_state())
    if damage == "reference_shape":
        state["controller"]["reference_q"] = np.zeros(2)
    elif damage == "reference_nonfinite":
        state["controller"]["reference_q"][task.index.qpos[0]] = np.nan
        state["task"]["servo_reference_q"][0] = np.nan
    elif damage == "reference_range":
        value = task.model.actuator_ctrlrange[task.index.actuators[0], 1] + .01
        state["controller"]["reference_q"][task.index.qpos[0]] = value
        state["task"]["servo_reference_q"][0] = value
    elif damage == "progress":
        state["controller"]["remaining_substeps"] = 51
    else:
        state["task"]["servo_reference_q"][0] += .001
    with pytest.raises(ValueError):
        task.set_state(state)
