"""Validate delta coordinates, fixed targets, replay and fail-closed safety."""

import argparse
import copy
import hashlib
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mink
import mujoco
import numpy as np

from feedingrobot.control.adapter import DeltaAdapter
from feedingrobot.control.safety import SafetyMonitor
from feedingrobot.sim.model import ROOT, RobotIndex, assemble_xml, load_json
from feedingrobot.sim.task import FeedingTask
from feedingrobot.scripts.report import write_report
from feedingrobot.scripts.validate_p1 import compare_values


def check_axes(task):
    initial = task.get_state()
    initial_p = task.data.site_xpos[task.index.tcp].copy()
    initial_r = mink.SO3.from_matrix(task.data.site_xmat[task.index.tcp].reshape(3, 3))
    task.step_delta(np.zeros(6))
    zero_p = task.data.site_xpos[task.index.tcp].copy()
    zero_r = mink.SO3.from_matrix(task.data.site_xmat[task.index.tcp].reshape(3, 3))
    rows = []
    original_solve = mink.solve_ik
    for axis in range(6):
        for sign in (-1, 1):
            task.set_state(initial)
            delta = np.zeros(6)
            delta[axis] = sign * (.001 if axis < 3 else .01)
            target_history, velocity_history = [], []
            def solve(*args, **kwargs):
                target_history.append(task.adapter.target.wxyz_xyz.copy())
                velocity = original_solve(*args, **kwargs)
                velocity_history.append(velocity.copy())
                return velocity
            with patch("feedingrobot.control.adapter.mink.solve_ik", side_effect=solve):
                result = task.step_delta(delta)
            assert not result["terminated"], result
            assert task.tick - initial["task"]["tick"] == 50
            np.testing.assert_allclose(task.data.time - initial["physics"][0], .05, rtol=0., atol=1e-12)
            assert task.adapter.executed_substeps == 50 and task.adapter.remaining_substeps == 0
            assert len(target_history) == len(velocity_history) == 50
            for target in target_history:
                np.testing.assert_array_equal(target, task.adapter.target.wxyz_xyz)
            np.testing.assert_allclose(task.adapter.target.translation(), initial_p + delta[:3],
                                       rtol=0., atol=1e-12)
            np.testing.assert_allclose(task.adapter.target.rotation().as_matrix(),
                                       (mink.SO3.exp(delta[3:]) @ initial_r).as_matrix(),
                                       rtol=0., atol=1e-12)
            np.testing.assert_array_equal(task.adapter.last_action["action"], delta)
            np.testing.assert_array_equal(task.adapter.last_result["actual_position_world_m"],
                                          task.data.site_xpos[task.index.tcp])
            assert task.adapter.last_result["time"] == task.data.time
            velocity = np.array(velocity_history)
            assert np.max(abs(velocity[:, task.index.dofs])) <= .8 + 1e-7
            assert np.max(abs(velocity[:, task.adapter.frozen_dofs])) <= 1e-7
            actual_r = mink.SO3.from_matrix(task.data.site_xmat[task.index.tcp].reshape(3, 3))
            response = (task.data.site_xpos[task.index.tcp] - zero_p if axis < 3
                        else (actual_r @ zero_r.inverse()).log())
            assert sign * response[axis % 3] > 0., (axis, sign, response)
            rows.append(dict(axis=axis, sign=sign, requested_delta=delta,
                             target_world_wxyz_xyz=task.adapter.target.wxyz_xyz.copy(),
                             zero_action_relative_response=response,
                             result=copy.deepcopy(task.adapter.last_result),
                             peak_joint_ik_speed_rad_s=float(np.max(abs(velocity[:, task.index.dofs])))))
    task.set_state(initial)
    return dict(status="passed", axes=rows, zero_action_drift_m=zero_p-initial_p)


def check_rotated_base():
    xml, robot, _ = assemble_xml(cameras=False)
    base_rotation = mink.SO3.exp(np.array([.3, -.2, .4]))
    base = xml.find("worldbody/body[@name='robot_base']")
    base.set("quat", " ".join(map(str, base_rotation.wxyz)))
    base.set("pos", "0.1 -0.05 0.02")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    index = RobotIndex(model, robot)
    data = mujoco.MjData(model)
    data.qpos[index.qpos] = robot["reset_q"]
    data.qpos[index.qpos[5]] -= .6  # Ensure TCP orientation does not commute with the delta.
    mujoco.mj_forward(model, data)
    adapter = DeltaAdapter(model, data, index, robot, load_json("configs/control.json"))
    delta = np.array([.001, -.002, .003, .01, -.02, .03])
    actual_p = data.site_xpos[index.tcp].copy()
    actual_r = data.site_xmat[index.tcp].reshape(3, 3).copy()
    rb = data.site_xmat[index.base].reshape(3, 3)
    adapter.set_delta(delta)
    expected_rotation = rb @ mink.SO3.exp(delta[3:]).as_matrix() @ rb.T @ actual_r
    np.testing.assert_allclose(adapter.target.translation(), actual_p + rb @ delta[:3], atol=1e-12)
    np.testing.assert_allclose(adapter.target.rotation().as_matrix(), expected_rotation, atol=1e-12)
    assert np.linalg.norm(expected_rotation - actual_r @ mink.SO3.exp(delta[3:]).as_matrix()) > 1e-3
    return dict(status="passed", delta=delta, base_rotation=rb,
                target_world_wxyz_xyz=adapter.target.wxyz_xyz)


def check_pending_replay(task):
    task.reset(layout=dict(bowl_offset_m=[.02, -.02], bean_offset_m=[0., 0.]))
    task.adapter.set_delta(np.array([.001, -.001, 0., .01, 0., 0.]))
    for _ in range(17):
        task.step_physics()
    middle = task.get_state()
    assert middle["controller"]["remaining_substeps"] == 33
    for _ in range(33):
        task.step_physics()
    expected = task.get_state()
    task.reset(layout=dict(bowl_offset_m=[-.02, .02], bean_offset_m=[0., 0.]))
    task.set_state(middle)
    restored = task.get_state()
    compare_values(middle, restored)
    for _ in range(33):
        task.step_physics()
    actual = task.get_state()
    for state in (expected, actual):
        state["task"]["physics_timing"] = dict(steps=state["task"]["physics_timing"]["steps"])
    compare_values(expected, actual)
    assert not task.terminated and task.adapter.executed_substeps == 50
    return dict(status="passed", saved_after_substeps=17, resumed_substeps=33,
                result=copy.deepcopy(task.adapter.last_result))


def check_safety_timing():
    config = load_json("configs/control.json")
    monitor = SafetyMonitor(config, .0004)
    deep = [dict(distance=-.0031, bean1_id=None, bean2_id=None)]
    reasons = [monitor.update(deep, deep, .001, (i+1)*.001) for i in range(5)]
    assert reasons == [None, None, None, None, "model_penetration"]
    np.testing.assert_allclose(monitor.penetration_duration_s, .005, rtol=0., atol=1e-12)
    saved = monitor.get_state()
    monitor.reset()
    monitor.set_state(saved)
    compare_values(saved, monitor.get_state())
    monitor.reset()
    assert monitor.update([], [dict(distance=-.00041, bean1_id="bean_000")], .001, .001) == "model_penetration"
    monitor.reset()
    assert monitor.update(deep, [], .001, .001) is None
    assert monitor.update([], [], .001, .002) is None
    assert monitor.penetration_duration_s == 0.
    return dict(status="passed", generic_penetration_confirm_s=.005,
                bean_penetration_immediate_limit_m=.0004)


def check_fail_closed(task):
    initial = task.get_state()
    action = np.array([.001, 0., 0., 0., 0., 0.])
    task.external_wrench = np.array([[20., 0., 0.], [0., 0., 0.],
                                    task.data.site_xpos[task.index.tcp]])
    force = task.step_delta(action)
    assert force["terminated"] and force["failure_reason"] == "contact_limit"
    assert 0 < task.adapter.executed_substeps < 50 and task.adapter.remaining_substeps == 0
    np.testing.assert_array_equal(task.adapter.last_action["action"], action)
    np.testing.assert_array_equal(task.adapter.last_result["target_world_wxyz_xyz"],
                                  task.adapter.last_action["target_world_wxyz_xyz"])
    force_evidence = dict(external_force_n=20., terminated_after_substeps=task.adapter.executed_substeps,
                         compensated_wrist_peak_n=task.substep_wrist_peak_n,
                         request=copy.deepcopy(task.adapter.last_action),
                         result=copy.deepcopy(task.adapter.last_result))
    task.set_state(initial)
    with patch("feedingrobot.control.adapter.mink.solve_ik",
               side_effect=mink.exceptions.NoSolutionFound("daqp")):
        failure = task.step_delta(action)
    assert failure["terminated"] and failure["failure_reason"] == "ik_failure"
    assert task.tick == initial["task"]["tick"] and task.adapter.remaining_substeps == 0
    ik_evidence = dict(terminated_before_physics=True, request=copy.deepcopy(task.adapter.last_action),
                       result=copy.deepcopy(task.adapter.last_result))
    task.set_state(initial)
    return dict(status="passed", force=force_evidence, ik_failure=ik_evidence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "outputs/p3/report.json"))
    args = parser.parse_args()
    task = FeedingTask(cameras=False)
    paths = ["configs/control.json", "src/feedingrobot/control/adapter.py", "src/feedingrobot/control/safety.py",
             "src/feedingrobot/sim/task.py", "src/feedingrobot/scripts/validate_p3.py", "tests/test_p3.py"]
    report = dict(stage="P3", status="passed", signature=task.signature, control_config=task.control_config,
                  input_sha256={path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
                  source_manifest_sha256=hashlib.sha256((ROOT / "third_party_manifest.json").read_bytes()).hexdigest(),
                  dependency_lock_sha256=hashlib.sha256((ROOT / "requirements.lock.txt").read_bytes()).hexdigest(),
                  axes=check_axes(task), rotated_base=check_rotated_base(),
                  pending_replay=check_pending_replay(task), safety=check_safety_timing(),
                  fail_closed=check_fail_closed(task),
                  semantics="Target remains unchanged; physical tracking residuals are reported, not relabelled.")
    write_report(args.output, report)
    print(f"P3 passed: {args.output}")


if __name__ == "__main__":
    main()
