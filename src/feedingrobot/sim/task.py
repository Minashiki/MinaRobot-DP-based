"""P1 physics owner: fixed joint holding, randomized initial state and replay."""

import copy
import hashlib
import json
import time

import mujoco
import numpy as np

from feedingrobot.random import numpy_generator
from feedingrobot.sim.beans import bean_state, bean_diagnostics, spawn_clearance, spawn_envelope
from feedingrobot.sim.contacts import ContactMonitor, read_contacts
from feedingrobot.sim.model import ROOT, asset_files, load_json, load_model
from feedingrobot.sim.sensors import wrist_state


class InvalidInitialState(ValueError):
    """Only predefined geometry/Bean settling failures permit another proposal."""

    def __init__(self, reason, diagnostics):
        super().__init__(reason)
        self.reason, self.diagnostics = reason, diagnostics


class FeedingTask:
    def __init__(self):
        self.model, self.index, self.robot_config, self.scene_config = load_model()
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        self.bean_acceptance = load_json("configs/acceptance.json")
        self.protocol = load_json("configs/protocol.json")
        self.default_bowl_pos = self.model.body_pos[self.index.bowl_body].copy()
        self.default_bowl_quat = self.model.body_quat[self.index.bowl_body].copy()
        digest = hashlib.sha256()
        for path in asset_files("panda"):
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes())
        compiled = np.empty(mujoco.mj_sizeModel(self.model), dtype=np.uint8)
        mujoco.mj_saveModel(self.model, None, compiled)
        digest.update(compiled.tobytes())
        self.signature = dict(schema_version=1, model_config_hash=digest.hexdigest(),
                              mujoco_version=mujoco.__version__)
        self.monitor = ContactMonitor(self.scene_config["contact_force_limit_n"])
        self.reset(seed=self.protocol["seed"], split="debug", episode_id=0,
                   layout={"bowl_offset_m": [0., 0.], "bean_offset_m": [0., 0.]})

    def _initialize_candidate(self, bowl_offset):
        m, d, idx = self.model, self.data, self.index
        mujoco.mj_resetData(m, d)
        base_rotation = np.zeros(9)
        mujoco.mju_quat2Mat(base_rotation, np.array(self.robot_config["base_quaternion"], dtype=float))
        m.body_pos[idx.bowl_body] = self.default_bowl_pos + base_rotation.reshape(3, 3) @ np.r_[bowl_offset, 0.]
        m.body_quat[idx.bowl_body] = self.default_bowl_quat
        self.tick = 0
        self.terminated, self.failure_reason = False, None
        self.external_wrench = None
        self.servo_reference_q = np.array(self.robot_config["reset_q"], dtype=float)
        self.servo_status, self.servo_fault = "holding", None
        self.scenario_state = dict(seed=self.seed, head_phase=0., parameters={"head_offset_m": [0., 0., 0.]},
                                   closure_start=None, future_events=[])
        self.physics_timing = dict(mj_step_s=0., forward_s=0., steps=0)
        self.reset_diagnostics = {}
        self.monitor.reset()
        self.contacts, self.applied_contacts = [], []
        self.substep_contact_peak_n = self.substep_wrist_peak_n = 0.
        d.qpos[idx.qpos] = self.servo_reference_q
        d.ctrl[idx.actuators] = self.servo_reference_q
        d.qpos[idx.bean_qpos[0]] = [2., 2., -.2455, 1., 0., 0., 0.]
        mujoco.mj_forward(m, d)
        started = time.monotonic()
        for _ in range(round(self.scene_config["settle_seconds"] / self.dt)):
            self.step_physics()
            if self.terminated:
                raise RuntimeError(f"Robot reset failed: {self.failure_reason}")
        self.reset_diagnostics["robot_settle_wall_s"] = time.monotonic() - started

    def reset(self, *, seed=42, split="debug", episode_id=0, layout=None):
        """Explicit offsets validate one proposal; otherwise use the episode's own stream."""
        self.seed, self.split, self.episode_id = int(seed), str(split), int(episode_id)
        self.proposal_rng = numpy_generator("proposal", self.split, self.episode_id, seed=self.seed)
        self.rejections = []
        half_width = self.protocol["proposal_half_width_m"]
        for candidate in range(self.protocol["max_candidates"]):
            self.candidate_index = candidate
            proposal = copy.deepcopy(layout) if layout is not None else dict(
                bowl_offset_m=self.proposal_rng.uniform(-half_width, half_width, 2).tolist(),
                bean_offset_m=self.proposal_rng.uniform(-half_width, half_width, 2).tolist())
            for key in ("bowl_offset_m", "bean_offset_m"):
                offset = np.asarray(proposal[key], dtype=float)
                if offset.shape != (2,) or not np.isfinite(offset).all() or np.any(np.abs(offset) > half_width):
                    raise ValueError(f"Invalid proposal offset: {key}")
            self._initialize_candidate(proposal["bowl_offset_m"])
            try:
                self._place_and_settle(proposal)
            except InvalidInitialState as error:
                self.rejections.append(dict(candidate_index=candidate, proposal=proposal,
                                            reason=error.reason, diagnostics=error.diagnostics,
                                            proposed_bean_local_m=self.reset_diagnostics["proposed_bean_local_m"].copy(),
                                            proposed_bean_world_m=self.reset_diagnostics["proposed_bean_world_m"].copy()))
                if layout is not None:
                    raise
                continue
            self.data.time, self.tick = 0., 0
            self.data.xfrc_applied[:] = 0
            self.data.qfrc_applied[:] = 0
            self.monitor.reset()
            self.substep_contact_peak_n = self.substep_wrist_peak_n = 0.
            self.contacts = read_contacts(self.model, self.data, self.index)
            self.applied_contacts = []
            self.reset_diagnostics.update(status="passed", proposal=proposal, candidate_index=candidate,
                                          rejections=copy.deepcopy(self.rejections), seed=self.seed,
                                          split=self.split, episode_id=self.episode_id)
            return self.snapshot()
        raise RuntimeError(f"No valid initial state after {self.protocol['max_candidates']} candidates")

    def _place_and_settle(self, proposal):
        m, d, idx = self.model, self.data, self.index
        local = np.array(self.scene_config["beans"]["reset_position_bowl_m"], dtype=float)
        local[:2] += proposal["bean_offset_m"]
        rotation = d.site_xmat[idx.bowl_frame].reshape(3, 3)
        position = d.site_xpos[idx.bowl_frame] + rotation @ local
        self.reset_diagnostics.update(proposed_bean_local_m=local.copy(), proposed_bean_world_m=position.copy())
        d.qpos[idx.bean_qpos[0]] = np.r_[position, self.scene_config["beans"]["reset_quaternion_wxyz"]]
        d.qvel[idx.bean_dofs] = 0.
        mujoco.mj_forward(m, d)
        envelope = spawn_envelope(self)
        if envelope["radial_extent_m"] > envelope["bottom_radius_m"] + 1e-12:
            raise InvalidInitialState("bean_envelope_outside_bottom", envelope)
        gap = self.bean_acceptance["spawn_gap_m"]
        for _ in range(100):
            clearance = spawn_clearance(self)
            if clearance >= gap - 1e-9:
                break
            d.qpos[idx.bean_qpos[0, :3]] += (gap - clearance + 1e-8) * rotation[:, 2]
            mujoco.mj_forward(m, d)
            if spawn_envelope(self)["rim_clearance_m"] < gap:
                raise InvalidInitialState("spawn_clearance", spawn_envelope(self))
        else:
            raise InvalidInitialState("spawn_clearance", dict(clearance_m=clearance))
        envelope = spawn_envelope(self)
        if envelope["bottom_clearance_m"] < gap - 1e-9 or envelope["rim_clearance_m"] < gap:
            raise InvalidInitialState("spawn_envelope", envelope)
        self.reset_diagnostics.update(spawn_clearance_m=clearance, spawn_envelope=envelope,
            bowl_frame_world=d.site_xpos[idx.bowl_frame].copy(),
            proposed_bean_local_m=local.copy(), born_bean_world_m=d.xpos[idx.bean_bodies[0]].copy())
        self._settle_beans()
        self.reset_diagnostics["settled_bean_local_m"] = rotation.T @ (
            d.xpos[idx.bean_bodies[0]] - d.site_xpos[idx.bowl_frame])

    def _settle_beans(self):
        cfg = self.bean_acceptance
        stable, max_penetration = 0., 0.
        started = time.monotonic()
        for step in range(round(cfg["max_settle_s"] / self.dt)):
            self.step_physics()
            if self.terminated or any(w.number for w in self.data.warning):
                raise RuntimeError(f"Physics failure during settling: {self.failure_reason}")
            if time.monotonic() - started > cfg["wall_budget_s"]:
                raise RuntimeError("Bean settling exceeded wall-clock budget")
            diag = bean_diagnostics(self)
            max_penetration = max(max_penetration, diag["max_penetration_m"])
            if diag["max_penetration_m"] > cfg["penetration_limit_m"]:
                raise InvalidInitialState("bean_penetration", diag)
            if not diag["in_bowl"].all():
                raise InvalidInitialState("bean_outside_bowl", diag)
            low = (diag["linear_speed_m_s"] < cfg["linear_speed_m_s"]) & (
                diag["angular_speed_rad_s"] < cfg["angular_speed_rad_s"])
            stable = stable + self.dt if low.all() and diag["bowl_supported"].all() else 0.
            if stable + 1e-12 >= cfg["stable_window_s"]:
                self.reset_diagnostics.update(bean_settle_s=(step+1)*self.dt, stable_window_s=stable,
                    max_penetration_m=max_penetration, final=diag,
                    bean_settle_wall_s=time.monotonic()-started)
                return
        raise InvalidInitialState("bean_settling_timeout", diag)

    def _write_drivers(self):
        m, d, idx = self.model, self.data, self.index
        d.ctrl[idx.actuators] = self.servo_reference_q
        d.xfrc_applied[:] = 0.
        d.qfrc_applied[:] = 0.
        if self.external_wrench is not None:
            force, torque, point = self.external_wrench
            d.xfrc_applied[idx.tool_body, :3] = force
            d.xfrc_applied[idx.tool_body, 3:] = torque + np.cross(point-d.xipos[idx.tool_body], force)
        cfg = self.scene_config["head"]
        desired = np.r_[self.scenario_state["parameters"]["head_offset_m"], 0., cfg["jaw_center_rad"]]
        joints = idx.head_joints
        dofs, qpos = m.jnt_dofadr[joints], m.jnt_qposadr[joints]
        command = np.array(cfg["kp"])*(desired-d.qpos[qpos]) - np.array(cfg["kd"])*d.qvel[dofs] + d.qfrc_bias[dofs]
        ids = idx.head_actuators
        d.ctrl[ids] = np.clip(command, m.actuator_ctrlrange[ids, 0], m.actuator_ctrlrange[ids, 1])

    def _terminate(self, reason):
        self.terminated, self.failure_reason = True, reason
        self.servo_status, self.servo_fault = "fault", reason

    def step_physics(self):
        if self.terminated:
            raise RuntimeError(f"Episode terminated: {self.failure_reason}; reset required")
        if not all(np.isfinite(a).all() for a in (self.data.qpos, self.data.qvel, self.data.ctrl)):
            self._terminate("nonfinite_state")
            return self.snapshot()
        self._write_drivers()
        started = time.perf_counter()
        mujoco.mj_step(self.model, self.data)
        self.physics_timing["mj_step_s"] += time.perf_counter()-started
        self.physics_timing["steps"] += 1
        self.tick += 1
        self.applied_contacts = read_contacts(self.model, self.data, self.index)
        overloaded = self.monitor.update(self.applied_contacts, self.dt, float(self.data.time))
        self.substep_contact_peak_n = self.monitor.last_peak_n
        applied_wrist = wrist_state(self.model, self.data, self.index)
        started = time.perf_counter()
        mujoco.mj_forward(self.model, self.data)
        self.physics_timing["forward_s"] += time.perf_counter()-started
        if (not all(np.isfinite(a).all() for a in (self.data.qpos, self.data.qvel, self.data.qacc))
                or any(w.number for w in self.data.warning)):
            self._terminate("numerical_failure")
            return self.snapshot()
        self.contacts = read_contacts(self.model, self.data, self.index)
        overloaded = self.monitor.update(self.contacts, 0., float(self.data.time)) or overloaded
        self.substep_contact_peak_n = max(self.substep_contact_peak_n, self.monitor.last_peak_n)
        sensors = wrist_state(self.model, self.data, self.index)
        self.substep_wrist_peak_n = float(max(np.linalg.norm(sensors["compensated_wrench"][:3]),
                                             np.linalg.norm(applied_wrist["compensated_wrench"][:3])))
        if overloaded or self.substep_wrist_peak_n > self.scene_config["wrist_force_limit_n"]:
            self._terminate("contact_limit")
        elif np.max(np.abs(self.data.qvel[self.index.dofs])) > self.scene_config["joint_speed_fault_rad_s"]:
            self._terminate("joint_speed_limit")
        return self.snapshot()

    def snapshot(self):
        return dict(time=float(self.data.time), tick=self.tick, terminated=self.terminated,
                    failure_reason=self.failure_reason, **bean_state(self.model, self.data, self.index))

    def get_state(self):
        kind = mujoco.mjtState.mjSTATE_INTEGRATION
        physics = np.empty(mujoco.mj_stateSize(self.model, kind))
        mujoco.mj_getState(self.model, self.data, physics, kind)
        fields = ("seed", "split", "episode_id", "tick", "external_wrench", "servo_reference_q",
                  "servo_status", "servo_fault", "scenario_state", "terminated", "failure_reason",
                  "contacts", "applied_contacts", "substep_contact_peak_n", "substep_wrist_peak_n",
                  "candidate_index", "rejections", "reset_diagnostics", "physics_timing")
        return copy.deepcopy(dict(signature=self.signature,
            layout=dict(body_pos=self.model.body_pos[self.index.bowl_body],
                        body_quat=self.model.body_quat[self.index.bowl_body]), physics=physics,
            boundary=dict(qacc=self.data.qacc, sensordata=self.data.sensordata),
            warnings=[(w.lastinfo, w.number) for w in self.data.warning],
            task={name: getattr(self, name) for name in fields}, monitor=vars(self.monitor),
            proposal_rng=self.proposal_rng.bit_generator.state))

    def set_state(self, state):
        if state["signature"] != self.signature:
            raise ValueError("Incompatible snapshot model/configuration/version")
        state = copy.deepcopy(state)
        mujoco.mj_resetData(self.model, self.data)
        self.model.body_pos[self.index.bowl_body] = state["layout"]["body_pos"]
        self.model.body_quat[self.index.bowl_body] = state["layout"]["body_quat"]
        kind = mujoco.mjtState.mjSTATE_INTEGRATION
        mujoco.mj_setState(self.model, self.data, state["physics"], kind)
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_setState(self.model, self.data, state["physics"], kind)
        self.data.qacc[:] = state["boundary"]["qacc"]
        self.data.sensordata[:] = state["boundary"]["sensordata"]
        for warning, (lastinfo, number) in zip(self.data.warning, state["warnings"]):
            warning.lastinfo, warning.number = lastinfo, number
        for name, value in state["task"].items():
            setattr(self, name, value)
        self.monitor.__dict__.update(state["monitor"])
        self.proposal_rng = numpy_generator("proposal", self.split, self.episode_id, seed=self.seed)
        self.proposal_rng.bit_generator.state = state["proposal_rng"]
