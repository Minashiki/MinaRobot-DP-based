"""Fixed base-frame delta targets using the legacy Mink IK core."""

import copy
import itertools

import mink
import numpy as np


class DeltaAdapter:
    def __init__(self, model, data, index, robot_config, control_config):
        self.model, self.data, self.index = model, data, index
        self.robot_config, self.control_config = robot_config, control_config
        self.substeps = int(control_config["physics_substeps"])
        if not np.isclose(self.substeps * model.opt.timestep, control_config["control_dt_s"],
                          rtol=0., atol=1e-12):
            raise ValueError("Delta duration must match the configured physical substeps")
        self.frame = mink.FrameTask(robot_config["tcp_site"], "site", position_cost=1.,
                                    orientation_cost=1., lm_damping=1e-4)
        self.frozen_dofs = np.setdiff1d(np.arange(model.nv), index.dofs)
        self.freeze = mink.DofFreezingTask(model, self.frozen_dofs.tolist())
        by_body = [[g for g in index.arm_geoms if model.geom_bodyid[g] == b]
                   for b in sorted(index.arm_bodies)]
        pairs = [(a, b) for a, b in itertools.combinations(by_body, 2) if a and b]
        obstacles = [g for g in range(model.ngeom)
                     if index.group(model, g) in {"table", "bowl", "mouth", "floor"}]
        pairs.append((index.arm_geoms, obstacles))
        self.limits = [mink.ConfigurationLimit(model),
                       mink.VelocityLimit(model, {n: robot_config["joint_velocity_limit"]
                                                 for n in robot_config["joints"]}),
                       mink.CollisionAvoidanceLimit(model, pairs,
                           minimum_distance_from_collisions=control_config["collision_minimum_distance_m"],
                           collision_detection_distance=control_config["collision_detection_distance_m"])]
        self.reference = mink.Configuration(model)
        self.reset()

    def reset(self, reference_q=None):
        q = self.data.qpos.copy()
        if reference_q is not None:
            q[self.index.qpos] = reference_q
        self.reference.update(q)
        self.target = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(self.data.site_xmat[self.index.tcp].reshape(3, 3)),
            self.data.site_xpos[self.index.tcp].copy())
        self.frame.set_target(self.target)
        self.status, self.fault, self.error_detail = "holding", None, None
        self.remaining_substeps = self.executed_substeps = self.action_count = self.ik_failures = 0
        self.last_ik_velocity = np.zeros(self.model.nv)
        self.last_action = None
        self.last_result = None

    def set_delta(self, action):
        if self.fault:
            raise RuntimeError(f"Reset required after {self.fault}")
        if self.remaining_substeps:
            raise RuntimeError("The previous delta action still has pending physical substeps")
        value = np.asarray(action, dtype=float)
        self.last_action = dict(action=value.copy(), time=float(self.data.time),
                                action_id=self.action_count)
        self.action_count += 1
        if value.shape != (6,) or not np.isfinite(value).all():
            self.stop("invalid_command")
            raise ValueError("Expected a finite six-dimensional delta in metres and radians")
        idx, d = self.index, self.data
        base_r = d.site_xmat[idx.base].reshape(3, 3)
        actual_p = d.site_xpos[idx.tcp].copy()
        actual_r = mink.SO3.from_matrix(d.site_xmat[idx.tcp].reshape(3, 3))
        self.target = mink.SE3.from_rotation_and_translation(
            mink.SO3.exp(base_r @ value[3:]) @ actual_r, actual_p + base_r @ value[:3])
        self.frame.set_target(self.target)
        self.last_action["target_world_wxyz_xyz"] = self.target.wxyz_xyz.copy()
        self.remaining_substeps, self.executed_substeps = self.substeps, 0
        self.status, self.error_detail, self.last_result = "active", None, None

    def stop(self, reason, fault=True):
        self.remaining_substeps = 0
        self.last_ik_velocity[:] = 0.
        self.status = "fault" if fault else "holding"
        if fault:
            self.fault = reason
        # Emergency holding does not overwrite the original action or TCP target.
        if np.isfinite(self.data.qpos).all():
            self.reference.update(self.data.qpos.copy())
        elif np.isfinite(self.data.qpos[self.index.qpos]).all():
            q = self.reference.q.copy()
            q[self.index.qpos] = self.data.qpos[self.index.qpos]
            self.reference.update(q)

    def update(self, dt):
        idx, cfg = self.index, self.control_config
        if self.status != "active":
            return self.reference.q[idx.qpos].copy()
        if not np.isclose(dt, self.model.opt.timestep, rtol=0., atol=1e-12):
            raise ValueError("IK must run once per configured physical substep")
        q = self.data.qpos.copy()
        q[idx.qpos] = self.reference.q[idx.qpos]
        self.reference.update(q)
        try:
            velocity = mink.solve_ik(self.reference, [self.frame], dt, solver="daqp",
                                     limits=self.limits, constraints=[self.freeze],
                                     damping=1e-5, safety_break=True)
            if not np.isfinite(velocity).all():
                raise ValueError("Non-finite IK result")
            if np.max(np.abs(velocity[self.frozen_dofs])) > cfg["frozen_velocity_tolerance"]:
                raise ValueError("IK moved a frozen degree of freedom")
            if np.max(np.abs(velocity[idx.dofs])) > (
                    self.robot_config["joint_velocity_limit"] + cfg["ik_velocity_tolerance_rad_s"]):
                raise ValueError("IK exceeded the joint velocity feasibility constraint")
            self.last_ik_velocity = velocity.copy()
            self.reference.integrate_inplace(velocity, dt)
            reference_q = self.reference.q[idx.qpos]
            bounds = self.model.actuator_ctrlrange[idx.actuators]
            tolerance = cfg["reference_tolerance_rad"]
            if (not np.isfinite(reference_q).all() or np.any(reference_q < bounds[:, 0] - tolerance)
                    or np.any(reference_q > bounds[:, 1] + tolerance)):
                self.error_detail = "Joint reference outside the position actuator ctrlrange"
                self.stop("reference_out_of_range")
                return self.reference.q[idx.qpos].copy()
            self.remaining_substeps -= 1
            self.executed_substeps += 1
            if self.remaining_substeps == 0:
                self.status = "completed"
        except (mink.exceptions.MinkError, ValueError, np.linalg.LinAlgError) as error:
            self.ik_failures += 1
            self.error_detail = str(error)
            self.stop("ik_failure")
        return self.reference.q[idx.qpos].copy()

    def record_result(self):
        """Called after physics, so residuals belong to the current boundary."""
        position = self.data.site_xpos[self.index.tcp].copy()
        rotation = self.data.site_xmat[self.index.tcp].reshape(3, 3)
        finite = np.isfinite(position).all() and np.isfinite(rotation).all()
        actual_r = mink.SO3.from_matrix(rotation) if finite else None
        self.last_result = dict(time=float(self.data.time),
                    target_world_wxyz_xyz=self.target.wxyz_xyz.copy(),
                    position_error_m=float(np.linalg.norm(
                        self.target.translation() - position)) if finite else None,
                    rotation_error_rad=float(np.linalg.norm(
                        (self.target.rotation() @ actual_r.inverse()).log())) if finite else None,
                    actual_position_world_m=position if finite else None,
                    actual_quaternion_world_wxyz=actual_r.wxyz.copy() if finite else None,
                    executed_substeps=self.executed_substeps,
                    remaining_substeps=self.remaining_substeps, status=self.status,
                    fault=self.fault, error_detail=self.error_detail)
        return copy.deepcopy(self.last_result)

    def get_state(self):
        fields = ("status", "fault", "error_detail", "remaining_substeps", "executed_substeps",
                  "action_count", "ik_failures", "last_ik_velocity", "last_action", "last_result")
        return copy.deepcopy(dict(reference_q=self.reference.q,
            target_world_wxyz_xyz=self.target.wxyz_xyz,
            **{name: getattr(self, name) for name in fields}))

    def set_state(self, state):
        state = copy.deepcopy(state)
        q = np.asarray(state["reference_q"], dtype=float)
        target = np.asarray(state["target_world_wxyz_xyz"], dtype=float)
        bounds = self.model.actuator_ctrlrange[self.index.actuators]
        tolerance = self.control_config["reference_tolerance_rad"]
        if (q.shape != (self.model.nq,) or not np.isfinite(q).all()
                or np.any(q[self.index.qpos] < bounds[:, 0] - tolerance)
                or np.any(q[self.index.qpos] > bounds[:, 1] + tolerance)):
            raise ValueError("Invalid restored joint reference")
        if (target.shape != (7,) or not np.isfinite(target).all()
                or not np.isclose(np.linalg.norm(target[:4]), 1., rtol=0., atol=1e-8)):
            raise ValueError("Invalid restored TCP target")
        remaining, executed = state["remaining_substeps"], state["executed_substeps"]
        if (not isinstance(remaining, int) or not isinstance(executed, int)
                or not 0 <= remaining <= self.substeps or not 0 <= executed <= self.substeps
                or state["status"] not in {"holding", "active", "completed", "fault"}
                or (state["status"] == "active" and (remaining == 0 or remaining + executed != self.substeps))
                or (state["status"] != "active" and remaining != 0)
                or (state["status"] == "completed" and executed != self.substeps)):
            raise ValueError("Invalid restored delta substep progress")
        self.reference.update(state.pop("reference_q"))
        self.target = mink.SE3(state.pop("target_world_wxyz_xyz"))
        self.frame.set_target(self.target)
        for name, value in state.items():
            setattr(self, name, value)
