"""Render-only legacy state conversion and offline P2 visibility measurements."""

import copy
from collections import Counter

import mujoco
import numpy as np

from feedingrobot.scripts.p2_legacy_replay import FLAGS, STAGES


CROPS = {"center": (12, 12), "top_left": (0, 0), "top_right": (0, 24),
         "bottom_left": (24, 0), "bottom_right": (24, 24)}


def object_names(model, label):
    kinds = {"body": (mujoco.mjtObj.mjOBJ_BODY, model.nbody),
             "joint": (mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
             "actuator": (mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
             "geom": (mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
             "site": (mujoco.mjtObj.mjOBJ_SITE, model.nsite),
             "sensor": (mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor),
             "equality": (mujoco.mjtObj.mjOBJ_EQUALITY, model.neq)}
    kind, count = kinds[label]
    names, ordinals = [], {}
    for index in range(count):
        name = mujoco.mj_id2name(model, kind, index)
        if label in ("geom", "site"):
            body = int(getattr(model, label + "_bodyid")[index])
            ordinal = ordinals.get(body, 0)
            ordinals[body] = ordinal + 1
            if name is None:
                name = f"body:{model.body(body).name}/{label}:{ordinal}"
        if name is None or name in names:
            raise ValueError(f"Unnamed or ambiguous {label} prevents audited mapping")
        names.append(name)
    return names


def remap_joint_vector(value, source, model, *, qpos=False):
    address = "jnt_qposadr" if qpos else "jnt_dofadr"
    size = "nq" if qpos else "nv"
    before = source[address]
    after = getattr(model, address)
    current = object_names(model, "joint")
    if set(current) != set(source["names"]["joint"]):
        raise ValueError("Legacy evidence cannot add or remove physical joints")
    output = np.zeros(getattr(model, size), dtype=np.asarray(value).dtype)
    for old, name in enumerate(source["names"]["joint"]):
        new = current.index(name)
        old_end = int(before[old + 1]) if old + 1 < len(before) else source[size]
        new_end = int(after[new + 1]) if new + 1 < len(after) else getattr(model, size)
        if old_end - before[old] != new_end - after[new]:
            raise ValueError(f"Joint state width differs: {name}")
        output[int(after[new]):new_end] = value[int(before[old]):old_end]
    return output


def remap_rows(value, old_names, new_names, width):
    value = np.asarray(value).reshape(len(old_names), width)
    output = np.zeros((len(new_names), width), dtype=value.dtype)
    for old, name in enumerate(old_names):
        if name not in new_names:
            raise ValueError(f"Legacy physical object is missing: {name}")
        output[new_names.index(name)] = value[old]
    return output.ravel()


def mapped_parts(sample, source, model):
    output = {}
    for name, value in sample["parts"].items():
        if name == "QPOS":
            value = remap_joint_vector(value, source, model, qpos=True)
        elif name in ("QVEL", "WARMSTART", "QFRC_APPLIED"):
            value = remap_joint_vector(value, source, model)
        elif name == "CTRL":
            value = remap_rows(value, source["names"]["actuator"], object_names(model, "actuator"), 1)
        elif name == "XFRC_APPLIED":
            value = remap_rows(value, source["names"]["body"], object_names(model, "body"), 6)
        elif name == "EQ_ACTIVE":
            value = remap_rows(value, source["names"]["equality"], object_names(model, "equality"), 1)
        flag = getattr(mujoco.mjtState, "mjSTATE_" + name)
        if len(value) != mujoco.mj_stateSize(model, flag):
            raise ValueError(f"Unsupported state dimension change: {name}")
        output[name] = np.asarray(value).copy()
    return output


def legacy_render_state(task, sample, source):
    """Explicit migration to a strict current snapshot, with inactive controller.

    Legacy adapter/event logic remains in the cache as provenance. It is never
    installed as a current controller or advanced through current physics.
    """
    state = task.get_state()
    if state["signature"]["schema_version"] != 2:
        raise ValueError("P2 render migration requires the current schema 2")
    parts = mapped_parts(sample, source, task.model)
    data = mujoco.MjData(task.model)
    for name in FLAGS:
        mujoco.mj_setState(task.model, data, parts[name], getattr(mujoco.mjtState, "mjSTATE_" + name))
    mujoco.mj_getState(task.model, data, state["physics"], mujoco.mjtState.mjSTATE_INTEGRATION)
    old = sample["state"]
    state["layout"] = dict(body_pos=sample["bowl_pos"].copy(), body_quat=sample["bowl_quat"].copy())
    state["boundary"]["qacc"] = remap_joint_vector(old["boundary"]["qacc"], source, task.model)
    sensors = object_names(task.model, "sensor")
    for index, name in enumerate(source["names"]["sensor"]):
        current = sensors.index(name)
        dim = int(source["sensor_dim"][index])
        if dim != task.model.sensor_dim[current]:
            raise ValueError(f"Sensor dimensions changed: {name}")
        start, target = int(source["sensor_adr"][index]), int(task.model.sensor_adr[current])
        state["boundary"]["sensordata"][target:target + dim] = old["boundary"]["sensordata"][start:start + dim]
    state["warnings"] = copy.deepcopy(old["warnings"])
    for name in state["task"].keys() & old["task"].keys():
        state["task"][name] = copy.deepcopy(old["task"][name])
    geoms = object_names(task.model, "geom")
    for name in ("contacts", "applied_contacts"):
        for contact in state["task"][name]:
            for key in ("geom1", "geom2"):
                contact[key] = geoms.index(source["names"]["geom"][contact[key]])
    state["monitor"] = copy.deepcopy(old["monitor"])
    controller = state["controller"]
    controller.update(reference_q=parts["QPOS"].copy(), status="holding", fault=None,
                      error_detail=None, remaining_substeps=0, executed_substeps=0,
                      last_action=None, last_result=None)
    tcp = source["names"]["site"].index(task.robot_config["tcp_site"])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, sample["site_xmat"][tcp])
    controller["target_world_wxyz_xyz"] = np.r_[quat, sample["site_xpos"][tcp]]
    state["task"].update(servo_reference_q=parts["QPOS"][task.index.qpos].copy(),
                         servo_status="holding", servo_fault=None, previous_eef_quat=None,
                         resetting=False)
    return state


def check_restored_geometry(task, sample, source):
    errors = {}
    for label, field in (("body", "body_xpos"), ("site", "site_xpos"), ("site", "site_xmat")):
        names = object_names(task.model, label)
        actual = getattr(task.data, "xpos" if field == "body_xpos" else field)
        error = max((float(np.max(np.abs(actual[names.index(name)] - sample[field][i])))
                     for i, name in enumerate(source["names"][label])), default=0.)
        if error > 1e-12:
            raise AssertionError(f"Restored {field} differs from the frozen state: {error}")
        errors[field] = error
    return errors


class BeanSegmentation:
    """Native 240px diagnostics; never exposed to policy observations."""

    def __init__(self, model, scene_option):
        self.model, self.option = model, scene_option
        self.renderer = mujoco.Renderer(model, height=240, width=240)
        self.renderer.enable_segmentation_rendering()
        self.bean = model.geom("bean_000_visual").id

    def measure(self, data, camera):
        self.renderer.update_scene(data, camera=camera, scene_option=self.option)
        actual = self.renderer.render().copy()
        scene = self.renderer.scene
        # Alter only transient draw geometry, never model/data. Every other draw
        # object is outside the clipping planes, leaving the identical Bean and
        # optical camera for an unobstructed silhouette denominator.
        for geom in scene.geoms[:scene.ngeom]:
            if geom.objtype != mujoco.mjtObj.mjOBJ_GEOM or geom.objid != self.bean:
                geom.pos[:] = [1e8, 1e8, 1e8]
        isolated = self.renderer.render().copy()
        mask = (actual[:, :, 0] == self.bean) & (actual[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
        whole = (isolated[:, :, 0] == self.bean) & (isolated[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
        occluders = actual[whole & ~mask]
        counts = {}
        for objid, objtype in occluders:
            if objtype == mujoco.mjtObj.mjOBJ_GEOM:
                name = self.model.geom(int(objid)).name or f"geom:{int(objid)}"
                counts[name] = counts.get(name, 0) + 1
        return mask, whole, counts

    def close(self):
        self.renderer.close()


def classify_frustum_points(points, tangent, near, far, *, crop_offset=(0, 0), crop_size=240):
    """Sufficient containment/exclusion proof for eight visual AABB corners.

    Coordinates use optical +X right, +Y up, -Z forward. An intersecting AABB
    alone cannot prove whether the actual mesh intersects the view.
    """
    points = np.asarray(points)
    x, y, depth = points[:, 0], points[:, 1], -points[:, 2]
    top, left = crop_offset
    lo_x, hi_x = (2 * left / 240 - 1) * tangent, (2 * (left + crop_size) / 240 - 1) * tangent
    lo_y, hi_y = (1 - 2 * (top + crop_size) / 240) * tangent, (1 - 2 * top / 240) * tangent
    planes = dict(near=depth - near, far=far - depth, left=x - lo_x * depth,
                  right=hi_x * depth - x, bottom=y - lo_y * depth, top=hi_y * depth - y)
    outside = [name for name, distances in planes.items() if np.all(distances < 0)]
    status = ("fully_outside_frustum" if outside else "fully_inside_frustum"
              if all(np.all(distances >= 0) for distances in planes.values()) else "intersection_unresolved")
    return dict(position=status, outside_planes=outside,
                proof="Visual mesh AABB corners against all six optical frustum planes; no area threshold uses this box")


def visual_frustum_status(model, data, geom, camera, *, crop_offset=(0, 0), crop_size=240):
    camera_id = model.camera(camera).id
    bounds = model.geom_aabb[geom]
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    local = bounds[:3] + signs * bounds[3:]
    world = local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    optical = (world - data.cam_xpos[camera_id]) @ data.cam_xmat[camera_id].reshape(3, 3)
    tangent = float(np.tan(np.deg2rad(model.cam_fovy[camera_id]) / 2))
    return classify_frustum_points(optical, tangent, model.vis.map.znear * model.stat.extent,
                                  model.vis.map.zfar * model.stat.extent,
                                  crop_offset=crop_offset, crop_size=crop_size)


def visibility(mask, whole, frustum=None):
    visible, unoccluded = int(mask.sum()), int(whole.sum())
    if visible > unoccluded:
        raise AssertionError("Visible pixels exceed the same-camera isolated silhouette")
    zero_reason = None
    if not unoccluded:
        position = frustum["position"] if frustum is not None else None
        zero_reason = {"fully_outside_frustum": "fully_outside_frustum",
                       "fully_inside_frustum": "inside_frustum_subpixel_unresolved",
                       "intersection_unresolved": "intersection_unresolved"}.get(position, "outside_view_or_subpixel_unresolved")
    return dict(visible_pixels=visible, unoccluded_pixels=unoccluded,
                visible_fraction=visible / unoccluded if unoccluded else None,
                no_unoccluded_pixels=unoccluded == 0,
                zero_denominator_reason=zero_reason)


def distributions(rows, field):
    values = np.array([row[field] for row in rows if row[field] is not None], dtype=float)
    if not len(values):
        return dict(count=0, zeros=0, min=None, q01=None, q05=None, q25=None)
    return dict(count=len(values), zeros=int(np.count_nonzero(values == 0)),
                min=float(values.min()), q01=float(np.quantile(values, .01)),
                q05=float(np.quantile(values, .05)), q25=float(np.quantile(values, .25)))


def sampled_intervals(rows, flags):
    """Runs of observed samples; spans do not prove intermediate continuity."""
    intervals, run = [], []
    for row, flag in zip(rows, flags):
        if flag:
            run.append(row)
        elif run:
            intervals.append(run)
            run = []
    if run:
        intervals.append(run)
    return [dict(sample_start_s=run[0]["time_s"], sample_end_s=run[-1]["time_s"],
                 sampled_span_s=run[-1]["time_s"] - run[0]["time_s"], sample_count=len(run),
                 max_sample_gap_s=max(np.diff([row["time_s"] for row in run]), default=0.),
                 start_sample=run[0]["sample_id"], end_sample=run[-1]["sample_id"],
                 intermediate_continuity_verified=False) for run in intervals]


def acceptance_counts(rows, k240, k216, minimum_stage_frames=3, expected_samples=None):
    """Five independent crop schemes, each allowing either complete camera view."""
    failures, counts = [], []
    cameras = ("agentview", "robot0_eye_in_hand")
    sample_ids = set(expected_samples) if expected_samples is not None else {row["sample_id"] for row in rows}
    if not sample_ids:
        return dict(status="failed", passed=False, reason="No evidence samples")
    records = Counter((row["sample_id"], row["crop"], row["camera"]) for row in rows)
    for sample in sorted(sample_ids):
        for crop in CROPS:
            for camera in cameras:
                count = records[(sample, crop, camera)]
                if count != 1:
                    failures.append(dict(sample_id=sample, crop=crop, camera=camera,
                                         record_count=count, expected=1))
    if failures:
        return dict(status="failed", passed=False, reason="Incomplete or duplicated evidence", failures=failures)
    if k240 is None or k216 is None:
        return dict(status="pending", passed=False, reason="K240/K216 require user calibration")
    for crop in CROPS:
        selected = [row for row in rows if row["crop"] == crop]
        samples = {}
        for row in selected:
            sufficient = row["area240"] >= k240 and row["area216"] >= k216
            samples.setdefault(row["sample_id"], []).append(sufficient)
        uncovered = [sample for sample, views in samples.items() if not any(views)]
        if uncovered:
            failures.append(dict(crop=crop, uncovered_samples=uncovered))
        for stage in STAGES:
            for camera in cameras:
                count = len({row["sample_id"] for row in selected if row["source"] == "legacy_replay"
                             and row["stage"] == stage and row["camera"] == camera
                             and row["area240"] >= k240 and row["area216"] >= k216})
                counts.append(dict(crop=crop, stage=stage, camera=camera, sufficient_frames=count))
                if count < minimum_stage_frames:
                    failures.append(dict(crop=crop, stage=stage, camera=camera,
                                         sufficient_frames=count, required=minimum_stage_frames))
    return dict(status="passed" if not failures else "failed", passed=not failures,
                joint_rule="OR(view0_area240>=K240 AND view0_area216>=K216, view1_area240>=K240 AND view1_area216>=K216)",
                crop_schemes_checked_independently=True, stage_counts=counts, failures=failures)
