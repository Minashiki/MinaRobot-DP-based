"""P1 physical equivalence, boundary proposals, replay and 50-episode coverage."""

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from feedingrobot.sim.model import ROOT, assemble_xml, load_json
from feedingrobot.sim.task import FeedingTask, InvalidInitialState
from feedingrobot.sim.contacts import read_contacts
from feedingrobot.scripts.report import write_report
from feedingrobot.scripts.validate_p0 import check_sources


def compare_values(left, right, path="state"):
    if isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            compare_values(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (first, second) in enumerate(zip(left, right)):
            compare_values(first, second, f"{path}[{index}]")
    elif isinstance(left, np.ndarray):
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12, err_msg=path)
    elif isinstance(left, (float, np.floating)):
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12, err_msg=path)
    else:
        assert left == right, path


def canonical_xml(root):
    records = {}
    def visit(element, path):
        for key, value in element.attrib.items():
            if element.tag == "mesh" and key == "file":
                value = Path(value).name
            try:
                value = tuple(float(part) for part in value.split())
            except ValueError:
                pass
            records[f"{path}@{key}"] = value
        for ordinal, child in enumerate(element):
            identity = child.get("name", child.get("class", str(ordinal)))
            visit(child, f"{path}/{child.tag}[{identity}]")
    visit(root, root.tag)
    return records


def reference_model():
    path = ROOT / "tests/reference/legacy_model.py"
    spec = importlib.util.spec_from_file_location("legacy_physical_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    def reference_json(relative):
        return json.loads((ROOT / "tests/reference" / Path(relative).name).read_text())
    module.load_json = reference_json
    model, index, config, scene = module.load_model("panda")
    # Audit-only extraction: stop the unchanged old assembler immediately before compilation.
    needle = '    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))'
    source = path.read_text()
    assert source.count(needle) == 1
    namespace = dict(module.__dict__)
    exec(compile(source.replace(needle, '    return root, cfg, scene_cfg\n' + needle), str(path), "exec"), namespace)
    namespace["ROOT"], namespace["load_json"] = ROOT, reference_json
    xml, _, _ = namespace["load_model"]("panda")
    return model, index, config, scene, xml


def _check_original_physics(task):
    check_sources()
    baseline, index, config, scene, old_xml = reference_model()
    new_xml, _, _ = assemble_xml(cameras=False)
    old_fields, new_fields = canonical_xml(old_xml), canonical_xml(new_xml)
    diff = {key: dict(before=old_fields.get(key), after=new_fields.get(key))
            for key in old_fields.keys() | new_fields.keys() if old_fields.get(key) != new_fields.get(key)}
    assert not diff, diff
    manifest = load_json("third_party_manifest.json")
    raw_xml = [entry["path"] for entry in manifest["files"]
               if entry["path"].startswith("assets/") and entry["path"].endswith(".xml")]
    for entry in manifest["files"]:
        if entry["path"] in raw_xml:
            assert entry["sha256"] == entry["source_sha256"]
    dimensions = {name: getattr(task.model, name) for name in ("nq", "nv", "ngeom", "nmesh", "nbody", "nu")}
    assert dimensions == dict(nq=19, nv=18, ngeom=233, nmesh=206, nbody=19, nu=12)
    assert all(getattr(baseline, name) == value for name, value in dimensions.items())
    assert baseline.names == task.model.names
    arrays = ["body_parentid", "body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
              "jnt_type", "jnt_qposadr", "jnt_dofadr", "jnt_range", "jnt_axis", "jnt_pos", "jnt_limited",
              "dof_damping", "dof_armature", "dof_frictionloss", "geom_type", "geom_size", "geom_pos", "geom_quat",
              "geom_friction", "geom_solref", "geom_solimp", "geom_margin", "geom_gap", "geom_contype", "geom_conaffinity",
              "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange", "actuator_gear",
              "actuator_trnid", "actuator_trntype", "actuator_dyntype", "actuator_biastype",
              "pair_geom1", "pair_geom2", "pair_friction", "pair_solref", "pair_solimp", "pair_margin", "pair_gap",
              "mesh_vert", "mesh_face", "mesh_pos", "mesh_quat", "sensor_type", "sensor_adr", "sensor_dim"]
    for name in arrays:
        np.testing.assert_array_equal(getattr(baseline, name), getattr(task.model, name), err_msg=name)
    reset = baseline.qpos0.copy()
    reset[index.qpos] = config["reset_q"]
    reset[baseline.jnt_qposadr[index.head_joints[-1]]] = scene["head"]["jaw_center_rad"]
    reset[index.bean_qpos[0, 3:]] = scene["beans"]["reset_quaternion_wxyz"]
    poses = {"qpos0": baseline.qpos0.copy(), "reset": reset}
    for sign in (-1, 1):
        pose = reset.copy()
        pose[index.qpos] = np.clip(pose[index.qpos] + sign*.1,
                                   baseline.jnt_range[index.joints, 0], baseline.jnt_range[index.joints, 1])
        if sign == 1:
            pose[index.bean_qpos[0, 3:]] = [1., 0., 0., 0.]
        poses[f"reset_{sign:+d}"] = pose
    errors = {}
    for name, pose in poses.items():
        matrices = []
        for model in (baseline, task.model):
            data = mujoco.MjData(model)
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            matrix = np.empty((model.nv, model.nv))
            mujoco.mj_fullM(model, data, matrix)
            matrices.append(matrix)
        np.testing.assert_allclose(*matrices, rtol=1e-10, atol=1e-12)
        errors[name] = float(np.max(np.abs(matrices[0]-matrices[1])))
    assert tuple(task.index.bean_ids) == ("bean_000",)
    assert sum(task.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) == 1
    assert not any('plate' in task.model.body(i).name for i in range(task.model.nbody))
    assert not any('plate' in task.model.geom(int(g)).name
                   for g in np.r_[task.model.pair_geom1, task.model.pair_geom2])
    return dict(dimensions=dimensions, xml_fields_compared=len(old_fields), xml_diff=diff,
                raw_asset_xml_byte_identical=raw_xml,
                compiled_arrays_compared=arrays, mass_matrix_max_errors=errors,
                mass_matrix_rtol=1e-10, mass_matrix_atol=1e-12, bean_count=1, plate_assembled=False,
                xml_input_sha256=hashlib.sha256(ET.tostring(new_xml)).hexdigest())


def check_physics(task):
    """Keep the exact P1 audit, then allow only the registered visual additions."""
    if not task.cameras:
        return _check_original_physics(task)
    baseline = FeedingTask(cameras=False)
    report = _check_original_physics(baseline)
    original_xml, _, _ = assemble_xml(cameras=False)
    visual_xml, _, _ = assemble_xml(cameras=True)
    camera_config = load_json("configs/cameras.json")
    stripped = copy.deepcopy(visual_xml)
    added_bodies = {camera["name"] + "_d435i" for camera in camera_config["cameras"]}
    removed = set()
    for parent in stripped.iter():
        for child in list(parent):
            if child.tag == "body" and child.get("name") in added_bodies:
                removed.add(child.get("name"))
                parent.remove(child)
    assert removed == added_bodies
    source = ET.parse(ROOT / camera_config["asset"]).getroot()
    added_assets = {"d435i_" + Path(mesh.get("file")).stem for mesh in source.findall("asset/mesh")}
    added_assets.update("d435i_" + material.get("name") for material in source.findall("asset/material"))
    removed = set()
    for child in list(stripped.find("asset")):
        if child.get("name") in added_assets:
            removed.add(child.get("name"))
            stripped.find("asset").remove(child)
    assert removed == added_assets
    mapping = stripped.find("visual/map")
    assert mapping.attrib == dict(znear=str(camera_config["znear_extent_fraction"]),
                                  zfar=str(camera_config["zfar_extent_fraction"]))
    stripped.find("visual").remove(mapping)
    assert canonical_xml(stripped) == canonical_xml(original_xml), "Unexpected camera assembly edits"

    old, new = baseline.model, task.model
    assert (old.nq, old.nv, old.nu, old.nsensor, old.npair) == (
        new.nq, new.nv, new.nu, new.nsensor, new.npair)
    body_map = np.array([new.body(old.body(i).name).id for i in range(old.nbody)])
    geom_map = []
    for body in range(old.nbody):
        old_ids = np.flatnonzero(old.geom_bodyid == body)
        new_ids = np.flatnonzero(new.geom_bodyid == body_map[body])
        assert len(old_ids) == len(new_ids), old.body(body).name
        geom_map.extend(zip(old_ids, new_ids))
    geom_map = np.array([new_id for _, new_id in sorted(geom_map)])
    for name in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat"):
        np.testing.assert_array_equal(getattr(old, name), getattr(new, name)[body_map], err_msg=name)
    np.testing.assert_array_equal(body_map[old.body_parentid], new.body_parentid[body_map])
    for name in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_friction", "geom_solref",
                 "geom_solimp", "geom_margin", "geom_gap", "geom_contype", "geom_conaffinity"):
        np.testing.assert_array_equal(getattr(old, name), getattr(new, name)[geom_map], err_msg=name)
    for kind in ("joint", "actuator", "sensor"):
        count = {"joint": old.njnt, "actuator": old.nu, "sensor": old.nsensor}[kind]
        assert [getattr(old, kind)(i).name for i in range(count)] == [
            getattr(new, kind)(i).name for i in range(count)]
    for name in ("jnt_type", "jnt_qposadr", "jnt_dofadr", "jnt_range", "jnt_axis", "jnt_pos", "jnt_limited",
                 "dof_damping", "dof_armature", "dof_frictionloss", "actuator_gainprm", "actuator_biasprm",
                 "actuator_ctrlrange", "actuator_forcerange", "actuator_gear", "actuator_trnid",
                 "actuator_trntype", "actuator_dyntype", "actuator_biastype", "sensor_type", "sensor_adr", "sensor_dim",
                 "pair_friction", "pair_solref", "pair_solimp", "pair_margin", "pair_gap"):
        np.testing.assert_array_equal(getattr(old, name), getattr(new, name), err_msg=name)
    for name in ("pair_geom1", "pair_geom2"):
        np.testing.assert_array_equal(geom_map[getattr(old, name)], getattr(new, name), err_msg=name)
    added = [new.body(name).id for name in added_bodies]
    np.testing.assert_array_equal(new.body_mass[added], 0.)
    np.testing.assert_array_equal(new.body_inertia[added], 0.)
    visual_geoms = np.flatnonzero(np.isin(new.geom_bodyid, added))
    assert len(visual_geoms) == 18 and new.nbody == old.nbody + 2
    assert new.ngeom == old.ngeom + 18 and new.nmesh == old.nmesh + 9 and new.ncam == old.ncam + 2
    assert not np.any(new.geom_contype[visual_geoms]) and not np.any(new.geom_conaffinity[visual_geoms])
    assert task.index.tool_mass == baseline.index.tool_mass == .035
    errors = {}
    for label, offset in (("qpos0", None), ("reset", 0.), ("reset_-1", -.1), ("reset_+1", .1)):
        qpos = old.qpos0.copy()
        if offset is not None:
            qpos[baseline.index.qpos] = np.clip(np.array(baseline.robot_config["reset_q"]) + offset,
                old.jnt_range[baseline.index.joints, 0], old.jnt_range[baseline.index.joints, 1])
            qpos[old.jnt_qposadr[baseline.index.head_joints[-1]]] = baseline.scene_config["head"]["jaw_center_rad"]
            qpos[baseline.index.bean_qpos[0, 3:]] = ([1., 0., 0., 0.] if offset > 0.
                else baseline.scene_config["beans"]["reset_quaternion_wxyz"])
        matrices = []
        for model in (old, new):
            data = mujoco.MjData(model)
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            matrix = np.empty((model.nv, model.nv))
            mujoco.mj_fullM(model, data, matrix)
            matrices.append(matrix)
        np.testing.assert_allclose(*matrices, rtol=1e-10, atol=1e-12)
        errors[label] = float(np.max(np.abs(matrices[0] - matrices[1])))
    report["visual_additions"] = dict(original_xml_equal_after_explicit_visual_removal=True,
        body_names=sorted(added_bodies), added_geoms=len(visual_geoms), added_meshes=9,
        zero_mass_and_collision=True, tool_mass_kg=task.index.tool_mass,
        mass_matrix_max_errors=errors, xml_input_sha256=hashlib.sha256(ET.tostring(visual_xml)).hexdigest())
    return report


def check_head_driver(task):
    saved = task.get_state()
    tree = ast.parse((ROOT / "tests/reference/legacy_task.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FeedingTask")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_write_drivers")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "legacy_head_driver", "exec"), namespace)
    task.scenario_state["parameters"]["head_offset_m"] = [.001, -.002, .003]
    task.data.qvel[task.model.jnt_dofadr[task.index.head_joints]] = [.01, -.02, .03, -.04, .05]
    mujoco.mj_forward(task.model, task.data)
    commands = []
    for tick, phase in ((0, 0.), (1000, 1.), (2000, -2.)):
        task.tick, task.scenario_state["head_phase"] = tick, phase
        namespace["_write_drivers"](task)
        old = task.data.ctrl[task.index.head_actuators].copy()
        task._write_drivers()
        np.testing.assert_array_equal(old, task.data.ctrl[task.index.head_actuators])
        commands.append(old)
    np.testing.assert_array_equal(commands[0], commands[1])
    np.testing.assert_array_equal(commands[1], commands[2])
    task.set_state(saved)
    return dict(legacy_pd_equal=True, fixed_driver_time_independent=True, nonzero_offset_and_velocity_tested=True)


def boundary_proposals(width):
    offsets = [[0., 0.], [-width, 0.], [width, 0.], [0., -width], [0., width],
               [-width, -width], [-width, width], [width, -width], [width, width]]
    rows = [dict(id=f"bowl_{i}", layout=dict(bowl_offset_m=offset, bean_offset_m=[0., 0.]), expected=True)
            for i, offset in enumerate(offsets)]
    rows += [dict(id=f"bean_{i}", layout=dict(bowl_offset_m=[0., 0.], bean_offset_m=offset), expected=i in (1, 3, 5))
             for i, offset in enumerate(offsets) if i != 0]
    rows += [dict(id=f"combined_{i}", layout=dict(bowl_offset_m=offset, bean_offset_m=offset), expected=i == 0)
             for i, offset in enumerate(offsets[5:])]
    assert len(rows) == 21
    return rows


def check_boundaries(task):
    results = []
    for row in boundary_proposals(task.protocol["proposal_half_width_m"]):
        try:
            task.reset(split="boundary", episode_id=len(results), layout=row["layout"])
            diagnostics = copy.deepcopy(task.reset_diagnostics)
            accepted, reason = True, None
            assert diagnostics["spawn_clearance_m"] >= .0005-1e-9
            assert diagnostics["max_penetration_m"] <= .0004
            assert diagnostics["stable_window_s"] >= .5-1e-12
            assert diagnostics["bean_settle_s"] <= 5.
            assert diagnostics["final"]["bowl_supported"].all()
            assert (diagnostics["final"]["linear_speed_m_s"] < .001).all()
            assert (diagnostics["final"]["angular_speed_rad_s"] < .1).all()
            # Whole bowl collision envelope remains over the finite tabletop.
            table = task.model.geom("table").id
            for geom in task.index.bowl_geoms:
                extent = np.abs(task.data.geom_xmat[geom].reshape(3, 3)) @ task.model.geom_size[geom]
                lo, hi = task.data.geom_xpos[geom]-extent, task.data.geom_xpos[geom]+extent
                assert np.all(lo[:2] >= task.data.geom_xpos[table, :2]-task.model.geom_size[table, :2])
                assert np.all(hi[:2] <= task.data.geom_xpos[table, :2]+task.model.geom_size[table, :2])
        except InvalidInitialState as error:
            accepted, reason, diagnostics = False, error.reason, error.diagnostics
        assert accepted == row["expected"], (row["id"], reason)
        results.append(dict(**row, accepted=accepted, reason=reason, diagnostics=diagnostics))
    return dict(base_count="9 bowl + 9 Bean - 1 shared centre = 17", combined_extremes=4, results=results)


def check_restore(task):
    width = task.protocol["proposal_half_width_m"]
    layout_a = dict(bowl_offset_m=[-width, -width], bean_offset_m=[0., 0.])
    layout_b = dict(bowl_offset_m=[width, width], bean_offset_m=[0., 0.])
    task.reset(split="restore", episode_id=0, layout=layout_a)
    original = task.get_state()
    task.reset(split="restore", episode_id=1, layout=layout_b)
    task.reset(split="restore", episode_id=0, layout=layout_a)
    rerun = task.get_state()
    np.testing.assert_array_equal(original["layout"]["body_pos"], rerun["layout"]["body_pos"])
    np.testing.assert_array_equal(original["physics"], rerun["physics"])
    task.external_wrench = np.array([[.001, 0., 0.], [0., 0., 0.], task.data.site_xpos[task.index.tcp]])
    task._write_drivers()
    mujoco.mj_forward(task.model, task.data)
    task.contacts = read_contacts(task.model, task.data, task.index)
    state_a = task.get_state()
    positions = task.data.geom_xpos.copy()
    rotations = task.data.geom_xmat.copy()
    sites = task.data.site_xpos.copy()
    contact_geometry = [(int(c.geom1), int(c.geom2), float(c.dist), c.pos.copy()) for c in task.data.contact]
    for _ in range(50):
        task.step_physics()
    expected = task.get_state()
    task.reset(split="restore", episode_id=1, layout=layout_b)
    events = []
    real_set, real_forward = mujoco.mj_setState, mujoco.mj_forward
    def setting(*args):
        events.append("setState")
        np.testing.assert_array_equal(task.model.body_pos[task.index.bowl_body], state_a["layout"]["body_pos"])
        return real_set(*args)
    def forwarding(*args):
        events.append("forward")
        return real_forward(*args)
    with patch.object(mujoco, "mj_setState", side_effect=setting), patch.object(mujoco, "mj_forward", side_effect=forwarding):
        task.set_state(state_a)
    assert events == ["setState", "forward", "setState"]
    compare_values(state_a, task.get_state())
    np.testing.assert_array_equal(task.data.geom_xpos, positions)
    np.testing.assert_array_equal(task.data.geom_xmat, rotations)
    np.testing.assert_array_equal(task.data.site_xpos, sites)
    actual_contacts = [(int(c.geom1), int(c.geom2), float(c.dist), c.pos.copy()) for c in task.data.contact]
    compare_values(contact_geometry, actual_contacts, "derived_contacts")
    for _ in range(50):
        task.step_physics()
    actual = task.get_state()
    timings = (copy.deepcopy(expected["task"]["physics_timing"]), copy.deepcopy(actual["task"]["physics_timing"]))
    for state in (expected, actual):
        timing = state["task"].pop("physics_timing")
        assert all(np.isfinite(value) and value >= 0 for value in timing.values())
    compare_values(expected, actual)
    assert timings[0]["steps"] == timings[1]["steps"]
    clone = np.random.default_rng()
    clone.bit_generator.state = state_a["proposal_rng"]
    np.testing.assert_array_equal(task.proposal_rng.normal(size=8), clone.normal(size=8))
    return dict(nonaccumulating_reset=True, full_forward_call_order=events, immediate_complete_state_equal=True,
                derived_geometry_and_contacts_equal=True, replay_steps=50, replay_equal=True,
                nonzero_applied_force_tested=True, rng_roundtrip=True, timing_diagnostics=timings)


def calibrate(task, directory):
    records = []
    for episode in range(task.protocol["calibration_episodes"]):
        task.reset(seed=task.protocol["seed"], split="calibration", episode_id=episode)
        records.append(copy.deepcopy(task.reset_diagnostics))
        if (episode+1) % 10 == 0:
            print(f"Calibration {episode+1}/{task.protocol['calibration_episodes']}", flush=True)
    proposals = np.array([row["proposed_bean_local_m"][:2] for row in records])
    settled = np.array([row["settled_bean_local_m"][:2] for row in records])
    all_proposals = np.array([proposal["proposed_bean_local_m"][:2]
                             for row in records for proposal in [*row["rejections"], row]])
    resolution = task.protocol["unique_position_resolution_m"]
    unique = len(np.unique(np.rint(settled/resolution).astype(np.int64), axis=0))
    assert unique > 1, "Randomized resets collapsed to one quantized settled position"
    radius = float(task.model.geom_size[task.index.bowl_bottom, 0])
    edges = np.arange(-radius, radius+task.protocol["histogram_bin_m"]/2, task.protocol["histogram_bin_m"])
    histogram, xedges, yedges = np.histogram2d(settled[:, 0], settled[:, 1], bins=(edges, edges))
    reasons = {}
    for row in records:
        for rejection in row["rejections"]:
            reasons[rejection["reason"]] = reasons.get(rejection["reason"], 0)+1
    write_report(directory / "calibration_initial_states.json", records)
    np.savetxt(directory / "settled_xy.csv", settled, delimiter=",", header="bowl_local_x_m,bowl_local_y_m", comments="")
    import csv
    with (directory / "proposal_xy.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["episode_id", "candidate_index", "bowl_offset_x_m", "bowl_offset_y_m",
                         "bean_local_x_m", "bean_local_y_m", "accepted", "reason"])
        for row in records:
            for proposal in [*row["rejections"], row]:
                writer.writerow([row["episode_id"], proposal["candidate_index"],
                    *proposal["proposal"]["bowl_offset_m"], *proposal["proposed_bean_local_m"][:2],
                    proposal is row, proposal.get("reason", "")])
    os.environ.setdefault("MPLCONFIGDIR", str(directory / "_matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].scatter(*(all_proposals*1000).T, color="gray", alpha=.4, label="All proposals")
    axes[0].scatter(*(proposals*1000).T, marker="+", label="Accepted proposals")
    axes[0].scatter(*(settled*1000).T, s=15, label="Settled")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Bean initial-state coverage")
    mesh = axes[1].pcolormesh(xedges*1000, yedges*1000, histogram.T, shading="flat")
    fig.colorbar(mesh, ax=axes[1], label="Episodes / 5 mm cell")
    axes[1].set_title("Settled positions: 50 episodes")
    for axis in axes:
        axis.set(xlabel="Bowl-local X (mm)", ylabel="Bowl-local Y (mm)", aspect="equal")
    fig.savefig(directory / "calibration_xy.png", dpi=160)
    plt.close(fig)
    return dict(split="calibration", episodes=len(records), unique_position_resolution_m=resolution,
                unique_settled_positions=unique, all_proposed_min_m=all_proposals.min(axis=0),
                all_proposed_max_m=all_proposals.max(axis=0),
                accepted_proposed_min_m=proposals.min(axis=0), accepted_proposed_max_m=proposals.max(axis=0),
                settled_min_m=settled.min(axis=0), settled_max_m=settled.max(axis=0), settled_range_m=np.ptp(settled, axis=0),
                histogram_counts=histogram.astype(int), histogram_edges_x_m=xedges, histogram_edges_y_m=yedges,
                histogram_outside_count=int(len(records)-histogram.sum()), rejection_reasons=reasons,
                total_candidates=len(records)+sum(len(row["rejections"]) for row in records),
                max_penetration_m=max(row["max_penetration_m"] for row in records),
                geometry_rule=task.protocol["geometry_rule"], formal_split_lists_frozen=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/p1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    task = FeedingTask()
    report = dict(stage="P1", passed=False, signature=task.signature, protocol=task.protocol,
                  dependency_lock_sha256=hashlib.sha256((ROOT / "requirements.lock.txt").read_bytes()).hexdigest(),
                  source_manifest_sha256=hashlib.sha256((ROOT / "third_party_manifest.json").read_bytes()).hexdigest(),
                  physics=check_physics(task), head_driver=check_head_driver(task))
    print("Physical XML, compiled model and mass matrix passed", flush=True)
    report["boundaries"] = check_boundaries(task)
    print("21 boundary proposals passed", flush=True)
    report["restore"] = check_restore(task)
    print("Cross-layout complete restore and 50-step replay passed", flush=True)
    report["calibration"] = calibrate(task, args.output)
    report["passed"] = True
    write_report(args.output / "report.json", report)
    print(f"P1 passed: {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
