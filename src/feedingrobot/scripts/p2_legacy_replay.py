"""Isolated, read-only replay of hash-frozen M4 commands into full snapshots.

Run this file as a subprocess, not as an imported current-project module: the
recorded implementation must own every legacy import. Only the output cache is
written; the legacy workspace and frozen input archive remain read-only.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import sys


STAGES = ("in_bowl", "scoop", "level", "carry", "deliver", "retract")
FLAGS = ("TIME", "QPOS", "QVEL", "ACT", "HISTORY", "WARMSTART", "CTRL",
         "QFRC_APPLIED", "XFRC_APPLIED", "EQ_ACTIVE", "MOCAP_POS", "MOCAP_QUAT",
         "USERDATA", "PLUGIN")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def semantic_stage(teacher_stage, phase=None):
    if teacher_stage in ("above", "pre_entry"):
        return "in_bowl"
    if teacher_stage == "entry" and phase != "ACQUIRE":
        return "carry"
    if teacher_stage in ("entry", "sweep"):
        return "scoop"
    if teacher_stage.startswith("wall_") or teacher_stage in ("clearance", "pickup_hold"):
        return "level"
    if teacher_stage in ("transport", "wait_ready", "wait_level", "recover"):
        return "carry"
    if teacher_stage in ("release_clear", "retract"):
        return "retract"
    return "deliver"


def verify_source(legacy_root, source_root, episode, parent):
    frozen = json.loads(parent.read_text())
    if frozen["status"] != "frozen":
        raise ValueError("M4 parent is not frozen")
    manifest = json.loads((episode / "manifest.json").read_text())
    if not manifest["success"] or manifest["truncated"] or manifest["failure_reason"]:
        raise ValueError("P2 requires the predetermined successful full M4 episode")
    hashes = {"parent_manifest": sha256(parent)}
    for relative, expected in manifest["input_hashes"].items():
        if sha256(source_root / relative) != expected:
            raise ValueError(f"Frozen input changed: {relative}")
    for name, expected in {**manifest["files_sha256"], "manifest.json": sha256(episode / "manifest.json")}.items():
        path = episode / name
        actual = sha256(path)
        if actual != expected or frozen["evidence_sha256"].get(path.relative_to(legacy_root).as_posix()) != actual:
            raise ValueError(f"Episode is not bound to its frozen M4 parent: {name}")
        hashes[name] = actual
    return manifest, hashes


def main():
    extractor_hash = sha256(__file__)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--episode", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-ticks", type=int, default=100)
    args = parser.parse_args()
    if args.sample_ticks <= 0:
        parser.error("sample-ticks must be positive")
    manifest, hashes = verify_source(args.legacy_root, args.source_root, args.episode, args.parent)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.source_root / "src"))
    import mujoco
    import numpy as np
    from unittest.mock import patch
    from feedingrobot.control.adapter import RobotAdapter
    from feedingrobot.data.replay import replay_episode
    from feedingrobot.experts import Teacher
    from feedingrobot.experts.geometry import teacher_geometry
    from feedingrobot.sim import model as legacy_model
    from feedingrobot.sim.events import evidence
    from feedingrobot.sim.task import FeedingTask

    # MJB contains mesh path strings. Use only the recorded workspace paths after
    # confirming their bytes, while all source/configuration comes from the archive.
    original_tostring = legacy_model.ET.tostring

    def recorded_mesh_paths(root, *positional, **keywords):
        if root.tag == "mujoco":
            for mesh in root.findall("asset/mesh"):
                archived = Path(mesh.get("file"))
                if archived.is_relative_to(args.source_root):
                    relative = archived.relative_to(args.source_root)
                    recorded = args.legacy_root / relative
                    if sha256(recorded) != manifest["input_hashes"][relative.as_posix()]:
                        raise ValueError(f"Recorded mesh changed: {relative}")
                    mesh.set("file", str(recorded))
        return original_tostring(root, *positional, **keywords)

    records, stage_changes, physical_changes, physical_trace = {}, [], [], []
    task_ref, teacher, active, previous, penultimate, pending = None, None, False, None, None, set()
    last_stage, last_support = None, None
    max_teacher_error = 0.
    seen_support = set()
    critical_physical_events = {}
    prescribed = {0: {"initial"}}
    for event in manifest["events"]:
        tick = round(event["time"] / manifest["dt"])
        for offset in (-1, 0, 1):
            prescribed.setdefault(max(0, tick + offset), set()).add(f"event:{event['name']}:{offset:+d}")

    def metadata(task):
        return dict(tick=task.tick, time_s=float(task.data.time), phase=task.logic.phase,
                    teacher_stage=teacher.stage, stage=semantic_stage(teacher.stage, task.logic.phase))

    def retain(record, reason):
        tick = record["meta"]["tick"]
        reasons = records[tick]["reasons"] if tick in records else []
        records[tick] = record
        records[tick]["reasons"] = reasons
        if reason not in records[tick]["reasons"]:
            records[tick]["reasons"].append(reason)

    def capture(task):
        state = task.get_state()
        parts = {}
        for name in FLAGS:
            flag = getattr(mujoco.mjtState, "mjSTATE_" + name)
            value = np.empty(mujoco.mj_stateSize(task.model, flag))
            mujoco.mj_getState(task.model, task.data, value, flag)
            parts[name] = value
        bowl = task.model.body("dynamic_bowl2").id
        return dict(meta=metadata(task), state=state, parts=parts,
                    body_xpos=task.data.xpos.copy(), body_xquat=task.data.xquat.copy(),
                    site_xpos=task.data.site_xpos.copy(), site_xmat=task.data.site_xmat.copy(),
                    bowl_pos=task.model.body_pos[bowl].copy(),
                    bowl_quat=task.model.body_quat[bowl].copy())

    original_restore = FeedingTask.set_state
    original_command = RobotAdapter.set_twist
    original_step = FeedingTask.step_physics

    def restore(task, state):
        nonlocal task_ref, teacher, active, previous
        original_restore(task, state)
        task_ref = task
        teacher = Teacher(task.robot_config, manifest["teacher_config"])
        teacher.reset(manifest["teacher_parameters"], geometry=teacher_geometry(task))
        signature = task.state_signature()
        # The frozen model/configuration is constant during this fixed episode.
        # Cache this pure serialization value to make per-boundary instrumentation
        # affordable; no dynamics, commands or event logic is changed.
        task.state_signature = lambda: copy.deepcopy(signature)
        active = True
        previous = capture(task)
        retain(previous, "initial")

    def command(adapter, twist, now, valid_until):
        nonlocal last_stage, max_teacher_error
        if active and adapter is task_ref.adapter:
            proposal = teacher.act(task_ref.provider.observe()["policy_obs"])
            error = float(np.max(np.abs(proposal - twist)))
            if not np.isfinite(error) or error > 1e-12:
                raise AssertionError(f"Recorded teacher command mismatch: {error}")
            max_teacher_error = max(max_teacher_error, error)
            if teacher.stage != last_stage:
                stage_changes.append(dict(tick=task_ref.tick, time_s=float(task_ref.data.time),
                                          previous=last_stage, teacher_stage=teacher.stage,
                                          stage=semantic_stage(teacher.stage, task_ref.logic.phase)))
                if penultimate is not None:
                    retain(penultimate, "teacher_transition:-1")
                retain(capture(task_ref), "teacher_transition:0")
                pending.add(task_ref.tick + 1)
                last_stage = teacher.stage
        return original_command(adapter, twist, now, valid_until)

    def step(task, *positional, **keywords):
        nonlocal previous, penultimate, last_support
        before = previous
        result = original_step(task, *positional, **keywords)
        if not active or task is not task_ref:
            return result
        physical = evidence(task)
        bottom_clearance = physical["bowl_clearance_m"] + task._event_rim - task.data.site_xpos[task.model.site("bowl_frame").id, 2]
        physical_trace.append(dict(tick=task.tick, time_s=float(task.data.time),
            teacher_stage=teacher.stage, phase=task.logic.phase,
            supported=bool(physical["supported"]), on_bowl=bool(physical["on_bowl"]),
            off_bowl=bool(physical["off_bowl"]), released=bool(physical["released"]),
            mouth_supported=bool(physical["mouth_supported"]),
            spoon_support_force_n=physical["spoon_support_force_n"],
            bottom_clearance_m=float(bottom_clearance)))
        support = tuple(bool(physical[key]) for key in ("supported", "on_bowl", "off_bowl", "mouth_supported", "released"))
        current = capture(task)
        if task.tick % args.sample_ticks == 0:
            retain(current, "uniform")
        for reason in prescribed.get(task.tick, ()):
            retain(current, reason)
        if task.tick in pending:
            retain(current, "transition:+1")
            pending.remove(task.tick)
        if last_support is not None and support != last_support:
            physical_changes.append(dict(tick=task.tick, time_s=float(task.data.time),
                                         before=last_support, after=support))
            changed = [(key, value) for key, old, value in zip(
                ("supported", "on_bowl", "off_bowl", "mouth_supported", "released"), last_support, support)
                       if old != value and (teacher.stage, key, value) not in seen_support]
            if changed:
                for key, value in changed:
                    seen_support.add((teacher.stage, key, value))
                if before is not None:
                    retain(before, "physical_support_stage_first:-1")
                retain(current, "physical_support_stage_first:0")
                pending.add(task.tick + 1)
        if ("pickup_physical" not in critical_physical_events and physical["supported"]
                and bottom_clearance > 0 and not physical["on_bowl"]):
            critical_physical_events["pickup_physical"] = dict(meta=current["meta"], evidence=copy.deepcopy(physical),
                bottom_clearance_m=float(bottom_clearance),
                support_min_force_n=task.bean_acceptance["support_min_force_n"],
                contact_min_force_n=task.task_config["contact_min_force_n"])
            if before is not None:
                retain(before, "pickup_physical:-1")
            retain(current, "pickup_physical:0")
            pending.add(task.tick + 1)
        if ("release_physical" not in critical_physical_events and teacher.stage.startswith("release")
                and last_support is not None and not last_support[4] and physical["released"]):
            critical_physical_events["release_physical"] = dict(meta=current["meta"], evidence=copy.deepcopy(physical))
            if before is not None:
                retain(before, "release_physical:-1")
            retain(current, "release_physical:0")
            pending.add(task.tick + 1)
        penultimate, previous = previous, current
        last_support = support
        return result

    with patch.object(legacy_model.ET, "tostring", recorded_mesh_paths), \
            patch.object(FeedingTask, "set_state", restore), \
            patch.object(RobotAdapter, "set_twist", command), \
            patch.object(FeedingTask, "step_physics", step):
        replay = replay_episode(args.episode)
    retain(previous, "final")
    if replay["max_physics_reference_error"] > 1e-10 or replay["max_observation_error"] > 1e-7:
        raise AssertionError("Strict legacy replay failed")
    model = task_ref.model
    names = {}
    for label, kind, count in (("body", mujoco.mjtObj.mjOBJ_BODY, model.nbody),
                               ("joint", mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
                               ("actuator", mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
                               ("geom", mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
                               ("site", mujoco.mjtObj.mjOBJ_SITE, model.nsite),
                               ("sensor", mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor),
                               ("equality", mujoco.mjtObj.mjOBJ_EQUALITY, model.neq)):
        names[label] = []
        ordinals = {}
        for i in range(count):
            name = mujoco.mj_id2name(model, kind, i)
            if label in ("geom", "site"):
                body = int(getattr(model, label + "_bodyid")[i])
                ordinal = ordinals.get(body, 0)
                ordinals[body] = ordinal + 1
                if name is None:
                    name = f"body:{model.body(body).name}/{label}:{ordinal}"
            if name is None or name in names[label]:
                raise ValueError(f"Unnamed or ambiguous {label} prevents audited mapping")
            names[label].append(name)
    metadata_out = dict(names=names, nq=model.nq, nv=model.nv, nbody=model.nbody,
                        jnt_qposadr=model.jnt_qposadr.copy(), jnt_dofadr=model.jnt_dofadr.copy(),
                        jnt_type=model.jnt_type.copy(), body_mocapid=model.body_mocapid.copy(),
                        sensor_adr=model.sensor_adr.copy(), sensor_dim=model.sensor_dim.copy(),
                        dimensions={name: getattr(model, name) for name in ("na", "nhistory", "nmocap", "nuserdata", "npluginstate")})
    output = dict(schema_version=1, purpose="P2 render-only evidence; never a new teacher demonstration",
                  sample_ticks=args.sample_ticks, dt=manifest["dt"], provenance=hashes,
                  source_root=str(args.source_root), episode=str(args.episode), replay=replay,
                  extractor_sha256=extractor_hash,
                  max_teacher_command_error=max_teacher_error, model=metadata_out,
                  stage_changes=stage_changes, physical_changes=physical_changes,
                  critical_physical_events=critical_physical_events,
                  physical_trace=physical_trace,
                  samples=[records[tick] for tick in sorted(records)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".pending")
    temporary.write_bytes(pickle.dumps(output, protocol=5))
    temporary.replace(args.output)
    print(json.dumps(dict(status="passed", samples=len(records), cache=str(args.output),
                         max_physics_reference_error=replay["max_physics_reference_error"],
                         max_observation_error=replay["max_observation_error"],
                         max_teacher_command_error=max_teacher_error)), flush=True)


if __name__ == "__main__":
    main()
