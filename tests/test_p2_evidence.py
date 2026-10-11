"""Auditable replay conversion, visibility diagnostics and complete P2 gates."""

import copy
import csv
import hashlib
import json
import os
import pickle

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import pytest
from PIL import Image

from feedingrobot.scripts.p2_evidence import (BeanSegmentation, CROPS, acceptance_counts,
    check_restored_geometry, distributions, legacy_render_state, object_names, sampled_intervals, visibility,
    classify_frustum_points)
from feedingrobot.scripts.p2_legacy_replay import FLAGS, STAGES, semantic_stage, sha256, verify_source
from feedingrobot.scripts.validate_p2 import (boundary_attribution, check_frozen_model, critical_event_acceptance,
    critical_event_visibility, frozen_main_sequence, gallery, occlusion_report, stage_evidence_frames)
from feedingrobot.sim.model import load_json
from feedingrobot.sim.task import FeedingTask
from feedingrobot.sim.vision import policy_scene_option
from feedingrobot.sim.images import crop_image


def complete_rows():
    return [dict(sample_id=f"{stage}_{frame}", source="legacy_replay", stage=stage,
                 camera=camera, crop=crop, area240=25, area216=16)
            for stage in STAGES for frame in range(3)
            for camera in ("agentview", "robot0_eye_in_hand") for crop in CROPS]


def test_pending_thresholds_and_missing_records_cannot_pass():
    rows = complete_rows()
    assert acceptance_counts(rows, None, None)["status"] == "pending"
    assert acceptance_counts([], 9, 9)["status"] == "failed"
    assert acceptance_counts(rows[:-1], 9, 9)["status"] == "failed"
    assert acceptance_counts(rows + [rows[-1]], 9, 9)["status"] == "failed"
    assert acceptance_counts([row for row in rows if row["camera"] == "agentview"], 9, 9)["status"] == "failed"
    assert acceptance_counts(rows, 9, 9, expected_samples=["missing"])["status"] == "failed"


def test_joint_coverage_requires_both_sizes_from_the_same_view():
    rows = complete_rows()
    assert acceptance_counts(rows, 9, 9)["passed"]
    for row in rows:
        if row["sample_id"] == "in_bowl_0" and row["crop"] == "center":
            if row["camera"] == "agentview":
                row["area240"], row["area216"] = 8, 25
            else:
                row["area240"], row["area216"] = 25, 8
    result = acceptance_counts(rows, 9, 9)
    assert not result["passed"]
    assert any(failure.get("uncovered_samples") == ["in_bowl_0"] for failure in result["failures"])


def test_five_crop_schemes_allow_different_sufficient_cameras():
    rows = complete_rows()
    for row in rows:
        if (row["crop"] == "top_left" and row["camera"] == "agentview"
                or row["crop"] == "top_right" and row["camera"] == "robot0_eye_in_hand"):
            row["area216"] = 0
    assert acceptance_counts(rows, 9, 9, minimum_stage_frames=0)["passed"]
    assert not acceptance_counts(rows, 9, 9, minimum_stage_frames=3)["passed"]


def test_boundary_annotations_preserve_measured_occluders_and_support_flags(tmp_path):
    ticks = [15600, *range(15699, 15704), *range(15707, 15711), 15800]
    ids = [f"legacy_{tick:06d}" for tick in ticks]
    comment = "舀起后入勺部分遮挡，阈值下限由此类结构性区间决定"
    config = dict(threshold_lock=dict(boundary_samples=ids, boundary_annotation=comment,
                                     attribution_note="User context and measured attribution remain separate"))
    rows = [dict(sample_id=f"legacy_{tick:06d}", tick=tick, time_s=tick / 1000, stage="level",
        source="legacy_replay", teacher_stage="wall_align", phase="ACQUIRE", camera=camera, crop=crop,
        area240=8 if camera == "agentview" else 3, area216=8 if camera == "agentview" else 3,
        unoccluded240=8 if camera == "agentview" else 124, unoccluded216=8 if camera == "agentview" else 124,
        visible_fraction240=1. if camera == "agentview" else 3 / 124,
        visible_fraction216=1. if camera == "agentview" else 3 / 124,
        occluding_geoms={} if camera == "agentview" else {"visual_dynamic_bowl2": 121},
        frustum_position240="fully_inside_frustum", frustum_position216="fully_inside_frustum",
        crop_visible_pixels_lost=0, rgb256="256.png", rgb240="240.png", rgb216="216.png")
        for tick in ticks for camera in ("agentview", "robot0_eye_in_hand") for crop in CROPS]
    cache = dict(critical_physical_events={"pickup_physical": {"meta": dict(tick=10789, time_s=10.789)}},
        physical_trace=[dict(tick=tick, time_s=tick / 1000, supported=True, off_bowl=False,
            on_bowl=tick == 15800, released=False, spoon_support_force_n=.003,
            bottom_clearance_m=.008) for tick in ticks])
    attribution = boundary_attribution(rows, cache, config, tmp_path)
    assert len(attribution["measured_rows"]) == 110
    for row in attribution["measured_rows"]:
        assert row["user_annotation"] == comment
        assert row["seconds_after_physical_pickup"] == pytest.approx(row["time_s"] - 10.789)
        assert row["physical_supported"] and not row["physical_off_bowl"] and not row["physical_released"]
        assert row["actual_occluding_geoms240"] == ({} if row["camera"] == "agentview" else {"visual_dynamic_bowl2": 121})
    occlusion_report(rows, tmp_path, [9, 25, 49], (8, 8), attribution)
    summaries = list(csv.DictReader((tmp_path / "occlusion_by_stage.csv").open()))
    fixed_partial = next(row for row in summaries if row["camera"] == "agentview" and row["crop"] == "center"
                         and row["stage"] == "level" and row["metric"] == "partial_occlusion240")
    assert fixed_partial["failing_sample_count"] == "0" and json.loads(fixed_partial["boundary_sample_ids"]) == []
    intervals = list(csv.DictReader((tmp_path / "occlusion_intervals.csv").open()))
    fixed_below9 = next(row for row in intervals if row["camera"] == "agentview" and row["crop"] == "center"
                        and row["metric"] == "below_candidate_9")
    assert json.loads(fixed_below9["boundary_sample_ids"]) == ids
    assert float(fixed_below9["sampled_span_s"]) == pytest.approx(.2)
    assert all(
        not geoms for geoms in json.loads(fixed_below9["boundary_actual_occluding_geoms240"]).values())


def test_explicit_six_critical_frames_apply_the_same_five_crop_joint_rule(tmp_path):
    cache = dict(critical_physical_events={name: {"meta": dict(tick=tick)}
                 for name, tick in (("pickup_physical", 10789), ("release_physical", 50630))})
    config = dict(protocol_status="locked", K240=8, K216=8, calibration_candidates_pixels=[9, 25, 49])
    rows = [dict(sample_id=f"legacy_{tick + offset:06d}", camera=camera, crop=crop,
        area240=8, area216=8, unoccluded240=8, unoccluded216=8,
        rgb256="256.png", rgb240="240.png", rgb216="216.png")
        for tick in (10789, 50630) for offset in (-1, 0, 1)
        for camera in ("agentview", "robot0_eye_in_hand") for crop in CROPS]
    events = critical_event_visibility(cache, rows, config, tmp_path)
    result = critical_event_acceptance(events, config)
    assert result["passed"] and result["checked_frames"] == 6 and result["checked_frame_crop_checks"] == 30
    rows[0]["area216"] = 7
    assert critical_event_acceptance(critical_event_visibility(cache, rows, config, tmp_path), config)["passed"]
    rows[5]["area216"] = 7
    assert not critical_event_acceptance(critical_event_visibility(cache, rows, config, tmp_path), config)["passed"]
    with pytest.raises(AssertionError, match="Missing critical"):
        critical_event_visibility(cache, rows[:-1], config, tmp_path)


def test_stage_evidence_selects_three_distinct_qualified_first_middle_last_frames(tmp_path):
    config = dict(protocol_status="locked", K240=8, K216=8)
    rows = [dict(sample_id=f"{stage}_{i}", source="legacy_replay", stage=stage, time_s=i / 10, tick=i,
        teacher_stage=stage, phase="ACQUIRE", camera=camera, crop=crop, area240=8, area216=8,
        rgb256=f"{stage}_{i}_256.png", rgb240=f"{stage}_{i}_240.png", rgb216=f"{stage}_{i}_216.png")
        for stage in STAGES for i in range(5) for camera in ("agentview", "robot0_eye_in_hand") for crop in CROPS]
    evidence = stage_evidence_frames(rows, config, tmp_path)
    assert len(evidence) == 60 and all(item["status"] == "passed" for item in evidence)
    assert [frame["tick"] for frame in evidence[0]["frames"]] == [0, 2, 4]
    for row in rows:
        if row["stage"] == "in_bowl" and row["camera"] == "agentview" and row["crop"] == "center" and row["tick"] < 3:
            row["area216"] = 7
    evidence = stage_evidence_frames(rows, config, tmp_path)
    assert evidence[0]["status"] == "failed" and evidence[0]["sufficient_frame_count"] == 2 and not evidence[0]["frames"]


def test_zero_pixel_diagnostics_and_quantiles_keep_zeroes():
    missing = visibility(np.zeros((240, 240), bool), np.zeros((240, 240), bool))
    assert missing["visible_fraction"] is None
    assert missing["zero_denominator_reason"] == "outside_view_or_subpixel_unresolved"
    assert "offscreen" not in missing
    distribution = distributions([dict(area=value) for value in (0, 0, 9, 16, 25)], "area")
    assert distribution["zeros"] == 2 and distribution["min"] == 0
    assert distribution["q01"] == 0 and distribution["q25"] == 0


def test_visual_frustum_proof_distinguishes_outside_and_inside_unresolved_pixels():
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    tiny = signs * .00001 + [0., 0., -1.]
    inside = classify_frustum_points(tiny, .4, .01, 10.)
    outside = classify_frustum_points(tiny + [1., 0., 0.], .4, .01, 10.)
    assert inside["position"] == "fully_inside_frustum"
    assert outside["position"] == "fully_outside_frustum" and "right" in outside["outside_planes"]
    empty = np.zeros((240, 240), bool)
    assert visibility(empty, empty, inside)["zero_denominator_reason"] == "inside_frustum_subpixel_unresolved"
    assert visibility(empty, empty, outside)["zero_denominator_reason"] == "fully_outside_frustum"


def test_216_crop_coordinates_and_frustum_use_the_native_240_grid():
    assert CROPS == dict(center=(12, 12), top_left=(0, 0), top_right=(0, 24),
                         bottom_left=(24, 0), bottom_right=(24, 24))
    image = np.arange(240 * 240).reshape(240, 240)
    for crop, (top, left) in CROPS.items():
        result = crop_image(image, size=216, offset=(top, left))
        assert result.shape == (216, 216)
        np.testing.assert_array_equal(result, image[top:top + 216, left:left + 216])
    near_right = np.array([[.939, -.001, -1.], [.941, .001, -1.]])
    assert classify_frustum_points(near_right, 1., .01, 10.)["position"] == "fully_inside_frustum"
    assert classify_frustum_points(near_right, 1., .01, 10., crop_offset=CROPS["center"],
                                   crop_size=216)["position"] == "fully_outside_frustum"
    assert classify_frustum_points(near_right, 1., .01, 10., crop_offset=CROPS["top_right"],
                                   crop_size=216)["position"] == "fully_inside_frustum"


def test_sampled_runs_report_endpoint_precision_without_continuity_claim():
    rows = [dict(sample_id=str(i), time_s=time) for i, time in enumerate((0., .001, .1, .3, .301))]
    result = sampled_intervals(rows, [False, True, True, False, True])
    assert result[0]["sample_start_s"] == .001
    assert result[0]["sample_end_s"] == .1
    assert result[0]["sampled_span_s"] == pytest.approx(.099)
    assert result[1]["sampled_span_s"] == 0
    assert all(not interval["intermediate_continuity_verified"] for interval in result)


@pytest.mark.parametrize("stage,phase,expected", [
    ("entry", "ACQUIRE", "scoop"), ("entry", "APPROACH", "carry"),
    ("wall_align", "ACQUIRE", "level"), ("pickup_hold", "TRANSPORT", "level"),
    ("release_roll", "RETRACT", "deliver"), ("release_clear", "RETRACT", "retract")])
def test_teacher_substages_define_the_six_semantic_stages(stage, phase, expected):
    assert semantic_stage(stage, phase) == expected


def test_source_verification_rejects_modified_frozen_records(tmp_path):
    archive, episode = tmp_path / "archive", tmp_path / "episode"
    archive.mkdir()
    episode.mkdir()
    (archive / "source.py").write_text("frozen source\n")
    (episode / "commands.json").write_text("[]")
    manifest = dict(success=True, truncated=False, failure_reason=None,
                    input_hashes={"source.py": sha256(archive / "source.py")},
                    files_sha256={"commands.json": sha256(episode / "commands.json")})
    (episode / "manifest.json").write_text(json.dumps(manifest))
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps(dict(status="frozen", evidence_sha256={
        f"episode/{name}": sha256(episode / name) for name in ("commands.json", "manifest.json")})))
    verify_source(tmp_path, archive, episode, parent)
    (episode / "commands.json").write_text("changed")
    with pytest.raises(ValueError, match="not bound"):
        verify_source(tmp_path, archive, episode, parent)


def test_resolution_comparison_rejects_modified_or_duplicate_frozen_states(tmp_path):
    sources = {}
    for label in ("source_report", "raw_cache", "source_layout_report"):
        source = tmp_path / label
        source.write_text("frozen")
        sources[label + "_path"] = str(source)
        sources[label + "_sha256"] = sha256(source)
    state = dict(physics=np.zeros(1))
    state_sha = hashlib.sha256(pickle.dumps(state, protocol=5)).hexdigest()
    records = [dict(sample_id=f"sample_{index}", state=copy.deepcopy(state), state_sha256=state_sha)
               for index in range(723)]
    path = tmp_path / "sequence.pkl"

    def write_sequence():
        path.write_bytes(pickle.dumps(dict(schema_version=1, samples=records, **sources), protocol=5))
        return dict(frozen_main_sequence=dict(path=str(path), sha256=sha256(path)))

    config = write_sequence()
    assert len(frozen_main_sequence(config)[0]["samples"]) == 723
    records[0]["state"]["physics"][0] = 1.
    with pytest.raises(ValueError, match="sample state changed"):
        frozen_main_sequence(write_sequence())
    records[0]["state"]["physics"][0] = 0.
    records[-1]["sample_id"] = records[0]["sample_id"]
    with pytest.raises(ValueError, match="duplicate sample IDs"):
        frozen_main_sequence(write_sequence())


def test_resolution_comparison_rejects_model_or_camera_changes():
    task = FeedingTask()
    try:
        compiled = np.empty(mujoco.mj_sizeModel(task.model), dtype=np.uint8)
        mujoco.mj_saveModel(task.model, None, compiled)
        sequence = dict(compiled_model_sha256=hashlib.sha256(compiled.tobytes()).hexdigest(),
                        camera_config=load_json("configs/cameras.json"))
        assert check_frozen_model(task, sequence) == sequence["compiled_model_sha256"]
        sequence["camera_config"]["hfov_deg"] += 1.
        with pytest.raises(ValueError, match="camera optics: hfov_deg"):
            check_frozen_model(task, sequence)
        sequence["compiled_model_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="compiled physical model"):
            check_frozen_model(task, sequence)
    finally:
        task.close()


def test_schema_migration_maps_added_visual_bodies_without_weakening_restore():
    old, current = FeedingTask(cameras=False), FeedingTask(cameras=True)
    try:
        old.data.xfrc_applied[old.index.tool_body] = [.001, .002, .003, .004, .005, .006]
        mujoco.mj_forward(old.model, old.data)
        parts = {}
        for name in FLAGS:
            flag = getattr(mujoco.mjtState, "mjSTATE_" + name)
            parts[name] = np.empty(mujoco.mj_stateSize(old.model, flag))
            mujoco.mj_getState(old.model, old.data, parts[name], flag)
        source = dict(names={name: object_names(old.model, name) for name in
                      ("body", "joint", "actuator", "geom", "site", "sensor", "equality")},
                      nq=old.model.nq, nv=old.model.nv, jnt_qposadr=old.model.jnt_qposadr.copy(),
                      jnt_dofadr=old.model.jnt_dofadr.copy(), sensor_adr=old.model.sensor_adr.copy(),
                      sensor_dim=old.model.sensor_dim.copy())
        sample = dict(state=old.get_state(), parts=parts,
            bowl_pos=old.model.body_pos[old.index.bowl_body].copy(),
            bowl_quat=old.model.body_quat[old.index.bowl_body].copy(),
            body_xpos=old.data.xpos.copy(), site_xpos=old.data.site_xpos.copy(),
            site_xmat=old.data.site_xmat.copy())
        original = copy.deepcopy(sample)
        converted = legacy_render_state(current, sample, source)
        assert converted["signature"] == current.signature
        assert converted["controller"]["status"] == "holding"
        assert converted["controller"]["remaining_substeps"] == 0
        current.set_state(converted)
        assert all(error == 0 for error in check_restored_geometry(current, sample, source).values())
        np.testing.assert_array_equal(current.data.xfrc_applied[current.index.tool_body], old.data.xfrc_applied[old.index.tool_body])
        for index, name in enumerate(object_names(current.model, "body")):
            if name not in source["names"]["body"]:
                np.testing.assert_array_equal(current.data.xfrc_applied[index], np.zeros(6))
        np.testing.assert_array_equal(sample["state"]["physics"], original["state"]["physics"])
        bad = copy.deepcopy(converted)
        bad["signature"] = old.signature
        with pytest.raises(ValueError, match="Incompatible"):
            current.set_state(bad)
    finally:
        current.close()
        old.close()


def test_native_segmentation_isolates_the_real_bean_without_changing_physics():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <camera name="camera" pos="0 0 1"/>
      <geom name="blocker" type="box" size=".14 .14 .01" pos="0 0 .15"/>
      <body name="bean"><freejoint/><geom name="bean_000_visual" type="ellipsoid" size=".08 .06 .04"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    qpos, positions = data.qpos.copy(), model.geom_pos.copy()
    diagnostic = BeanSegmentation(model, policy_scene_option())
    try:
        mask, whole, occluders = diagnostic.measure(data, "camera")
        assert mask.shape == whole.shape == (240, 240)
        result = visibility(mask, whole)
        assert result["visible_pixels"] == 0 and result["unoccluded_pixels"] > 0
        assert "blocker" in occluders
        np.testing.assert_array_equal(data.qpos, qpos)
        np.testing.assert_array_equal(model.geom_pos, positions)
    finally:
        diagnostic.close()


def test_gallery_enlargements_do_not_collide_across_sample_basenames(tmp_path):
    rows = []
    for tick, area, color in ((1, 9, (11, 22, 33)), (2, 25, (44, 55, 66))):
        sample = f"legacy_{tick:06d}"
        folder = tmp_path / "frames" / sample
        folder.mkdir(parents=True)
        for camera in ("agentview", "robot0_eye_in_hand"):
            rgb256, rgb240 = f"frames/{sample}/{camera}_256.png", f"frames/{sample}/{camera}_240.png"
            Image.new("RGB", (256, 256), color).save(tmp_path / rgb256)
            Image.new("RGB", (240, 240), color).save(tmp_path / rgb240)
            for crop in CROPS:
                rgb216 = f"frames/{sample}/{camera}_216_{crop}.png"
                Image.new("RGB", (216, 216), color).save(tmp_path / rgb216)
                rows.append(dict(sample_id=sample, tick=tick, time_s=tick * .001,
                    stage="scoop", teacher_stage="sweep", source="legacy_replay",
                    camera=camera, crop=crop, area240=area, area216=area,
                    rgb256=rgb256, rgb240=rgb240, rgb216=rgb216))
    selections = gallery(rows, tmp_path, [9, 25, 49], {})
    assert {item["sample_id"] for item in selections} == {"legacy_000001", "legacy_000002"}
    enlarged_by_original = {}
    for item in selections:
        for size in (240, 216):
            original, enlarged = item[f"rgb{size}"], item[f"rgb{size}_nearest6x"]
            enlarged_by_original[original] = enlarged
            with Image.open(tmp_path / original) as actual, Image.open(tmp_path / enlarged) as nearest:
                np.testing.assert_array_equal(nearest, actual.resize((size * 6, size * 6), Image.Resampling.NEAREST))
    assert len(set(enlarged_by_original.values())) == len(enlarged_by_original)
