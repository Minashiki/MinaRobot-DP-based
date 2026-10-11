"""Build auditable P2 RGB evidence; unset K240/K216 keep acceptance pending."""

import argparse
import copy
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import pickle

import mujoco
import numpy as np
from PIL import Image

from feedingrobot.scripts.p2_evidence import (BeanSegmentation, CROPS, acceptance_counts,
    check_restored_geometry, distributions, sampled_intervals, visibility, visual_frustum_status)
from feedingrobot.scripts.p2_legacy_replay import STAGES, sha256, verify_source
from feedingrobot.scripts.report import write_report
from feedingrobot.sim.images import crop_image, resize_area
from feedingrobot.sim.model import ROOT, load_json
from feedingrobot.sim.task import FeedingTask
from feedingrobot.sim.vision import CAMERA_FIELDS, VisionRenderer


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple))
                             else value for key, value in row.items()})


def frozen_main_sequence(config):
    binding = config["frozen_main_sequence"]
    path = ROOT / binding["path"]
    if sha256(path) != binding["sha256"]:
        raise ValueError("The predetermined 723-state sequence changed")
    sequence = pickle.loads(path.read_bytes())
    if sequence["schema_version"] != 1 or len(sequence["samples"]) != 723:
        raise ValueError("Resolution comparison requires the frozen v2 723-state sequence")
    for label in ("source_report", "raw_cache", "source_layout_report"):
        if sha256(Path(sequence[label + "_path"])) != sequence[label + "_sha256"]:
            raise ValueError(f"Frozen sequence source changed: {label}")
    records = sequence["samples"]
    if len({record["sample_id"] for record in records}) != len(records):
        raise ValueError("Frozen sequence contains duplicate sample IDs")
    for record in records:
        digest = hashlib.sha256(pickle.dumps(record["state"], protocol=5)).hexdigest()
        if digest != record["state_sha256"]:
            raise ValueError(f"Frozen sample state changed: {record['sample_id']}")
    return sequence, dict(path=str(path), sha256=binding["sha256"],
        source_report_sha256=sequence["source_report_sha256"],
        raw_cache_sha256=sequence["raw_cache_sha256"], samples=len(records),
        state_sha256={record["sample_id"]: record["state_sha256"] for record in records})


def check_frozen_model(task, sequence):
    compiled = np.empty(mujoco.mj_sizeModel(task.model), dtype=np.uint8)
    mujoco.mj_saveModel(task.model, None, compiled)
    digest = hashlib.sha256(compiled.tobytes()).hexdigest()
    if digest != sequence["compiled_model_sha256"]:
        raise ValueError("The compiled physical model or camera optics changed")
    current = load_json("configs/cameras.json")
    for key in ("asset", "hfov_deg", "render_size", "optical_center_m", "optical_quaternion_wxyz",
                "znear_extent_fraction", "zfar_extent_fraction", "cameras"):
        if current[key] != sequence["camera_config"][key]:
            raise ValueError(f"Resolution-only comparison changed camera optics: {key}")
    if current["render_size"] != [256, 256] or current["observation_size"] != 240 or current["crop_size"] != 216:
        raise ValueError("P2 comparison requires raw256, area240 and crop216")
    return digest


def legacy_cache(legacy_root, config, sequence):
    episode = legacy_root / config["legacy_episode"]
    archive = legacy_root / config["legacy_source"]
    parent = legacy_root / config["legacy_parent_manifest"]
    if sha256(parent) != config["legacy_parent_sha256"]:
        raise ValueError("The predetermined frozen M4 parent hash changed")
    _, provenance = verify_source(legacy_root, archive, episode, parent)
    extractor = Path(__file__).with_name("p2_legacy_replay.py")
    binding = dict(provenance=provenance, extractor_sha256=sha256(extractor),
                   sample_ticks=config["sample_ticks"], cache_schema_version=1)
    fingerprint = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    path = Path(sequence["raw_cache_path"])
    if path.name != f"raw_snapshots_{fingerprint[:16]}.pkl" or sha256(path) != sequence["raw_cache_sha256"]:
        raise ValueError("Resolution comparison must reuse the identical frozen v2 raw cache")
    cache = pickle.loads(path.read_bytes())  # Only this locally generated, hash-bound cache.
    if (cache["provenance"] != provenance or cache["extractor_sha256"] != binding["extractor_sha256"]
            or cache["sample_ticks"] != config["sample_ticks"] or cache["schema_version"] != 1):
        raise ValueError("Legacy snapshot cache binding differs; refusing silent reuse")
    if set(cache["critical_physical_events"]) != {"pickup_physical", "release_physical"}:
        raise ValueError("The frozen replay did not supply both real pickup/release boundaries")
    return cache, dict(path=str(path), sha256=sha256(path), fingerprint=fingerprint, **binding)


def save_image(output, relative, pixels, hashes):
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)
    hashes[relative] = sha256(path)
    return relative


def render_sample(task, renderer, segmentation, sample_id, meta, output, hashes):
    before = task.get_state()["physics"].copy()
    images = renderer.render_rgb(task.data)
    rows = []
    for camera, field in CAMERA_FIELDS.items():
        rgb256 = images[field]
        rgb240 = resize_area(rgb256, size=240)
        prefix = f"frames/{sample_id}/{camera}"
        path256 = save_image(output, f"{prefix}_256.png", rgb256, hashes)
        path240 = save_image(output, f"{prefix}_240.png", rgb240, hashes)
        mask, whole, occluders = segmentation.measure(task.data, camera)
        frustum240 = visual_frustum_status(task.model, task.data, segmentation.bean, camera)
        full = visibility(mask, whole, frustum240)
        for crop, offset in CROPS.items():
            crop_mask = crop_image(mask, size=216, offset=offset)
            crop_whole = crop_image(whole, size=216, offset=offset)
            frustum216 = visual_frustum_status(task.model, task.data, segmentation.bean, camera,
                                             crop_offset=offset, crop_size=216)
            measured = visibility(crop_mask, crop_whole, frustum216)
            path216 = save_image(output, f"{prefix}_216_{crop}.png", crop_image(rgb240, size=216, offset=offset), hashes)
            rows.append(dict(sample_id=sample_id, source=meta["source"], tick=meta["tick"],
                time_s=meta["time_s"], stage=meta["stage"], teacher_stage=meta.get("teacher_stage", ""),
                phase=meta.get("phase", ""), camera=camera, crop=crop,
                area240=full["visible_pixels"], area216=measured["visible_pixels"],
                unoccluded240=full["unoccluded_pixels"], unoccluded216=measured["unoccluded_pixels"],
                visible_fraction240=full["visible_fraction"], visible_fraction216=measured["visible_fraction"],
                no_unoccluded_pixels240=full["no_unoccluded_pixels"],
                no_unoccluded_pixels216=measured["no_unoccluded_pixels"],
                frustum_position240=frustum240["position"], frustum_position216=frustum216["position"],
                frustum_outside_planes240=frustum240["outside_planes"], frustum_outside_planes216=frustum216["outside_planes"],
                zero_unoccluded_reason240=full["zero_denominator_reason"],
                zero_unoccluded_reason216=measured["zero_denominator_reason"],
                crop_visible_pixels_lost=full["visible_pixels"] - measured["visible_pixels"],
                crop_unoccluded_pixels_lost=full["unoccluded_pixels"] - measured["unoccluded_pixels"],
                occluding_geoms=occluders, reasons=meta["reasons"],
                rgb256=path256, rgb240=path240, rgb216=path216))
    np.testing.assert_array_equal(task.get_state()["physics"], before,
                                  err_msg="RGB/segmentation modified the physics state")
    return rows


def render_all(cache, sequence, task, output):
    hashes, rows, restore_errors = {}, [], {}
    rgb_comparison = []
    source_report = Path(sequence["source_report_path"])
    source_hashes = json.loads(source_report.read_text())["evidence_sha256"]
    renderer = VisionRenderer(task.model)
    segmentation = BeanSegmentation(task.model, renderer.scene_option)
    old_samples = {sample["meta"]["tick"]: sample for sample in cache["samples"]}
    try:
        for index, record in enumerate(sequence["samples"]):
            state = copy.deepcopy(record["state"])
            # The compiled model is identical. Only the resolution configuration
            # hash changes; the public restoration interface stays strict.
            state["signature"] = copy.deepcopy(task.signature)
            task.set_state(state)
            np.testing.assert_array_equal(task.get_state()["physics"], record["state"]["physics"])
            if record["meta"]["source"] == "legacy_replay":
                sample = old_samples[record["meta"]["tick"]]
                for key, error in check_restored_geometry(task, sample, cache["model"]).items():
                    restore_errors[key] = max(error, restore_errors.get(key, 0.))
            rows.extend(render_sample(task, renderer, segmentation, record["sample_id"],
                                      record["meta"], output, hashes))
            for camera in CAMERA_FIELDS:
                relative = f"frames/{record['sample_id']}/{camera}_256.png"
                changed, maximum = 0, 0
                if hashes[relative] != source_hashes[relative]:
                    with Image.open(source_report.parent / relative) as old, Image.open(output / relative) as new:
                        difference = np.abs(np.asarray(new).astype(int) - np.asarray(old).astype(int))
                    changed, maximum = int(np.count_nonzero(difference)), int(difference.max())
                rgb_comparison.append(dict(sample_id=record["sample_id"], camera=camera,
                    max_channel_error=maximum, changed_channels=changed,
                    source_png_sha256=source_hashes[relative], current_png_sha256=hashes[relative]))
            if (index + 1) % 100 == 0:
                print(f"P2 frozen RGB {index + 1}/{len(sequence['samples'])}", flush=True)
        write_report(output / "layout_extremes.json",
                     json.loads(Path(sequence["source_layout_report_path"]).read_text()))
    finally:
        segmentation.close()
        renderer.close()
    write_report(output / "raw256_comparison_with_v2.json", rgb_comparison)
    return rows, hashes, restore_errors, rgb_comparison


def distribution_report(rows):
    result = []
    for source in sorted({row["source"] for row in rows}):
        for camera in CAMERA_FIELDS:
            for stage in (*STAGES, "all"):
                for crop in CROPS:
                    selected = [row for row in rows if row["source"] == source and row["camera"] == camera
                                and row["crop"] == crop and (stage == "all" or row["stage"] == stage)]
                    if selected:
                        result.append(dict(source=source, camera=camera, stage=stage, crop=crop,
                            area240=distributions(selected, "area240"), area216=distributions(selected, "area216"),
                            visible_fraction240=distributions(selected, "visible_fraction240"),
                            visible_fraction216=distributions(selected, "visible_fraction216")))
    return result


def gallery(rows, output, candidates, hashes, thresholds=(None, None)):
    selections = []
    for camera in CAMERA_FIELDS:
        selected = [row for row in rows if row["source"] == "legacy_replay" and row["camera"] == camera]
        for metric in ("area240", "area216"):
            for crop in (("center",) if metric == "area240" else tuple(CROPS)):
                choices = [row for row in selected if row["crop"] == crop]
                quantiles = [float(np.quantile([row[metric] for row in choices], q)) for q in (.01, .05, .25)]
                for label, value in [(f"candidate_{k}", k) for k in candidates] + list(zip(("q01", "q05", "q25"), quantiles)):
                    row = min(choices, key=lambda item: (abs(item[metric] - value), item["tick"]))
                    selections.append(dict(camera=camera, crop=crop, metric=metric, target=label,
                                           target_pixels=value, actual_pixels=row[metric], sample_id=row["sample_id"],
                                           time_s=row["time_s"], stage=row["stage"], teacher_stage=row["teacher_stage"],
                                           rgb256=row["rgb256"], rgb240=row["rgb240"], rgb216=row["rgb216"]))
        for crop in CROPS:
            choices = sorted([row for row in selected if row["crop"] == crop],
                             key=lambda item: (min(item["area240"], item["area216"]), item["tick"]))
            for row in choices[:max(1, math.ceil(len(choices) * .01))]:
                selections.append(dict(camera=camera, crop=crop, metric="min_area240_area216", target="worst_1_percent",
                                       target_pixels=None, actual_pixels=min(row["area240"], row["area216"]),
                                       time_s=row["time_s"], stage=row["stage"], teacher_stage=row["teacher_stage"],
                                       sample_id=row["sample_id"], rgb256=row["rgb256"], rgb240=row["rgb240"], rgb216=row["rgb216"]))
    for item in selections:
        for size in (240, 216):
            original = item[f"rgb{size}"]
            relative = f"gallery/{item['sample_id']}_{Path(original).stem}_nearest6x.png"
            if relative not in hashes:
                with Image.open(output / original) as image:
                    enlarged = image.resize((image.width * 6, image.height * 6), Image.Resampling.NEAREST)
                    save_image(output, relative, np.array(enlarged), hashes)
            item[f"rgb{size}_nearest6x"] = relative
    write_report(output / "calibration_gallery.json", selections)
    cards = []
    for item in selections:
        label = html.escape(f"{item['camera']} · {item['crop']} · {item['target']} · {item['metric']}={item['actual_pixels']} px · "
                            f"t={item['time_s']:.3f} s · {item['stage']} / {item['teacher_stage']}")
        images = "".join(f'<figure><figcaption>{caption}</figcaption><img src="../{html.escape(item[field])}" alt="{caption}"></figure>'
                         for field, caption in (("rgb240", "240 original"), ("rgb216", "216 original"),
                                                ("rgb240_nearest6x", "240 nearest ×6"), ("rgb216_nearest6x", "216 nearest ×6")))
        cards.append(f'<section><p>{label} · <a href="../{html.escape(item["rgb256"])}">256 original</a></p><div>{images}</div></section>')
    threshold_text = (f'Locked K240={thresholds[0]}, K216={thresholds[1]}. ' if all(k is not None for k in thresholds)
                      else 'K240/K216 remain pending until the user selects and freezes them. ')
    gallery_html = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>P2 RGB threshold calibration</title>'
        '<style>body{font:14px system-ui;margin:24px;background:#fafafa}section{border-top:1px solid #bbb;padding:16px 0}'
        'section>div{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}figure{margin:0}figcaption{margin-bottom:6px}'
        'img{display:block}a{color:#164b88}</style><h1>P2 RGB threshold calibration</h1>'
        '<p>Actual RGB originals and nearest-neighbor enlargements. ' + threshold_text +
        'Zero segmentation pixels can reflect occlusion, an object outside the view, or unresolved subpixel size; consult the full CSV and isolated silhouette counts.</p>'
        + "".join(cards) + '</html>')
    (output / "gallery").mkdir(exist_ok=True)
    path = output / "gallery/calibration_gallery.html"
    path.write_text(gallery_html)
    hashes["gallery/calibration_gallery.html"] = sha256(path)
    return selections


def boundary_attribution(rows, cache, config, output):
    lock = config["threshold_lock"]
    selected = [row for row in rows if row["sample_id"] in lock["boundary_samples"]]
    expected = {(sample, camera, crop) for sample in lock["boundary_samples"]
                for camera in CAMERA_FIELDS for crop in CROPS}
    if len(selected) != len(expected) or {(row["sample_id"], row["camera"], row["crop"]) for row in selected} != expected:
        raise AssertionError("Threshold boundary attribution requires every specified camera/crop row")
    pickup = cache["critical_physical_events"]["pickup_physical"]["meta"]
    physical = {entry["tick"]: entry for entry in cache["physical_trace"]}
    measured = []
    for row in selected:
        measured.append(dict(sample_id=row["sample_id"], stage=row["stage"], tick=row["tick"], time_s=row["time_s"],
            teacher_stage=row["teacher_stage"], phase=row["phase"],
            camera=row["camera"], crop=row["crop"], user_annotation=lock["boundary_annotation"],
            attribution_note=lock["attribution_note"], physical_pickup_tick=pickup["tick"],
            physical_pickup_time_s=pickup["time_s"], seconds_after_physical_pickup=row["time_s"] - pickup["time_s"],
            area240=row["area240"], area216=row["area216"],
            unoccluded240=row["unoccluded240"], unoccluded216=row["unoccluded216"],
            visible_fraction240=row["visible_fraction240"], visible_fraction216=row["visible_fraction216"],
            actual_occluding_geoms240=row["occluding_geoms"],
            crop_visible_pixels_lost=row["crop_visible_pixels_lost"],
            rgb256=row["rgb256"], rgb240=row["rgb240"], rgb216=row["rgb216"],
            **{f"physical_{key}": value for key, value in physical[row["tick"]].items() if key not in ("tick", "time_s")}))
    write_csv(output / "threshold_boundary_attribution.csv", measured)
    return dict(user_annotation=lock["boundary_annotation"], attribution_note=lock["attribution_note"],
                physical_pickup_tick=pickup["tick"], physical_pickup_time_s=pickup["time_s"], measured_rows=measured)


def occlusion_report(rows, output, candidates, thresholds, boundary):
    intervals, summaries, longest_both = [], [], []
    boundary_rows = boundary["measured_rows"]

    def annotation(sample_ids, camera, crop):
        measured = [row for row in boundary_rows if row["sample_id"] in sample_ids
                    and row["camera"] == camera and row["crop"] == crop]
        return dict(boundary_sample_ids=[row["sample_id"] for row in measured],
                    boundary_annotation=boundary["user_annotation"] if measured else "",
                    boundary_actual_occluding_geoms240={row["sample_id"]: row["actual_occluding_geoms240"] for row in measured})
    for camera in CAMERA_FIELDS:
        for crop in CROPS:
            for stage in STAGES:
                selected = sorted([row for row in rows if row["source"] == "legacy_replay"
                                  and row["camera"] == camera and row["crop"] == crop and row["stage"] == stage],
                                  key=lambda row: row["time_s"])
                metrics = {"partial_occlusion240": [row["visible_fraction240"] is not None
                           and row["visible_fraction240"] < 1 for row in selected],
                           "zero_visible240": [row["area240"] == 0 for row in selected],
                           "zero_visible216": [row["area216"] == 0 for row in selected],
                           "no_unoccluded_pixels240": [row["unoccluded240"] == 0 for row in selected],
                           "fully_outside_frustum240": [row["frustum_position240"] == "fully_outside_frustum" for row in selected],
                           "fully_outside_frustum216": [row["frustum_position216"] == "fully_outside_frustum" for row in selected],
                           "crop_visible_pixels_lost": [row["crop_visible_pixels_lost"] > 0 for row in selected]}
                for k in candidates:
                    metrics[f"below_candidate_{k}"] = [row["area240"] < k or row["area216"] < k for row in selected]
                if all(k is not None for k in thresholds):
                    metrics["below_locked_thresholds"] = [row["area240"] < thresholds[0] or row["area216"] < thresholds[1]
                                                          for row in selected]
                for metric, flags in metrics.items():
                    runs = sampled_intervals(selected, flags)
                    spans = np.array([run["sampled_span_s"] for run in runs])
                    span_distribution = dict(count=len(spans), zero_spans=int(np.count_nonzero(spans == 0)),
                        min=float(spans.min()) if len(spans) else None,
                        max=float(spans.max()) if len(spans) else None,
                        **{f"p{round(q * 100):02d}": float(np.quantile(spans, q)) if len(spans) else None
                           for q in (.25, .5, .75, .95)})
                    intervals.extend(dict(camera=camera, crop=crop, stage=stage, metric=metric, **run,
                        **annotation([row["sample_id"] for row in selected
                                      if run["sample_start_s"] <= row["time_s"] <= run["sample_end_s"]], camera, crop))
                                     for run in runs)
                    summaries.append(dict(camera=camera, crop=crop, stage=stage, metric=metric,
                        sample_count=len(selected), failing_sample_count=sum(flags),
                        summed_sampled_span_s=sum(run["sampled_span_s"] for run in runs),
                        longest_sampled_span_s=max((run["sampled_span_s"] for run in runs), default=0.),
                        sampled_spans_s=spans.tolist(), sampled_span_distribution=span_distribution,
                        intermediate_continuity_verified=False,
                        **annotation([row["sample_id"] for row, flag in zip(selected, flags) if flag], camera, crop)))
    for crop in CROPS:
        views = {}
        for row in rows:
            if row["source"] == "legacy_replay" and row["crop"] == crop:
                views.setdefault(row["sample_id"], {})[row["camera"]] = row
        ordered = sorted(views, key=lambda sample: next(iter(views[sample].values()))["time_s"])
        timeline = [dict(sample_id=sample, time_s=next(iter(views[sample].values()))["time_s"]) for sample in ordered]
        checks = [("positive_threshold_pixel_feasibility", 1, 1)] + [(f"candidate_{k}", k, k) for k in candidates]
        if all(k is not None for k in thresholds):
            checks.append(("locked_thresholds", *thresholds))
        for label, k240, k216 in checks:
            flags = [all(view["area240"] < k240 or view["area216"] < k216 for view in views[sample].values())
                     for sample in ordered]
            runs = sampled_intervals(timeline, flags)
            longest_both.append(dict(crop=crop, threshold=label, K240=k240, K216=k216,
                longest=max(runs, key=lambda run: run["sampled_span_s"], default=None),
                failing_samples=sum(flags), sample_count=len(ordered)))
    write_csv(output / "occlusion_intervals.csv", intervals)
    write_csv(output / "occlusion_by_stage.csv", summaries)
    write_report(output / "longest_both_views_below_threshold.json", longest_both)
    return longest_both


def critical_event_visibility(cache, rows, config, output):
    events, table = [], []
    for name, event in cache["critical_physical_events"].items():
        tick = event["meta"]["tick"]
        frames = []
        for offset in (-1, 0, 1):
            sample_id = f"legacy_{tick + offset:06d}"
            selected = [row for row in rows if row["sample_id"] == sample_id]
            if (len(selected) != len(CAMERA_FIELDS) * len(CROPS)
                    or {(row["camera"], row["crop"]) for row in selected} != {(camera, crop) for camera in CAMERA_FIELDS for crop in CROPS}):
                raise AssertionError(f"Missing critical physical boundary RGB: {name}/{offset:+d}")
            schemes = []
            for crop in CROPS:
                views = [row for row in selected if row["crop"] == crop]
                locked = None if config["K240"] is None or config["K216"] is None else any(
                    row["area240"] >= config["K240"] and row["area216"] >= config["K216"] for row in views)
                schemes.append(dict(crop=crop, views=[dict(camera=row["camera"], area240=row["area240"],
                    area216=row["area216"], unoccluded240=row["unoccluded240"], unoccluded216=row["unoccluded216"],
                    rgb256=row["rgb256"], rgb240=row["rgb240"], rgb216=row["rgb216"]) for row in views],
                    locked_threshold_joint_coverage=locked,
                    candidate_joint_coverage={str(k): any(row["area240"] >= k and row["area216"] >= k for row in views)
                                              for k in config["calibration_candidates_pixels"]}))
            frames.append(dict(offset_ticks=offset, tick=tick + offset, sample_id=sample_id, crop_schemes=schemes))
            table.extend(dict(event=name, offset_ticks=offset, sample_id=sample_id, camera=row["camera"],
                              crop=row["crop"], area240=row["area240"], area216=row["area216"],
                              unoccluded240=row["unoccluded240"], unoccluded216=row["unoccluded216"])
                         for row in selected)
        events.append(dict(event=name, physical_boundary=event, frames=frames))
    write_csv(output / "critical_event_areas.csv", table)
    return events


def critical_event_acceptance(events, config):
    frames = [frame for event in events for frame in event["frames"]]
    checks = [dict(sample_id=frame["sample_id"], crop=scheme["crop"],
                   joint_coverage=scheme["locked_threshold_joint_coverage"])
              for frame in frames for scheme in frame["crop_schemes"]]
    complete = len(frames) == 6 and len(checks) == 30
    pending = config["protocol_status"] != "locked" or any(config[key] is None for key in ("K240", "K216"))
    passed = complete and not pending and all(check["joint_coverage"] is True for check in checks)
    return dict(status="passed" if passed else "pending" if pending and complete else "failed", passed=passed,
                required_frames=6, checked_frames=len(frames), required_frame_crop_checks=30,
                checked_frame_crop_checks=len(checks), K240=config["K240"], K216=config["K216"], checks=checks)


def stage_evidence_frames(rows, config, output):
    evidence = []
    pending = config["protocol_status"] != "locked" or any(config[key] is None for key in ("K240", "K216"))
    for stage in STAGES:
        for camera in CAMERA_FIELDS:
            for crop in CROPS:
                chosen = [] if pending else sorted([row for row in rows if row["source"] == "legacy_replay"
                    and row["stage"] == stage and row["camera"] == camera and row["crop"] == crop
                    and row["area240"] >= config["K240"] and row["area216"] >= config["K216"]],
                    key=lambda row: row["time_s"])
                sufficient = len(chosen) >= 3
                frames = [chosen[0], chosen[len(chosen) // 2], chosen[-1]] if sufficient else []
                evidence.append(dict(stage=stage, camera=camera, crop=crop, required=3,
                    status="pending" if pending else "passed" if sufficient else "failed",
                    sufficient_frame_count=len(chosen), selection="first/middle/last qualifying sampled frames",
                    frames=[{key: row[key] for key in ("sample_id", "source", "tick", "time_s", "teacher_stage", "phase",
                        "area240", "area216", "rgb256", "rgb240", "rgb216")} for row in frames]))
    write_report(output / "stage_evidence_frames.json", evidence)
    return evidence


def main():
    source_paths = [path for path in (ROOT / "configs").rglob("*.json")]
    source_paths += [ROOT / relative for relative in (
        "src/feedingrobot/sim/task.py", "src/feedingrobot/sim/model.py", "src/feedingrobot/sim/beans.py",
        "src/feedingrobot/sim/contacts.py", "src/feedingrobot/sim/sensors.py", "src/feedingrobot/sim/vision.py",
        "src/feedingrobot/sim/images.py", "src/feedingrobot/control/adapter.py", "src/feedingrobot/control/safety.py",
        "src/feedingrobot/random.py", "src/feedingrobot/scripts/validate_p2.py", "src/feedingrobot/scripts/p2_evidence.py",
        "src/feedingrobot/scripts/p2_legacy_replay.py", "third_party_manifest.json", "requirements.lock.txt")]
    archived_inputs = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in source_paths}
    archived_hashes = {name: hashlib.sha256(content).hexdigest() for name, content in archived_inputs.items()}
    tool_hashes = {name: archived_hashes[f"src/feedingrobot/scripts/{name}"] for name in ("validate_p2.py", "p2_evidence.py")}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, default=Path("/home/minashiki/FeedingRobot_DPRL"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/p2")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_json("configs/vision_acceptance.json")
    if config["crop_schemes"] != list(CROPS):
        raise ValueError("Acceptance crop protocol differs from the five fixed schemes")
    for key in ("K240", "K216"):
        value = config[key]
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            raise ValueError(f"{key} must be a user-calibrated positive integer or null")
    if config["protocol_status"] == "locked" and any(config[key] is None for key in ("K240", "K216")):
        raise ValueError("Locked protocol must contain both user-calibrated thresholds")
    if config["protocol_status"] == "locked":
        lock = config["threshold_lock"]
        if (lock["K240"], lock["K216"]) != (config["K240"], config["K216"]):
            raise ValueError("Thresholds differ from the user's calibration decision")
        if sha256(ROOT / lock["calibration_report"]) != lock["calibration_report_sha256"]:
            raise ValueError("The user's threshold calibration report changed")
    if not config.get("version") or not config.get("change_reason"):
        raise ValueError("A protocol version and change reason are required")
    output_root = args.output.resolve()
    sequence, sequence_binding = frozen_main_sequence(config)
    cache, cache_binding = legacy_cache(args.legacy_root.resolve(), config, sequence)
    task = FeedingTask()
    try:
        compiled_digest = check_frozen_model(task, sequence)
        render_binding = dict(raw_cache_sha256=cache_binding["sha256"], model_signature=task.signature,
            frozen_main_sequence=sequence_binding, compiled_model_sha256=compiled_digest,
            camera_config_sha256=sha256(ROOT / "configs/cameras.json"),
            tools_sha256=tool_hashes,
            preprocessing_sha256=sha256(ROOT / "src/feedingrobot/sim/images.py"),
            visual_options_sha256=sha256(ROOT / "src/feedingrobot/sim/vision.py"), crop_schemes=CROPS,
            archived_input_sha256=archived_hashes)
        fingerprint = hashlib.sha256(json.dumps(dict(binding=render_binding, protocol=config), sort_keys=True).encode()).hexdigest()
        args.output = output_root / config["version"] / fingerprint[:16]
        args.output.mkdir(parents=True, exist_ok=True)
        for relative, content in archived_inputs.items():
            path = args.output / "inputs" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        write_report(args.output / "input_binding.json", dict(fingerprint=fingerprint, binding=render_binding, protocol=config))
        write_csv(args.output / "physical_support_1ms.csv", cache["physical_trace"])
        write_report(args.output / "physical_support_changes.json", cache["physical_changes"])
        write_report(args.output / "critical_physical_events.json", cache["critical_physical_events"])
        write_report(args.output / "teacher_stage_changes.json", cache["stage_changes"])
        measurement_file = args.output / "measurements.json"
        rows, hashes, restore_errors, rgb_comparison = render_all(cache, sequence, task, args.output)
        write_report(measurement_file, dict(binding=render_binding, rows=rows,
            evidence_sha256=hashes, restore_errors=restore_errors, raw256_comparison_with_v2=rgb_comparison))
    finally:
        task.close()
    write_csv(args.output / "visibility_all.csv", rows)
    distribution = distribution_report(rows)
    write_report(args.output / "visibility_distributions.json", distribution)
    selections = gallery(rows, args.output, config["calibration_candidates_pixels"], hashes,
                         thresholds=(config["K240"], config["K216"]))
    boundary = boundary_attribution(rows, cache, config, args.output)
    longest = occlusion_report(rows, args.output, config["calibration_candidates_pixels"], (config["K240"], config["K216"]), boundary)
    samples = [record["sample_id"] for record in sequence["samples"]]
    expected = {f"legacy_{sample['meta']['tick']:06d}" for sample in cache["samples"]}
    expected.update(f"layout_{proposal['id']}" for proposal in sequence["layout_proposals"] if proposal["accepted"])
    if len(samples) != 723 or set(samples) != expected:
        raise AssertionError("Frozen IDs differ from the unchanged 710 replay and 13 accepted layout states")
    if set(samples) != {row["sample_id"] for row in rows}:
        raise AssertionError("Rendered IDs differ from the complete raw replay and predefined accepted layout matrix")
    acceptance = acceptance_counts(rows, config["K240"], config["K216"], config["minimum_stage_frames_per_camera"],
                                   expected_samples=samples)
    if config["protocol_status"] != "locked" and acceptance["status"] == "passed":
        acceptance.update(status="pending", passed=False, reason="Threshold values have not been frozen by the user")
    candidate_results = {str(k): acceptance_counts(rows, k, k, config["minimum_stage_frames_per_camera"],
                                                 expected_samples=samples)
                         for k in config["calibration_candidates_pixels"]}
    joint_zero = []
    for crop in CROPS:
        views = {}
        for row in rows:
            if row["crop"] == crop:
                views.setdefault(row["sample_id"], []).append(row)
        joint_zero.append(dict(crop=crop, samples=[dict(sample_id=sample, time_s=items[0]["time_s"],
            source=items[0]["source"], stage=items[0]["stage"])
            for sample, items in views.items() if all(item["area240"] == 0 or item["area216"] == 0 for item in items)]))
    critical_areas = critical_event_visibility(cache, rows, config, args.output)
    critical_acceptance = critical_event_acceptance(critical_areas, config)
    stage_frames = stage_evidence_frames(rows, config, args.output)
    if acceptance["passed"] and (not critical_acceptance["passed"] or any(item["status"] != "passed" for item in stage_frames)):
        acceptance.update(status="failed", passed=False, reason="Explicit critical/stage evidence differs from all-sample acceptance")
    report = dict(stage="P2", status=acceptance["status"], passed=acceptance["passed"],
        evidence_generation="complete", protocol=config, acceptance=acceptance,
        candidate_threshold_diagnostics=candidate_results, raw_cache=cache_binding, render_binding=render_binding,
        same_main_sequence=sequence_binding, resolution_comparison_source_report=sequence["source_report_path"],
        frozen_sequence_rgb256_reproduction=sequence["rgb256_comparison"],
        raw256_comparison_with_v2=dict(frames=len(rgb_comparison),
            differing_frames=sum(item["changed_channels"] > 0 for item in rgb_comparison),
            max_channel_error=max(item["max_channel_error"] for item in rgb_comparison)),
        strict_legacy_replay=cache["replay"], max_teacher_command_error=cache["max_teacher_command_error"],
        legacy_samples=len(cache["samples"]), total_samples=len(samples), measurements=len(rows),
        restore_geometry_max_errors=restore_errors, distributions=distribution,
        native240_segmentation="Bean visual geom ID; same camera isolated render-scene silhouette; never policy input",
        zero_area_interpretation="Zero visible pixels alone does not prove occlusion; visual mesh AABB/frustum proofs separate definite exclusion, definite inclusion with unresolved subpixel rasterization, and unresolved intersection",
        interval_precision=config["interval_precision"], longest_both_views_below_threshold=longest,
        joint_zero_visible_pixel_diagnostics=joint_zero,
        positive_threshold_joint_coverage_possible=not any(item["samples"] for item in joint_zero),
        structural_joint_coverage=dict(status="failed" if any(item["samples"] for item in joint_zero)
                                       else "no_zero_pixel_gap_observed_in_sampled_evidence",
                                       criterion="Necessary pixel feasibility diagnostic; this does not replace user-calibrated K240/K216"),
        critical_physical_event_visibility=critical_areas,
        critical_event_acceptance=critical_acceptance, threshold_boundary_attribution=boundary,
        stage_evidence_frames=stage_frames,
        calibration_gallery_entries=len(selections), evidence_sha256=hashes,
        p4_reverification=dict(status="TODO", required_validation_demonstrations=10,
                              clause=config["p4_reverification"]))
    write_report(args.output / "report.json", report)
    write_report(output_root / "latest.json", dict(report=str(args.output / "report.json"),
                 report_sha256=sha256(args.output / "report.json"), fingerprint=fingerprint,
                 protocol_version=config["version"], status=report["status"]))
    print(json.dumps(dict(status=report["status"], passed=report["passed"], samples=len(samples),
                         measurements=len(rows), report=str(args.output / "report.json"))), flush=True)
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
