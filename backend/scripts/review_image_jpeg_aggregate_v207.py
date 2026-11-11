"""Review sealed v189 JPEG measurement arithmetic without opening image payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V189 = "image-jpeg-diagnostic-freeze-v189.json"
V42 = "image-regional-calibration-v42-execution-freeze.json"
V46 = "image-native-confirmation-v46-freeze.json"
FREEZE_SHA = "cd0333f18623441dba7177e4f79db69a31174713bc823822d5112a07dd4bba5e"
QUALITIES = (75, 90)
KINDS = ("ai", "original", "tone", "blur", "copy_move")
METHODS = ("current", "sensitive", "shared")
ARMS = ("baseline", "candidate")
PHOTOS, GROUPS, BATCH, PHASES = 59, 48, 25, 12
SEED, RESAMPLES = 20261108, 10_000
MODULE = "scripts.image_jpeg_diagnostic_v189"
MONITOR = "backend/scripts/image_jpeg_diagnostic_runtime_v189.py"
ASSET_NAMES = {
    "models/research/image-v18-relayformer-v2/image-only.pth",
    "models/weights/fused/fused_opensdi.pth",
    "models/weights/fused/convnext.safetensors",
}
LIMITS = {
    "seconds": 900,
    "rss": 6 * 1024**3,
    "reserve": 1536 * 1024**2,
    "start_ram": 6 * 1024**3,
    "start_disk": 3 * 1024**3,
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def verify_public(root=ROOT):
    panels = root / "docs/evaluation-panels"
    freeze_file = panels / V189
    require(sha(freeze_file) == FREEZE_SHA, "The published v189 freeze changed")
    freeze, calibration, confirmation = map(read, (freeze_file, panels / V42, panels / V46))
    require(
        sha(panels / V42) == freeze["v42_freeze_sha256"]
        and sha(panels / V46) == freeze["v46_freeze_sha256"]
        and freeze["training_chain_sha256"] == calibration["training_chain_sha256"]
        and freeze["candidate_checkpoint_sha256"]
        == calibration["final_checkpoint_sha256"]
        == confirmation["final_checkpoint_sha256"]
        and confirmation["candidate_policies"] == [freeze["sensitive_policy"]]
        and freeze["asset_sha256"] == {key: calibration["asset_sha256"][key] for key in ASSET_NAMES}
        and all(freeze["versions"][name] == value for name, value in calibration["runtime_versions"].items()),
        "Frozen v42/v46 checkpoint, comparator assets, policy or runtime commitments differ",
    )
    scope = {
        "schema": "chipmunk.image-jpeg-diagnostic-freeze/189",
        "source_role": "diagnostic",
        "photos": PHOTOS,
        "groups": GROUPS,
        "qualities": list(QUALITIES),
        "kinds": list(KINDS),
        "arms": list(ARMS),
        "methods": list(METHODS),
        "sensitive_policy": "p0.90-a0.01000",
        "useful_iou": 0.3,
        "useful_recall": 0.5,
        "bootstrap_draws": RESAMPLES,
        "bootstrap_seed": SEED,
        "device": {"baseline": "mps", "candidate": "cpu"},
        "dtype": "float32",
        "cpu_threads": 2,
        "views_per_arm_quality": PHOTOS * len(KINDS),
        "views_per_batch": BATCH,
        "batches": PHASES,
        "unique_model_calls": 1180,
        "exact_first_input_repeats": 4,
        "neural_calls_during_preparation": 0,
        "serving_admitted": False,
    }
    require(all(freeze[key] == value for key, value in scope.items()), "Prospective comparison scope changed")
    require(
        all(sha(root / key) == digest for key, digest in freeze["code_sha256"].items()),
        "A frozen v189 implementation or protocol file changed",
    )
    return freeze


def verified_rows(path, digest):
    content = path.read_bytes()
    require(hashlib.sha256(content).hexdigest() == digest, "Journal digest differs from its receipt")
    return [json.loads(line) for line in content.splitlines()]


def profile(folder, arguments, monitor_sha, published_sha=None):
    path = folder / "profile.json"
    digest = sha(path)
    if published_sha is not None:
        require(digest == published_sha, "Phase profile digest differs from the published result")
    record = read(path)
    require(
        record["schema"] == "chipmunk.image-jpeg-diagnostic-profile/189"
        and record["worker"] == MODULE
        and record["command"] == arguments[0]
        and record["worker_arguments"] == arguments
        and record["monitor_sha256"] == monitor_sha
        and record["limits"] == LIMITS
        and type(record["exit_code"]) is int
        and record["exit_code"] == 0
        and record["failure"] is None
        and 0 < record["duration_seconds"] <= LIMITS["seconds"],
        "Expected successful terminal profile",
    )
    samples = verified_rows(folder / "resources.jsonl", record["resource_sha256"])
    times = [sample["elapsed_s"] for sample in samples]
    require(
        samples and times == sorted(times) and 0 <= times[0] <= times[-1] <= record["duration_seconds"],
        "Incomplete or inconsistent resource timestamps",
    )
    require(
        max(s["tree_rss_bytes"] for s in samples) == record["sampled_tree_rss_lower_bound_bytes"]
        and all(
            s["violations"] == []
            and 0 <= s["tree_rss_bytes"] <= LIMITS["rss"]
            and min(s["ram_available_bytes"], s["disk_free_bytes"], s["raw_disk_free_bytes"])
            >= LIMITS["reserve"]
            for s in samples
        )
        and samples[0]["ram_available_bytes"] >= LIMITS["start_ram"]
        and min(samples[0]["disk_free_bytes"], samples[0]["raw_disk_free_bytes"]) >= LIMITS["start_disk"],
        "Resource journal does not support an unguarded completion",
    )
    return {"profile_sha256": digest, "resource_journal_sha256": record["resource_sha256"]}


def phase_support(number):
    start, stop = number * BATCH, min((number + 1) * BATCH, PHOTOS * len(KINDS))
    return [(index // len(KINDS), KINDS[index % len(KINDS)]) for index in range(start, stop)]


def phase(trial, quality, number, arm, expected_profiles, monitor_sha):
    name = f"{arm}-q{quality}-{number:02d}"
    score = arm in ARMS
    command = "score" if score else "readout"
    args = [command, str(trial), FREEZE_SHA, *([arm] if score else []), str(quality), str(number)]
    receipt = profile(trial / f"{name}-profile", args, monitor_sha, expected_profiles[name])
    completion = trial / name / "completed.json"
    done = read(completion)
    support = phase_support(number)
    require(
        done["freeze_sha256"] == FREEZE_SHA and done["phase"] == number and done["quality"] == quality,
        f"{name} completion belongs to another study",
    )
    if score:
        require(
            done["arm"] == arm
            and done["unique_forwards"] == len(support)
            and done["repeat_forwards"] == int(number == 0),
            f"{name} forward counts differ",
        )
    else:
        require(done["views"] == len(support) and done["neural_calls"] == 0, f"{name} readout counts differ")
    rows = verified_rows(trial / name / "rows.jsonl", done["rows_sha256"])
    require([(r["n"], r["kind"]) for r in rows] == support, f"{name} support is incomplete or out of order")
    return (
        name,
        rows,
        {**receipt, "completion_sha256": sha(completion), "row_journal_sha256": done["rows_sha256"]},
    )


def verify_sources(inputs):
    sources = inputs["sources"]
    pairs = [(source["component"], source["source_id"]) for source in sources]
    require(
        len(pairs) == PHOTOS
        and len({sid for _, sid in pairs}) == PHOTOS
        and pairs == sorted(pairs)
        and len({group for group, _ in pairs}) == GROUPS,
        "Wrong frozen source or group support",
    )
    for source in sources:
        changed = source["changed_pixels"]
        require(
            is_sha(source["masks_sha256"])
            and set(changed) == set(KINDS)
            and all(type(value) is int and value >= 0 for value in changed.values())
            and changed["ai"] > 0
            and changed["original"] == 0,
            "Invalid footprint commitments",
        )
        sizes = set()
        for quality in QUALITIES:
            views = source["inputs"][str(quality)]
            require(set(views) == set(KINDS), "Missing frozen source view")
            for view in views.values():
                size = view["source_size"]
                require(
                    len(size) == 2
                    and all(type(x) is int and x > 0 for x in size)
                    and is_sha(view["input_sha256"])
                    and is_sha(view["decoded_rgb_sha256"]),
                    "Invalid frozen input geometry or byte commitments",
                )
                sizes.add(tuple(size))
        require(
            len(sizes) == 1 and max(changed.values()) <= math.prod(next(iter(sizes))),
            "Inconsistent geometry or footprint coverage",
        )
    return sources


def verify_pair(quality, sources, current, sensitive, measured):
    for old, new, row in zip(current, sensitive, measured, strict=True):
        n, kind = row["n"], row["kind"]
        source = sources[n]
        binding = source["inputs"][str(quality)][kind]
        height, width = reversed(binding["source_size"])
        require(
            row["quality"] == quality
            and row["component"] == source["component"]
            and row["changed_pixels"] == source["changed_pixels"][kind]
            and old["binding"] == binding == new["binding"]
            and old["status"] == new["status"] == "ok"
            and old["raw_map_shape"] == [512, 512]
            and new["raw_map_shape"] == [height, width]
            and all(is_sha(producer["raw_map_sha256"]) for producer in (old, new)),
            "Producer/readout pair diverges from frozen source, quality or geometry",
        )
        score = old["details"]["image_score"]
        geometry = new["details"]["geometry"]
        require(
            type(score) is float
            and 0 <= score <= 1
            and old["details"]["image_gate"] is (score >= 0.5)
            and old["details"]["pixel_threshold"] == 0.5
            and geometry["source_size"] == binding["source_size"]
            and geometry["source_rgb_sha256"] == binding["decoded_rgb_sha256"]
            and row["methods"]["current"]["ai"]["pixels"] == height * width,
            "Committed model gate or source geometry differs",
        )
        if score < 0.5:
            require(
                row["methods"]["current"]["ai"]["predicted_positive_pixels"] == 0,
                "Closed current gate cannot retain pixels",
            )


@dataclass(frozen=True)
class Counts:
    total: int
    truth: int
    selected: int
    overlap: int

    @property
    def iou(self):
        union = self.truth + self.selected - self.overlap
        return Fraction(self.overlap, union) if union else Fraction(1)

    @property
    def area(self):
        return Fraction(self.selected, self.total)

    @property
    def recall(self):
        return Fraction(self.overlap, self.truth) if self.truth else None

    @property
    def useful(self):
        union = self.truth + self.selected - self.overlap
        return self.truth > 0 and 10 * self.overlap >= 3 * union and 2 * self.overlap >= self.truth


def check_number(actual, expected, where):
    if isinstance(expected, Fraction | float):
        require(
            type(actual) in (float, int) and math.isclose(actual, expected, abs_tol=2e-12, rel_tol=0),
            f"{where} arithmetic differs",
        )
    else:
        require(type(actual) is type(expected) and actual == expected, f"{where} differs")


def counts(record, total, truth, selected=None):
    measure = Counts(
        *(
            record[k]
            for k in ("pixels", "reference_positive_pixels", "predicted_positive_pixels", "intersection")
        )
    )
    require(
        all(type(n) is int for n in (measure.total, measure.truth, measure.selected, measure.overlap))
        and measure.total == total > 0
        and measure.truth == truth
        and 0 <= measure.overlap <= min(measure.truth, measure.selected)
        and measure.truth + measure.selected - measure.overlap <= total
        and (selected is None or measure.selected == selected),
        "Impossible or mismatched pixel contingency",
    )
    derived = {
        "false_positive_pixels": measure.selected - measure.overlap,
        "false_negative_pixels": measure.truth - measure.overlap,
        "iou": measure.iou,
        "pixel_f1": Fraction(2 * measure.overlap, truth + measure.selected)
        if truth + measure.selected
        else Fraction(1),
        "precision": Fraction(measure.overlap, measure.selected) if measure.selected else None,
        "recall": measure.recall,
        "flagged_area_fraction": measure.area,
    }
    for key, value in derived.items():
        check_number(record[key], value, key)
    return measure


def shared_bounds(first, second, shared):
    for capacity, left, right, common in (
        (first.truth, first.overlap, second.overlap, shared.overlap),
        (
            first.total - first.truth,
            first.selected - first.overlap,
            second.selected - second.overlap,
            shared.selected - shared.overlap,
        ),
    ):
        require(
            max(0, left + right - capacity) <= common <= min(left, right),
            "Shared counts cannot be an intersection within the truth stratum",
        )


def derive_row(row):
    kind, footprint = row["kind"], row["changed_pixels"]
    require(
        kind in KINDS
        and type(footprint) is int
        and footprint >= 0
        and (kind != "ai" or footprint > 0)
        and (kind != "original" or footprint == 0)
        and set(row["methods"]) == set(METHODS),
        "Invalid measurement role or footprint",
    )
    total = row["methods"]["current"]["ai"]["pixels"]
    require(0 <= footprint <= total, "Footprint exceeds source pixels")
    truth = footprint if kind == "ai" else 0
    has_ordinary = kind in KINDS[2:] and footprint > 0
    ai, ordinary = {}, {}
    for name in METHODS:
        record = row["methods"][name]
        require(is_sha(record["mask_sha256"]), "Selected-mask commitment missing")
        ai[name] = counts(record["ai"], total, truth)
        if has_ordinary:
            ordinary[name] = counts(record["ordinary"], total, footprint, ai[name].selected)
        else:
            require(record["ordinary"] is None, "Only changed ordinary controls have an ordinary reference")
    shared_bounds(*(ai[name] for name in METHODS))
    if has_ordinary:
        shared_bounds(*(ordinary[name] for name in METHODS))
    return ai, ordinary


def average(values):
    return sum(values, Fraction()) / len(values)


def bootstrap_weights(groups):
    generator = np.random.Generator(np.random.PCG64(SEED))
    sample = generator.integers(0, groups, size=(RESAMPLES, groups))
    return np.stack([np.bincount(row, minlength=groups) for row in sample])


def percentile(sorted_values, probability):
    rank = probability * (len(sorted_values) - 1)
    low = int(rank)
    weight = float(rank - low)
    upper = min(low + 1, len(sorted_values) - 1)
    return float((1 - weight) * sorted_values[low] + weight * sorted_values[upper])


def paired_interval(candidate, current, weights):
    delta = [new - old for new, old in zip(candidate, current, strict=True)]
    # One dot product per resample against group multiplicities. The v189 scorer
    # instead materializes a gathered 10,000 × 48 matrix of measured differences.
    estimates = np.sort(weights @ np.asarray(delta, dtype=np.float64) / len(delta))
    return {
        "difference": average(delta),
        "pointwise_95pct_interval": [percentile(estimates, Fraction(n, 40)) for n in (1, 39)],
    }


def recompute_quality(rows, components, quality, weights):
    support = [(n, kind) for n in range(len(components)) for kind in KINDS]
    require(
        [(row["n"], row["kind"]) for row in rows] == support
        and all(row["quality"] == quality and row["component"] == components[row["n"]] for row in rows),
        "Cannot aggregate incomplete or mismatched source/quality support",
    )
    measures, group_members = defaultdict(list), defaultdict(list)
    for row in rows:
        measures[row["kind"]].append(derive_row(row))
    for n, component in enumerate(components):
        group_members[component].append(n)
    group_order = [group_members[key] for key in sorted(group_members)]
    require(weights.shape == (RESAMPLES, len(group_order)), "Bootstrap must pair entire source groups")

    def measure_group(values, statistic):
        return [statistic([values[n] for n in members]) for members in group_order]

    methods, group_metrics = {}, {}
    current = [item[0]["current"] for item in measures["ai"]]
    for name in METHODS:
        edited = [item[0][name] for item in measures["ai"]]
        useful = measure_group(edited, lambda group: average([int(m.useful) for m in group]))
        edited_iou = measure_group(edited, lambda group: average([m.iou for m in group]))
        group_metrics[name] = {"useful": useful, "edited_iou": edited_iou}
        controls = {}
        for kind in KINDS[1:]:
            views = [ai[name] for ai, _ in measures[kind]]
            footprints = [ordinary[name] for _, ordinary in measures[kind] if name in ordinary]
            flagged_groups = measure_group(views, lambda group: int(any(m.selected for m in group)))
            selected_area = measure_group(views, lambda group: average([m.area for m in group]))
            group_metrics[name][f"{kind}_false_group"] = flagged_groups
            group_metrics[name][f"{kind}_highlighted_area"] = selected_area
            controls[kind] = {
                "photos": len(views),
                "flagged_photos": sum(m.selected > 0 for m in views),
                "flagged_groups": sum(flagged_groups),
                "group_mean_highlighted_area_fraction": average(selected_area),
                "ordinary_footprints_with_changed_pixels": len(footprints),
                "ordinary_footprints_overlapped": sum(m.overlap > 0 for m in footprints),
                "mean_iou_with_ordinary_edit_if_changed": average([m.iou for m in footprints])
                if footprints
                else None,
                "mean_ordinary_edit_recall_if_changed": average([m.recall for m in footprints])
                if footprints
                else None,
            }
        methods[name] = {
            "edited_photos": len(edited),
            "edited_photos_flagged": sum(m.selected > 0 for m in edited),
            "edited_photos_with_any_overlap": sum(m.overlap > 0 for m in edited),
            "edited_photos_usefully_localized": sum(m.useful for m in edited),
            "edited_photos_without_a_region": sum(m.selected == 0 for m in edited),
            "new_useful_locations": sum(
                new.useful and not old.useful for old, new in zip(current, edited, strict=True)
            ),
            "lost_useful_locations": sum(
                old.useful and not new.useful for old, new in zip(current, edited, strict=True)
            ),
            "group_mean_useful_fraction": average(useful),
            "group_mean_edited_iou": average(edited_iou),
            "controls": controls,
        }
    for name, metrics in group_metrics.items():
        methods[name]["paired_against_current"] = {
            metric: paired_interval(values, group_metrics["current"][metric], weights)
            for metric, values in metrics.items()
        }
    return {
        "quality": quality,
        "photos": len(components),
        "source_groups": len(group_order),
        "methods": methods,
    }


def compare(actual, expected, field="quality_results"):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"{field} fields differ")
        for name, value in expected.items():
            compare(actual[name], value, f"{field}.{name}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"{field} shape differs")
        for n, (observed, target) in enumerate(zip(actual, expected, strict=True)):
            compare(observed, target, f"{field}[{n}]")
    else:
        check_number(actual, expected, field)


def review(trial, published, result_sha, output):
    freeze = verify_public()
    trial = trial.resolve()
    require(
        is_sha(result_sha) and sha(published) == result_sha == sha(trial / "summary.json"),
        "Published and local aggregate must match the exact expected SHA-256",
    )
    require(sha(trial / "freeze.json") == FREEZE_SHA, "Sealed trial has a different prospective freeze")
    report = read(published)
    contract = {
        "schema": "chipmunk.image-jpeg-diagnostic-result/189",
        "freeze_sha256": FREEZE_SHA,
        "unique_model_calls": 1180,
        "repeat_model_calls": 4,
        "unassessed_calls": 0,
        "serving_admitted": False,
        "calibration_performed": False,
        "ordinary_controls_have_zero_ai_truth": True,
        "scope": "Same SD2 editing family. Qualities share the same 48 source groups. Paired group bootstrap intervals "
        "are conditional and pointwise. Cue localization is separate from whole-file authorship confidence.",
    }
    for name, value in contract.items():
        check_number(report[name], value, name)
    profiles = report["phase_profile_sha256"]
    names = {
        f"{arm}-q{quality}-{number:02d}"
        for arm in (*ARMS, "readout")
        for quality in QUALITIES
        for number in range(PHASES)
    }
    require(
        set(profiles) == {"prepare", *names} and all(is_sha(value) for value in profiles.values()),
        "Published result lacks the complete 73-profile commitment",
    )
    monitor = freeze["code_sha256"][MONITOR]
    # Do not open a measurement journal until the aggregate and terminal seal are verified.
    receipts = {"seal": profile(trial / "seal-profile", ["seal", str(trial), FREEZE_SHA], monitor)}
    inputs_path = trial / "inputs.json"
    require(
        sha(inputs_path) == freeze["private_input_bindings_sha256"], "Frozen private input manifest changed"
    )
    inputs = read(inputs_path)
    require(isinstance(inputs["raw_root"], str), "Prepared raw root must be recorded")
    receipts["prepare"] = profile(
        trial.parent / f"{trial.name}-prepare-profile",
        ["prepare", str(trial), inputs["raw_root"]],
        monitor,
        profiles["prepare"],
    )
    sources = verify_sources(inputs)
    components = [source["component"] for source in sources]
    weights = bootstrap_weights(GROUPS)
    calculated = {}
    for quality in QUALITIES:
        measurements = []
        for number in range(PHASES):
            views = {}
            for arm in (*ARMS, "readout"):
                name, rows, receipt = phase(trial, quality, number, arm, profiles, monitor)
                views[arm] = rows
                receipts[name] = receipt
            verify_pair(quality, sources, views["baseline"], views["candidate"], views["readout"])
            measurements.extend(views["readout"])
        calculated[str(quality)] = recompute_quality(measurements, components, quality, weights)
    compare(report["quality_results"], calculated)
    result = {
        "schema": "chipmunk.image-jpeg-aggregate-review/207",
        "v189_freeze_sha256": FREEZE_SHA,
        "published_result_sha256": result_sha,
        "v42_freeze_sha256": freeze["v42_freeze_sha256"],
        "v46_freeze_sha256": freeze["v46_freeze_sha256"],
        "review_code_sha256": sha(Path(__file__)),
        "paired_phases_verified": len(QUALITIES) * PHASES,
        "producer_row_commitments_verified": 1180,
        "readout_rows_recomputed": 590,
        "matched_published_aggregate": True,
        "float_absolute_tolerance": 2e-12,
        "phase_receipts": receipts,
        "image_or_mask_pixels_replayed": False,
        "neural_calls": 0,
        "serving_admitted": False,
        "scope": "Local arithmetic review of complete recorded measurements. The JPEG qualities reuse 48 source groups. "
        "Intervals are paired, pointwise and conditional on the fixed sources and models.",
    }
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check-public")
    audit = commands.add_parser("review")
    for name in ("trial", "published"):
        audit.add_argument(name, type=Path)
    audit.add_argument("result_sha256")
    audit.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "check-public":
        freeze = verify_public()
        print(
            json.dumps(
                {
                    "v189_freeze_sha256": FREEZE_SHA,
                    "v42_freeze_sha256": freeze["v42_freeze_sha256"],
                    "v46_freeze_sha256": freeze["v46_freeze_sha256"],
                    "frozen_code_files_verified": len(freeze["code_sha256"]),
                }
            )
        )
    else:
        result = review(args.trial, args.published, args.result_sha256, args.output)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "published_result_sha256",
                        "paired_phases_verified",
                        "readout_rows_recomputed",
                        "matched_published_aggregate",
                    )
                }
            )
        )


if __name__ == "__main__":
    main()
