"""Verify local stream-presence decisions and native source clocks without models."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import wave
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_bytes())


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root: Path, parent: Path, published: Path | None = None) -> dict:
    freeze_path = root / "stream-presence-freeze.json"
    freeze = load(freeze_path)
    points_path = root / "points.json"
    points = load(points_path)
    fit_path, confirm_path = root / "fit-private.json", root / "confirm-private.json"
    dev, held = load(fit_path), load(confirm_path)
    archives = load(root / "natural-source-plan.json")
    parent_manifest = load(parent / "manifest.json")
    if (points["freeze_sha256"] != sha(freeze_path) or points["development_sha256"] != sha(fit_path)
            or freeze["parent_manifest_sha256"] != sha(parent / "manifest.json")
            or freeze["natural_selection_sha256"] != sha(root / "natural-source-plan.json")):
        raise ValueError("Frozen source or fitting commitments differ")
    if Counter(row["panel"] for row in dev + held) != freeze["cases"]:
        raise ValueError("A planned analysis row is missing or additional")
    if len({row["key"] for row in dev + held}) != len(dev + held):
        raise ValueError("A planned audio row was duplicated")
    dev_groups = {row["group"] for row in dev}
    held_groups = {row["group"] for row in held if row["panel"] == "constructed_confirmation"}
    expected_development = set()
    for language, quota in {"et": 1, "fi": 2, "lt": 1, "nl": 3, "sl": 1}.items():
        choices = [group["group"] for group in parent_manifest["groups"] if group["language"] == language]
        expected_development.update(sorted(choices, key=lambda value: hashlib.sha256(f"separation/1:{value}".encode()).hexdigest())[:quota])
    if len(dev_groups) != 8 or len(held_groups) != 13 or dev_groups & held_groups or any(row["panel"] != "development" for row in dev):
        raise ValueError("Development and confirmation sources overlap or have changed")
    if dev_groups != expected_development:
        raise ValueError("Development groups do not match the frozen salted roster")
    by_key = {row["key"]: row for row in dev + held}
    expected_roster = [(f"c{row['id']:03d}", row["pcm_sha256"]) for row in parent_manifest["rows"]
                       if not row["role_rotation"] and row["condition"] in {"two", "three_quiet", "three_faint"}]
    natural_sources = {row["group"]: row["sha256"] for row in freeze["native_sources"]}
    natural_verified = 0
    for group, room in enumerate(archives["rooms"]):
        source = root / "natural-raw" / f"{room['meeting']}.wav"
        if sha(source) != natural_sources[group]:
            raise ValueError("The archived microphone source has changed")
        with wave.open(str(source), "rb") as file:
            if file.getparams()[:3] != (1, 2, 16000):
                raise ValueError("Incorrect natural microphone PCM geometry")
            for ordinal, case in enumerate(room["cases"]):
                key = f"n{group}-{ordinal:02d}"
                row = by_key[key]
                start = 160 * case["begin_centiseconds"]
                file.setpos(start)
                data = file.readframes(48000)
                digest = hashlib.sha256(data).hexdigest()
                if (len(data) != 96000 or row["panel"] != "natural_confirmation" or row["group"] != group or row["delivery"] != "native_mono"
                        or row["expected_voices"] != case["expected_voices"] or row["start_sample"] != start or digest != row["pcm_sha256"]):
                    raise ValueError("Natural source interval or activity-label binding differs")
                natural_verified += 1
                expected_roster.append((key, digest))
    if (len(expected_roster) != len(by_key)
            or hashlib.sha256(str(expected_roster).encode()).hexdigest() != freeze["input_roster_sha256"]):
        raise ValueError("Reconstructed input roster differs from the frozen schedule")
    for key, digest in expected_roster:
        row = by_key[key]
        if row["pcm_sha256"] != digest:
            raise ValueError("Output does not refer to the planned PCM")
        if key.startswith("c"):
            index = int(key[1:])
            source = parent / "pcm" / f"{index:03d}.pcm"
            original = parent_manifest["rows"][index]
            panel = "development" if original["group"] in expected_development else "constructed_confirmation"
            if (source.stat().st_size != 96000 or sha(source) != digest or row["panel"] != panel
                    or any(row.get(field) != value for field, value in original.items())):
                raise ValueError("Constructed source or known voice-count label differs")

    fitted = {arm: defaultdict(list) for arm in points["points"]}
    for row in dev:
        if row["expected_voices"] == 2:
            for arm, score in row["observation"]["scores"].items():
                fitted[arm][row["group"]].append(score if score is not None else 0.)
    if {arm: sorted(max(values) for values in groups.values())[-2] for arm, groups in fitted.items()} != points["points"]:
        raise ValueError("Operating points differ from development controls")
    verified_rollups = 0
    aggregates = {}
    for stage, rows, path in (("fit", dev, fit_path), ("confirm", held, confirm_path)):
        report = load(root / f"{stage}-aggregate.json")
        if (report["private_response_sha256"] != sha(path) or report["freeze_sha256"] != sha(freeze_path)
                or report["points_sha256"] != sha(points_path) or report["forwards"] != 3 * len(rows)):
            raise ValueError("Saved aggregate has a mismatched response commitment")
        rollups, groups = defaultdict(Counter), defaultdict(set)
        for row in rows:
            obs = row["observation"]
            active = obs["activity_samples"]
            stable = obs["minimum_stability"]
            novel = obs["minimum_unexplained_energy"]
            rms = obs["minimum_relative_rms"]
            if type(active) is not int or not 0 <= active <= 48000 or any(not math.isfinite(value) for value in (rms, stable, novel)):
                raise ValueError("Invalid saved signal measures")
            passed_activity = active >= 2560 and rms >= .04
            scores = {"activity": active / 48000, "aligned_stability": stable, "stable_independent": min(stable, novel)}
            if not passed_activity:
                scores = dict.fromkeys(scores)
            if obs["scores"] != scores or obs["qualified"] is not passed_activity or obs["raw_energy"] is not (rms >= .1):
                raise ValueError("Saved scores differ from the frozen measurement rules")
            flags = {"raw_energy": rms >= .1, **{arm: value is not None and value > points["points"][arm] for arm, value in scores.items()}}
            label = "two" if row["expected_voices"] == 2 else "three"
            for panel in (row["panel"], f"{row['panel']}/{row['delivery']}"):
                tally = rollups[panel]
                tally[f"planned_{label}"] += 1
                tally[f"activity_qualified_{label}"] += int(passed_activity)
                for arm, flag in flags.items():
                    tally[f"{arm}_{label}_flag"] += int(flag)
                    if flag:
                        groups[f"{panel}/{arm}/{label}"].add(row["group"])
                if row["panel"] != "natural_confirmation":
                    reference = row["reference_audit"]
                    if label == "two":
                        tally[f"unmatched_{reference['extra_output_category']}"] += 1
                    else:
                        recovered = reference["third_recovered"]
                        tally["reference_third_recovered"] += int(recovered)
                        tally["reference_third_recovered_and_compound_flagged"] += int(recovered and flags["stable_independent"])
        def advances(counts):
            return (2 * counts["stable_independent_three_flag"] >= counts["planned_three"] > 0
                    and 6 * counts["stable_independent_two_flag"] <= counts["planned_two"] and counts["planned_two"] > 0)

        advancement = advances(rollups["constructed_confirmation"]) and advances(rollups["natural_confirmation"])
        if (dict(rollups) != report["totals"] or {key: len(items) for key, items in groups.items()} != report["affected_groups"]
                or report["advance"] is not advancement):
            raise ValueError("Published decision arithmetic differs from the private receipt")
        verified_rollups += len(rollups)
        aggregates[stage] = report
    if published is not None:
        public = load(published)
        if public["freeze"] != freeze or public["points"] != points or any(public[stage] != data for stage, data in aggregates.items()):
            raise ValueError("Public result differs from the locally verified records")
    return {"verified_pcm_ranges": len(by_key), "verified_natural_source_clocks": natural_verified,
            "verified_development_groups": len(dev_groups), "verified_constructed_confirmation_groups": len(held_groups),
            "verified_condition_rollups": verified_rollups, "verified_advancement_outcome": aggregates["confirm"]["advance"]}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("run", type=Path)
    cli.add_argument("parent", type=Path)
    cli.add_argument("--published", type=Path)
    opts = cli.parse_args()
    print(json.dumps(verify(opts.run, opts.parent, opts.published), indent=2, sort_keys=True))
