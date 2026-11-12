"""Freeze a separate comparison using retained controls after a Mac restart."""

from __future__ import annotations

import argparse
import json
import wave
from collections import Counter
from pathlib import Path

from scripts import audio_spectra_background_transfer as trial
from scripts import prepare_audio_channel_speech_gap_confirmation as room_plan
from scripts.audio_short_edit_cases_v172 import exact_source, sources

BASE = trial.ROOT / "backend/data/independent-evaluation"
POINT_SOURCE = BASE / "audio-matched-background-panel-v248/bind-00/output"
POINT_SCORE = BASE / "audio-matched-background-score-v250"
GENUINE_COUNT = 16
SYNTHETIC_QUOTAS = {"clone": 3, "expressive": 3, "stock": 3}
sha, read, write_new = trial.sha, trial.read, trial.write_new


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def retained_point_controls():
    cases_path = POINT_SOURCE / "private-cases.jsonl"
    cases_public = trial.PANEL / "audio-matched-background-panel-complete-v248.json"
    baseline_public = trial.PANEL / "audio-matched-background-score-complete-v250.json"
    manifest, scoring = read(cases_public), read(baseline_public)
    freeze_path = POINT_SCORE / "freeze.json"
    if (sha(cases_path) != manifest["private_cases_sha256"]
            or sha(freeze_path) != scoring["freeze_sha256"]
            or read(freeze_path)["v248_case_journal_sha256"] != sha(cases_path)):
        raise ValueError("The retained point PCM and baseline have different commitments")
    decisions = {}
    for index, profile_sha in enumerate(scoring["baseline_profile_sha256"]):
        batch = POINT_SCORE / "baseline" / f"extract-{index:02}"
        profile = batch / "profile.json"
        receipt = batch / "output/result.json"
        journal = batch / "output/private-observations.jsonl"
        if (sha(profile) != profile_sha or sha(receipt) != read(profile)["worker_receipt_sha256"]
                or read(receipt)["private_observations_sha256"] != sha(journal)
                or read(receipt)["freeze_sha256"] != scoring["freeze_sha256"]):
            raise ValueError("The retained baseline batch differs from its public receipt")
        for row in records(journal):
            if row["case"] in decisions:
                raise ValueError("Duplicate point baseline case")
            decisions[row["case"]] = row
    result = []
    for row in records(cases_path):
        if row["arm"] != "genuine" or row["status"] != "prepared":
            continue
        case = row["case"]
        old = decisions[case]
        source = POINT_SOURCE / "prepared" / f"{case}.wav"
        with wave.open(str(source), "rb") as stream:
            if stream.getparams()[:4] != (1, 2, trial.RATE, trial.COUNT):
                raise ValueError("Retained point WAV format differs")
            pcm = stream.readframes(trial.COUNT)
        if (sha(source) != row["wav_sha256"] or sha(pcm) != row["pcm_sha256"]
                or any(row[k] != old[k] for k in ("case", "condition", "arm"))):
            raise ValueError("Retained point WAV, PCM and baseline differ")
        raw = old.get("raw")
        decision = old["decision"] if raw is not None else "unassessed"
        expected_windows = [[0, trial.COUNT]] if raw is not None else []
        expected_status = "ok" if raw is not None else "not_applicable"
        if (old["route_status"] != expected_status or old["windows"] != expected_windows
                or decision not in {"ai", "original", "inconclusive", "unassessed"}):
            raise ValueError("The retained point uses different speech admission")
        result.append((pcm, {"family": "genuine", "group": row["group"], "condition": row["condition"],
                             "source_sha256": row["source_wav_sha256"], "source_pcm_sha256": row["source_prefix_sha256"],
                             "baseline": {"pcm_sha256": sha(pcm), "raw": raw, "decision": decision}}))
    if len(result) != 115 or len({row["group"] for _, row in result}) != 23:
        raise ValueError("The retained genuine point cohort is incomplete")
    return result, {"point_source_result_sha256": sha(cases_public),
                    "point_baseline_result_sha256": sha(baseline_public),
                    "point_unavailable_views": 5}


def retained_confirmation_foregrounds():
    public = read(trial.PANEL / "audio-short-edit-materialization-complete-v169c.json")
    manifest = sources.MEDIA / "holdout/private-manifest.json"
    if sha(manifest) != public["roles"]["holdout"]["private_manifest_sha256"]:
        raise ValueError("The retained confirmation foreground manifest differs")
    retained = [row for row in read(manifest)["foregrounds"] if row["status"] == "qualified"]
    selected = [row for row in retained if row["lane"] == "genuine"]
    if len(selected) != GENUINE_COUNT or len({row["voice"] for row in selected}) != 8:
        raise ValueError("Expected eight genuine speakers with two sources each")
    for family, count in SYNTHETIC_QUOTAS.items():
        eligible = sorted((row for row in retained if row["lane"] == family),
                          key=lambda row: sha(f"chipmunk/spectra-recovery/1:{row['slot']}".encode()))
        picked = eligible[:count]
        if len({row["voice"] for row in picked}) != count or len({row["group_id"] for row in picked}) != count:
            raise ValueError("Confirmation synthetic voices must be distinct within each family")
        selected.extend(picked)
    return [(exact_source(row, "holdout"), row) for row in selected], sha(manifest)


def reconstructed_room_cuts(ami):
    annotations = ami / "ami_public_manual_1.6.2.zip"
    prior = read(trial.PANEL / "audio-channel-speech-gap-transfer-2026-11-11.json")
    if sha(annotations) != trial.ANNOTATION_SHA256:
        raise ValueError("The pinned AMI manual annotations differ")
    chosen = room_plan.choose(annotations)
    excluded = {room: [start * room_plan.study.STEP for _, start, _ in chosen[room]]
                for room, _, _ in trial.ROOMS.values()}
    cuts = trial.verified_room_cuts(ami, annotations, prior["participant_disjoint_confirmation"]["freeze"]["sources"], excluded)
    if [details["annotated_overlap_ms"] for _, details in cuts.values()] != [0, 2400]:
        raise ValueError("The deterministic room cuts differ from the recorded pre-restart annotation counts")
    return cuts, {"room_selection_code_sha256": sha(Path(room_plan.__file__)),
                  "ami_annotation_sha256": trial.ANNOTATION_SHA256}


def prepare(folder, assets, ami):
    verified_assets = trial.pinned_assets(assets)
    point, point_binding = retained_point_controls()
    held, held_manifest = retained_confirmation_foregrounds()
    cuts, room_binding = reconstructed_room_cuts(ami)
    point_groups = {row["group"] for _, row in point}
    point_fingerprints = {row[key] for _, row in point for key in ("source_sha256", "source_pcm_sha256")}
    held_fingerprints = {fingerprint for raw, row in held for fingerprint in (row["original_sha256"], sha(raw))}
    if point_groups & {row["group_id"] for _, row in held} or point_fingerprints & held_fingerprints:
        raise ValueError("Point and confirmation foregrounds share a source identity")
    (folder / "pcm").mkdir(parents=True)
    rows = []

    def save(raw, role, condition, family, group, baseline=None, **evidence):
        index = len(rows)
        (folder / "pcm" / f"{index:03}.pcm").write_bytes(raw)
        rows.append({"id": index, "role": role, "condition": condition, "family": family, "group": group,
                     "pcm_sha256": sha(raw), "baseline": baseline, **evidence})

    for raw, row in point:
        save(raw, "point", **row)
    for clean, row in held:
        metadata = {"source_sha256": row["original_sha256"], "source_pcm_sha256": sha(clean)}
        save(clean, "prospective", "clean", row["lane"], row["group_id"], **metadata)
        for condition, (raw, background) in cuts.items():
            pcm, scale, gain = trial.mix(clean, raw)
            save(pcm, "prospective", condition, row["lane"], row["group_id"], **metadata,
                 clean_pcm_sha256=sha(clean), background=background, scale=scale, common_gain=gain)
    for condition, (raw, background) in cuts.items():
        save(raw, "prospective", f"isolated_{condition}", "background", condition, background=background)
    write_new(folder / "roster.json", rows)
    binding = {
        "schema": "chipmunk.audio-independent-background-recovery-comparison/1",
        "runner_sha256": sha(Path(trial.__file__)),
        "recovery_preparation_sha256": sha(Path(__file__)),
        "roster_sha256": sha(folder / "roster.json"),
        "rule": trial.RULE, "candidate_assets": verified_assets,
        "incumbent_assets": trial.incumbent.ASSETS,
        "incumbent_calibration_sha256": sha(trial.incumbent.CALIBRATION),
        "confirmation_foreground_manifest_sha256": held_manifest,
        "prospective_backgrounds": [details for _, details in cuts.values()],
        "planned": dict(Counter(row["role"] for row in rows)),
        "input_samples": trial.COUNT, "candidate_samples": trial.SHAPE,
        **point_binding, **room_binding,
    }
    write_new(folder / "freeze.json", binding)
    print(binding["planned"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("folder", "assets", "ami"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    prepare(args.folder, args.assets, args.ami)
