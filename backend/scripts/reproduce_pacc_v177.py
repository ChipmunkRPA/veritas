"""Portable independent replay of the publisher-prepared PACC-P genuine controls.

Qualification and inference are separate, explicit commands. All per-input
records stay in the caller's private output directory; no old freeze is changed.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import tarfile
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import audio_pacc_false_alarm_v177 as score_reference
from scripts import audio_pacc_sample_v176 as source

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "docs/evaluation-panels/audio-pacc-sample-freeze-v176.json"
SCORE_FREEZE = ROOT / "docs/evaluation-panels/audio-pacc-false-alarm-freeze-v177.json"
REFERENCE = ROOT / "docs/evaluation-panels/audio-pacc-false-alarm-result-v177.json"
ROUTE_FREEZE = ROOT / "docs/evaluation-panels/audio-matched-edit-diagnostic-freeze-v174.json"
COMMITMENTS = {
    FREEZE: "e9aed0644d698d6e0fcd239520984565b1ed552e43d4666594a8bdc8daf2a073",
    SCORE_FREEZE: "eb36498738dcb9d95189ac6a7eeaf3c8c3938c867223ea100249a8a18bfc76b8",
    ROUTE_FREEZE: "896d996b55b5dbb012dc075018f1c699503f49576f81241dee06a501c4923748",
    REFERENCE: "048d96cd24cd834d95186ac325358234e50c564deddf1e913ba858a6999a95ad",
}
RATE_CONTEXT = "source_context"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    # Names, per-source scores and fingerprints must remain in this private directory.
    with path.open("x") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify_code():
    for path, digest in COMMITMENTS.items():
        require(path.is_file() and source.sha(path) == digest, "Public reference commitment differs")
    for freeze in (FREEZE, SCORE_FREEZE, ROUTE_FREEZE):
        for name, digest in read(freeze)["code_sha256"].items():
            path = ROOT / name
            require(path.is_file() and source.sha(path) == digest, f"Frozen scoring source differs: {name}")


def environment():
    names = read(SCORE_FREEZE)["runtime_packages"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    binaries = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    executable_hashes = {name: source.sha(Path(path)) if path else None for name, path in binaries.items()}
    return {"platform": platform.platform(), "python": platform.python_version(), "packages": versions,
            "packages_match_v177": versions == names, "ffmpeg_sha256": executable_hashes,
            "ffmpeg_matches_v176": executable_hashes == read(FREEZE)["ffmpeg_tool_sha256"]}


def source_files(directory):
    frozen = read(FREEZE)
    archive = directory / "pacc-p_sample.tar"
    require(archive.is_file() and not archive.is_symlink() and archive.stat().st_size == source.SAMPLE_SIZE
            and source.sha(archive) == frozen["sample_sha256"] == source.SAMPLE_SHA256,
            "Pinned publisher sample is missing or differs")
    for name, digest in frozen["publisher_metadata_sha256"].items():
        path = directory / name
        require(path.is_file() and not path.is_symlink() and source.sha(path) == digest,
                f"Publisher metadata differs: {name}")
    require(f"{source.SAMPLE_SHA256}  {archive.name}" in (directory / "SHA256SUMS").read_text().splitlines(),
            "Publisher checksum record differs")
    return archive


def source_plan(directory):
    """Build the full sample roster without the historical private voice catalogue."""
    archive_path = source_files(directory)
    with tarfile.open(archive_path, "r:") as archive:
        entries = archive.getmembers()
        require(len(entries) == len({entry.name for entry in entries}) == 838 and all(e.isfile() for e in entries),
                "Publisher sample layout differs")
        sources = source.rows(archive, "sources/metadata.csv")
        keys = {(row["pool"], row["file_id"]) for row in sources}
        require(len(keys) == len(sources) == 14 and Counter(row["pool"] for row in sources) == source.POOLS,
                "Publisher source roster differs")
        credits = source.sha(directory / "attribution.csv")
        treatments = {}
        for condition, snr in source.CONDITIONS.items():
            if snr is None:
                continue
            rows = source.rows(archive, f"pacc-p/{condition}/params.csv")
            treatment = {(row["pool"], row["file_id"]): row for row in rows}
            require(len(rows) == len(treatment) == len(keys) and treatment.keys() == keys
                    and source.digest(source.member(archive, f"pacc-p/{condition}/attribution.csv")) == credits,
                    "Publisher treatment roster or attribution differs")
            treatments[condition] = treatment
        credit_rows = source.rows(archive, "pacc-p/babble_snr00/attribution.csv")
        require(len(credit_rows) == 4696 and {row["licence"] for row in credit_rows} == source.SOURCE_LICENSES,
                "Publisher credit inventory differs")
        plan = []
        for index, row in enumerate(sorted(sources, key=lambda item: (item["pool"], item["file_id"]))):
            pool, file_id = row["pool"], row["file_id"]
            family = "AMI" if pool == "AMI" else "VCTK"
            context = row["ami_meeting"] if family == "AMI" else row["vctk_utterance"]
            require(bool(context and row["speaker_id"]), "Publisher dependence metadata missing")
            plan.append({"slot": f"source-{index:02d}", "pool": pool, "family": family,
                         RATE_CONTEXT: context, "source_rate": int(row["source_sample_rate"]),
                         "members": {"clean": source.match_member(row["file"], ("sources", pool), file_id),
                                     **{name: source.aligned_params(treatments[name][pool, file_id], name, row, snr)
                                        for name, snr in source.CONDITIONS.items() if snr is not None}}})
        ami = {item[RATE_CONTEXT] for item in plan if item["pool"] == "AMI"}
        first = {item[RATE_CONTEXT] for item in plan if item["pool"] == "VCTK_mic1"}
        second = {item[RATE_CONTEXT] for item in plan if item["pool"] == "VCTK_mic2"}
        require(len(ami) == 4 and len(first) == len(second) == 5 and first == second,
                "Publisher source contexts and microphone pairings differ")
    return plan


def qualify(publisher, output):
    require(output.is_dir() and not output.is_symlink() and not any(output.iterdir()),
            "Output must be an existing empty private directory")
    require(not output.resolve().is_relative_to(publisher.resolve()) and not publisher.resolve().is_relative_to(output.resolve()),
            "Publisher and output directories must be separate")
    verify_code()
    plan = source_plan(publisher)  # All publisher commitments are checked before any decoding.
    os.chmod(output, 0o700)
    media = output / "prepared"
    media.mkdir(mode=0o700)
    observed = []
    with tarfile.open(publisher / "pacc-p_sample.tar", "r:") as archive, TemporaryDirectory(dir=output) as scratch:
        for item in plan:
            for condition in source.CONDITIONS:
                name = item["members"][condition]
                incoming = Path(scratch) / ("input" + Path(name).suffix)
                outgoing = media / f"{item['slot']}-{condition}.wav"
                row = {key: item[key] for key in ("slot", "family", "pool")}
                row["condition"] = condition
                try:
                    row.update(source.prepare(incoming, outgoing, source.member(archive, name), item["source_rate"]))
                    os.chmod(outgoing, 0o600)
                except (OSError, ValueError) as exc:
                    row.update(status="unavailable", error_type=type(exc).__name__)
                observed.append(row)
    write(output / "plan.private.json", plan)
    write(output / "qualification.private.json", observed)
    write(output / "qualification.json", {"planned_cases": len(observed),
          "status_counts": dict(Counter(row["status"] for row in observed)), "neural_calls": 0,
          "source_sha256": source.SAMPLE_SHA256, "historical_catalogue_checked": False,
          "plan_sha256": source.sha(output / "plan.private.json"),
          "observations_sha256": source.sha(output / "qualification.private.json"), "environment": environment()})


def compare(rows, plan):
    """Compare source-family aggregate directions and route transitions, not scores."""
    require([(r["slot"], r["condition"]) for r in rows] ==
            [(item["slot"], condition) for item in plan for condition in source.CONDITIONS],
            "Each planned case requires one ordered result")
    reference = read(REFERENCE)["genuine_controls"]
    by_key = {(row["slot"], row["condition"]): row for row in rows}
    groups = {}
    for group in ("AMI", "VCTK", "VCTK_mic1", "VCTK_mic2"):
        items = [item for item in plan if group in {item["family"], item["pool"]}]
        current = {condition: [by_key[item["slot"], condition] for item in items] for condition in source.CONDITIONS}
        conditions, pairs = {}, {}
        for condition, values in current.items():
            metric = score_reference.condition_metrics(values)
            expected = reference[group]["by_condition"][condition]
            conditions[condition] = {"directions": metric["directions"], "route_statuses": metric["route_statuses"],
                                     "matches_reference": all(metric[key] == expected[key] for key in ("directions", "route_statuses"))}
            if condition != "clean":
                pair = score_reference.paired_metrics(current["clean"], values)
                old = reference[group]["paired_vs_clean"][condition]
                keys = ("planned_pairs", "direction_transitions", "route_status_transitions")
                pairs[condition] = {**{key: pair[key] for key in keys},
                                    "matches_reference": all(pair[key] == old[key] for key in keys)}
        groups[group] = {"by_condition": conditions, "paired_vs_clean": pairs}
    return groups


def run(output, models):
    require(output.is_dir() and not output.is_symlink() and not (output / "scores.private.json").exists()
            and not (output / "comparison.json").exists(), "Qualified output required; never overwrite an earlier run")
    verify_code()
    lock = read(output / "qualification.json")
    require(lock.get("source_sha256") == source.SAMPLE_SHA256 and lock.get("planned_cases") == 98
            and lock.get("plan_sha256") == source.sha(output / "plan.private.json")
            and lock.get("observations_sha256") == source.sha(output / "qualification.private.json"),
            "Qualification commitments differ")
    plan, qualified = read(output / "plan.private.json"), read(output / "qualification.private.json")
    require(len(plan) == 14 and len({p["slot"] for p in plan}) == 14 and len(qualified) == 98
            and Counter(p["pool"] for p in plan) == source.POOLS
            and all(p["family"] == ("AMI" if p["pool"] == "AMI" else "VCTK") for p in plan)
            and all(re.fullmatch(r"source-[0-9]{2}", p["slot"]) for p in plan)
            and all(q["status"] in {"qualified", "unavailable"} for q in qualified)
            and [(q["slot"], q["family"], q["pool"], q["condition"]) for q in qualified]
            == [(p["slot"], p["family"], p["pool"], c) for p in plan for c in source.CONDITIONS],
            "Complete ordered qualification plan required")
    require(not models.is_symlink() and not (output / "prepared").is_symlink(), "Direct model and prepared directories required")
    expected = read(SCORE_FREEZE)["local_asset_commitments"]
    for name, digest in expected.items():
        path = models / name
        require(path.is_file() and not path.is_symlink() and source.sha(path) == digest, f"Pinned model asset differs: {name}")
    # These imports and model calls occur only in the explicit scoring command.
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from app.detectors.audio.decode import read_wav
    from app.detectors.audio.upgraded import SpeechActivity
    from app.engine.boundary import Deadline
    from scripts.audio_matched_edit_diagnostic_v174 import score_voice
    from scripts.run_audio_specialists_v15 import load_voice

    model = load_voice({"weights": str(models), "assets": expected})
    activity = SpeechActivity(models / "silero_vad.jit")
    rows = []
    for item in qualified:
        row = {key: item[key] for key in ("slot", "family", "condition")}
        if item["status"] != "qualified":
            row.update(route_status="preparation_unavailable", decision="unavailable")
        else:
            path = output / "prepared" / f"{item['slot']}-{item['condition']}.wav"
            require(path.is_file() and not path.is_symlink() and source.sha(path) == item["wav_sha256"],
                    "Prepared WAV differs from the qualification")
            pcm = read_wav(path.read_bytes())
            require(len(pcm) == source.SAMPLES and source.digest(pcm.astype("<f4").tobytes()) == item["pcm_sha256"],
                    "Prepared PCM differs from the qualification")
            try:
                row.update(score_voice(pcm, model, activity, Deadline(120)))
            except Exception as exc:  # Record a failed input without treating it as an ordinary speech-gate rejection.
                row.update(route_status="scoring_error", decision="unavailable", error_type=type(exc).__name__)
        rows.append(row)
    groups = compare(rows, plan)
    all_match = all(entry["matches_reference"] for group in groups.values()
                    for section in group.values() for entry in section.values())
    write(output / "scores.private.json", rows)
    write(output / "comparison.json", {"planned_cases": len(rows), "planned_pairs": 84, "groups": groups,
          "all_aggregate_status_counts_match": all_match,
          "historical_catalogue_checked": False, "environment": environment(),
          "scope": "Independent aggregate replay; original per-input results and historical independence are not verified."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("qualify", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publisher", type=Path)
    parser.add_argument("--models", type=Path)
    args = parser.parse_args()
    if args.command == "qualify":
        require(args.publisher is not None and args.models is None, "Qualification requires only --publisher and --output")
        qualify(args.publisher, args.output)
    else:
        require(args.models is not None and args.publisher is None, "Scoring requires only --models and --output")
        run(args.output, args.models)


if __name__ == "__main__":
    main()
