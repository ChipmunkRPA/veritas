"""Replay fixed-gain audio repeat detection using only public AMI microphone WAVs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import wave
from contextlib import ExitStack
from pathlib import Path
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np

from app.evidence import audio_exact_repetition, audio_gain_repetition, audio_gain_repetition_binding
from scripts import audio_gain_repetition_transfer as trial

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "docs/evaluation-panels/audio-gain-repetition-transfer-2026-11-13.json"
PUBLIC = ROOT / "docs/evaluation-panels/audio-gain-public-replay-2026-11-13.json"
MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
COHORTS = {"previous": ("ES2004a", "IS1009a"), "additional": ("ES2005a", "ES2008a")}
SOURCE_MAX = 40 * 1024**2
CODE = (Path(__file__), Path(trial.__file__), Path(audio_exact_repetition.__file__),
        Path(audio_gain_repetition.__file__), Path(audio_gain_repetition_binding.__file__))


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def schedule():
    return {"schema": "chipmunk.ami-gain-replay-plan/1", "cohorts": {key: list(value) for key, value in COHORTS.items()},
            "sources": {f"{session}.{mic}.wav": f"{MIRROR}/{session}/audio/{session}.{mic}.wav"
                        for meetings in COHORTS.values() for session in meetings for mic in trial.MICROPHONES},
            "window_starts_s": list(trial.GRID), "window_samples": trial.DURATION,
            "variants": [*trial.RATIOS, "exact_copy", *trial.CONTROLS], "boundary_tolerance_samples": trial.BOUNDARY_TOLERANCE,
            "production_decode_window_indices_per_session": [0, 11],
            "prior_receipt_sha256": trial.digest(PANEL),
            "code_sha256": {str(path.relative_to(ROOT)): trial.digest(path) for path in CODE}}


def receive(url, path):
    request = Request(url, headers={"Accept-Encoding": "identity", "User-Agent": "Chipmunk-Verify-public-AMI-replay/1"})
    minimum = 2 * (trial.GRID[-1] * trial.RATE + trial.DURATION)
    deadline = time.monotonic() + 120
    digest = hashlib.sha256()
    length = 0
    partial = path.with_suffix(".part")
    with build_opener(NoRedirect()).open(request, timeout=20) as response:
        size = response.headers.get("Content-Length")
        if (response.status != 200 or response.geturl() != url or response.headers.get("Content-Encoding", "identity") != "identity"
                or (size is not None and (not size.isdecimal() or not minimum <= int(size) <= SOURCE_MAX))):
            raise ValueError(f"Unexpected AMI response for {path.name}")
        with partial.open("xb") as stream:
            while chunk := response.read1(256 * 1024):
                length += len(chunk)
                if time.monotonic() > deadline or length > SOURCE_MAX:
                    raise ValueError(f"AMI source exceeded download bounds: {path.name}")
                digest.update(chunk)
                stream.write(chunk)
        if not minimum <= length <= SOURCE_MAX or (size is not None and int(size) != length):
            raise ValueError(f"Incomplete AMI source: {path.name}")
    return {"bytes": length, "sha256": digest.hexdigest(), "url": url}


def pin_matches(expected, actual):
    return all(expected[key] == actual[key] for key in ("bytes", "sha256", "url"))


def acquire(directory, plan, expected):
    if trial.read(directory / "plan.json") != plan:
        raise ValueError("Source plan changed after freezing")
    records = {}
    prior = trial.read(PANEL)["freeze"]["confirmation_sources"]
    for name, url in plan["sources"].items():
        path = directory / name
        if path.exists():
            raise ValueError(f"Use a new scratch directory: {name}")
        record = receive(url, path)
        pin = prior.get(name) or (expected or {}).get(name)
        if pin is not None and not pin_matches(pin, record):
            raise ValueError(f"First-party AMI source differs from its published receipt: {name}")
        path.with_suffix(".part").rename(path)
        records[name] = record
        print(f"Verified source {name} ({record['bytes']} bytes)", flush=True)
    trial.write(directory / "sources.json", records)


def measure(directory, plan, pins, ffmpeg):
    rows = []
    for cohort, meetings in COHORTS.items():
        for session in meetings:
            with ExitStack() as stack:
                readers = []
                for microphone in trial.MICROPHONES:
                    name = f"{session}.{microphone}.wav"
                    path = directory / name
                    record = {"bytes": path.stat().st_size, "sha256": trial.digest(path), "url": plan["sources"][name]}
                    if not pin_matches(pins[name], record):
                        raise ValueError(f"AMI source does not match its pinned receipt: {name}")
                    wav = stack.enter_context(wave.open(str(path), "rb"))
                    trial.check_profile(wav, 1)
                    readers.append(wav)
                if readers[0].getnframes() != readers[1].getnframes():
                    raise ValueError(f"Microphone sample clocks differ: {session}")
                for index, seconds in enumerate(trial.GRID):
                    start = seconds * trial.RATE
                    stereo = np.column_stack([trial.decode_wav(wav, start, trial.DURATION)[:, 0] for wav in readers])
                    for variant, samples, truth in trial.deliveries(stereo, index):
                        row, _ = trial.score(samples, truth, ffmpeg, production_check=index in {0, 11})
                        rows.append({"cohort": cohort, "meeting": session, "source_start_sample": start, "variant": variant, **row})
            print(f"Replayed {session}", flush=True)
    return rows


def report(rows):
    return {cohort: {
        "positive": trial.totals(row for row in rows if row["cohort"] == cohort and row["variant"] in trial.RATIOS),
        "exact_copy": trial.totals(row for row in rows if row["cohort"] == cohort and row["variant"] == "exact_copy"),
        "control": trial.totals(row for row in rows if row["cohort"] == cohort and row["variant"] in trial.CONTROLS),
        "by_meeting": {session: trial.totals(row for row in rows if row["meeting"] == session) for session in meetings},
    } for cohort, meetings in COHORTS.items()}


def run(directory, plan, pins, ffmpeg, expected):
    if set(pins) != set(plan["sources"]):
        raise ValueError("Source manifest does not cover the complete frozen roster")
    rows = measure(directory, plan, pins, ffmpeg)
    counts = report(rows)
    prior = trial.read(PANEL)
    if (counts["previous"]["positive"] != prior["confirmation_gain_positives"]
            or counts["previous"]["control"] != prior["confirmation_no_relation_controls"]
            or counts["previous"]["by_meeting"] != prior["counts"]["confirmation"]["by_source_group"]):
        raise ValueError("Recomputed outcomes differ from the earlier AMI comparison")
    readout = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    # Preserve native and mono input hashes separately so a different FFmpeg build is visible.
    native = [{key: value for key, value in row.items() if key != "mono_pcm_sha256"} for row in rows]
    native_digest = trial.sha(json.dumps(native, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    result = {"schema": "chipmunk.ami-gain-public-replay/1", "plan": plan, "sources": pins, "counts": counts,
              "deliveries": len(rows), "production_decode_checks": sum(row["production_decode_checked"] for row in rows),
              "readout_sha256": trial.sha(readout), "readout_without_mono_pcm_sha256": native_digest,
              "ffmpeg_version": subprocess.check_output([ffmpeg, "-version"], text=True, timeout=5).splitlines()[0],
              "numpy_version": np.__version__}
    if expected and any(expected[key] != result[key] for key in ("plan", "sources", "counts", "deliveries",
                            "production_decode_checks", "readout_without_mono_pcm_sha256")):
        raise ValueError("Public replay differs from its committed outcomes")
    trial.write(directory / "result.json", result)
    print(json.dumps({"deliveries": len(rows), "counts": counts, "full_readout_match":
                      expected is not None and expected["readout_sha256"] == result["readout_sha256"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "acquire", "replay"))
    parser.add_argument("scratch", type=Path)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    args = parser.parse_args()
    plan = schedule()
    expected = trial.read(PUBLIC) if PUBLIC.exists() else None
    if expected and expected["plan"] != plan:
        raise ValueError("Use the revision linked in the public replay documentation")
    if args.stage == "freeze":
        args.scratch.mkdir(mode=0o700, exist_ok=True)
        trial.write(args.scratch / "plan.json", plan)
    elif args.stage == "acquire":
        acquire(args.scratch, plan, expected["sources"] if expected else None)
    else:
        if not args.ffmpeg:
            parser.error("An FFmpeg executable is required to replay the mono comparison")
        if trial.read(args.scratch / "plan.json") != plan:
            raise ValueError("Replay plan differs from the prospective freeze")
        pins = expected["sources"] if expected else trial.read(args.scratch / "sources.json")
        run(args.scratch, plan, pins, args.ffmpeg, expected)


if __name__ == "__main__":
    main()
