"""Measure source-bound competing-speech context on the consumed PACC-P sample."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import types
import zipfile
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

from scripts import audio_pacc_false_alarm_v177 as prior
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = prior.ROOT, prior.BASE, prior.PANEL
NAME = "audio-overlap-context-v180"
TRIAL, SOURCE = BASE / NAME, prior.source.ARCHIVE / NAME
PUBLIC = PANEL / "audio-overlap-context-freeze-v180.json"
PRIOR_RESULT = PANEL / "audio-pacc-false-alarm-result-v177.json"
METHOD = ROOT / "docs/audio-overlap-context-v180-method.md"
MODULE = "scripts.audio_overlap_context_v180"
OUTPUT = TRIAL / "authored/output"
ARCHIVE_NAME = "sherpa-onnx-pyannote-segmentation-3-0"
ASSETS = {
    f"{ARCHIVE_NAME}.tar.bz2": "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488",
    "model.onnx": "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
    "LICENSE": "14d7016ad68e7394d6e6b78d96cc2ae431c905287b89674cfdf021e79e62b8ba",
}
WHEEL = "onnxruntime-1.23.2-cp312-cp312-macosx_13_0_arm64.whl"
WHEEL_SHA = "b8f029a6b98d3cf5be564d52802bb50a8489ab73409fa9db0bf583eabb7c2321"
RATE = prior.source.SAMPLE_RATE
WINDOW, RECEPTIVE, STEP = 160000, 991, 270
FRAMES = (WINDOW - RECEPTIVE) // STEP + 1
CASE_COUNT = prior.CASE_COUNT
RUN_SAMPLES = RATE // 4
read, write, sha, require = prior.read, prior.write, prior.sha, prior.require


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def weights():
    path = SOURCE / f"{ARCHIVE_NAME}.tar.bz2"
    require(
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == 6958444
        and sha(path) == ASSETS[path.name],
        "Exact publisher segmentation release archive required",
    )
    payloads = {}
    with tarfile.open(path, "r:bz2") as archive:
        for name in ("model.onnx", "LICENSE"):
            item = archive.getmember(f"{ARCHIVE_NAME}/{name}")
            require(item.isfile() and 0 < item.size < 8_000_000, "Unexpected model release member")
            with archive.extractfile(item) as stream:
                raw = stream.read()
            require(len(raw) == item.size and digest(raw) == ASSETS[name], "Exact model and license required")
            payloads[name] = raw
    return payloads["model.onnx"]


def verify_runtime(root):
    wheel = root / WHEEL
    site = root / "site"
    require(
        wheel.is_file() and sha(wheel) == WHEEL_SHA and site.resolve(strict=True) == site,
        "Exact publisher wheel and real isolated runtime directory required",
    )
    with zipfile.ZipFile(wheel) as archive:
        (record_path,) = [name for name in archive.namelist() if name.endswith(".dist-info/RECORD")]
        records = list(csv.reader(io.StringIO(archive.read(record_path).decode())))
        for path, expected, size in records:
            if not path.startswith("onnxruntime/"):
                continue
            item = site / path
            require(
                item.is_relative_to(site)
                and item.is_file()
                and item.resolve(strict=True) == item
                and item.stat().st_size == int(size),
                "Installed runtime file is unavailable",
            )
            actual = base64.urlsafe_b64encode(bytes.fromhex(sha(item))).rstrip(b"=").decode()
            require(expected == f"sha256={actual}", "Installed runtime bytes differ from verified wheel")
    return site


def input_rows():
    public = read(PRIOR_RESULT)
    require(
        public == prior.readout()
        and public["planned_cases"] == CASE_COUNT
        and public["holdout_wavs_read"] == 0,
        "Exact completed v177 genuine controls required",
    )
    plan, prepared = prior.prepared()
    stage = prior.TRIAL / "authored/output/private-observations.jsonl"
    decisions = prior.baseline.speech.planning.reviewed.lines(stage)
    require(
        [prior.identity(x) for x in decisions] == [prior.identity(x) for x in prepared],
        "Exact paired serving and PCM receipts required",
    )
    pools = {row["slot"]: row["pool"] for row in plan}
    return [
        (pcm, decision["decision"], pools[pcm["slot"]])
        for pcm, decision in zip(prepared, decisions, strict=True)
    ]


def context(root):
    verify_runtime(root)
    paths = (Path(__file__), METHOD, ROOT / "backend/tests/test_audio_overlap_context_v180.py")
    return {
        "schema": "chipmunk.audio-overlap-context-freeze/180",
        "prior_serving_receipt_sha256": sha(PRIOR_RESULT),
        "prior_serving_freeze_sha256": sha(prior.PUBLIC),
        "assets": ASSETS,
        "runtime_wheel": WHEEL,
        "runtime_wheel_sha256": WHEEL_SHA,
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in paths},
        "planned_cases": CASE_COUNT,
        "input_sample_rate": RATE,
        "model_window_samples": WINDOW,
        "receptive_field_samples": RECEPTIVE,
        "receptive_field_shift_samples": STEP,
        "minimum_contiguous_overlap_samples": RUN_SAMPLES,
        "model_calls_per_case": 1,
        "holdout_access_permitted": False,
        "scope": "Retrospective competing-speech context on previously evaluated genuine PACC-P inputs only.",
    }


def freeze(root):
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve every prior attempt")
    lock = context(root)
    weights()
    require(len(input_rows()) == CASE_COUNT, "Complete bounded comparison required")
    TRIAL.mkdir(mode=0o700)
    write(TRIAL / "freeze.json", lock)
    write(PUBLIC, lock)


def verify(root):
    require(
        context(root) == read(TRIAL / "freeze.json") == read(PUBLIC), "Exact overlap method freeze required"
    )
    return input_rows()


def overlap_record(logp, samples):
    require(
        logp.shape == (1, FRAMES, 7) and np.isfinite(logp).all() and RECEPTIVE <= samples <= WINDOW,
        "Unexpected finite segmentation response shape",
    )
    # Keep outputs with their full convolutional receptive field inside source audio.
    measured = (samples - RECEPTIVE) // STEP + 1
    probabilities = np.exp(logp[0, :measured].astype(np.float64))
    require(
        np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=2e-5), "Expected normalized log probabilities"
    )
    classes = np.argmax(probabilities, axis=1)
    overlap = classes >= 4
    endpoints = np.flatnonzero(np.diff(np.r_[False, overlap, False])).reshape(-1, 2)
    first_cell = (RECEPTIVE - STEP) // 2
    spans = [[first_cell + int(a) * STEP, first_cell + int(b) * STEP] for a, b in endpoints]
    longest = max((end - start for start, end in spans), default=0)
    return {
        "eligible_frames": measured,
        "overlap_frames": int(overlap.sum()),
        "silence_frames": int(np.count_nonzero(classes == 0)),
        "mean_overlap_mass": float(probabilities[:, 4:].sum(axis=1).mean()),
        "longest_overlap_samples": longest,
        "diagnostic_context": longest >= RUN_SAMPLES,
        "private_intervals_samples": spans,
    }


def key(row):
    return row["slot"], row["family"], row["condition"]


def summary(rows, inputs):
    require(
        [key(row) for row in rows] == [key(row) for row, _, _ in inputs],
        "Every frozen input must have one ordered overlap receipt",
    )
    groups = {}
    for group in ("AMI", "VCTK", "VCTK_mic1", "VCTK_mic2"):
        groups[group] = {}
        for condition in prior.source.CONDITIONS:
            panel = [
                row for row in rows if group in (row["family"], row["pool"]) and row["condition"] == condition
            ]
            flagged = [row for row in panel if row["diagnostic_context"]]
            groups[group][condition] = {
                "planned": len(panel),
                "overlap_context": len(flagged),
                "original_serving_directions": dict(Counter(row["prior_direction"] for row in panel)),
                "directions_with_overlap_context": dict(Counter(row["prior_direction"] for row in flagged)),
                "median_overlap_fraction": median(
                    row["overlap_frames"] / row["eligible_frames"] for row in panel
                ),
                "median_overlap_mass": median(row["mean_overlap_mass"] for row in panel),
            }
    require(
        all(
            row["pool"] == pool and row["prior_direction"] == decision
            for row, (_, decision, pool) in zip(rows, inputs, strict=True)
        ),
        "Prior serving status or public grouping differs",
    )
    return groups


def worker(root, output):
    stage = TRIAL / "authored"
    require(
        output == OUTPUT and (stage / "resources.jsonl").is_file() and (stage / "worker.log").is_file(),
        "Exact guarded overlap worker required",
    )
    rows = verify(root)
    sys.path.insert(0, str(root / "site"))
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(weights(), sess_options=options, providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    require(
        ort.__version__ == "1.23.2"
        and Path(ort.__file__).resolve().is_relative_to(root / "site")
        and session.get_providers() == ["CPUExecutionProvider"]
        and int(metadata["sample_rate"]) == RATE
        and int(metadata["window_size"]) == WINDOW
        and int(metadata["receptive_field_size"]) == RECEPTIVE
        and int(metadata["receptive_field_shift"]) == STEP
        and int(metadata["num_classes"]) == 7
        and int(metadata["num_speakers"]) == 3,
        "Exact verified CPU model and native timing required",
    )
    output.mkdir(mode=0o700)
    observations = []
    for item, decision, pool in rows:
        pcm = prior.read_pcm(item)
        source = np.pad(pcm, (0, WINDOW - len(pcm)))[None, None, :]
        [response] = session.run(["y"], {"x": source})
        observations.append(
            {
                "slot": item["slot"],
                "family": item["family"],
                "condition": item["condition"],
                "pool": pool,
                "prior_direction": decision,
                **overlap_record(response, len(pcm)),
            }
        )
    journal = output / "private-observations.json"
    write(journal, observations)
    write(
        output / "result.json",
        {
            "schema": "chipmunk.audio-overlap-context-result/180",
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "private_observations_sha256": sha(journal),
            "new_neural_calls": len(observations),
            "planned_cases": CASE_COUNT,
            "genuine_controls": summary(observations, rows),
            "holdout_wavs_read": 0,
            "defaults_changed": False,
        },
    )


def run(root):
    def launch(argv, **kwargs):
        require(
            argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored", str(OUTPUT)],
            "Exact monitored segmentation command required",
        )
        return subprocess.Popen([*argv[:3], MODULE, "worker", str(root), str(OUTPUT)], **kwargs)

    proxy = types.SimpleNamespace(
        Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        prior.runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": lambda: verify(root),
            "prerequisites": lambda *_: True,
            "subprocess": proxy,
            "__file__": __file__,
        },
    )("authored")


def readout(root):
    expected = verify(root)
    stage = TRIAL / "authored"
    journal = OUTPUT / "private-observations.json"
    rows = read(journal)
    result = prior.baseline.speech.qualification_profile(stage, CASE_COUNT, {}, Path(__file__).name)
    require(
        result["schema"] == "chipmunk.audio-overlap-context-result/180"
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["private_observations_sha256"] == sha(journal)
        and result["planned_cases"] == CASE_COUNT
        and result["genuine_controls"] == summary(rows, expected)
        and result["holdout_wavs_read"] == 0
        and result["defaults_changed"] is False,
        "Exact completed guarded overlap result required",
    )
    profile = read(stage / "profile.json")
    return {
        **result,
        "resource_profile_sha256": sha(stage / "profile.json"),
        "maximum_sampled_descendant_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
        "worker_duration_seconds": profile["duration_s"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "worker", "readout"))
    parser.add_argument("runtime_root", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    require((args.command == "worker") == (args.output is not None), "Only the worker takes an output path")
    root = args.runtime_root.resolve(strict=True)
    if args.command == "worker":
        worker(root, args.output)
    elif args.command == "readout":
        print(json.dumps(readout(root), indent=2, allow_nan=False))
    else:
        {"freeze": freeze, "run": run}[args.command](root)
