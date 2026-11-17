"""Measure recoverable short soundtrack differences using an earlier video copy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from scripts import video_lip_correspondence as receipt
from scripts.video_visual_speech_evaluation import overlaps

SCHEMA = "chipmunk.video-short-phrase-comparison/1"
MODES = ("pcm_pair", "pcm_aac", "aac_pair")


def compact_counts(rows):
    result = defaultdict(Counter)
    for row in rows:
        counter = result[(row["cohort"], row["pair"], row["role"])]
        counter["deliveries"] += 1
        counter["outside_scope"] += not row["eligible"]
        old, extra, truth = row["previous"], row["short"], row["truth"]
        counter["original_alerts"] += old is not None
        counter["added_alerts"] += extra is not None
        if truth:
            counter["original_correct"] += overlaps(old and old["samples"], truth)
            counter["combined_correct"] += overlaps((old or extra or {}).get("samples"), truth)
            counter["added_wrong"] += bool(extra) and not overlaps(extra["samples"], truth)
    return {"/".join(group): row for group, row in sorted(result.items())}


def trial(inputs, sources, output):
    import numpy as np

    from app.detectors.video.decode import probe_frames
    from app.evidence import (
        video_reference_short_phrases,
        video_reference_sound_difference,
        video_reference_soundtrack,
    )
    from scripts.export_video_lip_correspondence import make_soundtracks, waveform
    from scripts.video_lip_delivery_transfer import qualified_48k

    paths = (Path(video_reference_short_phrases.__file__), Path(video_reference_sound_difference.__file__), Path(__file__))
    method = {p.name: receipt.sha(p) for p in paths}
    rows = []
    with (ZipFile(sources / "lombardgrid_audio.zip") as audio, ZipFile(sources / "lombardgrid_front.zip") as pictures,
          TemporaryDirectory() as directory):
        path = Path(directory) / "picture.mov"
        carrier = Path(directory) / "audio.m4a"
        for plan_path in inputs:
            plan = receipt.read(plan_path)
            sources_selected = [row for row in plan["rows"] if row["split"] not in {"fit", "calibration"}]
            for index, row in enumerate(sources_selected, 1):
                target = row["target"]
                path.write_bytes(pictures.read(target["front"]))
                assert receipt.sha(path) == target["front_sha256"]
                native = probe_frames(path, cap=120, timeout_s=18)
                profile = video_reference_soundtrack.native_difference_profile((path, path), (native, native))
                original = waveform(audio, target)
                original_aac, _, codec = qualified_48k(original, carrier)
                stationary = dict(zip(MODES, (original, original, original_aac), strict=True))
                for role, pcm in make_soundtracks(audio, row).items():
                    aac, _, delivered = qualified_48k(pcm, carrier)
                    for pair, samples in zip(MODES, (pcm, aac, aac), strict=True):
                        previous = candidate = None
                        if profile is not None:
                            first = stationary[pair]
                            shifted, _ = video_reference_soundtrack.measure(
                                first.astype(np.float32) / 32768, samples.astype(np.float32) / 32768,
                                window=video_reference_soundtrack.SHORT_WINDOW,
                                step=video_reference_soundtrack.SHORT_STEP, min_rms=video_reference_soundtrack.QUIET_RMS,
                            )
                            if not shifted:
                                _, old = video_reference_sound_difference.measure(first, samples, recover_supported=True, short_native=True)
                                if old is not None and video_reference_sound_difference.strong_flanks(old):
                                    previous = old
                                else:
                                    candidate = video_reference_short_phrases.locate(first, samples)
                        rows.append({"cohort": row["split"], "picture_sha256": target["front_sha256"], "speaker": row["speaker"],
                                     "pair": pair, "role": role, "truth": target["span"] if role in {"different_words", "same_words"} else None,
                                     "size": len(samples), "submitted_pcm_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
                                     "profile": profile, "eligible": profile is not None,
                                     "aac_reference_sha256": codec["file_sha256"] if pair == "aac_pair" else None,
                                     "aac_submitted_sha256": delivered["file_sha256"] if pair != "pcm_pair" else None,
                                     "previous": previous, "short": candidate})
                if index % 6 == 0:
                    print(f"{plan_path.name} {index}/{len(sources_selected)} paired source clips", flush=True)
    packet = {"schema": SCHEMA, "method": method, "sources": [receipt.sha(p) for p in inputs], "rows": rows,
              "summary": compact_counts(rows)}
    receipt.save(output, packet)
    print(json.dumps(packet["summary"], indent=2))


def native(folder, benchmark, output):
    from app.evidence import video_reference
    from app.evidence.video_reference_short_phrases import BASIS

    trial_packet = receipt.read(benchmark)
    cases = json.loads((folder / "manifest.json").read_bytes())["cases"]
    indexed = {}
    for case in cases:
        _, _, codec, role = case["id"].split("-", 3)
        indexed[case["source_sha256"], codec, role] = case

    def measure(row):
        ref_codec = "aac48" if row["pair"] == "aac_pair" else "pcm"
        test_codec = "pcm" if row["pair"] == "pcm_pair" else "aac48"
        earlier = indexed[row["picture_sha256"], ref_codec, "original"]
        submitted = indexed[row["picture_sha256"], test_codec, row["role"]]
        paths = [folder / "clips" / f"{case['id']}.mov" for case in (earlier, submitted)]
        record, assets = video_reference.compare(*paths, include_channel_timing=True, include_trimmed=True,
            include_local_area=True, include_measured_interval=True, include_trim_reencoded=True,
            include_soundtrack_reencode=True, include_hd_soundtrack_reencode=True, include_quiet_soundtrack=True,
            include_short_soundtrack=True, include_sound_difference=True, recover_sound_difference=True,
            include_short_native_sound_difference=True, include_short_phrase_difference=True, include_picture_insertion=True)
        if (record["earlier"]["sha256"], record["submitted"]["sha256"]) != (earlier["input_sha256"], submitted["input_sha256"]):
            raise ValueError("A decoded comparison used a different source")
        result = record["sound_difference"]
        span = result.get("range")
        if span:
            added = span.get("measurement_basis") == BASIS
            expected = row["short" if added else "previous"]
            if (expected is None or span["samples"] != expected["samples"] or record["picture"]["changed_frames"]
                    or len(span["paired_pictures"]) != 3 or not assets):
                raise ValueError("The reported range differs from the measured input or is missing source evidence")
        elif row["short"] or row["previous"]:
            raise ValueError(f"The native comparison could not replay a measured range on {submitted['id']}")
        return {"earlier": earlier["input_sha256"], "submitted": submitted["input_sha256"], "codec": row["pair"],
                "role": row["role"], "truth": row["truth"], "status": result["status"], "reason": result.get("reason"),
                "profiles": result.get("audio_presentation"), "interval": span["samples"] if span else None,
                "method": span.get("measurement_basis", "existing") if span else None}

    rows = [row for row in trial_packet["rows"] if row["cohort"] == "fresh"]
    if not rows or len(rows) * 2 != len(cases) * 3:
        raise ValueError("The prepared delivery set differs from the frozen comparison")
    with ThreadPoolExecutor(max_workers=2) as pool:
        checked = []
        for index, result in enumerate(pool.map(measure, rows), 1):
            checked.append(result)
            if index % 36 == 0:
                print(f"Verified exact earlier and submitted files on {index}/{len(rows)} native comparisons", flush=True)
    receipt.save(output, {"schema": SCHEMA, "benchmark": receipt.sha(benchmark),
                          "manifest": receipt.sha(folder / "manifest.json"), "rows": checked})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    measure = sub.add_parser("measure")
    measure.add_argument("sources", type=Path)
    measure.add_argument("output", type=Path)
    measure.add_argument("inputs", nargs="+", type=Path)
    replay = sub.add_parser("native")
    for name in ("folder", "benchmark", "output"):
        replay.add_argument(name, type=Path)
    check = sub.add_parser("verify")
    check.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.mode == "measure":
        trial(args.inputs, args.sources, args.output)
    elif args.mode == "native":
        native(args.folder, args.benchmark, args.output)
    else:
        packet = receipt.read(args.receipt)
        assert packet["schema"] == SCHEMA and compact_counts(packet["rows"]) == packet["summary"]
        print(json.dumps(packet["summary"], indent=2))
