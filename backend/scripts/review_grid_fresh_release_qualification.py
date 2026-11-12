# SPDX-License-Identifier: Apache-2.0
"""Reopen native files and independently replay source commitments for the fresh release panel."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from scripts.review_grid_single_clip_release_timing import RGB_SIZE, digest, native, read, sha


def oracle_match(ref, pcm, center):
    start = center - 320
    template = ref[start : center + 640].astype(float)
    template -= template.mean()
    lo = max(0, start - 6400)
    hi = min(len(pcm) - len(template), start + 6400)
    windows = np.lib.stride_tricks.sliding_window_view(pcm[lo : hi + len(template)], len(template))
    values = []
    for offset in range(0, len(windows), 256):
        samples = windows[offset : offset + 256].astype(float)
        samples -= samples.mean(axis=1, keepdims=True)
        denominator = np.linalg.norm(samples, axis=1) * np.linalg.norm(template)
        values.extend(np.clip((samples @ template) / np.maximum(1e-12, denominator), -1, 1))
    best = int(np.argmax(values))
    alternative = max((value for j, value in enumerate(values) if abs(j - best) >= 640), default=-1)
    if values[best] < 0.94 or values[best] - alternative < 0.08:
        return None
    return lo + best, float(values[best]), float(values[best] - alternative)


def localization(case, result, proof, ref, pcm):
    delay = case["shift"]
    span = case["span_samples"]
    current = False
    if delay:
        start, end = (Fraction(s, 640) for s in span)
        for flag in result["current"]["ranges"]:
            a, b = map(Fraction, flag["extent_ticks"])
            lag = flag["relative_lag_ticks"]
            overlap = max(0, min(end, b) - max(start, a))
            union = max(end, b) - min(start, a)
            current |= lag is not None and abs(lag - delay) <= 2 and 5 * overlap >= 2 * union
    single = result["blind"]
    blind = bool(
        delay
        and single["status"] == "local_timing_change"
        and abs(single["audio_minus_picture_change_samples"] - delay * 640) <= 1280
        and span[0] <= single["events"][1]["acoustic_anchor"] < span[1]
    )
    chosen = result["oracle"]
    replayed = (
        oracle_match(ref, pcm, proof["original_audio_center"]) if proof["status"] == "supported" else None
    )
    assert (chosen is None) == (replayed is None)
    if chosen:
        offset, correlation, margin = replayed
        assert chosen["matched_samples"] == [offset, offset + 960]
        assert abs(chosen["correlation"] - correlation) < 1e-7 and abs(chosen["margin"] - margin) < 1e-7
        assert chosen["timing_difference"] == (abs(chosen["offset_samples"]) >= 1920)
    exact = bool(
        delay
        and chosen
        and chosen["timing_difference"]
        and abs(chosen["offset_samples"] - delay * 640) <= 1280
        and span[0] <= chosen["matched_samples"][0] < chosen["matched_samples"][1] <= span[1]
    )
    assert [current, blind, exact] == result["correct"]


def audit(run):
    freeze, sound, visual, labels, media, readout = (
        read(run / (stem + ".json"))
        for stem in ("freeze", "audio-labels", "visual-labels", "label-freeze", "media", "readout")
    )
    lock = digest(run / "freeze.json")
    assert lock == sound["freeze_sha256"] == visual["freeze_sha256"] == labels["freeze_sha256"]
    assert labels["audio_sha256"] == digest(run / "audio-labels.json") and labels["visual_sha256"] == digest(
        run / "visual-labels.json"
    )
    assert media["labels_sha256"] == digest(run / "label-freeze.json") and readout["media_sha256"] == digest(
        run / "media.json"
    )
    assert len(freeze["roster"]) == len(sound["cases"]) == len(visual["cases"]) == len(labels["cases"]) == 34
    assert not {r["group"] for r in labels["cases"] if r["split"] == "held"} & {
        r["group"] for r in labels["cases"] if r["split"] == "development"
    }
    originals, totals = {}, Counter()
    for source, a, v, label in zip(
        freeze["roster"], sound["cases"], visual["cases"], labels["cases"], strict=True
    ):
        assert source["id"] == a["id"] == v["id"] == label["id"]
        path = Path(source["path"])
        with ZipFile(source["source_archive"]) as archive:
            info = archive.getinfo(source["member"])
            assert (
                source["member_crc32"] == info.CRC
                and sha(archive.read(info)) == digest(path) == source["source_sha256"]
            )
        pts, tb, pictures, pcm = native(path)
        picture_hashes = [sha(pictures[k * RGB_SIZE : (k + 1) * RGB_SIZE]) for k in range(75)]
        assert pts == v["pts"] and Fraction(v["time_base"]) == tb
        assert len(pcm) == a["samples"] and a["pcm_sha256"] == sha(pcm.tobytes())
        if a.get("range_samples"):
            lo, hi = a["range_samples"]
            assert hi - lo == 240 and a["burst_pcm_sha256"] == sha(pcm[lo:hi].tobytes())
            totals["source_audio_brackets"] += 1
        for crop in v.get("native_crops", []):
            k = crop["frame"]
            raw = pictures[k * RGB_SIZE : (k + 1) * RGB_SIZE]
            x0, y0, x1, y1 = crop["xyxy"]
            assert (
                crop["pts"] == pts[k] and Fraction(crop["time_base"]) == tb and sha(raw) == crop["rgb_sha256"]
            )
            assert (
                sha(np.frombuffer(raw, np.uint8).reshape(288, 360, 3)[y0:y1, x0:x1].tobytes())
                == crop["crop_sha256"]
            )
            totals["source_native_mouth_crops"] += 1
        if label["status"] == "supported":
            assert label["original_audio_center"] == sum(a["range_samples"]) // 2
            assert label["original_picture_midpoint"] == sum(v["bracket_frames"]) * 320
        originals[source["id"]] = picture_hashes, pcm
        totals["original_pictures"] += len(pts)
        totals["original_samples"] += len(pcm)
    expected_cases = {
        (r["id"], role)
        for r in labels["cases"]
        if r["split"] == "held" and r["status"] in {"supported", "lexical_negative"}
        for role in (
            ("original", "gain", "audio_early", "audio_late") if r["status"] == "supported" else ("original",)
        )
    }
    assert len(expected_cases) == len(media["cases"]) == len(readout["rows"])
    assert {(r["id"], r["role"]) for r in media["cases"]} == expected_cases
    by_id = {r["id"]: r for r in labels["cases"]}
    for case, result in zip(media["cases"], readout["rows"], strict=True):
        assert (case["id"], case["role"], case["shift"] * 640) == (
            result["id"],
            result["role"],
            result["shift_samples"],
        )
        file = Path(case["file"])
        assert digest(file) == case["file_sha256"]
        pts, _, pictures, pcm = native(file)
        prior_rgb, ref = originals[case["id"]]
        assert (
            [sha(pictures[k * RGB_SIZE : (k + 1) * RGB_SIZE]) for k in range(75)]
            == prior_rgb
            == case["native_rgb"]
        )
        expected = ref.copy()
        if case["role"] != "original":
            start, stop = case["span_samples"]
            d = case["shift"] * 640
            expected[start:stop] = ref[start - d : stop - d] if d else np.rint(ref[start:stop] * 0.72)
            if d:
                assert result["shift_samples"] == d
                totals["exact_copied_edit_views"] += 1
        assert np.array_equal(expected, pcm) and sha(pcm.tobytes()) == case["pcm_sha256"]
        localization(case, result, by_id[case["id"]], ref, pcm)
        totals["localization_dispositions_replayed"] += 3
        if (oracle := result["oracle"]) is not None:
            start, stop = oracle["source_samples"]
            a, b = oracle["matched_samples"]
            assert stop - start == b - a == 960 and a - start == oracle["offset_samples"]
            assert sha(ref[start:stop].tobytes()) == oracle["source_slice_sha256"]
            assert sha(pcm[a:b].tobytes()) == oracle["matched_slice_sha256"]
            if result["correct"][2]:
                assert case["span_samples"][0] <= a < b <= case["span_samples"][1]
                assert oracle["offset_samples"] == result["shift_samples"] and np.array_equal(
                    ref[start:stop], pcm[a:b]
                )
            totals["replayed_oracle_slice_pairs"] += 1
        totals["scored_view_pictures"] += len(pts)
        totals["scored_view_samples"] += len(pcm)
    output = {
        "verified": True,
        "source_views": len(freeze["roster"]),
        "scored_views": len(media["cases"]),
        **totals,
        "freeze_sha256": lock,
        "labels_sha256": digest(run / "label-freeze.json"),
        "readout_sha256": digest(run / "readout.json"),
    }
    with (run / "independent-audit.json").open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(json.dumps(output), flush=True)


def aggregate(run, target):
    audit_record = read(run / "independent-audit.json")
    label_file = run / "label-freeze.json"
    readout_file = run / "readout.json"
    assert audit_record["verified"] and audit_record["labels_sha256"] == digest(label_file)
    assert audit_record["readout_sha256"] == digest(readout_file)
    by_cohort = defaultdict(Counter)
    labels = read(label_file)["cases"]
    for row in labels:
        by_cohort[row["split"]][row["status"]] += 1
    table = defaultdict(Counter)
    for row in read(readout_file)["rows"]:
        group = table[row["role"]]
        current, blind, oracle = row["current"], row["blind"], row["oracle"]
        group["views"] += 1
        group["current_assessed"] += bool(current["measured_windows"])
        group["current_alerted"] += bool(current["ranges"])
        group["blind_assessed"] += blind["status"] != "unassessed"
        group["blind_alerted"] += blind["status"] == "local_timing_change"
        group["reference_assessed"] += oracle is not None
        group["reference_alerted"] += bool(oracle and oracle["timing_difference"])
        for arm, hit in zip(("current", "blind", "reference"), row["correct"], strict=True):
            group[arm + "_localized"] += hit
    assert sum(row["views"] for row in table.values()) == audit_record["scored_views"]
    result = {
        "schema": "chipmunk.grid-fresh-release-public/1",
        "source": "GRID corpus, 10.5281/zenodo.3625687, archived native clips and word alignments",
        "source_group_count": {
            split: len({row["group"] for row in labels if row["split"] == split}) for split in by_cohort
        },
        "separate_observations": dict(by_cohort),
        "deliveries": dict(table),
        "independent_audit": audit_record,
        "reference_use": "Known source waveform and a chosen mouth event were supplied. Timing comparison only.",
    }
    with target.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--aggregate", type=Path)
    args = parser.parse_args()
    if args.aggregate:
        aggregate(args.run, args.aggregate)
    else:
        audit(args.run)
