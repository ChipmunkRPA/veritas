"""Replay the denser earlier-copy audio comparison on the frozen stock voice banks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.review_audio_stock_voice_transfer import require, sha256, useful

SCHEMA = "chipmunk.audio-dense-clean-reference/1"
PANELS = Path(__file__).resolve().parents[2] / "docs/evaluation-panels"
PLAN = "audio-dense-clean-reference-plan-2026-11-13.json"
SAMPLES = 96000
SYNTHETIC = ("piper", "kitten")
SPLICES = ("same_reader_splice", "different_reader_splice")
CONTROLS = ("untouched", "local_gain", "global_eq")
UNMATCHED = ("unrelated", "masker_only")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def summarize(rows, arm):
    counts = {}
    for level in dict.fromkeys(row["snr_db"] for row in rows):
        selected = [row for row in rows if row["snr_db"] == level]
        edits = [row for row in selected if row["truth"]]
        flags = [row for row in edits if row[arm]["reported"]]
        controls = [row for row in selected if row["kind"] in CONTROLS]
        matches = [row for row in flags if useful(row[arm]["range"], row["truth"])]
        counts[str(level)] = {
            "useful_synthetic": sum(row["kind"] in SYNTHETIC for row in matches),
            "synthetic_total": sum(row["kind"] in SYNTHETIC for row in edits),
            "misplaced_synthetic": sum(row["kind"] in SYNTHETIC and row not in matches for row in flags),
            "useful_ordinary_splices": sum(row["kind"] in SPLICES for row in matches),
            "ordinary_splices_total": sum(row["kind"] in SPLICES for row in edits),
            "control_flags": sum(row[arm]["reported"] for row in controls),
            "controls_total": len(controls),
            "wrong_reference_admissions": sum(row["kind"] in UNMATCHED
                                              and row[arm]["status"] != "reference_match_not_supported" for row in selected),
            "wrong_reference_total": sum(row["kind"] in UNMATCHED for row in selected),
            "synthetic_groups": sorted({row["group"] for row in matches if row["kind"] in SYNTHETIC}),
            "synthetic_unassessed": sum(row["kind"] in SYNTHETIC and row[arm]["status"] in
                                        {"reference_match_not_supported", "local_support_not_available"} for row in selected),
        }
    return counts


def summary(rows):
    arms = {arm: summarize(rows, arm) for arm in ("baseline", "dense")}
    baseline, dense = (arms[arm]["-5"] for arm in arms)
    groups = {row["group"] for row in rows if row["snr_db"] == -5 and row["kind"] in SYNTHETIC
              and row["dense"]["reported"] and useful(row["dense"]["range"], row["truth"])
              and (not row["baseline"]["reported"] or not useful(row["baseline"]["range"], row["truth"]))}
    passed = (dense["useful_synthetic"] >= baseline["useful_synthetic"] + 2 and len(groups) >= 2
              and dense["control_flags"] <= baseline["control_flags"] + 1
              and dense["misplaced_synthetic"] <= baseline["misplaced_synthetic"] + 1
              and all(not counts["wrong_reference_admissions"] for counts in arms["dense"].values()))
    return {**arms, "added_synthetic_groups_at_minus_5": sorted(groups), "advances_to_local_trial": passed}


def review(data):
    require(data["schema"] == SCHEMA and sha256(data["preassessment_sha256"]), "Missing study commitment")
    require(data["preassessment_sha256"] == digest((PANELS / PLAN).read_bytes()), "Frozen study plan differs")
    require(data["point"] == 0.31758884, "The local operating point changed")
    for role, noise in (("development", (None, 0, -5)), ("assessment", (0, -5))):
        panel = data["panels"][role]
        rows = panel["rows"]
        expected = {(group, level, kind) for group in range(8) for level in noise
                    for kind in (*SYNTHETIC, *SPLICES, *CONTROLS, "unrelated", *(("masker_only",) if level is not None else ()))}
        require(len(rows) == len(expected) and {(r["group"], r["snr_db"], r["kind"]) for r in rows} == expected,
                "Cases do not match the frozen roster")
        for row in rows:
            require(all(sha256(row[k]) for k in ("earlier_sha256", "submitted_sha256")), "Missing original source hash")
            truth = row["truth"]
            require((truth is not None) == (row["kind"] in (*SYNTHETIC, *SPLICES)), "Case truth does not match its role")
            if truth:
                require(0 <= truth[0] < truth[1] <= SAMPLES, "Invalid ground truth clock")
            for arm, width in (("baseline", 11520), ("dense", 15360)):
                result = row[arm]
                chosen = result["range"]
                state = result["status"]
                covered = {"local_change", "no_local_change"} if arm == "baseline" else {"measured"}
                require(type(result["reported"]) is bool and state in covered | {
                    "reference_match_not_supported", "local_support_not_available",
                } and (chosen is not None) == (state == ("local_change" if arm == "baseline" else "measured")),
                        "Invalid assessment state")
                if chosen is not None:
                    require(0 <= chosen[0] < chosen[1] <= SAMPLES and chosen[1] - chosen[0] == width,
                            "Location outside its frozen clock")
                    require(type(result["score"]) in (int, float) and math.isfinite(result["score"]), "Invalid local score")
                require((result["score"] is None) == (chosen is None), "Incomplete measurement")
                reported = chosen is not None and (arm == "baseline" or result["score"] > data["point"])
                require(result["reported"] == reported, "Selection does not replay")
        require(panel["summary"] == summary(rows), "The reported comparison does not replay")
    fit = data["panels"]["development"]["rows"]
    require(max(row["dense"]["score"] for row in fit
                if row["kind"] in (*CONTROLS, *UNMATCHED) and row["dense"]["score"] is not None) == data["point"],
            "The operating point does not match development controls")
    return data["panels"]["assessment"]["summary"]


def source_bank(folder, role):
    import numpy as np

    stock = role == "development"
    name = "audio-stock-voice-transfer" if stock else "audio-random-clock-stock"
    published = json.loads((PANELS / f"{name}-2026-11-13.json").read_bytes())
    frozen = (folder / ("freeze.json" if stock else "placements-freeze.json")).read_bytes()
    require(digest(frozen) == published["native_freeze_sha256"], "Source roster changed")
    rows = json.loads(frozen)["cases" if stock else "native_manifest_rows"]
    if stock:
        native = dry = np.load(folder / "signals.npy", mmap_mode="r", allow_pickle=False)
    else:
        with np.load(folder / "placements-native.npz", allow_pickle=False) as archive:
            native, dry = archive["native"], archive["dry"]
    sources, spans = {}, {}
    for row, source, clean in zip(rows, native, dry, strict=True):
        require(source.shape == (SAMPLES,) and source.dtype == np.dtype("<i2") and digest(source.tobytes()) == row["pcm_sha256"],
                "Native source differs")
        if not stock and row["dry_pcm_sha256"] is not None:
            require(clean.dtype == np.dtype("<i2") and digest(clean.tobytes()) == row["dry_pcm_sha256"], "Clean source differs")
        group = row["group"]
        if stock and type(group) is int:
            spans[group] = row["sources"]["authored_interval"]
            if row["kind"] == "untouched" and row["condition"] == "dry":
                sources[group] = source
        elif not stock and type(group) is int:
            if row["kind"] == "untouched":
                sources[group] = clean
            if row["edit_clock"]:
                spans[group] = row["edit_clock"]
    for row, submitted in zip(rows, native, strict=True):
        group, kind = row["group"], row["kind"]
        if group not in sources:
            continue
        level = (None if row["condition"] == "dry" else int(row["condition"].removesuffix("dB"))) if stock else row["snr"]
        yield group, kind, level, spans[group], sources[group], submitted


def cases(folder, role):
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    planned = list(source_bank(folder, role))
    sources = {row[0]: row[-2] for row in planned}
    lowpass = butter(2, 2300, fs=16000, output="sos")
    for group, kind, level, truth, earlier, submitted in planned:
        yield group, kind, level, truth if kind != "untouched" else None, earlier, submitted
        if kind == "untouched":
            start, end = truth
            gain = earlier.astype(np.float64)
            gain[start:end] *= .7
            equalized = sosfiltfilt(lowpass, earlier.astype(np.float64))
            masker = submitted.astype(np.int32) - earlier
            for change, signal in (("local_gain", gain), ("global_eq", equalized)):
                processed = (np.rint(signal).astype(np.int32) + masker).clip(-32768, 32767).astype("<i2")
                yield group, change, level, None, earlier, processed
            yield group, "unrelated", level, None, sources[(group + 1) % 8], submitted
            if level is not None:
                yield group, "masker_only", level, None, earlier, masker.astype("<i2")


def score(folder, role):
    from app.evidence import audio_clean_speech, audio_clean_speech_dense

    rows = []
    for group, kind, level, truth, earlier, submitted in cases(folder, role):
        baseline_state, baseline = audio_clean_speech.locate(earlier, submitted)
        dense_state, dense = audio_clean_speech_dense.measure(earlier, submitted)
        rows.append({
            "group": group, "kind": kind, "snr_db": level, "truth": truth,
            "earlier_sha256": digest(earlier.tobytes()), "submitted_sha256": digest(submitted.tobytes()),
            "baseline": {"status": baseline_state, "reported": baseline is not None,
                         "range": baseline and baseline["samples"], "score": baseline and baseline["relative_drop"]},
            "dense": {"status": dense_state,
                      "reported": dense is not None and dense["relative_drop"] > audio_clean_speech_dense.POINT,
                      "range": dense and dense["samples"], "score": dense and dense["relative_drop"]},
        })
    return {"rows": rows, "summary": summary(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("score", "review"))
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--assessment", type=Path)
    args = parser.parse_args()
    if args.phase == "score":
        require(args.development is not None and args.assessment is not None, "Both frozen source banks are required")
        from app.evidence.audio_clean_speech_dense import POINT

        data = {"schema": SCHEMA, "preassessment_sha256": digest((PANELS / PLAN).read_bytes()),
                "point": POINT, "panels": {"development": score(args.development, "development"),
                                           "assessment": score(args.assessment, "assessment")}}
        result = review(data)
        with args.receipt.open("x") as output:
            json.dump(data, output, sort_keys=True, separators=(",", ":"))
            output.write("\n")
    else:
        result = review(json.loads(args.receipt.read_bytes()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
