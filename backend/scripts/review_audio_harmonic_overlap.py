"""Reconstruct the source sample bindings and abstention arithmetic without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import wave
from collections import Counter
from pathlib import Path

SAMPLES = 48_000


def sha(body):
    return hashlib.sha256(body).hexdigest()


def read(folder, name, expected=None):
    body = (folder / name).read_bytes()
    if expected is not None and sha(body) != expected:
        raise ValueError(f"Wrong commitment for {name}")
    return json.loads(body)


def spans(value):
    previous = 0
    for start, end in value:
        if not 0 <= previous <= start < end <= SAMPLES or end - start < 960 or (start - 160) % 320 or (end - 160) % 320:
            raise ValueError("Measured span is outside the 20 ms continuity grid")
        previous = end
    return sum(end - start for start, end in value) // 16


def synthetic(row):
    return row["family"] not in {"genuine", "background"}


def source_cut(folder, row):
    path = folder / f"{row['room']}.Array1-01.wav"
    with wave.open(str(path), "rb") as raw:
        if raw.getparams()[:3] != (1, 2, 16000) or row["time_base"] != "1/16000":
            raise ValueError("Room sample grid differs")
        if row["source_end_sample"] - row["source_start_sample"] != SAMPLES:
            raise ValueError("Native source interval is not exactly three seconds")
        raw.setpos(row["source_start_sample"])
        return raw.readframes(SAMPLES)


def verify_mix(row, waveform):
    clean = struct.unpack(f"<{SAMPLES}h", waveform[row["clean_pcm_sha256"]])
    noise = struct.unpack(f"<{SAMPLES}h", waveform[row["background_sha256"]])
    coefficient = row["scale"]
    body = [a + coefficient * b for a, b in zip(clean, noise, strict=True)]
    snr = 10 * math.log10(sum(x*x for x in clean) / math.fsum((coefficient * x)**2 for x in noise))
    gain = min(1., 31_129 / max(map(abs, body)))
    exact = struct.pack(f"<{SAMPLES}h", *(round(x * gain) for x in body))
    if abs(snr + 5.) > 1e-10 or abs(gain - row["common_gain"]) > 1e-13 or sha(exact) != row["pcm_sha256"]:
        raise ValueError("Independent signed 16-bit remixture differs")


def verify_selection(folder, roster, fit, freeze):
    observed = read(folder, "fit-spans-private.json")
    dev = [row for row in roster if row["role"] == "development"]
    for row, got in zip(dev, observed, strict=True):
        measured = spans(got["spans"])
        if {k: got[k] for k in row} != row or measured != got["duration_ms"]:
            raise ValueError("A development measurement differs from its frozen roster or sample spans")
    native = [row for row in observed if row["condition"] == "native"]
    marked = []
    rule = freeze["advance_rule"]
    for cutoff in freeze["acoustic_points_ms"]:
        passing = [row for row in native if row["duration_ms"] >= cutoff]
        negatives = Counter(row["room"] for row in passing if row["label"] == "minimal")
        if all(value <= rule["development_maximal_minimal_false_each_room"] for value in negatives.values()):
            marked.append([sum(row["label"] == "high" for row in passing), -sum(negatives.values()), cutoff])
    acoustic = max(marked)[-1] if marked else None
    ai = [row for row in observed if row["baseline"]["decision"] == "ai"]
    thresholds = {}
    for name in fit["independent_points"]:
        point = None
        best = None
        for cutoff in sorted({row["independent"] for row in ai}):
            selected = [row for row in ai if row["independent"] < cutoff
                        and (name != "both" or (acoustic is not None and row["duration_ms"] >= acoustic))]
            correct = len({row["group"] for row in selected if not synthetic(row)})
            lost = sum(synthetic(row) for row in selected)
            if (correct and lost <= rule["maximum_synthetic_ai_lost"]
                    and not any(synthetic(row) and row["condition"] == "clean" for row in selected)
                    and (best is None or (correct, -lost) > best)):
                best, point = (correct, -lost), cutoff
        thresholds[name] = point
    if (acoustic, marked, thresholds, len(native), len(ai)) != (
            fit["acoustic_point_ms"], fit["acoustic_operating_points_considered"], fit["independent_points"],
            fit["native_windows"], fit["development_ai_directions"]):
        raise ValueError("Independent development operating point calculation differs")
    return len(dev)


def check(folder, ami, development_ami):
    result = read(folder, "aggregate.json")
    freeze = read(folder, "freeze.json", result["freeze_sha256"])
    roster = read(folder, "roster.json", freeze["roster_sha256"])
    fit = read(folder, "fit.json", result["fit_sha256"])
    observed = read(folder, "confirmation.json", result["confirmation_sha256"])
    incumbent = read(folder, "baseline.json", result["baseline_sha256"])
    evidence = read(folder, "evidence-private.json", result["evidence_sha256"])
    if (freeze != result["freeze"] or fit != result["fit"] or observed["fit_sha256"] != result["fit_sha256"]
            or observed["incumbent_sha256"] != result["baseline_sha256"]
            or any(item["freeze_sha256"] != result["freeze_sha256"] for item in (fit, observed, incumbent))):
        raise ValueError("The measurement outputs do not share one freeze")
    if dict(Counter(row["role"] for row in roster)) != freeze["planned"]:
        raise ValueError("Input role totals differ")
    selection = verify_selection(folder, roster, fit, freeze)
    waveform, native, mixes = {}, 0, 0
    paths = {"confirmation": ami, "development": development_ami}
    for room, info in freeze["rooms"].items():
        path = paths[info["role"]] / f"{room}.Array1-01.wav"
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != info["archive_sha256"]:
                raise ValueError("Exact room recording commitment differs")
    for index, row in enumerate(roster):
        raw = (folder / "pcm" / f"{index:03}.pcm").read_bytes()
        if index != row["id"] or len(raw) != SAMPLES * 2 or sha(raw) != row["pcm_sha256"]:
            raise ValueError("Frozen PCM sample binding differs")
        waveform[row["pcm_sha256"]] = raw
        if "source_start_sample" in row:
            if source_cut(paths[row["role"]], row) != raw:
                raise ValueError("Original microphone channel does not reproduce the recorded source samples")
            native += 1
    cases = [row for row in roster if row["role"] == "confirmation"]
    review_evidence, true_false, audio_counts = [], Counter(), Counter()
    decisions, speech_counts = [], Counter()
    for row, got, ref in zip(cases, observed["rows"], incumbent["rows"], strict=True):
        if (row["id"], row["pcm_sha256"]) != (got["id"], got["pcm_sha256"]) or (
                ref["id"], ref["pcm_sha256"], ref["decision"]) != (row["id"], row["pcm_sha256"], got["reference"]):
            raise ValueError("Current or independent decision is attached to a different source")
        if row["condition"] == "mixed":
            verify_mix(row, waveform)
            mixes += 1
        duration = spans(got["spans"])
        if got["reference"] == "ai":
            if not isinstance(got["independent"], (int, float)) or not math.isfinite(got["independent"]):
                raise ValueError("An AI window lacks a finite independent score")
            from review_audio_spectra_background_transfer import graph_sha

            if graph_sha(waveform[row["pcm_sha256"]]) != got["model_input_sha256"]:
                raise ValueError("Independent pre-emphasis and repeat input hash differs")
        elif got["independent"] is not None or got["model_input_sha256"] is not None:
            raise ValueError("An unselected window contains an independent result")
        mark = fit["acoustic_point_ms"] is not None and duration >= fit["acoustic_point_ms"]
        if row["condition"] == "native":
            audio_counts[(row["room"], row["label"])] += mark
            for label, seq in (("cue", got["spans"]), ("annotated", row["annotated_overlap"])):
                audio_counts[label] += sum(b - a for a, b in seq)
            audio_counts["shared"] += sum(max(0, min(b, d) - max(a, c)) for a, b in got["spans"] for c, d in row["annotated_overlap"])
        if row["condition"] == "mixed" and row["family"] == "genuine":
            speech_counts[row["room"]] += got["reference"] != "unassessed"
        veto = {name: False for name in result["policies"]}
        if got["reference"] == "ai":
            true_false["synthetic_ai" if synthetic(row) else "false"] += 1
            veto["overlap_only"] = mark
            for name, point in fit["independent_points"].items():
                veto[name] = point is not None and got["independent"] < point and (name != "both" or mark)
        if mark or any(veto.values()):
            review_evidence.append({"id": row["id"], "pcm_sha256": row["pcm_sha256"], "time_base": "1/16000",
                                    "inspected": [0, SAMPLES], "observed_overlap_cue": got["spans"], "withheld": veto})
        decisions.append((row, veto))
    if review_evidence != evidence or dict(true_false) != {k: v for k, v in result["baseline"].items() if v}:
        raise ValueError("Source evidence intervals or baseline count differ")
    if result["native_localization"] != {f"{key}_samples": audio_counts[key] for key in ("shared", "cue", "annotated")}:
        raise ValueError("Native annotation overlap arithmetic differs")
    for cell in result["room_acoustic"]:
        room = cell["room"]
        if (cell["high_marked"], cell["minimal_marked"]) != (audio_counts[(room, "high")], audio_counts[(room, "minimal")]):
            raise ValueError("Native source operating point counts differ")
    rule = freeze["advance_rule"]
    coverage = all(speech_counts[cell["room"]] >= rule["minimum_genuine_assessed_in_each_mixed_room"] for cell in result["room_acoustic"])
    acoustic = (sum(cell["high_marked"] for cell in result["room_acoustic"]) >= rule["minimum_high_marked_confirmation"]
                and all(cell["high_marked"] >= rule["minimum_high_marked_each_room"] for cell in result["room_acoustic"])
                and sum(cell["minimal_marked"] for cell in result["room_acoustic"]) <= rule["maximal_minimal_false_confirmation"])
    if (coverage, acoustic) != (result["genuine_coverage_pass"], result["acoustic_pass"]):
        raise ValueError("Independent source coverage criteria differ")
    for name, outcome in result["policies"].items():
        removed = [row for row, veto in decisions if veto[name]]
        false = [row for row in removed if not synthetic(row)]
        true = [row for row in removed if synthetic(row)]
        unique = len({row["group"] for row in false})
        clean = sum(row["condition"] == "clean" for row in true)
        eligible = (len(false) >= rule["minimum_corrected_genuine_ai"]
                    and len(false) >= math.ceil(true_false["false"] * rule["minimum_fraction_genuine_ai_corrected"])
                    and unique >= rule["minimum_corrected_genuine_groups"]
                    and len(true) <= rule["maximum_synthetic_ai_lost"]
                    and true_false["synthetic_ai"] - len(true) >= math.ceil(
                        true_false["synthetic_ai"] * rule["minimum_fraction_synthetic_ai_preserved"])
                    and clean <= rule["maximum_clean_synthetic_ai_lost"])
        if (len(false), len(true), unique, clean, eligible, coverage and eligible and (acoustic or name == "independent_disagreement")) != (
                outcome["false_withheld"], outcome["true_withheld"], outcome["false_groups"], outcome["clean_true_withheld"],
                outcome["passes_decision"], outcome["advance"]):
            raise ValueError("Independent abstention decision differs")
    expected_forwards = true_false["false"] + true_false["synthetic_ai"]
    if result["independent_forwards"] != expected_forwards + int(expected_forwards > 0):
        raise ValueError("Independent inference count differs from the selected AI windows and deterministic repeat")
    return {"checked_pcm": len(roster), "development_point_views": selection,
            "exact_mixes": mixes, "verified_native_cuts": native,
            "verified_independent_graph_inputs": expected_forwards, "verified_evidence_rows": len(evidence),
            "advancing_policies": [name for name, data in result["policies"].items() if data["advance"]]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("folder", "ami", "development_ami"):
        parser.add_argument(argument, type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.folder, args.ami, args.development_ami), sort_keys=True))
