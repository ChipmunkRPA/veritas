"""Audit traffic enhancement PCM, scoring arithmetic and grouped counts without model inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import struct
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = 48_000
FAMILIES = {"clone", "expressive", "stock"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def samples(path, expected):
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, "PCM commitment differs")
    return struct.unpack(f"<{SAMPLES}h", data)


def db(num, den):
    return 10 * math.log10(max(num, 1e-18) / max(den, 1e-18))


def power(values):
    return sum(value * value for value in values)


def quality(value, ref):
    mean_v, mean_r = statistics.mean(value), statistics.mean(ref)
    projected = math.fsum((a - mean_v) * (b - mean_r) for a, b in zip(value, ref, strict=True))
    ref_power = math.fsum((b - mean_r) ** 2 for b in ref)
    factor = projected / ref_power
    residual = math.fsum((a - mean_v - factor * (b - mean_r)) ** 2 for a, b in zip(value, ref, strict=True))
    return db(factor**2 * ref_power, residual)


def independent_direction(row, calibration):
    raw = row["raw"]
    if raw is None:
        require(row["decision"] == "unassessed", "Missing raw response cannot be classified")
        return "unassessed", 0
    require(0 <= raw <= 1, "Invalid classifier response")
    raw = max(1e-6, min(1 - 1e-6, raw))
    fit = calibration["parameters"]
    prob = 1 / (1 + math.exp(-(fit["slope"] * (math.log(raw) - math.log1p(-raw)) + fit["intercept"])))
    ranks = []
    for label, score in enumerate((prob, 1 - prob)):
        guards = calibration["guards"][str(label)]
        if (1 + sum(m is None or m >= score for m in guards)) / (1 + len(guards)) > calibration["alpha"]:
            ranks.append(label)
    if ranks == [0] and prob <= .2 + 1e-12:
        require(row["probe"] == "computed" and row["stability_change"] is not None,
                "Original-side response lacks a completed stability check")
        decision = "inconclusive" if row["stability_change"] > math.log(2) else "original"
        forwards = 2
    else:
        require(row["probe"] == "not_required_for_decision" and row["stability_change"] is None,
                "Unexpected classifier stability response")
        decision = "ai" if ranks == [1] and prob >= .8 else "inconclusive"
        forwards = 1
    require(row["decision"] == decision, "Classified direction differs from frozen calibration arithmetic")
    return decision, forwards


def audit(folder):
    plan_path, roster_path = folder / "freeze.json", folder / "roster.json"
    enhanced_path, score_path = folder / "enhancement.json", folder / "scores.json"
    plan, roster = read(plan_path), read(roster_path)
    enhancement, scores, aggregate = read(enhanced_path), read(score_path), read(folder / "aggregate.json")
    plan_sha = digest(plan_path)
    require(plan_sha == aggregate["freeze_sha256"] == scores["freeze_sha256"] == enhancement["freeze_sha256"]
            and aggregate["freeze"] == plan and plan["roster_sha256"] == digest(roster_path)
            and digest(enhanced_path) == scores["enhancement_sha256"] == aggregate["enhancement_sha256"]
            and digest(score_path) == aggregate["score_sha256"], "Stage commitments differ")
    require(all(digest(ROOT / name) == sha for name, sha in plan["code_sha256"].items()), "Frozen implementation differs")
    clean = {r["id"]: samples(folder / "pcm" / f"{r['id']:03}.pcm", r["pcm_sha256"])
             for r in roster if r["condition"] == "clean"}
    require(len(clean) == plan["foreground_groups"] and len(roster) == plan["planned"], "Frozen case denominator differs")
    expected_views = {(i, level, tape) for i in clean for level in ("-5", "-10") for tape in plan["traffic_tapes"]}
    measured_views, noises, cells, medians = set(), {}, defaultdict(Counter), defaultdict(list)
    gains, flags, newly_assessed, clean_quality = [], [], [], []
    total_forwards, noise_error_db, largest_roundtrip_delta = 0, 0., 0
    calibration = read(ROOT / "backend/app/detectors/audio/calibrations/local-speech-v3.json")
    rule = plan["rule"]
    for i, (row, enhancement_row, score) in enumerate(zip(roster, enhancement["rows"], scores["rows"], strict=True)):
        require(i == row["id"] == enhancement_row["id"] == score["id"]
                and row["pcm_sha256"] == score["pcm_sha256"]
                and enhancement_row["enhanced_sha256"] == score["enhanced_sha256"], "Case binding differs")
        current = samples(folder / "pcm" / f"{i:03}.pcm", row["pcm_sha256"])
        altered = samples(folder / "enhanced" / f"{i:03}.pcm", score["enhanced_sha256"])
        suppression = db(power(current), power(altered))
        require(abs(suppression - enhancement_row["suppression_db"]) < 1e-9, "Wrong measured suppression")
        orig, orig_forwards = independent_direction(score["current"], calibration)
        enhanced, new_forwards = independent_direction(score["enhanced"], calibration)
        total_forwards += orig_forwards + new_forwards
        use_enhanced = suppression >= rule["minimum_suppression_db"] and enhanced == "ai" and orig in {"inconclusive", "unassessed"}
        combined = "ai" if use_enhanced else orig
        level, tape, family = row["condition"], row["tape"], row["family"]
        for scene in {"all", tape}:
            for arm, direction in (("current", orig), ("enhanced", enhanced), ("combined", combined)):
                cells[(level, scene, family, arm)][direction] += 1
        if use_enhanced:
            flags.append(row)
        if level == "isolated":
            require(row["noise_sha256"] == row["pcm_sha256"], "Isolated background differs from source excerpt")
            continue
        foreground = clean[row["clean_id"]]
        after = quality(altered, foreground)
        require(abs(after - enhancement_row["enhanced_si_sdr_db"]) < 1e-8, "Wrong acoustic fidelity value")
        if level == "clean":
            clean_quality.append(after)
            continue
        key = (row["clean_id"], level, tape)
        require(key not in measured_views and key in expected_views, "Unexpected or repeated matched input")
        measured_views.add(key)
        noise_hash = row["noise_sha256"]
        if noise_hash not in noises:
            noises[noise_hash] = samples(folder / "noise" / f"{noise_hash}.pcm", noise_hash)
        noise = noises[noise_hash]
        noise_multiplier = math.sqrt(power(foreground) / power(noise)) * 10 ** (-int(level) / 20)
        mixtures = [a + b * noise_multiplier for a, b in zip(foreground, noise, strict=True)]
        gain = min(1., .95 * 32768 / max(abs(sample) for sample in mixtures))
        require(abs(gain - row["gain"]) < 1e-13, "Common attenuation differs")
        reconstructed = [round(sample * gain) for sample in mixtures]
        worst = max(abs(a - b) for a, b in zip(current, reconstructed, strict=True))
        require(worst == 0, "Frozen mixture does not reconstruct exactly")
        largest_roundtrip_delta = max(largest_roundtrip_delta, worst)
        residual = [a - gain * b for a, b in zip(current, foreground, strict=True)]
        snr = db(gain**2 * power(foreground), power(residual))
        noise_error_db = max(noise_error_db, abs(snr - int(level)))
        improvement = after - quality(current, foreground)
        require(abs(improvement - enhancement_row["si_sdr_gain_db"]) < 1e-8, "Wrong acoustic improvement")
        medians[(level, tape)].append(improvement)
        if orig == "unassessed" and enhanced != "unassessed" and family in FAMILIES:
            newly_assessed.append(row)
        if use_enhanced and family in FAMILIES:
            gains.append(row)
    require(measured_views == expected_views and total_forwards == scores["voice_forwards"], "Matched views or classifier forward counts differ")
    require(enhancement["frames_per_view"] == 188 and enhancement["onnx_forwards"] == 188 * (len(roster) + 1), "Enhancement forward count differs")
    require(len(aggregate["counts"]) * 3 == len(cells), "Published group coverage differs")
    for row in aggregate["counts"]:
        for arm in ("current", "enhanced", "combined"):
            require(dict(cells[(row["condition"], row["tape"], row["family"], arm)]) == row[arm], "Published arm totals differ")
    outcomes = aggregate["conditional"]
    expected = {"new_true_ai": len(gains), "gained_groups": len({r["group"] for r in gains}),
                "gained_tapes": len({r["tape"] for r in gains}),
                "new_false_ai": sum(r["family"] == "genuine" and r["condition"] in {"-5", "-10"} for r in flags),
                "new_clean_ai": sum(r["condition"] == "clean" for r in flags),
                "new_isolated_ai": sum(r["condition"] == "isolated" for r in flags),
                "enhanced_newly_assessed_synthetic": len(newly_assessed),
                "enhanced_newly_assessed_tapes": len({r["tape"] for r in newly_assessed})}
    require(all(outcomes[k] == v for k, v in expected.items()), "Conditional counts differ")
    clean_median = statistics.median(clean_quality)
    require(abs(clean_median - aggregate["acoustic"]["median_clean_si_sdr_db"]) < 1e-8, "Clean acoustic aggregate differs")
    for (level, tape), values in medians.items():
        require(abs(statistics.median(values) - aggregate["acoustic"]["median_noisy_si_sdr_gains_db"][f"{level}:{tape}"]) < 1e-8,
                "Noisy acoustic aggregate differs")
    checks = {"true_additions": expected["new_true_ai"] >= rule["minimum_true_additions"],
              "gained_voice_groups": expected["gained_groups"] >= rule["minimum_gained_voice_groups"],
              "gained_tapes": expected["gained_tapes"] >= rule["minimum_tapes_with_gain"],
              "false_additions": expected["new_false_ai"] <= rule["maximum_false_additions"],
              "clean_additions": expected["new_clean_ai"] <= rule["maximum_clean_additions"],
              "isolated_additions": expected["new_isolated_ai"] <= rule["maximum_isolated_additions"],
              "newly_assessed": expected["enhanced_newly_assessed_synthetic"] >= rule["minimum_newly_assessed_synthetic"],
              "newly_assessed_tapes": expected["enhanced_newly_assessed_tapes"] >= rule["minimum_tapes_with_new_assessment"],
              "clean_fidelity": clean_median >= rule["minimum_clean_median_si_sdr_db"],
              "noisy_fidelity": all(sum(statistics.median(medians[(level, tape)]) >= rule["minimum_median_noisy_si_sdr_gain_db"]
                                        for tape in plan["traffic_tapes"]) >= rule["minimum_tapes_per_level_with_si_sdr_gain"]
                                    for level in ("-5", "-10"))}
    require(outcomes["checks"] == checks and outcomes["advance"] == all(checks.values()), "Advancement criteria differ")
    return {"verified_original_enhanced_pairs": len(roster), "reconstructed_mixtures": len(measured_views),
            "maximum_snr_error_db": noise_error_db, "max_reconstruction_error_samples": largest_roundtrip_delta,
            "verified_voice_forwards": total_forwards, "verified_decisions": 2 * len(roster), "aggregate_sha256": digest(folder / "aggregate.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--published", type=Path)
    args = parser.parse_args()
    result = audit(args.folder)
    if args.published:
        require(digest(args.published) == result["aggregate_sha256"], "Published aggregate differs")
    print(json.dumps(result, sort_keys=True))
