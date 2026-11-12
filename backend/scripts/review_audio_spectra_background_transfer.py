"""Recheck exact waveforms, independently calibrated directions and aggregate counts."""

import argparse
import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path

NATIVE, GRAPH = 48_000, 64_600


def sha(body):
    return hashlib.sha256(body).hexdigest()


def load(folder, name, digest=None):
    body = (folder / name).read_bytes()
    if digest is not None and sha(body) != digest:
        raise ValueError(f"Commitment mismatch in {name}")
    return json.loads(body)


def array(raw):
    return struct.unpack(f"<{NATIVE}h", raw)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def graph_sha(body):
    samples = [sample / 32768 for sample in array(body)]
    multiplier = f32(.97)
    filtered = [samples[0], *(a - f32(multiplier * b) for a, b in zip(samples[1:], samples[:-1], strict=True))]
    return sha(struct.pack(f"<{GRAPH}f", *filtered, *filtered[:GRAPH - NATIVE]))


def audit(folder):
    result = load(folder, "aggregate.json")
    freeze_sha = result["freeze_sha256"]
    lock = load(folder, "freeze.json", freeze_sha)
    roster = load(folder, "roster.json", lock["roster_sha256"])
    point = load(folder, "point.json", result["point_sha256"])
    baseline = load(folder, "baseline.json", result["baseline_sha256"])
    confirm = load(folder, "confirm.json", result["confirmation_sha256"])
    evidence = load(folder, "evidence-private.json", result["evidence_sha256"])
    if (result["freeze"] != lock or confirm["points_sha256"] != result["point_sha256"] or
            confirm["baseline_sha256"] != result["baseline_sha256"] or
            any(stage["freeze_sha256"] != freeze_sha for stage in (point, baseline, confirm)) or
            dict(Counter(row["role"] for row in roster)) != lock["planned"]):
        raise ValueError("The reviewed stages do not share one exact manifest")
    raw_by_hash = {}
    for index, row in enumerate(roster):
        body = (folder / "pcm" / f"{index:03}.pcm").read_bytes()
        if len(body) != NATIVE * 2 or row["id"] != index or sha(body) != row["pcm_sha256"]:
            raise ValueError(f"Exact native PCM differs at {index}")
        raw_by_hash[row["pcm_sha256"]] = body
    regenerated = 0
    for row in roster:
        if "clean_pcm_sha256" not in row:
            continue
        clean = array(raw_by_hash[row["clean_pcm_sha256"]])
        background = array(raw_by_hash[row["background"]["pcm_sha256"]])
        scale, gain = row["scale"], row["common_gain"]
        mixed = [a + scale * b for a, b in zip(clean, background, strict=True)]
        ratio = math.fsum(x*x for x in clean) / math.fsum((scale * x)**2 for x in background)
        expected_gain = min(1.0, 31_129 / max(map(abs, mixed)))
        if abs(10 * math.log10(ratio) + 5) > 1e-10 or abs(gain - expected_gain) > 1e-13:
            raise ValueError("Fresh mixture differs from frozen SNR or gain")
        if sha(struct.pack(f"<{NATIVE}h", *(round(x * gain) for x in mixed))) != row["pcm_sha256"]:
            raise ValueError("Fresh paired PCM differs from exact reconstruction")
        regenerated += 1
    eligible = [r for r in point["rows"] if r["family"] in {"genuine", "background"} and r["decision"] == "inconclusive"]
    genuine_groups = {r["group"] for r in eligible if r["family"] == "genuine"}
    boundary = {"candidate": max(r["candidate"] for r in eligible), "raw": max(r["raw"] for r in eligible),
                "eligible_views": len(eligible), "genuine_groups": len(genuine_groups)}
    if boundary != point["points"] or len(genuine_groups) < lock["rule"]["minimum_point_genuine_groups"]:
        raise ValueError("Operating point differs from the separately committed controls")
    checked = {r["id"]: r for r in baseline["rows"]}
    counts, ranks, flags = defaultdict(Counter), defaultdict(list), []
    for role, stage in (("point", point), ("transfer", confirm)):
        expected = [row for row in roster if (row["role"] == "point") == (role == "point")]
        assessed = 0
        for row, got in zip(expected, stage["rows"], strict=True):
            prior = row["baseline"] or checked.pop(row["id"])
            if (row["id"], row["family"], row["group"], row["pcm_sha256"], prior["decision"], prior.get("raw")) != (
                    got["id"], got["family"], got["group"], got["pcm_sha256"], got["decision"], got["raw"]):
                raise ValueError("Measured data differ from the source or incumbent record")
            ready = got["decision"] != "unassessed"
            assessed += ready
            digest = graph_sha(raw_by_hash[row["pcm_sha256"]]) if ready else None
            if digest != got["model_input_sha256"] or ready != (got["candidate"] is not None):
                raise ValueError("Candidate and incumbent input routing differ")
            if role == "point":
                continue
            old, condition, family = got["decision"], row["condition"], row["family"]
            alternate = ready and got["candidate"] > boundary["candidate"]
            new = old == "inconclusive" and alternate
            raw_point = old == "inconclusive" and got["raw"] > boundary["raw"]
            counts[(row["role"], condition, family)].update(planned=1, assessed=ready, incumbent_ai=old == "ai",
                 incumbent_original=old == "original", incumbent_inconclusive=old == "inconclusive", unassessed=not ready,
                 candidate_alone_ai=alternate, combined_ai=old == "ai" or new, raw_point_combined_ai=old == "ai" or raw_point)
            if ready and family != "background":
                ranks[(row["role"], condition)].append((family != "genuine", got["raw"], got["candidate"]))
            if new:
                flags.append({"id": row["id"], "role": row["role"], "condition": condition, "family": family,
                              "pcm_sha256": row["pcm_sha256"], "first_sample": 0, "end_sample": NATIVE,
                              "time_base": "1/16000", "candidate_input_sha256": digest})
        if stage["forwards"] != assessed + 1:
            raise ValueError("Forward count does not include all assessed inputs and exact repeat")
    rows = [{"role": r, "condition": c, "family": f, **stats} for (r, c, f), stats in sorted(counts.items())]
    if checked or rows != result["counts"] or flags != evidence or len(flags) != result["source_evidence_intervals"]:
        raise ValueError("Independent source intervals or report counts differ")
    for cell in result["ranking"]:
        values = ranks[(cell["role"], cell["condition"])]
        for i, key in ((1, "raw_auc"), (2, "candidate_auc")):
            positive, negative = ([v[i] for v in values if v[0] == target] for target in (True, False))
            ties = sum(a == b for a in positive for b in negative)
            wins = sum(a > b for a in positive for b in negative)
            computed = (wins + ties / 2) / (len(positive) * len(negative)) if positive and negative else None
            if computed != cell[key]:
                raise ValueError("Independent rank comparison differs")
    false = [item for item in flags if item["role"] == "prospective" and item["family"] in {"genuine", "background"}]
    rooms = ("mostly_single_interferer", "overlapping_interferers")
    gains = [item for item in flags if item["role"] == "prospective" and item["condition"] in rooms and item["family"] not in {"genuine", "background"}]
    groups = len({roster[item["id"]]["group"] for item in gains})
    families = len({item["family"] for item in gains})
    rule = lock["rule"]
    checks = {"genuine": len(false) <= rule["additional_genuine_max"],
              "clean": sum(r["condition"] == "clean" for r in false) <= rule["additional_clean_genuine_max"],
              "isolated": sum(r["condition"].startswith("isolated_") for r in false) <= rule["isolated_genuine_max"],
              "new_groups": groups >= rule["minimum_new_noisy_synthetic_groups"],
              "new_families": families >= rule["minimum_new_noisy_synthetic_families"],
              "scenes_with_gain": len({r["condition"] for r in gains}) >= rule["minimum_prospective_scenes_with_gain"],
              "genuine_assessed": all(counts[("prospective", c, "genuine")]["assessed"] >= rule["minimum_genuine_assessed_each_prospective_scene"] for c in rooms)}
    if (checks, all(checks.values()), groups, families) != (result["checks"], result["advance"], result["prospective_gained_groups"], result["prospective_gained_families"]):
        raise ValueError("Independent advancement decision differs")
    return {"verified_native_waveforms": len(roster), "exact_remixes": regenerated,
            "replayed_model_inputs": point["forwards"] + confirm["forwards"] - 2,
            "verified_intervals": len(evidence), "advance": result["advance"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.folder), sort_keys=True))
