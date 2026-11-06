"""Fresh confirmation of a narrowly scoped possible-background-transition lead.

The first study suggested steady background benefit. This explicitly post-study
follow-up freezes that question, unchanged method and threshold, new recording
groups, and disjoint donor pairs before looking at any new outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import urllib.parse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from evaluate_background_continuity import REVISION, ROOT, fetch, order, save, wilson
from scipy.signal import resample_poly

from app.bench.audio_continuity import candidates
from app.core.canonical import canonical_hash
from app.detectors.audio.decode import read_wav, wav_bytes
from app.evidence.audio_background import _candidates, measurements

FOLDER = ROOT / "fresh-followup"


def source_group(row):
    name = Path(row["wavename"]).stem
    if row["dataset"] == "UrbanSound8K":
        group = name.split("-")[0]
    elif row["dataset"] == "TUTASC2019Dev":
        group = "-".join(name.split("-")[:-2])
    else:
        group = name.rsplit("_", 1)[0]
    return row["dataset"] + ":" + group


def prepare():
    FOLDER.mkdir(exist_ok=True)
    for name in ("metadata", "source", "inputs"):
        (FOLDER / name).mkdir(exist_ok=True)
    prior = json.loads((ROOT / "protocol.json").read_bytes())
    used = {r["source_group"] for r in prior["selection"]}
    rows = [r for r in csv.DictReader((ROOT / "datasplit_dev.csv").open()) if r["usage"] == "validation"]
    categories = {
        "engine": (24, lambda r: r["event"] == "engine_idling"),
        "air_conditioner": (4, lambda r: r["event"] == "air_conditioner"),
        "music": (24, lambda r: r["event"] == "street_music"),
        "home": (12, lambda r: r["scene"] == "home"),
        "outdoor": (12, lambda r: r["scene"] in {"park", "street_traffic", "street_pedestrian"}),
    }
    selected = []
    for category, (count, predicate) in categories.items():
        groups = defaultdict(list)
        for index, row in enumerate(rows):
            group = source_group(row)
            if group not in used and predicate(row):
                groups[group].append((index, row))
        keys = sorted(groups, key=lambda k: order("fresh-followup:" + k))[:count]
        if len(keys) != count:
            raise RuntimeError(f"Not enough fresh source groups for {category}")
        for index, key in enumerate(keys):
            offset, row = min(groups[key], key=lambda item: order("fresh-followup:" + item[1]["wavename"]))
            selected.append(
                {
                    "source_group": key,
                    "category": category,
                    "pair_id": f"{category}:{index // 2:02d}",
                    "pair_side": index % 2,
                    "row_offset": offset * 5,
                    "expected": row,
                }
            )
    threshold = json.loads((ROOT / "thresholds.json").read_bytes())["thresholds"][
        "stationary_spectral_continuity"
    ]
    protocol = {
        "schema": "chipmunk.background-continuity-followup/1",
        "selection": selected,
        "revision": REVISION,
        "prior_study_sha256": canonical_hash(prior),
        "threshold": threshold,
        "method_sha256": hashlib.sha256(
            Path(__file__).parents[1].joinpath("app/bench/audio_continuity.py").read_bytes()
        ).hexdigest(),
        "claim": "Possible background transition for manual review; no AI-origin or edit-boundary proof.",
        "why_followup": "Post-first-study hypothesis: stationarity-gated continuity may improve steady background transition leads. Confirm on entirely fresh source groups.",
        "cases": ["original_left", "original_right", "gain_only", "hard_splice", "crossfade_splice"],
        "source_policy": "Recording/location groups disjoint from entire first study; disjoint source pairs; all conditions clustered by pair.",
        "derivation": "First4s at16kHz mono; -24dBFSRMS,peakcap0.8; donor tail matchedto recipienttail RMS; boundary2s; crossfade100ms aroundboundary.",
        "uncertainty": "Paired cluster bootstrap10000replicates over disjoint source pairs; original-control FPR uncertainty by pair(anyflag).",
        "decision_policy": "No retuning. Consider manual-review lead only if fresh paired localized-recall difference lower95%bound>0 and untouched-control FPR point<=10%, upper95%<=20%. Show category results and abstention; no all-background/AI-origin claim.",
    }
    save(FOLDER / "protocol.json", protocol)

    def download(item):
        cached = FOLDER / "metadata" / f"{item['row_offset']}.json"
        if cached.exists():
            data = json.loads(cached.read_bytes())
        else:
            query = urllib.parse.urlencode(
                {
                    "dataset": "EnvSDD/EnvSDD",
                    "config": "default",
                    "split": "validation",
                    "offset": item["row_offset"],
                    "length": 1,
                }
            )
            data = json.loads(fetch("https://datasets-server.huggingface.co/rows?" + query))
            save(cached, data)
        row = data["rows"][0]["row"]
        url = row["audio"][0]["src"]
        if row["filename"] != item["expected"]["wavename"] or row["label"] != "real" or REVISION not in url:
            raise RuntimeError("Changed source selection")
        path = FOLDER / "source" / f"{item['row_offset']}.wav"
        if not path.exists():
            path.write_bytes(fetch(url))
        raw = path.read_bytes()
        pcm, rate = sf.read(io.BytesIO(raw), dtype="float64", always_2d=True)
        pcm = pcm.mean(axis=1)
        if rate != 16000:
            divisor = math.gcd(rate, 16000)
            pcm = resample_poly(pcm, 16000 // divisor, rate // divisor)
        if len(pcm) < 64000 or np.sqrt(np.mean(pcm[:64000] ** 2)) < 1e-6:
            raise RuntimeError("Short/quiet source; do not substitute after scores")
        pcm = pcm[:64000]
        gain = min(10 ** (-24 / 20) / np.sqrt(np.mean(pcm * pcm)), 0.8 / max(np.abs(pcm)))
        data = wav_bytes(pcm * gain)
        sha = hashlib.sha256(data).hexdigest()
        (FOLDER / "inputs" / f"{sha}.wav").write_bytes(data)
        print("fresh source", item["category"], item["source_group"], flush=True)
        return {
            **item,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_path": str(path.relative_to(ROOT)),
            "source_url": url.split("?")[0],
            "source_sample_rate": rate,
            "normalization_gain": float(gain),
            "input_path": str((FOLDER / "inputs" / f"{sha}.wav").relative_to(ROOT)),
            "input_sha256": sha,
        }

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(download, selected))
    cases = []
    pairs = defaultdict(list)
    for record in records:
        pairs[record["pair_id"]].append(record)
    for pair_id, pair in pairs.items():
        left, right = sorted(pair, key=lambda r: r["pair_side"])
        x, y = [read_wav((ROOT / r["input_path"]).read_bytes()).astype(np.float64) for r in (left, right)]
        gain = float(np.sqrt(np.mean(x[32000:] ** 2)) / max(np.sqrt(np.mean(y[32000:] ** 2)), 1e-9))
        hard = np.concatenate([x[:32000], gain * y[32000:]])
        fade = hard.copy()
        t = np.linspace(0, 1, 1600)
        fade[31200:32800] = (1 - t) * x[31200:32800] + t * gain * y[31200:32800]
        gainonly = x.copy()
        gainonly[32000:] *= 0.25
        for kind, pcm, boundaries in [
            ("original_left", x, []),
            ("original_right", y, []),
            ("gain_only", gainonly, []),
            ("hard_splice", hard, [32000]),
            ("crossfade_splice", fade, [32000]),
        ]:
            peak_gain = 0.8 / max(0.8, float(np.max(np.abs(pcm))))
            data = wav_bytes(pcm * peak_gain)
            sha = hashlib.sha256(data).hexdigest()
            (FOLDER / "inputs" / f"{sha}.wav").write_bytes(data)
            cases.append(
                {
                    "id": pair_id + ":" + kind,
                    "pair_id": pair_id,
                    "category": left["category"],
                    "case": kind,
                    "source_groups": [left["source_group"], right["source_group"]],
                    "input_path": str((FOLDER / "inputs" / f"{sha}.wav").relative_to(ROOT)),
                    "input_sha256": sha,
                    "source_samples": 64000,
                    "sample_rate": 16000,
                    "boundary_samples": boundaries,
                    "donor_gain": gain if boundaries else None,
                    "final_peak_gain": peak_gain,
                    "crossfade_interval": {"start_sample": 31200, "end_sample": 32800}
                    if kind == "crossfade_splice"
                    else None,
                    "source_intervals": (
                        [
                            {
                                "source_sha256": left["source_sha256"],
                                "start_sample": 0,
                                "end_sample": 32800 if kind == "crossfade_splice" else 32000,
                            },
                            {
                                "source_sha256": right["source_sha256"],
                                "start_sample": 31200 if kind == "crossfade_splice" else 32000,
                                "end_sample": 64000,
                            },
                        ]
                        if boundaries
                        else [
                            {
                                "source_sha256": right["source_sha256"]
                                if kind == "original_right"
                                else left["source_sha256"],
                                "start_sample": 0,
                                "end_sample": 64000,
                            }
                        ]
                    ),
                }
            )
    save(
        FOLDER / "manifest.json",
        {"protocol_sha256": canonical_hash(protocol), "records": records, "cases": cases},
    )
    print({"sources": len(records), "pairs": len(pairs), "cases": len(cases)}, flush=True)


def score():
    manifest = json.loads((FOLDER / "manifest.json").read_bytes())
    output = []
    for case in manifest["cases"]:
        raw = (ROOT / case["input_path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != case["input_sha256"]:
            raise RuntimeError("Changed source")
        pcm = read_wav(raw)
        rows = candidates(pcm)
        baseline = [
            {"boundary_sample": r["change_sample"], "score": abs(r["value"])}
            for r in _candidates(measurements(pcm))
            if r["category"] == "energy_change"
        ]
        output.append({"id": case["id"], "windows": rows, "baseline": baseline})
    save(FOLDER / "scores.json", {"manifest_sha256": canonical_hash(manifest), "rows": output})


def evaluate():
    manifest = json.loads((FOLDER / "manifest.json").read_bytes())
    scores = json.loads((FOLDER / "scores.json").read_bytes())
    protocol = json.loads((FOLDER / "protocol.json").read_bytes())
    joined = [
        {**c, **s} for c, s in zip(manifest["cases"], scores["rows"], strict=True) if c["id"] == s["id"]
    ]
    if len(joined) != len(manifest["cases"]) or scores["manifest_sha256"] != canonical_hash(manifest):
        raise RuntimeError("Source bindings mismatch")
    for row in joined:
        row["baseline_positive"] = bool(row["baseline"])
        row["new_positive"] = any(
            w["stationary_spectral_distance"] > protocol["threshold"] for w in row["windows"]
        )
        row["baseline_localized"] = any(
            any(abs(w["boundary_sample"] - b) <= 4000 for b in row["boundary_samples"])
            for w in row["baseline"]
        )
        row["new_localized"] = any(
            w["stationary_spectral_distance"] > protocol["threshold"]
            and any(abs(w["boundary_sample"] - b) <= 4000 for b in row["boundary_samples"])
            for w in row["windows"]
        )
    cohorts = {}
    for category in ["all", *sorted({r["category"] for r in joined})]:
        subset = [r for r in joined if category == "all" or r["category"] == category]
        cohorts[category] = {}
        for kind in ["original", "gain_only", "hard_splice", "crossfade_splice"]:
            rows = [r for r in subset if r["case"].startswith(kind)]
            metric = {"n": len(rows)}
            for method in ["baseline", "new"]:
                metric[method + "_flags"] = sum(r[method + "_positive"] for r in rows)
                metric[method + "_localized"] = sum(r[method + "_localized"] for r in rows)
            cohorts[category][kind] = metric
    grouped = defaultdict(list)
    for row in joined:
        grouped[row["pair_id"]].append(row)
    pairs = list(grouped.values())
    rng = np.random.default_rng(20261104)
    differences = []
    false_rates = []
    for _ in range(10000):
        sample = [row for index in rng.integers(0, len(pairs), len(pairs)) for row in pairs[index]]
        cut = [r for r in sample if r["case"] == "hard_splice"]
        original = [r for r in sample if r["case"].startswith("original")]
        differences.append(np.mean([int(r["new_localized"]) - int(r["baseline_localized"]) for r in cut]))
        false_rates.append(np.mean([r["new_positive"] for r in original]))
    pair_any = sum(any(r["new_positive"] for r in pair if r["case"].startswith("original")) for pair in pairs)
    difference_ci = [float(v) for v in np.quantile(differences, [0.025, 0.975])]
    false_ci = [float(v) for v in np.quantile(false_rates, [0.025, 0.975])]
    false_point = cohorts["all"]["original"]["new_flags"] / cohorts["all"]["original"]["n"]
    pair_ci = wilson(pair_any, len(pairs))
    passed = difference_ci[0] > 0 and false_point <= 0.1 and pair_ci["high"] <= 0.2
    output = {
        "protocol_sha256": canonical_hash(protocol),
        "manifest_sha256": canonical_hash(manifest),
        "scores_sha256": canonical_hash(scores),
        "cohorts": cohorts,
        "paired_localized_recall_difference_ci95": difference_ci,
        "untouched_control_fpr_cluster_ci95": false_ci,
        "untouched_pair_any_false_flag_wilson95": pair_ci,
        "eligible_as_manual_review_lead": passed,
        "eligible_as_ai_origin_detector": False,
        "deployment_eligible": False,
        "threshold": protocol["threshold"],
        "source_group_overlap_with_first_study": sorted(
            {r["source_group"] for r in manifest["records"]}
            & {r["source_group"] for r in json.loads((ROOT / "protocol.json").read_bytes())["selection"]}
        ),
    }
    save(FOLDER / "evaluation.json", output)
    print(json.dumps(output, indent=2))


def review():
    """Supplemental additive comparison for the bounded product presentation."""
    from app.evidence.audio_continuity import select

    manifest = json.loads((FOLDER / "manifest.json").read_bytes())
    scores = json.loads((FOLDER / "scores.json").read_bytes())
    if scores["manifest_sha256"] != canonical_hash(manifest):
        raise RuntimeError("Source bindings mismatch")
    records = []
    for case, scored in zip(manifest["cases"], scores["rows"], strict=True):
        if case["id"] != scored["id"]:
            raise RuntimeError("Changed score identity")
        selected = select(scored["windows"])
        localized = lambda rows, case=case: any(  # noqa: E731
            abs(row["boundary_sample"] - boundary) <= 4000
            for row in rows for boundary in case["boundary_samples"]
        )
        records.append({"id": case["id"], "pair_id": case["pair_id"], "case": case["case"],
                        "baseline_flag": bool(scored["baseline"]), "selected_flag": bool(selected),
                        "baseline_localized": localized(scored["baseline"]),
                        "selected_localized": localized(selected), "selected": selected})
    groups = defaultdict(list)
    for record in records:
        groups[record["pair_id"]].append(record)
    pairs = list(groups.values())
    rng = np.random.default_rng(20261105)
    gains, extra_false = [], []
    for _ in range(10000):
        sample = [row for index in rng.integers(0, len(pairs), len(pairs)) for row in pairs[index]]
        cuts = [row for row in sample if row["case"] == "hard_splice"]
        controls = [row for row in sample if row["case"].startswith("original")]
        gains.append(np.mean([row["selected_localized"] and not row["baseline_localized"] for row in cuts]))
        extra_false.append(np.mean([row["selected_flag"] and not row["baseline_flag"] for row in controls]))
    counts = {}
    for kind in ("original", "gain_only", "hard_splice", "crossfade_splice"):
        subset = [row for row in records if row["case"].startswith(kind)]
        counts[kind] = {"n": len(subset),
                        "baseline_flags": sum(row["baseline_flag"] for row in subset),
                        "selected_flags": sum(row["selected_flag"] for row in subset),
                        "combined_flags": sum(row["baseline_flag"] or row["selected_flag"] for row in subset),
                        "baseline_localized": sum(row["baseline_localized"] for row in subset),
                        "selected_localized": sum(row["selected_localized"] for row in subset),
                        "combined_localized": sum(row["baseline_localized"] or row["selected_localized"] for row in subset)}
    output = {"analysis": "Post-result supplemental product-scope comparison; does not replace the frozen preregistered outcome.",
              "manifest_sha256": canonical_hash(manifest), "scores_sha256": canonical_hash(scores),
              "product_selection": "At most four strongest one-second supports, positions at least one second apart; existing energy leads preserved.",
              "counts": counts, "rows": records,
              "additional_localized_fraction_ci95": [float(v) for v in np.quantile(gains, [.025, .975])],
              "additional_original_flag_fraction_ci95": [float(v) for v in np.quantile(extra_false, [.025, .975])],
              "claim": "Possible sound-content transition for listening review; no AI-origin or exact splice proof."}
    save(FOLDER / "product-selection.json", output)
    print(json.dumps({k: v for k, v in output.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "score", "evaluate", "review"])
    args = parser.parse_args()
    {"prepare": prepare, "score": score, "evaluate": evaluate, "review": review}[args.stage]()
