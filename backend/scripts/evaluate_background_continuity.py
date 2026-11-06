"""Fresh, source-grouped environmental-audio evaluation. Research only.

Prepare freezes source selection and derivations before any scores. Score never
downloads or selects data. Evaluate freezes thresholds on calibration controls
and reports untouched test controls separately from synthetic-origin auditing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from app.bench.audio_continuity import candidates
from app.core.canonical import canonical_hash, canonical_json
from app.detectors.audio.decode import read_wav, wav_bytes
from app.evidence.audio_background import _candidates, measurements

ROOT = Path(__file__).resolve().parents[1] / "data/background-evaluation-v1"
REVISION = "b0832726deb55484dbce869c6f09c41f7e9e5fbb"
SEED = "chipmunk-background-continuity-20261104-v1"
RATE = 16000
SAMPLES = 64000


def save(path, document):
    data = canonical_json(document)
    if path.exists() and path.read_bytes() != data:
        raise RuntimeError(f"Refusing to replace frozen {path}")
    path.write_bytes(data)


def fetch(url):
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 5:
                raise
            delay = min(60, max(5 * 2**attempt, int(error.headers.get("Retry-After", "0") or 0)))
            print("temporary HTTP", error.code, "waiting", delay, "seconds", flush=True)
            time.sleep(delay)


def order(value):
    return hashlib.sha256((SEED + ":" + value).encode()).hexdigest()


def category(row):
    if row["scene"] in {"office", "library"}:
        return "indoor_ambience"
    if row["event"] == "air_conditioner":
        return "steady_machinery"
    if row["event"] == "street_music":
        return "recorded_music"
    if row["scene"] in {"park", "street_traffic", "street_pedestrian"}:
        return "outdoor_ambience"
    return None


def group(row):
    name = row["wavename"]
    # Keep adjacent UrbanSound excerpts and DCASE recording locations together.
    value = name.split("-")[0] if row["dataset"] == "UrbanSound8K" else "-".join(name.split("-")[:-2])
    return row["dataset"] + ":" + value


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    for folder in ("source", "metadata", "inputs"):
        (ROOT / folder).mkdir(exist_ok=True)
    metadata = ROOT / "datasplit_dev.csv"
    if not metadata.exists():
        metadata.write_bytes(
            fetch("https://raw.githubusercontent.com/apple-yinhan/EnvSDD/main/datasplit_dev.csv")
        )
    rows = [r for r in csv.DictReader(metadata.open()) if r["usage"] == "validation"]
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if category(row):
            groups[(category(row), group(row))].append((index, row))
    selected = []
    counts = {
        "indoor_ambience": (10, 4),
        "steady_machinery": (30, 10),
        "recorded_music": (30, 10),
        "outdoor_ambience": (30, 10),
    }
    for kind, (number, calibration) in counts.items():
        keys = sorted([k for k in groups if k[0] == kind], key=lambda k: order(k[1]))[:number]
        if len(keys) != number:
            raise RuntimeError(f"Insufficient source groups for {kind}: {len(keys)}")
        for position, key in enumerate(keys):
            index, row = min(groups[key], key=lambda pair: order(pair[1]["wavename"]))
            selected.append(
                {
                    "source_group": key[1],
                    "category": kind,
                    "split": "calibration" if position < calibration else "test",
                    "row_offset": index * 5,
                    "expected": row,
                }
            )
    protocol = {
        "schema": "chipmunk.background-continuity-evaluation/1",
        "seed": SEED,
        "dataset": "EnvSDD/EnvSDD",
        "revision": REVISION,
        "upstream_split": "validation",
        "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "selection": selected,
        "source_policy": "One excerpt per UrbanSound recording or DCASE location; calibration/test group-disjoint.",
        "audio_policy": "First 4 seconds, 16kHz mono; common deterministic -24dBFS RMS normalization with peak cap0.8. Sources shorter than4s excluded before scores; no replacement.",
        "cases": [
            "original",
            "gain_only",
            "real_splice",
            "crossfade_splice",
            "synthetic_insert",
            "synthetic_full",
        ],
        "splice_policy": "At2.0s; real donor is next source within same category/split. Full-window donor and local splice-segment RMS matched; crossfade100ms.",
        "methods": ["baseline_energy_12db", "spectral_continuity", "stationary_spectral_continuity"],
        "threshold_policy": "For new methods use calibration original maximum score 95th percentile (higher), never test; baseline12dB fixed.",
        "localization_tolerance_s": 0.25,
        "admission": {
            "minimum_test_real_sources": 50,
            "max_test_false_flag_rate": 0.10,
            "max_false_flag_wilson_upper": 0.20,
            "min_splice_recall_gain_vs_baseline": 0.15,
        },
        "limitations": [
            "Publisher labels and curated subset do not prove recording provenance.",
            "Office/library recordings are real indoor scenes, not isolated verified room-tone tracks.",
            "No trained environmental detector is evaluated here. Spectral continuity measures changes, not AI origin.",
            "Generated full environmental sounds are an origin-discrimination negative audit for these edit cues.",
            "Adjacent excerpts and generated variants remain in the same recording/location group.",
            "Local evaluation only; no deployment authorization or production claim.",
        ],
        "sources": [
            {
                "url": "https://github.com/apple-yinhan/EnvSDD",
                "purpose": "Official source metadata and generation labels",
            },
            {
                "url": "https://huggingface.co/datasets/EnvSDD/EnvSDD",
                "license_declaration": "CC-BY-4.0; upstream recording terms retained; local research only",
            },
            {
                "url": "https://www.isca-archive.org/interspeech_2025/yin25_interspeech.pdf",
                "purpose": "Environmental sound detection scope",
            },
            {
                "url": "https://www.isca-archive.org/interspeech_2024/xie24d_interspeech.pdf",
                "purpose": "Partial generated environmental audio localization",
            },
            {
                "url": "https://openaccess.thecvf.com/content/CVPR2025/html/Smeu_Circumventing_Shortcuts_in_Audio-visual_Deepfake_Detection_Datasets_with_Unsupervised_Learning_CVPR_2025_paper.html",
                "purpose": "Silence shortcut risk",
            },
        ],
    }
    save(ROOT / "protocol.json", protocol)

    def download(item):
        cached = ROOT / "metadata" / f"{item['row_offset']}.json"
        if cached.exists():
            response = json.loads(cached.read_bytes())
        else:
            query = urllib.parse.urlencode(
                {
                    "dataset": "EnvSDD/EnvSDD",
                    "config": "default",
                    "split": "validation",
                    "offset": item["row_offset"],
                    "length": 5,
                }
            )
            response = json.loads(fetch("https://datasets-server.huggingface.co/rows?" + query))
            save(cached, response)
        records = []
        for index in (0, 1, 4):
            entry = response["rows"][index]
            row = entry["row"]
            url = row["audio"][0]["src"]
            if (
                row["filename"] != item["expected"]["wavename"]
                or REVISION not in url
                or row["label"] != ("real" if index == 0 else "fake")
            ):
                raise RuntimeError("Dataset row order or revision changed; refusing selection drift")
            destination = ROOT / "source" / f"row-{entry['row_idx']}.wav"
            if not destination.exists():
                destination.write_bytes(fetch(url))
            data = destination.read_bytes()
            values, rate = sf.read(io.BytesIO(data), dtype="float64", always_2d=True)
            values = values.mean(axis=1)
            if rate != RATE:
                import math

                divisor = math.gcd(rate, RATE)
                values = resample_poly(values, RATE // divisor, rate // divisor)
            record = {
                **{k: v for k, v in item.items() if k != "expected"},
                **{k: v for k, v in row.items() if k != "audio"},
                "row_idx": entry["row_idx"],
                "source_path": str(destination.relative_to(ROOT)),
                "source_sha256": hashlib.sha256(data).hexdigest(),
                "source_url": url.split("?")[0],
                "original_sample_rate": rate,
                "decoded_samples": len(values),
            }
            if len(values) < SAMPLES:
                record["excluded_reason"] = "shorter_than_frozen_four_second_window"
            elif np.sqrt(np.mean(values[:SAMPLES] ** 2)) < 1e-6:
                record["excluded_reason"] = "no_usable_signal"
            else:
                values = values[:SAMPLES]
                scale = min(10 ** (-24 / 20) / np.sqrt(np.mean(values * values)), 0.8 / max(np.abs(values)))
                wav = wav_bytes(values * scale)
                sha = hashlib.sha256(wav).hexdigest()
                path = ROOT / "inputs" / f"{sha}.wav"
                path.write_bytes(wav)
                record.update(
                    {
                        "input_sha256": sha,
                        "input_path": str(path.relative_to(ROOT)),
                        "normalization_gain": float(scale),
                    }
                )
            records.append(record)
        print("source", item["category"], item["source_group"], flush=True)
        return records

    records = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for batch in pool.map(download, selected):
            records.extend(batch)
    cases = []

    def add(values, base, kind, contributors, source_spans, truth):
        gain = 0.8 / max(0.8, float(np.max(np.abs(values))))
        values = values * gain
        wav = wav_bytes(values)
        sha = hashlib.sha256(wav).hexdigest()
        (ROOT / "inputs" / f"{sha}.wav").write_bytes(wav)
        cases.append(
            {
                "id": f"{base['row_idx']}:{kind}",
                "case": kind,
                "split": base["split"],
                "category": base["category"],
                "source_group": base["source_group"],
                "input_path": f"inputs/{sha}.wav",
                "input_sha256": sha,
                "samples": len(values),
                "sample_rate": RATE,
                "final_peak_gain": gain,
                "contributors": contributors,
                "source_spans": source_spans,
                "truth": truth,
            }
        )

    by_group = defaultdict(list)
    for record in records:
        by_group[record["source_group"]].append(record)
    usable = [record for record in records if record["label"] == "real" and not record.get("excluded_reason")]
    for base in usable:
        x = read_wav((ROOT / base["input_path"]).read_bytes()).astype(np.float64)
        span = {
            "input_start_sample": 0,
            "input_end_sample": SAMPLES,
            "source_sha256": base["source_sha256"],
            "source_start_sample": 0,
            "source_end_sample": SAMPLES,
            "coordinate_space": "derived-16khz-mono",
        }
        add(
            x,
            base,
            "original",
            [base["source_group"]],
            [span],
            {"edited": False, "ai_generated": False, "boundaries": []},
        )
        gain = x.copy()
        gain[SAMPLES // 2 :] *= 0.25
        add(
            gain,
            base,
            "gain_only",
            [base["source_group"]],
            [span],
            {"edited": True, "content_splice": False, "ai_generated": False, "boundaries": [32000]},
        )
        peers = sorted(
            [r for r in usable if r["category"] == base["category"] and r["split"] == base["split"]],
            key=lambda r: r["source_group"],
        )
        donor = peers[
            (next(i for i, r in enumerate(peers) if r["source_group"] == base["source_group"]) + 1)
            % len(peers)
        ]
        y = read_wav((ROOT / donor["input_path"]).read_bytes()).astype(np.float64)

        def match_gain(part, x=x):
            return float(np.sqrt(np.mean(x[32000:] ** 2)) / max(np.sqrt(np.mean(part * part)), 1e-9))

        donor_gain = match_gain(y[32000:])
        tail = y[32000:] * donor_gain
        spliced = np.concatenate([x[:32000], tail])
        spans = [
            {**span, "input_end_sample": 32000, "source_end_sample": 32000},
            {
                **span,
                "input_start_sample": 32000,
                "source_sha256": donor["source_sha256"],
                "source_start_sample": 32000,
            },
        ]
        for kind, audio in [("real_splice", spliced), ("crossfade_splice", spliced.copy())]:
            case_spans = spans
            if kind == "crossfade_splice":
                width = 1600
                blend = np.linspace(0, 1, width)
                audio[31200:32800] = (1 - blend) * x[31200:32800] + blend * y[31200:32800] * donor_gain
                case_spans = [
                    {**spans[0], "input_end_sample": 31200, "source_end_sample": 31200},
                    {
                        **span,
                        "input_start_sample": 31200,
                        "input_end_sample": 32800,
                        "source_start_sample": 31200,
                        "source_end_sample": 32800,
                        "blend": "linear1_to0",
                    },
                    {
                        **span,
                        "source_sha256": donor["source_sha256"],
                        "input_start_sample": 31200,
                        "input_end_sample": 32800,
                        "source_start_sample": 31200,
                        "source_end_sample": 32800,
                        "blend": "linear0_to1",
                        "gain": donor_gain,
                    },
                    {
                        **spans[1],
                        "input_start_sample": 32800,
                        "source_start_sample": 32800,
                        "gain": donor_gain,
                    },
                ]
            else:
                case_spans = [spans[0], {**spans[1], "gain": donor_gain}]
            add(
                audio,
                base,
                kind,
                [base["source_group"], donor["source_group"]],
                case_spans,
                {
                    "edited": True,
                    "content_splice": True,
                    "ai_generated": False,
                    "boundaries": [32000],
                    "crossfade_samples": 1600 if kind == "crossfade_splice" else 0,
                },
            )
        for fake in by_group[base["source_group"]]:
            if fake["label"] != "fake" or fake.get("excluded_reason"):
                continue
            z = read_wav((ROOT / fake["input_path"]).read_bytes()).astype(np.float64)
            suffix = fake["generative_model"]
            fspan = {**span, "source_sha256": fake["source_sha256"]}
            add(
                z,
                base,
                "synthetic_full:" + suffix,
                [base["source_group"]],
                [fspan],
                {"edited": False, "ai_generated": True, "boundaries": []},
            )
            fake_gain = match_gain(z[32000:])
            add(
                np.concatenate([x[:32000], z[32000:] * fake_gain]),
                base,
                "synthetic_insert:" + suffix,
                [base["source_group"]],
                [
                    spans[0],
                    {**fspan, "input_start_sample": 32000, "source_start_sample": 32000, "gain": fake_gain},
                ],
                {"edited": True, "content_splice": True, "ai_generated": True, "boundaries": [32000]},
            )
    manifest = {
        "schema": protocol["schema"],
        "protocol_sha256": canonical_hash(protocol),
        "records": records,
        "cases": cases,
        "counts": {
            "records": len(records),
            "excluded": sum(bool(r.get("excluded_reason")) for r in records),
            "cases": len(cases),
            "real_sources": len(usable),
            "real_sources_by_split": dict(Counter(r["split"] for r in usable)),
        },
    }
    save(ROOT / "manifest.json", manifest)
    print(json.dumps(manifest["counts"]), flush=True)


def score():
    manifest = json.loads((ROOT / "manifest.json").read_bytes())
    output = []
    for i, case in enumerate(manifest["cases"]):
        data = (ROOT / case["input_path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != case["input_sha256"]:
            raise RuntimeError("Changed evaluation input")
        pcm = read_wav(data)
        windows = candidates(pcm)
        bins = measurements(pcm)
        baseline = [
            {"boundary_sample": c["change_sample"], "score": abs(c["value"])}
            for c in _candidates(bins)
            if c["category"] == "energy_change"
        ]
        output.append(
            {
                "id": case["id"],
                "windows": windows,
                "baseline_energy_events": baseline,
                "scores": {
                    "baseline_energy_12db": max((x["score"] for x in baseline), default=0.0),
                    "spectral_continuity": max((x["spectral_distance"] for x in windows), default=0.0),
                    "stationary_spectral_continuity": max(
                        (x["stationary_spectral_distance"] for x in windows), default=0.0
                    ),
                },
            }
        )
        if i % 50 == 0:
            print("scored", i, flush=True)
    save(
        ROOT / "scores.json",
        {
            "manifest_sha256": canonical_hash(manifest),
            "methods_sha256": hashlib.sha256(
                Path(__file__).parents[1].joinpath("app/bench/audio_continuity.py").read_bytes()
            ).hexdigest(),
            "rows": output,
        },
    )


def wilson(k, n):
    if not n:
        return None
    p = k / n
    z = 1.95996398454
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    delta = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"low": float(mid - delta), "high": float(mid + delta)}


def evaluate():
    manifest = json.loads((ROOT / "manifest.json").read_bytes())
    scores = json.loads((ROOT / "scores.json").read_bytes())
    protocol = json.loads((ROOT / "protocol.json").read_bytes())
    if scores["manifest_sha256"] != canonical_hash(manifest):
        raise RuntimeError("Evaluation source binding changed")
    joined = [
        {**c, **s} for c, s in zip(manifest["cases"], scores["rows"], strict=True) if c["id"] == s["id"]
    ]
    if len(joined) != len(manifest["cases"]):
        raise RuntimeError("Score identifiers disagree")
    controls = [r for r in joined if r["split"] == "calibration" and r["case"] == "original"]
    thresholds = {
        method: 12.0
        if method == "baseline_energy_12db"
        else float(np.quantile([r["scores"][method] for r in controls], 0.95, method="higher"))
        for method in protocol["methods"]
    }
    threshold_record = {
        "protocol_sha256": canonical_hash(protocol),
        "calibration_source_groups": sorted(r["source_group"] for r in controls),
        "thresholds": thresholds,
        "positive_rule": "score > threshold for calibrated methods; score >= 12 for fixed baseline",
    }
    save(ROOT / "thresholds.json", threshold_record)

    def positive(row, method):
        return (
            row["scores"][method] >= 12
            if method == "baseline_energy_12db"
            else row["scores"][method] > thresholds[method]
        )

    def localized(row, method):
        windows = (
            row["baseline_energy_events"]
            if method == "baseline_energy_12db"
            else [
                {
                    "boundary_sample": w["boundary_sample"],
                    "score": w[
                        "spectral_distance"
                        if method == "spectral_continuity"
                        else "stationary_spectral_distance"
                    ],
                }
                for w in row["windows"]
            ]
        )
        return any(
            (w["score"] >= 12 if method == "baseline_energy_12db" else w["score"] > thresholds[method])
            and any(abs(w["boundary_sample"] - b) <= 4000 for b in row["truth"]["boundaries"])
            for w in windows
        )

    cohorts = {}
    for kind in ["all", *sorted({r["category"] for r in joined})]:
        selected = [r for r in joined if r["split"] == "test" and (kind == "all" or r["category"] == kind)]
        cohorts[kind] = {}
        for method in thresholds:
            group_metrics = {}
            for case in [
                "original",
                "gain_only",
                "real_splice",
                "crossfade_splice",
                "synthetic_full",
                "synthetic_insert",
            ]:
                rows = [r for r in selected if r["case"].split(":")[0] == case]
                count = sum(positive(r, method) for r in rows)
                # Generated variants share source groups; group aggregation avoids inflated certainty.
                grouped = defaultdict(list)
                for row in rows:
                    grouped[row["source_group"]].append(positive(row, method))
                group_count = sum(any(v) for v in grouped.values())
                group_metrics[case] = {
                    "n": len(rows),
                    "flagged": count,
                    "flag_rate": count / len(rows) if rows else None,
                    "source_groups": len(grouped),
                    "any_variant_flagged_source_groups": group_count,
                    "source_group_wilson_ci95": wilson(group_count, len(grouped)),
                    "localized": sum(localized(r, method) for r in rows)
                    if rows and rows[0]["truth"]["boundaries"]
                    else None,
                }
            cohorts[kind][method] = group_metrics
    decisions = {}
    for method in thresholds:
        m = cohorts["all"][method]
        base = cohorts["all"]["baseline_energy_12db"]
        n = m["real_splice"]["n"]
        gain = (m["real_splice"]["localized"] - base["real_splice"]["localized"]) / n if n else 0
        original = m["original"]
        a = protocol["admission"]
        admitted = (
            method != "baseline_energy_12db"
            and original["n"] >= a["minimum_test_real_sources"]
            and original["flag_rate"] <= a["max_test_false_flag_rate"]
            and original["source_group_wilson_ci95"]["high"] <= a["max_false_flag_wilson_upper"]
            and gain >= a["min_splice_recall_gain_vs_baseline"]
        )
        decisions[method] = {
            "admitted_as_descriptive_continuity_check": admitted,
            "localized_real_splice_recall_gain": gain,
            "eligible_as_ai_origin_detector": False,
        }
    output = {
        "protocol_sha256": canonical_hash(protocol),
        "manifest_sha256": canonical_hash(manifest),
        "scores_sha256": canonical_hash(scores),
        "thresholds_sha256": canonical_hash(threshold_record),
        "cohorts": cohorts,
        "decisions": decisions,
        "deployment_eligible": False,
        "scope": "Small source-held-out local evaluation; continuity flags are not probabilities of AI generation. Music and indoor-scene cohorts retained separately.",
    }
    save(ROOT / "evaluation.json", output)
    print(
        json.dumps(
            {
                "thresholds": thresholds,
                "counts": manifest["counts"],
                "all": cohorts["all"],
                "decisions": decisions,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "score", "evaluate"])
    args = parser.parse_args()
    {"prepare": prepare, "score": score, "evaluate": evaluate}[args.stage]()
