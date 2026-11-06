"""Audit frozen face-swap and audio/video timing comparisons on fresh groups."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from prepare_video_upgrade_eval import sha
from run_face_swap_upgrade_eval import CHECKS
from run_video_av_controls import SYNC
from summarize_face_swap_upgrade_eval import compare


def rows(path: Path, manifest: Path, items: list[dict], *, face: bool = False) -> list[dict]:
    with path.open() as file:
        header = json.loads(next(file))
        assert header.get("run", header)["manifest_sha256"] == sha(manifest)
        values = [json.loads(line) for line in file]
    assert [row["id"] for row in values] == [item["id"] for item in items]
    for row, item in zip(values, items, strict=True):
        keys = ("group", "label", "family", "split") if face else ("group", "category", "split", "shift_samples")
        assert all(row[key] == item[key] for key in keys)
        expected = item.get("matched", {}).get("sha256") if face else item["sha256"]
        assert row.get("sha256") == expected
    return values


def face_comparison(old: Path, fresh: Path) -> dict:
    input_rows = []
    for manifest, baseline, candidate in (
        (old / "face-swap/manifest.json", old / "face-swap/scores-mps.jsonl", fresh / "dfds/old.jsonl"),
        (fresh / "face-swap/manifest.json", fresh / "face-swap/scores-mps.jsonl", fresh / "dfds/fresh.jsonl"),
    ):
        items = json.loads(manifest.read_text())["items"]
        current = rows(baseline, manifest, items, face=True)
        challenger = rows(candidate, manifest, items, face=True)
        input_rows.append([{**r, "scores": {**r.get("scores", {}), "candidate-dfds": {"prob": c["prob"], "status": c["status"]}}}
                           for r, c in zip(current, challenger, strict=True)])
    prior, test = input_rows
    assert {r["group"] for r in prior}.isdisjoint(r["group"] for r in test)
    reference = [r for r in prior if r["split"] == "calibration"]
    assert all(r["split"] == "test" for r in test)
    return {candidate: compare(reference + test, (CHECKS[0], candidate))
            for candidate in (CHECKS[1], "candidate-dfds")}


def peaks(row: dict, rule: dict) -> list[int]:
    offsets = []
    search = rule["lag_ticks"]
    near = rule["min_abs_ticks"] - 1
    for window in (row.get("measurement") or {}).get("profiles", []):
        values = window["curves"]["onset"]
        best = max(range(len(values)), key=values.__getitem__)
        if (abs(best - search) >= rule["min_abs_ticks"] and values[best] >= rule["peak"]
                and values[best] - max(values[search-near:search+near+1]) >= rule["zero_margin"]
                and values[best] - max(x for j, x in enumerate(values) if abs(j - best) > near) >= rule["other_peak_margin"]):
            offsets.append(best - search)
    return offsets


def timing_comparison(folder: Path) -> dict:
    manifest = folder / "manifest.json"
    items = json.loads(manifest.read_text())["items"]
    baseline = rows(folder / "current.jsonl", manifest, items)
    rule = json.loads((folder / "mesh-rule.json").read_text())
    assert rule["development_sha256"] == sha(folder / "mesh-development-cpu.jsonl")
    summary: dict[str, dict] = {}
    for split, name in (("development", "mesh-development-cpu.jsonl"), ("test", "mesh-test.jsonl"),
                        ("publisher-example", "mesh-publisher.jsonl")):
        candidates = rows(folder / name, manifest, [r for r in items if r["split"] == split])
        counters: dict[str, Counter] = {"current": Counter(), "candidate": Counter()}
        for old, new in zip((r for r in baseline if r["split"] == split), candidates, strict=True):
            entry = old.get("checks", {}).get(SYNC, {})
            tracks = [u["quality"] for u in entry.get("unit_scores", []) if "@" not in u["unit_ref"]]
            base_offsets = [int(q["baseline_offset"]) for q in tracks if "baseline_offset" in q]
            new_offsets = peaks(new, rule)
            category = old["category"]
            for method, offsets, covered in (("current", base_offsets, bool(tracks)),
                                             ("candidate", new_offsets, bool((new.get("measurement") or {}).get("profiles")))):
                counts = counters[method]
                counts[f"{category}/total"] += 1
                counts[f"{category}/measured"] += covered
                if split == "publisher-example":
                    flagged = any(q["flagged"] for q in tracks) if method == "current" else bool(offsets)
                    counts[f"{category}/flagged"] += flagged
                else:
                    offsets = [x for x in offsets if abs(x) >= rule["min_abs_ticks"]]
                    shift = old["shift_samples"] // 640
                    right = shift != 0 and any(abs(x - shift) <= rule["tolerance_ticks"] for x in offsets)
                    counts[f"{category}/correct_shift"] += right
                    counts[f"{category}/wrong_shift"] += bool(offsets) and not right
        summary[split] = {name: dict(counts) for name, counts in counters.items()}
    return {"manifest_sha256": sha(manifest), "rule": rule, "sets": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prior", type=Path)
    parser.add_argument("fresh", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()
    summary = {"timing": timing_comparison(args.fresh / "av")}
    if not args.timing_only:
        summary["face"] = face_comparison(args.prior, args.fresh)
    with args.output.open("x") as out:
        out.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
