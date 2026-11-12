"""Measure an archived independent voice model on fixed overlapping native contexts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import time
from collections import defaultdict
from pathlib import Path

from scripts import audio_partial_residual_transfer as previous
from scripts import audio_periodic_excitation_transfer as bound
from scripts import audio_spectra_background_transfer as spectra

RATE, CONTEXT = 16_000, 48_000
SCHEMA = "chipmunk.audio-spectra-context-transfer/1"
RAW, CONTRAST, UNION = "spectra_max", "spectra_edge_contrast", "incumbent_or_contrast"
BASELINE, ROOM_BASELINE = bound.BASELINE, bound.ROOM_BASELINE
METHOD = {
    "inputs": "Exact previous -5 dB matched four-second PCM. Use original confidence_cal genuine controls and disjoint confirmation roles. Apply unchanged points to a separate retained six-second AMI strong-competing-speech panel. All sources had prior project exposure.",
    "context": "Only fixed native three-second windows. Start at 0 and 1 seconds for four-second clips, and at 0,1,2,3 seconds for six-second clips. Use archived 0.97 pre-emphasis and repeat samples to graph length 64,600. Source ranges remain only the actual three seconds.",
    "arms": "Maximum independent model score and maximum absolute difference of two adjacent overlapping context scores. A difference names only the exclusive 1-second edge from the stronger context. Shared 2-second audio and input interaction are unresolved. The contrast is a hypothesis about location, not an attribution of the model response to those samples.",
    "point": "Strictly exceed each new arm's maximum of the independent genuine operating-point controls. Use no model or head fitting. Fixed union flags incumbent or edge-contrast result and retains both ranges if both flag. The union counts a useful location if either retained range meets the overlap criterion.",
    "evaluation": "Count matched genuine splice groups and known useful source spans (at least .30 IoU and .50 edit coverage). Separately report zero-, one- and two-false-group ranking comparisons and paired score ordering. A 3-second range cannot localize an .8-second edit under this definition, so also count its raw detection flags.",
}
read, write, sha, require = bound.read, bound.write, bound.sha, bound.require


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sources(args):
    prior, rows, rooms = bound.bind(args.parent, args.meeting)
    matched = [(r, previous.pcm_bytes(args.parent, r)) for r in rows if r["condition"] == "speech_minus5db"
               and ((r["role"] == "confidence_cal" and not r["label"]) or r["role"] == "confirmation")]
    ami = [(r, raw) for r, raw in rooms if r["panel"] == "confirmation" and r["condition"] == "room"]
    lock = {"schema": SCHEMA, "method": METHOD, "code_sha256": sha(Path(__file__)),
            "parent": prior["parent"], "meeting": prior["meeting"], "weights": spectra.pinned_assets(args.weights),
            "runtime": {name: importlib.metadata.version(name) for name in ("numpy", "onnxruntime")},
            "matched_views": len(matched), "held_ami_views": len(ami), "fits": 0}
    require(len(ami) == 12 and sum(r["role"] == "confirmation" for r, _ in matched) == 44, "Held cohort changed")
    return lock, matched, ami


def read_views(session, raw):
    extent = len(raw)//2
    entries = []
    for start in range(0, extent-CONTEXT+1, RATE):
        end = start + CONTEXT
        pcm = raw[2*start:2*end]
        score, model_sha = spectra.candidate_score(session, pcm)
        entries.append({"start": start, "end": end, "score": score, "pcm_sha256": digest(pcm), "input_sha256": model_sha})
    require(len(entries) in {2, 4}, "Source context extent differs")
    return entries


def best_values(windows):
    chosen = max(windows, key=lambda row: row["score"])
    differences = []
    for a, b in zip(windows, windows[1:]):
        delta = a["score"]-b["score"]
        differences.append((abs(delta), [a["start"], b["start"]] if delta >= 0 else [a["end"], b["end"]]))
    edge, edge_range = max(differences, key=lambda value: value[0])
    return {RAW: (chosen["score"], [chosen["start"], chosen["end"]]), CONTRAST: (edge, edge_range)}


def qualified(values, truth, points):
    return {name: {"score": score, "range": span, "flag": score > points[name], "located": previous.useful_range(*span, truth)}
            for name, (score, span) in values.items()}


def test_points(args, matched):
    session = spectra.start_session(args.weights)
    selected = []
    for row, raw in matched:
        if row["role"] == "confidence_cal":
            windows = read_views(session, raw)
            selected.append({"id": row["id"], "pcm_sha256": digest(raw), "windows": windows})
    points = {arm: max(best_values(row["windows"])[arm][0] for row in selected) for arm in (RAW, CONTRAST)}
    write(args.run / "selection.json", {"freeze_sha256": sha(args.run / "freeze.json"), "points": points, "controls": selected,
                                        "forwards": sum(len(r["windows"]) for r in selected)})
    print(f"Froze points from {len(selected)} genuine strong-interference clips", flush=True)


def score_held(args, matched, ami):
    seal = read(args.run / "selection.json")
    require(seal["freeze_sha256"] == sha(args.run / "freeze.json"), "Point-selection commitment changed")
    session = spectra.start_session(args.weights)
    prior_matched = {r["id"]: r for r in read(args.parent / "decisions.json")}
    prior_ami = {r["id"]: r for r in read(args.parent / "meeting-replay.json")["decisions"]}
    shorter = {r["id"]: r for r in read(args.meeting / "confirmation.json")["rows"]}
    point = read(args.meeting / "points.json")["points"]["short_odd"]["point"]
    held = []
    for panel, cases in (("matched", matched), ("ami", ami)):
        for row, raw in cases:
            if row.get("role") == "confidence_cal":
                continue
            identifier, truth = row["id"], row["truth"]
            windows = read_views(session, raw)
            checks = qualified(best_values(windows), truth, seal["points"])
            if panel == "matched":
                group = row["pair_id"] or row["background_group"]
                arm = row["arm"]
                old = prior_matched[identifier]["checks"][BASELINE]
                checks[BASELINE] = old | {"located": old["best_range_useful"]}
            else:
                group, arm = row["host_group"], row["kind"]
                old = prior_ami[identifier]["checks"][BASELINE]
                checks[BASELINE] = old | {"located": old["location"]}
                value, spans = shorter[identifier]["features"]["short_odd"]
                checks[ROOM_BASELINE] = {"score": value, "range": spans, "flag": value > point,
                                         "located": previous.useful_range(*spans, truth)}
            merged = [checks[name] for name in (BASELINE, CONTRAST) if checks[name]["flag"]]
            checks[UNION] = {"flag": bool(merged), "located": any(check["located"] for check in merged),
                             "range": [check["range"] for check in merged]}
            exclusive = [(a["start"], b["start"]) for a, b in zip(windows, windows[1:])]
            exclusive += [(a["end"], b["end"]) for a, b in zip(windows, windows[1:])]
            held.append({"panel": panel, "id": identifier, "group": group, "arm": arm, "truth": truth, "family": row["family"],
                         "pcm_sha256": digest(raw), "windows": windows, "checks": checks,
                         "edge_geometry_supports_truth": any(previous.useful_range(*span, truth) for span in exclusive)})
            if len(held) % 12 == 0:
                print(f"Scored {len(held)} held clips", flush=True)
    write(args.run / "confirmation.json", {"selection_sha256": sha(args.run / "selection.json"), "rows": held,
                                           "forwards": sum(len(row["windows"]) for row in held)})


def summarize(held):
    grouped = defaultdict(list)
    for row in held:
        grouped[row["panel"]].append(row)
    results = {}
    for panel, rows in grouped.items():
        baseline = BASELINE if panel == "matched" else ROOM_BASELINE
        alternatives = (baseline, RAW, CONTRAST)
        controls = previous.HUMAN if panel == "matched" else ("ordinary",)
        results[panel] = {"fixed": {name: bound.counts(rows, name) for name in rows[0]["checks"]},
                          "equal_false_groups": {str(n): {name: bound.counts(rows, name, n) for name in alternatives} for n in (0, 1, 2)},
                          "paired": {name: bound.pair_ranking(rows, name, controls) for name in alternatives},
                          "truths_reachable_by_exclusive_edge": sum(r["edge_geometry_supports_truth"] for r in rows)}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "selection", "confirmation", "report"))
    parser.add_argument("run", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--parent", type=Path, default=bound.PARENT)
    parser.add_argument("--meeting", type=Path, default=bound.MEETING)
    args = parser.parse_args()
    import onnxruntime as ort

    ort.disable_telemetry_events()
    lock, matched, ami = sources(args)
    frozen = args.run / "freeze.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        write(frozen, lock)
    else:
        require(read(frozen) == lock, "Frozen model, runtime or exact sources changed")
        started = time.monotonic()
        if args.phase == "selection":
            test_points(args, matched)
        elif args.phase == "confirmation":
            score_held(args, matched, ami)
        else:
            held = read(args.run / "confirmation.json")
            require(held["selection_sha256"] == sha(args.run / "selection.json"), "Held scores used different points")
            result = summarize(held["rows"])
            files = ("freeze", "selection", "confirmation")
            write(args.run / "result.json", lock | {"outcomes": result, "commitments": {k: sha(args.run / f"{k}.json") for k in files},
                                                     "forwards": held["forwards"] + read(args.run / "selection.json")["forwards"]})
            print({name: r["fixed"] for name, r in result.items()})
        print(f"{args.phase} completed in {time.monotonic()-started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
