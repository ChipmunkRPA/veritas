"""Freeze visual speech disagreement on separate controls and count held locations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts import video_lip_correspondence as receipt
from scripts import video_visual_speech_audio as audio
from scripts import video_visual_speech_train as training

ARMS = ("relative", "alternative")
BUDGETS = (0, 1, 2)
METHOD = {
    "fit": "Predict visually distinct phone groups from native mouth image sequences and original publisher alignments.",
    "review": "Compare audible events with the visual prediction within the proposed region and its two flanks. Fit a shared "
              "offset and check the neighboring offsets inside the region before reporting a local disagreement.",
    "selection": "Keep previously reported locations first. Freeze one scoring arm with separate carrier points from the "
                 "reserved calibration speakers. Allow up to two added calibration speaker groups among ordinary or delay "
                 "controls combined before maximizing correct locations. Evaluate the frozen rule on distinct speakers.",
}


def sha(body):
    return hashlib.sha256(body).hexdigest()


def overlaps(span, truth):
    if span is None:
        return False
    a, b = span
    c, d = truth
    return 5 * (min(b, d) - max(a, c)) >= 2 * (max(b, d) - min(a, c))


def measure(visual, acoustic, fit, priority, *, fresh_plan=None):
    import numpy as np
    import torch
    from safetensors.torch import load_file

    from app.detectors.video import viseme_patch_alignment as align
    from app.detectors.video.viseme_patch import MOUTH_GROUPS, VisualSpeech, normalize

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    learned = receipt.read(fit / "fit.json.gz")
    file = fit / "visual-speech.safetensors"
    assert receipt.sha(file) == learned["weight"]
    model = VisualSpeech().eval()
    model.load_state_dict(load_file(file), strict=True)
    bundle = receipt.read(visual / "capture.json.gz")
    tracks = receipt.read(acoustic)
    expected = learned["plan"]
    if fresh_plan is not None:
        assert receipt.read(fresh_plan)["fit_plan"] == expected
        expected = receipt.sha(fresh_plan)
    assert expected == bundle["plan"] == tracks["plan"]
    assert tuple(align.ARMS) == ARMS
    captures = {r["picture_sha256"]: r for r in bundle["rows"]}
    measured, classes = [], defaultdict(Counter)
    with torch.inference_mode():
        for row in tracks["rows"]:
            split = row["split"]
            if split not in priority:
                continue
            digest = row["picture_sha256"]
            sample = captures[digest]
            assert sample["speaker"] == row["speaker"] and sample["pts"] == row["pts"] and sample["time_base"] == row["time_base"]
            patch = visual / sample["file"]
            assert receipt.sha(patch) == sample["sha256"]
            with np.load(patch, allow_pickle=False) as frames:
                mask, clock = frames["valid"], frames["samples"]
                mouth = normalize(frames["patches"], mask)
                posteriors = model(torch.from_numpy(mouth[None]))[0].softmax(dim=-1).numpy()
                target = frames["targets"]
            usable = align.observed_context(mask)
            for index, group in enumerate(MOUTH_GROUPS):
                truth = (target == index) & usable
                classes[split][group] += int(truth.sum())
                classes[split][group + "_correct"] += int(((posteriors.argmax(1) == index) & truth).sum())
            for carrier, cases in row["cases"].items():
                for name, info in sorted(cases.items()):
                    choices = align.locate(posteriors, clock, mask, info["events"], row["samples"])
                    fields = ("picture_sha256", "speaker", "split", "pts", "time_base", "truth", "samples", "condition")
                    measured.append({key: row[key] for key in fields} | {"carrier": carrier, "role": name,
                        "audio_sha256": info["audio_sha256"], "carrier_sha256": info["carrier_sha256"], "baseline": info["baseline"],
                        "baseline_status": info["baseline_status"], "valid_picture_frames": int(mask.sum()),
                        "visible_sound_events": len(info["events"]), "choices": choices})
    return measured, classes


def classify(row):
    role = row["role"]
    if role in {"original", "gain"}:
        return "ordinary"
    if role in {"global_early", "global_late"}:
        return "constant_delay"
    return role


def points(rows, arm, budget):
    by_actor = defaultdict(dict)
    for row in rows:
        if classify(row) != "different_words" and not row["baseline"] and (reading := row["choices"].get(arm)):
            group = by_actor[row["carrier"]]
            score = reading["score"]
            group[row["speaker"]] = max(score, group.get(row["speaker"], 0.))
    choices = {}
    for carrier in audio.CARRIERS:
        values = sorted(by_actor[carrier].values(), reverse=True)
        choices[carrier] = values[budget] if len(values) > budget else 0.
    return choices


def propose(row, selected):
    reading = row["choices"].get(selected["arm"])
    if reading and reading["score"] > selected["points"][row["carrier"]]:
        return reading["span"]
    return None


def counts(rows, selected):
    groups = defaultdict(Counter)
    for row in rows:
        kind = classify(row)
        counter = groups[(row["split"], row["carrier"], kind)]
        old = row["baseline"]
        new = old or propose(row, selected)
        counter.update({"files": 1, "baseline_alerts": bool(old), "combined_alerts": bool(new),
                        "additional_alerts": bool(new and old is None), "no_candidate_range": not row["choices"].get(selected["arm"])})
        if kind == "different_words":
            useful, correct = overlaps(old, row["truth"]), overlaps(new, row["truth"])
            counter.update({"baseline_located": useful, "combined_located": correct,
                            "new_wrong_location": bool(new and old is None and not correct)})
    return {"/".join(key): value for key, value in sorted(groups.items())}


def selected_point(calibration):
    choices = []
    for arm in ARMS:
        for budget in BUDGETS:
            selected = {"arm": arm, "budget": budget, "points": points(calibration, arm, budget)}
            groups = defaultdict(set)
            for row in calibration:
                if not row["baseline"] and propose(row, selected):
                    groups[classify(row)].add(row["speaker"])
            controls = groups["ordinary"] | groups["constant_delay"] | groups["same_words"]
            totals = counts(calibration, selected)
            true = sum(n["combined_located"] - n["baseline_located"] for key, n in totals.items() if key.endswith("/different_words"))
            wrong = sum(n["new_wrong_location"] for key, n in totals.items() if key.endswith("/different_words"))
            choices.append({"selected": selected, "summary": totals, "extra_correct": true,
                            "wrong": wrong, "extra_control_groups": len(controls)})
    allowed = [row for row in choices if row["extra_control_groups"] <= 2]
    winner = max(allowed, key=lambda row: (row["extra_correct"], -row["extra_control_groups"], -row["wrong"],
                                           -row["selected"]["budget"], -ARMS.index(row["selected"]["arm"])))
    return winner["selected"], choices


def freeze(visual, acoustic, fit, output):
    rows, classes = measure(visual, acoustic, fit, {"calibration"})
    assert sum(row["role"] == "original" for row in rows) == 26
    winner, alternatives = selected_point(rows)
    code = (Path(__file__), Path(__file__).resolve().parents[1] / "app/detectors/video/viseme_patch_alignment.py")
    value = {"schema": training.SCHEMA, "method": METHOD, "fit": receipt.sha(fit / "fit.json.gz"),
             "weight": receipt.sha(fit / "visual-speech.safetensors"), "rows": rows,
             "selected": winner, "alternatives": alternatives, "visual_classes": classes,
             "source": {"visual": receipt.sha(visual / "capture.json.gz"), "acoustic": receipt.sha(acoustic)},
             "code": {p.name: receipt.sha(p) for p in code}}
    receipt.save(output, value)
    print(json.dumps({"winner": winner, "controls": counts(rows, winner), "visual_classes": classes}, indent=2))


def assess(visual, acoustic, fit, frozen, output, fresh_plan=None):
    candidate = receipt.read(frozen)
    assert candidate["method"] == METHOD and candidate["fit"] == receipt.sha(fit / "fit.json.gz")
    if fresh_plan is None:
        assert candidate["source"] == {"visual": receipt.sha(visual / "capture.json.gz"), "acoustic": receipt.sha(acoustic)}
    selected, alternatives = selected_point(candidate["rows"])
    assert selected == candidate["selected"] and alternatives == candidate["alternatives"]
    code = (Path(__file__), Path(__file__).resolve().parents[1] / "app/detectors/video/viseme_patch_alignment.py")
    assert candidate["code"] == {p.name: receipt.sha(p) for p in code}
    rows, classes = measure(visual, acoustic, fit, {"first", "second"} if fresh_plan is None else {"fresh"}, fresh_plan=fresh_plan)
    expected = 43 if fresh_plan is None else len(receipt.read(fresh_plan)["rows"])
    assert sum(row["role"] == "original" for row in rows) == 2 * expected
    result = {"schema": training.SCHEMA, "freeze": receipt.sha(frozen), "selected": selected, "rows": rows,
              "visual_classes": classes, "summary": counts(rows, selected)}
    receipt.save(output, result)
    print(json.dumps(result["summary"], indent=2))


def review(path):
    value = receipt.read(path)
    if "summary" in value:
        assert counts(value["rows"], value["selected"]) == value["summary"]
        print(json.dumps(value["summary"], indent=2))
    else:
        selected, alternatives = selected_point(value["rows"])
        assert selected == value["selected"] and alternatives == value["alternatives"]
        print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode, fields in (("freeze", ("visual", "acoustic", "fit", "output")),
                         ("assess", ("visual", "acoustic", "fit", "frozen", "output")), ("review", ("path",))):
        cmd = sub.add_parser(mode)
        for name in fields:
            cmd.add_argument(name, type=Path)
        if mode == "assess":
            cmd.add_argument("--fresh-plan", type=Path)
    args = vars(parser.parse_args())
    {"freeze": freeze, "assess": assess, "review": review}[args.pop("mode")](**args)
