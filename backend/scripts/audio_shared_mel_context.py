"""Compare shared and per-window Mel moments for local noisy speech review."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from collections import Counter
from pathlib import Path

from scripts import audio_interval_learning as interval
from scripts import audio_sine_resynthesis_transfer as sine
from scripts import audio_trimmed_mel_normalization as dense
from scripts import review_audio_stock_voice_transfer as stock

source = interval.source
PANELS = source.PANELS
SINE_PARENT = PANELS / "audio-sine-resynthesis-transfer-2026-11-14.json"
NORMALIZATION_PARENT = PANELS / "audio-trimmed-mel-normalization-2026-11-13.json"
ARMS = ("local", "shared_scale", "shared_center_scale")
SCHEMA = "chipmunk.audio-shared-mel-context/1"
METHOD = {
    "source": "Use previously scored native six-second clips and their frozen labels. Choose the point on the 44 severe stock "
              "mixtures. Hold out the random-clock and three-generator panels, plus the SINE original/variant pairs for "
              "splicing and same-word resynthesis under competing speech, and eight room-only controls.",
    "geometry": "Use the unchanged two four-second model contexts, 0 and 2 s, and original waveform preprocessing and activity. "
                "Use median frame responses in one-second windows every half second, with the lower median where both contexts cover a window.",
    "normalization": "Compare native per-context moments with shared scale only and shared center and scale. Pool raw Mel "
                     "frames with centers before 3 s from the first context and at or after 3 s from the second. Estimate "
                     "per-band center from the pooled values. Estimate shared within-context variance from the pooled "
                     "values after centering each retained split at its own mean, with denominator n-2. Only the affine "
                     "normalization changes, and original context waveform conditioning is retained.",
    "selection": "For each arm set a strict point at the larger of the highest eligible isolated background score and the "
                 "second-highest new ordinary group maximum. Keep prior serving flags for the supplement comparison. "
                 "Declare useful at 50% edit coverage and 0.30 temporal IoU.",
    "decision": "Consider fresh confirmation if a supplement adds two useful results across two generator families with "
                "at most one new ordinary source group per panel and no new room-only flag. Same-word resynthesis is a "
                "processing control for localized spoken-content change and has a separate count.",
}
WIDTH, STARTS = source.GRIDS["dense"]


def input_rows(panel):
    if panel == "families":
        previous = source.read(sine.POINTS_PARENT)["held"]["families"]
        return [{"pcm_sha256": r["pcm_sha256"], "group": r["group"], "kind": r["kind"], "family": r["family"],
                 "truth": r["truth"], "existing_scores": r["scores"]["existing"]} for r in previous]
    by_role = {role: [{"group": group, "kind": kind, "family": kind if truth else None, "truth": truth, "pcm_sha256": pcm,
                       "existing_scores": scores}
                      for group, kind, truth, pcm, scores in interval.stock_sources.prior_rows(role)]
               for role in ("development", "random_placement")}
    parent = sine.prior.read(SINE_PARENT)
    by_role["sine"] = [
        {"group": r["id"], "kind": r["role"] if r["side"] != "original" else "untouched",
         "family": "sine" if r["truth"] else None, "truth": r["truth"], "pcm_sha256": r["pcm_sha256"],
         "existing_scores": r["scores"]["existing"]}
        for r in parent["records"] if r["condition"] == "competing_speech" and r["role"] != "edit"
    ]
    return by_role[panel]


def ledger_roles():
    _roster, pins = source.sources()
    return {"stock_development": source.path_sha(interval.stock_sources.prior_path("development")),
            "stock_holdout": source.path_sha(interval.stock_sources.prior_path("random_placement")),
            "three_generator_roster": pins, "sine": source.path_sha(SINE_PARENT),
            "old_points": source.path_sha(sine.POINTS_PARENT), "geometry_parent": source.path_sha(NORMALIZATION_PARENT)}


def windows(model, pcm, speech):
    import numpy as np
    import torch

    from app.detectors.audio.content import window_in_speech
    from app.detectors.audio.upgraded import prepare_voice_window
    from app.third_party.kaldi.fbank import fbank

    if not any(window_in_speech(speech, start / stock.RATE, (start + WIDTH) / stock.RATE) for start in STARTS):
        return {arm: [None] * len(STARTS) for arm in ARMS}, None
    contexts = {}
    for start in source.CUTS:
        waveform = torch.from_numpy(prepare_voice_window(pcm[start:start + 64000]))
        nonzero = torch.nonzero(waveform).flatten()
        extent = int(nonzero[-1]) + 1 if len(nonzero) else len(waveform)
        mel = fbank(waveform[None, :max(extent, 400)] * 32768., num_mel_bins=80,
                   sample_frequency=stock.RATE, dither=0., window_type="povey")
        contexts[start] = mel
    split = []
    for start, mel in contexts.items():
        anchor = start + 200 + torch.arange(len(mel)) * 160
        split.append(mel[anchor < 48000 if start == 0 else anchor >= 48000])
    reference = torch.cat(split)
    source.check(all(len(part) >= 2 for part in split), "Insufficient raw Mel support")
    shared_mean = reference.mean(0, keepdim=True)
    shared_var = sum((part - part.mean(0)).square().sum(0) for part in split) / (len(reference)-2)

    def predict(arm, offset):
        mel = contexts[offset]
        mean = shared_mean if arm == "shared_center_scale" else mel.mean(0, keepdim=True)
        var = mel.var(0, keepdim=True) if arm == "local" else shared_var
        normalized = (mel - mean) / torch.sqrt(var + 1e-7)
        count = len(normalized) // 2 * 2
        frames = normalized[:count].reshape(1, count // 2, 160)
        mask = torch.ones(1, frames.shape[1], dtype=torch.long)
        states = model.backbone(input_features=frames, attention_mask=mask, output_hidden_states=True).hidden_states
        weighted = torch.stack(states).mul(model.layer_weights.softmax(dim=0)[:, None, None, None]).sum(0)[0]
        return model.head(weighted).squeeze(-1).cpu().numpy()

    with torch.inference_mode():
        parity = False
        readouts = {}
        for arm in ARMS:
            def evaluate(offset, samples, arm=arm):
                nonlocal parity
                result = predict(arm, offset)
                if arm == "local" and not parity:
                    expected = dense.frame_scores(model, samples, robust=False)
                    source.check(np.array_equal(expected, result), "Native local normalization differs")
                    parity = True
                return result

            readouts[arm] = [round(v, 8) if v is not None else None
                             for v in dense.capture_windows(pcm, speech, evaluate)]
    source.check(parity and len(reference) <= 600, "Shared moment clock differs")
    return readouts, [len(s) for s in split]


def candidate(row, arm, point=None):
    scores = row["views"][arm]
    top = max(((value, -i) for i, value in enumerate(scores) if value is not None), default=None)
    if top is None or (point is not None and top[0] <= point):
        return None
    return top[0], [STARTS[-top[1]], STARTS[-top[1]] + WIDTH]


def existing(row):
    scores = row["existing_scores"]
    if row.get("existing_summary"):
        return scores[1] if scores is not None and scores[0] > stock.SHORT_POINT else None
    top = stock.strongest({"scores": scores}, range(len(stock.WINDOWS)))
    return stock.WINDOWS[-top[1]] if top is not None and top[0] > stock.SHORT_POINT else None


def freeze_point(rows):
    points = {}
    for arm in ARMS:
        ordinary, isolated = {}, []
        for row in rows:
            top = candidate(row, arm)
            if row["truth"] or top is None or existing(row):
                continue
            value = top[0]
            group = row["group"]
            if row["kind"] in interval.BACKGROUNDS:
                isolated.append(value)
            else:
                ordinary[group] = max(ordinary.get(group, -math.inf), value)
        source.check(len(ordinary) >= 2 and isolated, "No calibration control support")
        points[arm] = max(sorted(ordinary.values(), reverse=True)[1], max(isolated))
    return points


def tally(rows, arm, point=None, supplement=False):
    counts, ordinary, added, new_ordinary, lost = Counter(), set(), [], set(), []
    for index, row in enumerate(rows):
        before = existing(row)
        measured = None if arm == "existing" else candidate(row, arm, point)
        observed = before if arm == "existing" or (supplement and before) else measured[1] if measured else None
        truth = row["truth"]
        if truth:
            counts["edits"] += 1
            previous_useful = bool(before and stock.useful(before, truth))
            useful = bool(observed and stock.useful(observed, truth))
            counts["useful"] += useful
            counts["misplaced"] += bool(observed and not useful)
            counts["unflagged_edits"] += observed is None
            if useful and not previous_useful:
                added.append([index, row["group"], row["family"], observed, truth])
            if previous_useful and not useful:
                lost.append(index)
        elif row["kind"] in interval.BACKGROUNDS:
            counts["background"] += 1
            counts["background_flags"] += bool(observed)
        elif row["kind"] == "resyn":
            counts["same_words"] += 1
            counts["same_words_flags"] += bool(observed)
        else:
            counts["ordinary"] += 1
            counts["ordinary_flags"] += bool(observed)
            if observed:
                ordinary.add(row["group"])
                if not before:
                    new_ordinary.add(row["group"])
        checked = bool(row["existing_scores"] is not None) if row.get("existing_summary") else any(
            value is not None for value in row["existing_scores"])
        if arm != "existing":
            checked = bool(candidate(row, arm) or (supplement and checked))
        counts["unassessed"] += not checked
    return {"counts": dict(counts), "ordinary_groups": len(ordinary), "new_ordinary_groups": sorted(new_ordinary),
            "added_useful": added, "lost_useful": lost}


def verify_rows(panel, rows):
    expected = input_rows(panel)
    for row, original in zip(rows, expected, strict=True):
        source.check({k: row[k] for k in original} == original and row["existing_summary"] == (panel in {"families", "sine"}),
                     "A measured source or comparison score differs from its public receipt")
        readouts, pool = row["views"], row["pooled_frames"]
        source.check(set(readouts) == set(ARMS)
                     and all(len(v) == len(STARTS) for v in readouts.values())
                     and all(value is None or math.isfinite(value) for values in readouts.values() for value in values),
                     "Incomplete window records")
        admission = [[value is not None for value in readouts[arm]] for arm in ARMS]
        source.check(admission[0] == admission[1] == admission[2], "A variant changed speech admission")
        source.check((pool is None and not any(admission[0])) or (len(pool or ()) == 2 and all(2 <= x <= 300 for x in pool)),
                     "Invalid pooled frame support")


def review(receipt):
    lock, panels = receipt["freeze"], receipt["panels"]
    source.check(lock["schema"] == SCHEMA and lock["method"] == METHOD
                 and set(panels) == {"development", "random_placement", "families", "sine"}
                 and set(lock["bindings"]) == {"stock_development", "stock_holdout", "three_generator_roster",
                                               "sine", "old_points", "geometry_parent"}, "Missing study inputs")
    for panel, rows in panels.items():
        verify_rows(panel, rows)
    points = freeze_point(panels["development"])
    source.check(points == receipt["points"], "Frozen points differ")
    return {panel: {"existing": tally(rows, "existing"),
                    **{arm: {"alone": tally(rows, arm, points[arm]),
                             "supplement": tally(rows, arm, points[arm], supplement=True)} for arm in ARMS}}
            for panel, rows in panels.items()}


def native_run(args):
    import numpy as np

    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free
    from scripts.prepare_audio_specialists_v15 import ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    lock = {"schema": SCHEMA, "method": METHOD, "bindings": ledger_roles(), "asset_hashes": ASSETS,
            "runner_sha256": source.path_sha(Path(__file__)),
            "runtime": {name: importlib.metadata.version(name) for name in sine.RUNTIME}}
    args.path.mkdir(mode=0o700)
    source.write(args.path / "freeze.json", lock)
    roster, native, _ = interval.native_rows(args.archive)
    sine_parent = source.read(SINE_PARENT)
    sine.audit_native(args.sine, sine_parent)
    for _row, pcm in sine.native_rows(args.sine, sine_parent["freeze"]):
        raw = pcm.tobytes()
        native[source.digest(raw)] = raw
    families = input_rows("families")
    if [r["pcm_sha256"] for r in families] != [r["pcm_sha256"] for r in roster["families"]]:
        known = {r["pcm_sha256"] for r in roster["families"]}
        source.check({r["pcm_sha256"] for r in families} == known, "Held sources differ")
    network_free()
    model = load_voice({"weights": args.weights, "assets": ASSETS})
    activity = SpeechActivity(args.weights / "silero_vad.jit")
    panels = {}
    for panel in ("development", "random_placement", "families", "sine"):
        panel_rows = families if panel == "families" else input_rows(panel)
        rows = []
        for row in panel_rows:
            raw = native[row["pcm_sha256"]]
            source.check(source.digest(raw) == row["pcm_sha256"], "Native mixture hash differs")
            pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
            views, pool = windows(model, pcm, activity.spans(pcm))
            rows.append(row | {"views": views, "pooled_frames": pool, "existing_summary": panel in {"families", "sine"}})
            if len(rows) % 12 == 0:
                print(f"{panel} {len(rows)}/{len(panel_rows)}", flush=True)
        panels[panel] = rows
        source.write(args.path / f"{panel}.json", rows)
        if panel == "development":
            points = freeze_point(rows)
            source.write(args.path / "points.json", points)
            print(f"Frozen points {points}", flush=True)
    receipt = {"freeze": lock, "panels": panels, "points": points}
    receipt["outcome"] = review(receipt)
    source.write(args.path / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("score", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--sine", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--weights", type=Path, default=source.ROOT / "models/weights/audio-upgrade")
    args = parser.parse_args()
    if args.phase == "score" and (args.sine is None or args.archive is None):
        parser.error("--sine and --archive are required for native scoring")
    receipt = native_run(args) if args.phase == "score" else source.read(args.path)
    outcome = review(receipt)
    source.check(outcome == receipt["outcome"], "Saved results differ")
    print(json.dumps(outcome, sort_keys=True))


if __name__ == "__main__":
    main()
