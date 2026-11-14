"""Compare two interval-trained speech heads on retained, noisy PCM mixtures."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from scripts import audio_source_isolation_diagnostic as source
from scripts import audio_voice_attenuation_transfer as stock_sources
from scripts import review_audio_stock_voice_transfer as stock

RATE, SAMPLES = stock.RATE, source.SAMPLES
WIDTH, STEP = 12800, 3200
STARTS = tuple(range(0, SAMPLES - WIDTH + 1, STEP))
METHOD = {
    "input": "Six seconds of continuous 16 kHz mono PCM. Two original four-second contexts start at zero and two seconds. Align native affine frame responses at 280 + 320i samples from each context origin. Drop the 200 ms nearest an internal context boundary, then average duplicate frame centers. Check 800 ms ranges every 200 ms with at least half detected speech.",
    "features": "Mean and 20th/80th response percentiles inside the proposed range, mean minus available near flanks and outside median, difference between the half-range means, and speech occupancy. No absolute position, source identity or known mix level enters the head.",
    "fit": "Fit clip-balanced ridge regression with penalty 0.1, an unpenalized intercept and scale floor 0.001. The clock head learns known overlap as IoU if the candidate covers at least half the edit, else zero. The paired head adds a unit-weight loss on feature differences between each edited range and the identical clock of its matched ordinary splices. Fit uses only the 40 separate development recordings from the isolation diagnostic.",
    "point": "Use 44 severe-noise or isolated recordings from the first stock-voice test. Preserve current flags. Set the strict candidate point at the larger of the second-highest unflagged genuine group maximum and the highest unflagged isolated background response. Fit and seal points before computing held features.",
    "held": "Verify both previously exposed held PCM banks. Reuse exact serving two-second scores on randomized stock clips. Rerun the five-window serving response on the 52 clips with three generators because their saved earlier comparator used only two windows. Report added and standalone useful locations, misses, wrong ranges, assessment, ordinary false flags and isolated backgrounds. A useful location covers half the edit and reaches 0.30 IoU.",
    "decision": "For a new-source follow-up, require at least two additional useful locations across two named generator checkpoints, at most one additional ordinary group in either held bank, and zero additional isolated-background flags. This replay cannot establish generalization to production recordings.",
}
ARMS = ("clock", "paired")
BACKGROUNDS = {"background", "background_only", "isolated"}
SCHEMA = "chipmunk.audio-interval-learning/1"
PANEL_SIZES = {"fit": 40, "calibration": 44, "random_placement": 43, "families": 52}
HELD_FAMILIES = {"clone": "Chatterbox", "expressive": "ElevenLabs v4", "stock": "Kokoro"}


def native_rows(archive):
    prior, pins = source.sources()
    rows, data = {}, {}
    for role, archive_name in (("development", "audio-stock-voice-transfer"),
                               ("random_placement", "audio-random-clock-stock")):
        panel = "calibration" if role == "development" else role
        pins[panel] = source.path_sha(stock_sources.prior_path(role))
        waveforms = stock_sources.signals(role, archive / f"{archive_name}-2026-11-13")
        rows[panel] = []
        for group, kind, truth, pcm, scores in stock_sources.prior_rows(role):
            data[pcm] = waveforms[pcm]
            rows[panel].append({"group": group, "kind": kind, "family": kind if truth else None,
                                "truth": truth, "pcm_sha256": pcm, "existing_scores": scores})
    for row in prior:
        panel = "families" if row["role"] == "assessment" else "fit"
        pcm = row["pcm_sha256"]
        data[pcm] = next(source.source_data(source.DATA / row["bank"], row))
        family = HELD_FAMILIES[row["family"]] if panel == "families" and row["truth"] else None
        rows.setdefault(panel, []).append({"group": row["group"], "kind": row["kind"], "family": family,
                                           "truth": row["truth"], "pcm_sha256": pcm, "existing_scores": None})
    source.check({key: len(items) for key, items in rows.items()} == PANEL_SIZES, "Panel size changed")
    source.check(len(data) == sum(PANEL_SIZES.values()), "Mixture bytes repeat across panels")
    return rows, data, pins


def design(values, activity, capture):
    import numpy as np

    spans = activity.spans(values)
    by_center = {}
    if spans:
        for origin in source.CUTS:
            for frame, response in enumerate(capture.forward(values[origin:origin + 64000])):
                center = origin + 280 + frame * 320
                if (origin == 0 or center >= origin + STEP) and (origin != 0 or center < 64000 - STEP):
                    by_center.setdefault(center, []).append(response)
    centers = np.array(sorted(by_center), dtype=np.int32)
    score = np.array([np.mean(by_center[x], dtype=np.float64) for x in centers])
    output = []
    for start in STARTS:
        end = start + WIDTH
        speech = sum(max(0., min(end / RATE, b) - max(start / RATE, a)) for a, b in spans) * RATE / WIDTH
        intersection = (start <= centers) & (centers < end)
        inside = score[intersection]
        if speech < .5 - 1e-8 or len(inside) < 36:
            output.append(None)
            continue
        flank = score[((start - STEP * 2 <= centers) & (centers < start)) |
                      ((end <= centers) & (centers < end + STEP * 2))]
        outside = score[~intersection]
        mean = inside.mean()
        lower, upper = np.quantile(inside, [.2, .8])
        middle = start + WIDTH // 2
        first = score[intersection & (centers < middle)].mean()
        second = score[intersection & (centers >= middle)].mean()
        output.append([mean, lower, upper, mean - flank.mean(), mean - np.median(outside), abs(first - second), speech])
    return output, spans


def native_inference(rows, data, weights, panel):
    import numpy as np

    from app.detectors.audio.calibration import logit
    from app.detectors.audio.content import window_in_speech
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free
    from scripts.prepare_audio_specialists_v15 import ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    network_free()
    model = load_voice({"weights": weights, "assets": ASSETS})
    capture, activity = source.FrameCapture(model), SpeechActivity(weights / "silero_vad.jit")
    output = []
    for row in rows:
        pcm = np.frombuffer(data[row["pcm_sha256"]], dtype="<i2").astype(np.float32) / 32768
        features, spans = design(pcm, activity, capture)
        scores = row["existing_scores"]
        if panel == "families":
            scores = [logit(model.predict([pcm[a:b]])[0]) if window_in_speech(spans, a / RATE, b / RATE) else None
                      for a, b in stock.WINDOWS]
        output.append(row | {"existing_scores": scores, "features": features})
        if len(output) % 8 == 0 or len(output) == len(rows):
            print(f"{panel}: {len(output)}/{len(rows)} native recordings", flush=True)
    return output


def interval(index):
    return [STARTS[index], STARTS[index] + WIDTH]


def target(index, truth):
    if truth is None:
        return 0.
    a, b = interval(index)
    overlap = max(0, min(b, truth[1]) - max(a, truth[0]))
    return overlap / (max(b, truth[1]) - min(a, truth[0])) if 2 * overlap >= truth[1] - truth[0] else 0.


def fit(rows):
    import numpy as np

    available = [[(x, target(i, row["truth"])) for i, x in enumerate(row["features"]) if x is not None] for row in rows]
    data = np.array([x for clip in available for x, _ in clip])
    weight = np.array([1. / len(clip) for clip in available for _ in clip])
    mean = np.average(data, axis=0, weights=weight)
    scale = np.maximum(.001, np.sqrt(np.average((data - mean)**2, axis=0, weights=weight)))
    design_matrix = np.column_stack((np.ones(len(data)), (data - mean) / scale))
    labels = np.array([y for clip in available for _, y in clip])
    lhs = design_matrix.T @ (design_matrix * weight[:, None]) + np.diag([0., *([.1 * weight.sum()] * len(mean))])
    rhs = design_matrix.T @ (labels * weight)
    paired_lhs, paired_rhs = lhs.copy(), rhs.copy()
    for edited in (r for r in rows if r["truth"]):
        for control in (r for r in rows if r["group"] == edited["group"] and r["kind"].startswith("ordinary_")):
            pairs = [(np.r_[0., (np.array(x) - np.array(z)) / scale], y)
                     for i, (x, z) in enumerate(zip(edited["features"], control["features"], strict=True))
                     if x is not None and z is not None and (y := target(i, edited["truth"])) > 0]
            if pairs:
                matrix, labels = np.array([x for x, _ in pairs]), np.array([y for _, y in pairs])
                divisor = 2 * len(pairs)  # Each edit contributes at most one paired clip's weight.
                paired_lhs += matrix.T @ matrix / divisor
                paired_rhs += matrix.T @ labels / divisor
    return {"mean": mean.tolist(), "scale": scale.tolist(),
            "weights": {"clock": np.linalg.solve(lhs, rhs).tolist(), "paired": np.linalg.solve(paired_lhs, paired_rhs).tolist()}}


def summarize(row, coefficients):
    def best(arm):
        bias, *weights = coefficients["weights"][arm]
        return max(((round(bias + sum(w * (v - m) / s for w, v, m, s in zip(
            weights, features, coefficients["mean"], coefficients["scale"], strict=True)), 10), -i)
                    for i, features in enumerate(row["features"]) if features is not None), default=None)

    current = stock.strongest({"scores": row["existing_scores"]}, range(len(stock.WINDOWS)))
    scores = {"existing": [current[0], list(stock.WINDOWS[-current[1]])] if current else None}
    for arm in ARMS:
        top = best(arm)
        scores[arm] = [top[0], interval(-top[1])] if top else None
    return {key: row[key] for key in ("pcm_sha256", "kind", "group", "truth", "family")} | {"scores": scores}


def selected(row, arm, point):
    result = row["scores"][arm]
    return result[1] if result is not None and result[0] > point else None


def calibrate(readings):
    points = {}
    for arm in ARMS:
        ordinary, background = {}, []
        for row in readings:
            score = row["scores"][arm]
            if row["truth"] or score is None or selected(row, "existing", stock.SHORT_POINT):
                continue
            if row["kind"] in BACKGROUNDS:
                background.append(score[0])
            else:
                ordinary[row["group"]] = max(ordinary.get(row["group"], -math.inf), score[0])
        source.check(len(ordinary) >= 2, "Insufficient ordinary groups to freeze the point")
        points[arm] = max(sorted(ordinary.values(), reverse=True)[1], max(background, default=-math.inf))
    return points


def tally(rows, arm, point, supplement=False):
    counts, groups, additions = Counter(), set(), []
    for row in rows:
        current = selected(row, "existing", stock.SHORT_POINT)
        chosen = (current if supplement else None) or selected(row, arm, point)
        truth = row["truth"]
        kind = "edit" if truth else "background" if row["kind"] in BACKGROUNDS else "ordinary"
        counts[kind] += 1
        if chosen is None:
            available = row["scores"][arm] is not None or (supplement and row["scores"]["existing"] is not None)
            counts[f"{'unflagged' if available else 'unassessed'}_{kind}"] += 1
            continue
        useful = truth is not None and stock.useful(chosen, truth)
        outcome = "useful" if useful else "misplaced" if truth else "flagged_" + kind
        counts[outcome] += 1
        if kind == "ordinary":
            groups.add(row["group"])
        if not current and supplement and useful:
            additions.append({"family": row["family"], "group": row["group"], "range": chosen})
    return dict(counts, ordinary_groups=len(groups), additional_useful=additions)


def report(panels, points):
    results = {name: {"current": tally(rows, "existing", stock.SHORT_POINT),
                      **{arm: {"standalone": tally(rows, arm, points[arm]),
                               "supplement": tally(rows, arm, points[arm], supplement=True)} for arm in ARMS}}
               for name, rows in panels.items()}
    eligible = {}
    for arm in ARMS:
        gains = [gain for panel in results.values() for gain in panel[arm]["supplement"]["additional_useful"]]
        eligible[arm] = len(gains) >= 2 and len({gain["family"] for gain in gains}) >= 2 and all(
            panel[arm]["supplement"]["ordinary_groups"] <= panel["current"]["ordinary_groups"] + 1
            and panel[arm]["supplement"].get("flagged_background", 0) == panel["current"].get("flagged_background", 0)
            for panel in results.values())
    return {"panels": results, "eligible_for_new_sources": eligible}


def review(receipt):
    source.check(receipt["schema"] == SCHEMA and receipt["method"] == METHOD, "Changed method")
    panels = receipt["held"] | {"calibration": receipt["calibration"]}
    source.check({key: len(rows) for key, rows in panels.items()}
                 == {key: count for key, count in PANEL_SIZES.items() if key != "fit"}, "Incomplete evaluation panels")
    roster = []
    candidate_ranges = [interval(i) for i in range(len(STARTS))]
    for rows in panels.values():
        for row in rows:
            roster.append(row["pcm_sha256"])
            source.check(set(row["scores"]) == {"existing", *ARMS}, "Missing response")
            for arm, value in row["scores"].items():
                if value is None:
                    continue
                score, span = value
                allowed = stock.WINDOWS if arm == "existing" else candidate_ranges
                source.check(math.isfinite(score) and tuple(span) in (tuple(r) for r in allowed), "Invalid response or range")
    source.check(len(set(roster)) == len(roster) and all(stock.sha256(sha) for sha in roster), "Invalid mixture commitments")
    for role, panel in (("development", "calibration"), ("random_placement", "random_placement")):
        parent = stock_sources.prior_path(role)
        source.check(source.path_sha(parent) == receipt["source_commitments"][panel], "Parent study changed")
        for row, (group, kind, truth, sha, scores) in zip(panels[panel], stock_sources.prior_rows(role), strict=True):
            expected = None
            strongest = stock.strongest({"scores": scores}, range(len(stock.WINDOWS)))
            if strongest:
                expected = [strongest[0], list(stock.WINDOWS[-strongest[1]])]
            source.check((row["group"], row["kind"], row["truth"], row["pcm_sha256"], row["scores"]["existing"])
                         == (group, kind, truth, sha, expected), "Prior source label or current response differs")
    points = calibrate(receipt["calibration"])
    source.check(points == receipt["points"] and report(receipt["held"], points) == receipt["results"],
                 "Points or results do not replay")
    return receipt["results"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "fit", "calibrate", "random_placement", "families", "report", "review"))
    parser.add_argument("path", type=Path, help="Run directory or public receipt for review")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--weights", type=Path, default=source.ROOT / "models/weights/audio-upgrade")
    args = parser.parse_args()
    if args.phase == "review":
        print(json.dumps(review(source.read(args.path)), indent=2, sort_keys=True))
        return
    if args.archive is None:
        parser.error("--archive is required for native phases")
    import importlib.metadata

    from scripts.prepare_audio_specialists_v15 import ASSETS

    rows, data, pins = native_rows(args.archive)
    code = (Path(__file__), Path(source.__file__), Path(stock_sources.__file__), Path(stock.__file__),
            source.ROOT / "backend/app/detectors/audio/upgraded.py")
    freeze = {"method": METHOD, "schema": SCHEMA, "source_commitments": pins,
              "roster_sha256": source.digest(json.dumps(rows, sort_keys=True).encode()),
              "code_sha256": {path.name: source.path_sha(path) for path in code}, "assets": ASSETS,
              "runtime": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "transformers", "pyloudnorm")}}
    root = args.path
    freeze_path, fit_path, cal_path = (root / name for name in ("freeze.json", "fit.json", "calibration.json"))
    if args.phase == "freeze":
        root.mkdir(mode=0o700)
        source.write(freeze_path, freeze)
        print("Frozen 40 fit, 44 point-setting, and 95 held exact PCM inputs")
        return
    source.check(source.read(freeze_path) == freeze, "Frozen methods, inputs or runtime changed")
    if args.phase == "fit":
        source.check(not fit_path.exists(), "Fit is already sealed")
        measured = native_inference(rows["fit"], data, args.weights, "fit")
        source.write(fit_path, fit(measured))
        return
    coefficients = source.read(fit_path)
    if args.phase == "calibrate":
        source.check(not cal_path.exists(), "Point calibration is already sealed")
        measured = native_inference(rows["calibration"], data, args.weights, "calibration")
        readings = [summarize(row, coefficients) for row in measured]
        source.write(cal_path, {"fit_sha256": source.path_sha(fit_path), "readings": readings, "points": calibrate(readings)})
        return
    cal = source.read(cal_path)
    source.check(cal["fit_sha256"] == source.path_sha(fit_path) and cal["points"] == calibrate(cal["readings"]),
                 "Fit or calibration changed")
    if args.phase in ("random_placement", "families"):
        out = root / f"{args.phase}.json"
        source.check(not out.exists(), "Held panel was already scored")
        measured = native_inference(rows[args.phase], data, args.weights, args.phase)
        source.write(out, {"calibration_sha256": source.path_sha(cal_path), "readings": [summarize(row, coefficients) for row in measured]})
        return
    held = {}
    for panel in ("random_placement", "families"):
        saved = source.read(root / f"{panel}.json")
        source.check(saved["calibration_sha256"] == source.path_sha(cal_path), "Held response used a different calibration")
        held[panel] = saved["readings"]
    source.write(root / "result.json", freeze | {"calibration": cal["readings"], "held": held,
                  "points": cal["points"], "results": report(held, cal["points"])})


if __name__ == "__main__":
    main()
