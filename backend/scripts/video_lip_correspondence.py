"""Test timing-local speech and lip-shape correspondence fitted on genuine clips."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.video-lip-correspondence/1"
ARMS = ("shape", "shape_motion")
POINTS = {"tight": 0, "wider": 2}
ROLES = {
    "original": "ordinary", "gain": "ordinary", "same_words": "same_word",
    "different_words": "content", "global_early": "whole_delay", "global_late": "whole_delay",
}
NEGATIVE_HOPS = (12, 18, 24, 32)
RIDGE = .25
FLANK = 7680
MIN_CONTEXT = 4
MIN_LOCAL = 5
MIN_ACTIVITY = .22
METHOD = {
    "fit": "Learn paired speech-to-mouth compatibility from original picture and acoustic samples only. "
           "Use time-mismatched observations from the same file as alternatives, with comparable speech activity.",
    "shape": "Compare local acoustic classes with visible width, opening, and three measures of lip shape.",
    "shape_motion": "Also use signed motion and the change in the acoustic class distribution.",
    "review": "Require one supported lag in each flank and persistent disagreement across neighboring lags within the interval.",
    "calibration": "Use distinct speakers, with matched same-word and constant-delay controls. "
                   "Choose the method that adds the most correct locations on calibration edits at the two frozen control budgets.",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def save(path, payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    with path.open("xb") as stream:
        stream.write(gzip.compress(encoded, mtime=0))


def visual(frames, arm):
    import numpy as np

    from app.evidence.video_viseme_articulation import visual_features

    fitted = visual_features(frames, shape=True)
    if fitted is None:
        return None
    times, indices, values = fitted
    columns = (1, 4, 6, 7, 8, 0, 3)
    if arm == "shape_motion":
        columns += (2, 5)
    values = np.asarray(values)[:, columns]
    if arm == "shape_texture":
        texture = [frames[index].get("texture") for index in indices]
        valid = np.asarray([row is not None for row in texture])
        if valid.sum() < 20:
            return None
        return np.asarray(times)[valid], np.asarray(indices)[valid], np.column_stack((values[valid], [
            row for row in texture if row is not None]))
    return np.asarray(times), np.asarray(indices), values


def acoustic(groups, arm):
    import numpy as np

    from app.evidence.video_viseme_ranked import acoustic as smooth

    nearby = smooth(groups)
    dense = np.asarray([row if row is not None else [0.] * 6 for row in nearby])
    active = dense[:, -1] >= MIN_ACTIVITY
    values = dense[:, :5]
    if arm == "shape_motion":
        change = np.zeros_like(values)
        change[2:-2] = values[4:] - values[:-4]
        values = np.column_stack((values, change))
        qualified = np.zeros_like(active)
        qualified[2:-2] = active[:-4] & active[4:]
        active &= qualified
    return values, active, dense[:, -1]


def nearest(times, target):
    import numpy as np

    from app.evidence.video_viseme_timing import MAX_NEAREST

    right = np.searchsorted(times, target).clip(0, len(times) - 1)
    left = (right - 1).clip(0)
    index = np.where(abs(times[left] - target) <= abs(times[right] - target), left, right)
    return index, abs(times[index] - target) <= MAX_NEAREST


def contrasts(row, arm):
    import numpy as np

    from app.evidence.video_viseme_ranked import CENTER, STRIDE

    video = visual(row["frames"], arm)
    if video is None:
        return None
    time, _, picture = video
    voice, active, activity = acoustic(row["deliveries"]["original"]["groups"], arm)
    clock = CENTER + STRIDE * np.arange(len(voice))
    assigned, usable = nearest(time, clock - STRIDE)
    target, foil = [], []
    for index in np.flatnonzero((active & usable) & (np.arange(len(active)) % 2 == 0)):
        left_and_right = [index + direction * hop for hop in NEGATIVE_HOPS for direction in (-1, 1)]
        options = [i for i in left_and_right if 0 <= i < len(voice) and active[i] and usable[i]
                   and abs(activity[i] - activity[index]) <= .35]
        if not options:
            continue
        best = min(options, key=lambda i: (abs(activity[i] - activity[index]), abs(i - index), i))
        target.append(index)
        foil.append(best)
    if not target:
        return None
    return (voice[target], picture[assigned[target]], voice[foil], picture[assigned[foil]])


def fit(rows, arm):
    import numpy as np

    parts = [(row["speaker"], contrast) for row in rows if (contrast := contrasts(row, arm)) is not None]
    if len({speaker for speaker, _ in parts}) < 8:
        raise ValueError("Too few fit speakers with native paired observations")
    voice = np.concatenate([arrays[0] for _, arrays in parts])
    picture = np.concatenate([arrays[1] for _, arrays in parts])
    a_mean, v_mean = voice.mean(axis=0), picture.mean(axis=0)
    a_scale, v_scale = voice.std(axis=0).clip(.04), picture.std(axis=0).clip(.04)

    def standard(values, mean, scale):
        return ((values - mean) / scale).clip(-4., 4.)

    def product(a, v):
        return np.einsum("ij,ik->ijk", a, v).reshape(len(a), -1)

    pieces = defaultdict(list)
    for speaker, (a, v, shifted_a, shifted_v) in parts:
        a, shifted_a = (standard(record, a_mean, a_scale) for record in (a, shifted_a))
        v, shifted_v = (standard(record, v_mean, v_scale) for record in (v, shifted_v))
        positive = product(a, v)
        # Observe mismatches by moving each stream independently in time.
        negative = np.concatenate((product(shifted_a, v), product(a, shifted_v)))
        delta = np.concatenate((positive, positive)) - negative
        pieces[speaker].append((delta.T @ delta / len(delta), delta.mean(axis=0), (positive.mean(axis=0) + negative.mean(axis=0)) / 2))
    pooled = [np.mean([row[i] for row in group], axis=0) for group in pieces.values() for i in range(3)]
    gram, target, midpoint = (np.mean(pooled[i::3], axis=0) for i in range(3))
    weights = np.linalg.solve(gram + RIDGE * np.eye(len(target)), target)
    center = float(midpoint @ weights)
    scale = max(.05, float(target @ weights))
    return {
        "arm": arm, "clips": len(parts), "speakers": len(pieces), "positive_pairs": len(voice),
        "audio_center": a_mean.round(10).tolist(), "audio_scale": a_scale.round(10).tolist(),
        "picture_center": v_mean.round(10).tolist(), "picture_scale": v_scale.round(10).tolist(),
        "weights": weights.reshape(len(a_mean), len(v_mean)).round(10).tolist(),
        "center": round(center, 10), "scale": round(scale, 10),
    }


def matches(frames, groups, head):
    import numpy as np

    from app.evidence.video_viseme_ranked import CENTER, STRIDE
    from app.evidence.video_viseme_timing import LAGS

    if (motion := visual(frames, head["arm"])) is None:
        return None
    times, indices, picture = motion
    voice, active, _ = acoustic(groups, head["arm"])
    voice = ((voice - head["audio_center"]) / head["audio_scale"]).clip(-4., 4.)
    picture = ((picture - head["picture_center"]) / head["picture_scale"]).clip(-4., 4.)
    clock = CENTER + STRIDE * np.arange(len(voice))
    frame, overlap = nearest(times, clock[:, None] - np.asarray(LAGS))
    include = active & overlap.all(axis=1) & (np.arange(len(active)) % 2 == 0)
    voice, frame = voice[include], frame[include]
    score = np.einsum("ij,jk,ilk->il", voice, np.asarray(head["weights"]), picture[frame], optimize=True)
    values = np.rint((score - head["center"]) / head["scale"] * 1_000_000)
    return clock[include], values.astype(np.int64), indices[frame]


def locate(row, groups, head):
    import numpy as np

    from app.evidence.video_viseme_timing import LAGS, STEP, WIDTHS

    observed = matches(row["frames"], groups, head)
    if observed is None or len(observed[0]) < 2 * MIN_CONTEXT + MIN_LOCAL:
        return {"status": "insufficient_events"}
    clock, scores, frames = observed
    cumulative = np.vstack((np.zeros((1, scores.shape[1]), np.int64), np.cumsum(scores, axis=0)))
    proposals = []
    spans = 0
    for width in WIDTHS:
        for start in range(STEP, row["samples"] - width - STEP + 1, STEP):
            stop = start + width
            lo, inside, outside, hi = np.searchsorted(clock, (start - FLANK, start, stop, stop + FLANK))
            lengths = inside - lo, outside - inside, hi - outside
            if lengths[0] < MIN_CONTEXT or lengths[2] < MIN_CONTEXT or lengths[1] < MIN_LOCAL:
                continue
            spans += 1
            a, b, c = ((cumulative[j] - cumulative[i]) / (size * 1_000_000)
                       for i, j, size in zip((lo, inside, outside), (inside, outside, hi), lengths, strict=True))
            lane = max(range(len(LAGS)), key=lambda i: (min(a[i], c[i]), a[i] + c[i], -abs(LAGS[i])))
            flank = min(a[lane], c[lane])
            local = max(b[max(0, lane - 1):lane + 2])
            if flank > 0 and local < 0:
                proposals.append({
                    "status": "measured", "score": round(flank - local, 7), "span": [start, stop],
                    "baseline_lag": LAGS[lane], "change": 0, "before_frame": int(frames[inside - 1, lane]),
                    "after_frame": int(frames[outside, lane]), "frames": sorted(set(map(int, frames[inside:outside, lane]))),
                })
    if not spans:
        return {"status": "insufficient_context"}
    if not proposals:
        return {"status": "no_supported_change"}
    winner = min(proposals, key=lambda item: (-item["score"], item["span"]))
    return winner


def overlaps(a, b):
    start, end = a
    lo, hi = b
    return 5 * (min(end, hi) - max(start, lo)) >= 2 * (max(end, hi) - min(start, lo))


def choose(row, role, arm, point):
    readings = row["cases"][role]
    incumbent = readings["incumbent"]
    if incumbent is not None:
        return incumbent
    candidate = readings.get(arm)
    return candidate if candidate and candidate["status"] == "measured" and candidate["score"] > point else None


def totals(rows, arm, point):
    counts = defaultdict(Counter)
    for row in rows:
        for role, category in ROLES.items():
            if role not in row["cases"]:
                continue
            incumbent = row["cases"][role]["incumbent"]
            assessed = row["cases"][role]["incumbent_assessed"]
            candidate = row["cases"][role].get(arm)
            if candidate is not None:
                assessed |= candidate["status"] in {"measured", "no_supported_change"}
            flag = choose(row, role, arm, point)
            useful = bool(flag and category == "content" and overlaps(flag["span"], row["truth"]))
            same_word = choose(row, "same_words", arm, point) if useful else None
            counts[category].update(total=1, assessed=assessed, flags=bool(flag), added=bool(flag and not incumbent),
                                    useful=useful, misplaced=bool(flag and category == "content" and not useful),
                                    confirmed=useful and not (same_word and overlaps(same_word["span"], flag["span"])))
    return {key: dict(row) for key, row in counts.items()}


def readouts(captured, incumbent, points, heads):
    from scripts import video_lip_articulation_transfer as prior
    from scripts.video_lip_content_transfer import choose as prior_choose

    by_source = {row["picture_sha256"]: row for row in incumbent}
    output = []
    for row in captured:
        previous = by_source[row["picture_sha256"]]
        if (row["samples"], row["truth"]) != (previous["samples"], previous["truth"]):
            raise ValueError("Prior comparison did not use the same native inputs")
        result = {field: row[field] for field in ("speaker", "picture_sha256", "samples", "truth", "condition", "time_base")}
        result["cases"] = {}
        for role, delivery in row["deliveries"].items():
            if delivery["pcm_sha256"] != previous["pcm"][role]:
                raise ValueError("Prior soundtrack digest differs")
            readings = previous["readings"][role]
            prior_flag = prior_choose(readings, "aperture_classes", points, "tight")
            assessed = any(readings[key]["status"] in {"measured", "no_supported_change"}
                           for key in ("sparse", "aperture", "aperture_classes"))
            scores = {arm: prior.parent.evidence(row, locate(row, delivery["groups"], head)) for arm, head in heads.items()}
            result["cases"][role] = {"pcm_sha256": delivery["pcm_sha256"], "incumbent": prior_flag,
                                    "incumbent_assessed": assessed, **scores}
        output.append(result)
    return output


def calibration(rows):
    table = {}
    for arm in ARMS:
        by_speaker = dict.fromkeys({row["speaker"] for row in rows}, 0.)
        for row in rows:
            for role, readings in row["cases"].items():
                if ROLES[role] == "content" or readings["incumbent"] is not None:
                    continue
                by_speaker[row["speaker"]] = max(by_speaker[row["speaker"]], readings[arm].get("score", 0.))
        maxima = sorted(by_speaker.values(), reverse=True)
        table[arm] = {level: {"point": maxima[n], "outcomes": totals(rows, arm, maxima[n])} for level, n in POINTS.items()}
    return table


def select(points):
    def order(item):
        arm, level = item
        stats = points[arm][level]["outcomes"]
        controls = sum(stats[role]["added"] for role in ("ordinary", "same_word", "whole_delay"))
        return stats["content"]["confirmed"], stats["content"]["useful"], -controls, -ARMS.index(arm), level == "tight"

    arm, level = max(((arm, level) for arm, values in points.items() for level in values), key=order)
    return {"arm": arm, "level": level, "point": points[arm][level]["point"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    calibrate = sub.add_parser("fit")
    for field in ("training", "calibration", "incumbent", "output"):
        calibrate.add_argument(field, type=Path)
    test = sub.add_parser("assess")
    for field in ("freeze", "held", "incumbent", "output"):
        test.add_argument(field, type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.action == "verify":
        packet = read(args.receipt)
        assert packet["schema"] == SCHEMA and packet["method"] == METHOD
        assert packet["freeze"]["points"] == calibration(packet["freeze"]["rows"])
        assert packet["freeze"]["selected"] == select(packet["freeze"]["points"])
        assert packet["outcomes"] == results(packet["rows"], packet["freeze"]["points"])
        print(json.dumps(packet["outcomes"], indent=2))
        return

    incumbent = read(args.incumbent)
    if args.action == "fit":
        training, cal = read(args.training), read(args.calibration)
        fit_groups = {row["speaker"] for row in training["rows"]}
        cal_groups = {row["speaker"] for row in cal["rows"]}
        if (training["phase"], cal["phase"]) != ("fit", "calibration") or fit_groups & cal_groups:
            raise ValueError("Fit and calibration inputs do not have disjoint speaker roles")
        assert training["plan_sha256"] == cal["plan_sha256"] == incumbent["plan_sha256"]
        assert sha(args.training) == incumbent["fit_sha256"] and sha(args.calibration) == incumbent["calibration_sha256"]
        heads = {arm: fit(training["rows"], arm) for arm in ARMS}
        readings = readouts(cal["rows"], incumbent["calibration"], incumbent["points"], heads)
        points = calibration(readings)
        freeze = {"schema": SCHEMA, "method": METHOD, "heads": heads, "points": points, "rows": readings,
                  "selected": select(points), "fit_speakers": sorted(fit_groups), "plan_sha256": training["plan_sha256"],
                  "triplet": {field: sha(getattr(args, field)) for field in ("training", "calibration", "incumbent")}}
        save(args.output, freeze)
        print(json.dumps(points, indent=2))
    else:
        freeze, held = read(args.freeze), read(args.held)
        assert held["phase"] == "held" and freeze["plan_sha256"] == held["plan_sha256"] == incumbent["plan_sha256"]
        assert sha(args.held) == incumbent["held_sha256"]
        assert freeze["points"] == calibration(freeze["rows"])
        assert not {row["speaker"] for row in held["rows"]}.intersection(
            set(freeze["fit_speakers"]) | {row["speaker"] for row in freeze["rows"]})
        readings = readouts(held["rows"], incumbent["cohort"], incumbent["points"], freeze["heads"])
        counts = results(readings, freeze["points"])
        packet = {"schema": SCHEMA, "method": METHOD, "freeze": freeze, "rows": readings, "outcomes": counts,
                  "held_sha256": sha(args.held), "incumbent_sha256": sha(args.incumbent)}
        save(args.output, packet)
        print(json.dumps(counts, indent=2))


def results(readings, points):
    return {"incumbent": totals(readings, "incumbent", 0)} | {
        f"{arm}/{level}": totals(readings, arm, record["point"])
        for arm, levels in points.items() for level, record in levels.items()
    }


if __name__ == "__main__":
    main()
