"""Separate localization geometry, interference and blind separation on short voice edits."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend/data/independent-evaluation"
PANELS = ROOT / "docs/evaluation-panels"
SAMPLES, RATE = 96000, 16000
SCHEMA = "chipmunk.audio-source-isolation-diagnostic/1"
GRIDS = {"stock": (32000, (16000, 48000)), "dense": (16000, tuple(range(0, 80001, 8000)))}
VIEWS = ("mixture", "blind", "ideal_mask", "exact_foreground")
FIT = {"early_development": 14, "later_development": 26}
CUTS = (0, 32000)
OVERLAP = 32000
ROLE_BANKS = (
    ("early_development", "audio-remote-neighbor-final-2026-11-12", "audio-remote-neighbor-transfer-2026-11-12", "fresh"),
    ("later_development", "audio-variable-boundary-2026-11-12", "audio-variable-boundary-transfer-2026-11-12", None),
    ("assessment", "audio-remote-neighbor-replication-2026-11-12", "audio-remote-neighbor-replication-2026-11-12", None),
)
PROTOCOL = {
    "source": "Use the retained 52-view three-generator -5 dB comparison as the held panel. Fit points from two distinct retained splice panels (40 files, of which 28 genuine, including four isolated captures). Their rooms and exact foreground identities are disjoint from held. No held labels or scores set a point. All panels had earlier project exposure.",
    "geometry": "Compare the two original 2.0 s intervals at 1 and 3 s and a 1.0 s grid every 0.5 s. The short range is a source interval, not an attribution of classifier internals. Both use the unchanged half-window speech-admission rule and checkpoint preprocessing. Preserve all failed or skipped windows.",
    "dense_compute": "One unchanged four-second forward at starts 0 and 2 s supplies affine native frame logits. Take the median of the frames with centers inside each one-second interval. Where both contexts contain that interval, take the lower median. Use the same waveform admission rule. The encoder still sees up to four seconds. This is a distinct change in readout and geometry.",
    "mixture": "Unchanged submitted samples. Reproduce the earlier stock-window measurement and fixed point separately.",
    "blind": "Apply the pinned two-output separator to the first and last four seconds of the same submitted samples. Match stream assignment by absolute centered correlation across the two-second overlap and require at least .25 on both streams. Align polarity and linearly crossfade the overlap. A stream is eligible if its whole-output RMS is at least 8 percent of original RMS, followed by the same speech admission. Take the strongest supported anonymous stream at each window. No truth enters routing or scoring.",
    "ideal_mask": "Benchmark-only. Use the known isolated foreground and background to estimate a squared ideal-ratio mask in a 512-sample Hann STFT at 128-sample hops. Multiply the mixture spectrum by the mask and retain the mixture phase. Reconstruct and round native PCM; this comparison cannot run on an ordinary upload.",
    "exact_foreground": "Benchmark-only. Score the exact retained pre-mix foreground. A silent foreground in a room-only control is deliberately unassessed. This is a diagnostic comparison, not an upper bound on all detectors.",
    "points": "Choose the maximum assessed genuine development score separately for each view and grid, and flag strictly above. Compare each fixed point and a retrospective equal-false-group ranking. Count known impossible geometry, unassessed geometry and best supported overlap before the point.",
    "gate": "Blind must add at least two useful held edits across at least two generator families at no more than one additional ordinary group versus the fixed stock comparator, without losing any existing useful location or flagging an isolated room. Exact foreground and ideal-mask results can never enter serving.",
}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def path_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, result):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def source_data(folder, row):
    fields = (("", "pcm_sha256"), ("-foreground", "foreground_sha256"), ("-background", "background_sha256"))
    for suffix, key in fields:
        raw = (folder / "pcm" / f"{row['id']}{suffix}.pcm").read_bytes()
        check(len(raw) == 2 * SAMPLES and digest(raw) == row[key], "Exact source bytes differ")
        yield raw


def sources():
    rows, parents, ancestry = [], {}, {}
    for role, bank, publication, only in ROLE_BANKS:
        folder = DATA / bank
        public = PANELS / (publication + ".json")
        pins = read(public)["commitments"]
        for name in ("freeze", "roster"):
            check(path_sha(folder / f"{name}.json") == pins[name], "A published source commitment differs")
        parents[bank] = {"public_sha256": path_sha(public), "roster_sha256": pins["roster"], "freeze_sha256": pins["freeze"]}
        selected = [r for r in read(folder / "roster.json") if only is None or r.get("cohort") == only]
        ancestry[role] = ({r["room"] for r in selected},
                          {r[side]["canonical_speaker"] for r in selected for side in ("host", "donor") if r.get(side)})
        current = {}
        if role == "assessment":
            check(path_sha(folder / "current.json") == pins["current"], "Earlier model score receipt differs")
            current = {r["id"]: r for r in read(folder / "current.json")["records"]}
            parents[bank]["incumbent_sha256"] = pins["current"]
        for row in selected:
            mixed, first, second = source_data(folder, row)
            check_mix(mixed, first, second)
            expected = current.get(row["id"])
            if expected:
                check(expected["pcm_sha256"] == row["pcm_sha256"] and all(
                    digest(mixed[2 * x["start"]:2 * x["end"]]) == x["pcm_sha256"] for x in expected["windows"]
                ), "Earlier exact window bytes differ")
            else:
                check(role != "assessment", "A held input lacks its earlier model receipt")
            rows.append({"role": role, "id": row["id"], "bank": bank, "group": row["group"], "kind": row["arm"],
                         "family": row.get("family") or (row.get("donor") or {}).get("family"), "truth": row["truth"],
                         **{key: row[key] for key in ("pcm_sha256", "foreground_sha256", "background_sha256")},
                         "incumbent": expected["windows"] if expected else None})
    check(Counter(r["role"] for r in rows) == FIT | {"assessment": 52}, "Incomplete declared source panel")
    held_rooms, held_speakers = ancestry["assessment"]
    check(all(not (rooms & held_rooms or speakers & held_speakers) for role, (rooms, speakers) in ancestry.items()
              if role != "assessment"), "A held room or source identity occurs in development")
    check(len({r["pcm_sha256"] for r in rows}) == len(rows), "Duplicate mixture bytes across roles")
    return rows, parents


def check_mix(mixed, first, second):
    import numpy as np

    pcm = [np.frombuffer(value, dtype="<i2").astype(np.int32) for value in (mixed, first, second)]
    check(np.array_equal(pcm[0], pcm[1] + pcm[2]), "Isolated stems do not exactly reconstruct the submitted mixture")


def assets(args):
    from scripts.audio_separation_v14 import CARD_SHA, WEIGHT_SHA
    from scripts.prepare_audio_specialists_v15 import ASSETS

    check({name: path_sha(args.voice / name) for name in ASSETS} == ASSETS, "Pinned voice checkpoint changed")
    pinned = {"pytorch_model.bin": WEIGHT_SHA, "README.md": CARD_SHA}
    check({name: path_sha(args.separator / name) for name in pinned} == pinned, "Pinned separator changed")
    return {"voice": ASSETS, "separator": pinned}


def freeze(args):
    rows, parents = sources()
    lock = {"schema": SCHEMA, "method": PROTOCOL, "inputs": parents, "assets": assets(args),
            "runner_sha256": path_sha(Path(__file__)),
            "separator_code_sha256": path_sha(Path(__file__).with_name("audio_separation_v14.py")),
            "voice_code_sha256": path_sha(ROOT / "backend/app/detectors/audio/upgraded.py"),
            "prior_point": read(DATA / ROLE_BANKS[-1][1] / "freeze.json")["points"]["current"],
            "runtime": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "transformers", "pyloudnorm")},
            "roster_sha256": digest(json.dumps(rows, sort_keys=True).encode())}
    return lock, rows


def quantize(data):
    import numpy as np

    peak = float(np.max(np.abs(data)))
    scale = min(1., .97 / max(peak, 1e-10))
    return np.rint(np.clip(data * scale, -1., 32767 / 32768) * 32768).astype("<i2").tobytes()


def ideal_mask(mixture, foreground, background):
    import numpy as np
    from scipy.signal import istft, stft

    params = {"fs": RATE, "nperseg": 512, "noverlap": 384}
    mix, clean, noise = [stft(v, **params)[2] for v in (mixture, foreground, background)]
    powers = np.abs(clean) ** 2
    ratio = powers / np.maximum(powers + np.abs(noise) ** 2, 1e-12)
    _, output = istft(mix * ratio, **params)
    check(len(output) == SAMPLES and np.isfinite(output).all(), "Ideal reconstruction lost the native clock")
    return output


def centered_correlation(first, second):
    import numpy as np

    first, second = first.astype(np.float64), second.astype(np.float64)
    first, second = first - first.mean(), second - second.mean()
    return float(np.dot(first, second) / max(1e-12, np.linalg.norm(first) * np.linalg.norm(second)))


def separate(mixture, separator):
    import numpy as np

    from scripts.audio_separation_v14 import separate as invoke

    left, right = [invoke(mixture[start:start + 64000], separator) for start in CUTS]
    correlations = [[centered_correlation(a[-OVERLAP:], b[:OVERLAP]) for b in right] for a in left]
    choice = max(((0, 1), (1, 0)), key=lambda pair: sum(abs(correlations[i][j]) for i, j in enumerate(pair)))
    agreement = [correlations[i][j] for i, j in enumerate(choice)]
    if min(map(abs, agreement)) < .25:
        return [], agreement
    fade = np.linspace(0., 1., OVERLAP, dtype=np.float32)
    result = []
    for i, j in enumerate(choice):
        a, b = left[i], right[j] * (1 if agreement[i] >= 0 else -1)
        estimates = np.r_[a[:OVERLAP], a[-OVERLAP:] * (1 - fade) + b[:OVERLAP] * fade, b[-OVERLAP:]]
        if np.linalg.norm(estimates) >= .08 * np.linalg.norm(mixture):
            result.append((str(i), estimates))
    return result, agreement


def acoustic_quality(original, estimate):
    from scripts.audio_separation_third_voice_pilot import signal_quality

    if not original.any() or not estimate.any():
        return None
    return signal_quality(estimate.astype("float64"), original.astype("float64"))


class FrameCapture:
    def __init__(self, model):
        self.model = model
        self.record = None
        model.backbone.register_forward_hook(self.capture, with_kwargs=True)

    def capture(self, _module, _args, kwargs, output):
        import torch

        mask = kwargs["attention_mask"]
        states = output.hidden_states
        weighted = torch.stack(states).mul(self.model.layer_weights.softmax(dim=0)[:, None, None, None]).sum(0)[0]
        scores = self.model.head(weighted).squeeze(-1)[mask[0].bool()]
        check(len(scores) <= 199 and torch.isfinite(scores).all().item(), "Unexpected frame response")
        self.record = scores.cpu().numpy()

    def forward(self, data):
        import torch

        from app.detectors.audio.upgraded import prepare_voice_window

        self.record = None
        values = prepare_voice_window(data)
        with torch.inference_mode():
            raw = float(self.model(torch.from_numpy(values)[None]).item())
        check(self.record is not None and abs(float(self.record.mean()) - raw) < .0001, "Native frame reduction differs")
        return self.record


def candidates(row, model, activity, separator, frame_capture, cached, calls):
    import numpy as np

    from app.detectors.audio.calibration import logit
    from app.detectors.audio.content import window_in_speech

    raw = tuple(source_data(DATA / row["bank"], row))
    mixture, fg, bg = [np.frombuffer(value, dtype="<i2").astype(np.float32) / 32768. for value in raw]
    blind, overlap = separate(mixture, separator)
    oracle = ideal_mask(mixture, fg, bg)
    quality = {"mixture": acoustic_quality(fg, mixture), "ideal_mask": acoustic_quality(fg, oracle),
               "blind": [acoustic_quality(fg, view) for _, view in blind]}
    views = {"mixture": [("0", raw[0])], "ideal_mask": [("0", quantize(oracle))],
             "exact_foreground": [("0", raw[1])], "blind": [(key, quantize(v)) for key, v in blind]}
    scores = {}
    for view, alternatives in views.items():
        scores[view] = {grid: [[] for _ in starts] for grid, (_, starts) in GRIDS.items()}
        for stream, waveform in alternatives:
            values = np.frombuffer(waveform, dtype="<i2").astype(np.float32) / 32768.
            spans = activity.spans(values)
            if not spans:
                for grid, (width, starts) in GRIDS.items():
                    for i, start in enumerate(starts):
                        pin = digest(waveform[2 * start:2 * (start + width)])
                        scores[view][grid][i].append({"stream": stream, "score": None, "input_sha256": pin})
                continue
            context_values = []
            for start in CUTS:
                data = values[start:start + 64000]
                pin = digest(waveform[2 * start:2 * (start + 64000)])
                if pin not in cached["context"]:
                    cached["context"][pin] = frame_capture.forward(data)
                    calls[f"{view}/context"] += 1
                context_values.append((start, cached["context"][pin]))
            for grid, (width, starts) in GRIDS.items():
                for i, start in enumerate(starts):
                    stop = start + width
                    pin = digest(waveform[2 * start:2 * stop])
                    response = None
                    if window_in_speech(spans, start / RATE, stop / RATE):
                        if grid == "dense":
                            measured = []
                            for offset, frame_scores in context_values:
                                if not (offset <= start and stop <= offset + 64000):
                                    continue
                                centers = offset + 280 + np.arange(len(frame_scores)) * 320
                                selected = frame_scores[(centers >= start) & (centers < stop)]
                                if len(selected) >= 40:
                                    measured.append(float(np.median(selected)))
                            response = min(measured) if measured else None
                        else:
                            if pin not in cached["stock"]:
                                probability, = model.predict([values[start:stop]])
                                cached["stock"][pin] = logit(probability)
                                calls[f"{view}/stock"] += 1
                            response = cached["stock"][pin]
                    scores[view][grid][i].append({"stream": stream, "score": response, "input_sha256": pin})
    if row["incumbent"]:
        for supplied, (new,) in zip(row["incumbent"], scores["mixture"]["stock"], strict=True):
            old, current = supplied["score"], new["score"]
            check(new["input_sha256"] == supplied["pcm_sha256"] and (
                old is None and current is None or old is not None and current is not None and abs(old - current) <= .0005
            ), "Stock arm failed native replay")
    public = {k: row[k] for k in ("role", "bank", "kind", "family", "truth", "pcm_sha256")}
    public.update(group=digest(f"{row['bank']}:{row['group']}".encode()), id=digest(row["pcm_sha256"].encode()),
                  measurements=scores, overlap_correlations=overlap, foreground_si_sdr_db=quality)
    return public


def best(row, view, grid):
    width, starts = GRIDS[grid]
    choices = [(item["score"], [start, start + width], item["stream"])
               for start, streams in zip(starts, row["measurements"][view][grid], strict=True)
               for item in streams if item["score"] is not None]
    return max(choices, key=lambda value: value[0], default=None)


def useful(span, truth):
    if truth is None:
        return False
    a, b = span
    x, y = truth
    shared = max(0, min(b, y) - max(a, x))
    return shared * 2 >= y - x and 10 * shared >= 3 * (b - a + y - x - shared)


def points(rows):
    result = {}
    for view in VIEWS:
        for grid in GRIDS:
            ordinary = [best(row, view, grid) for row in rows if row["truth"] is None]
            qualified = [value[0] for value in ordinary if value is not None]
            check(len(qualified) >= 8, "Too few assessed controls for a fixed point")
            result[f"{view}/{grid}"] = {"point": max(qualified), "genuine_assessed": len(qualified), "genuine_planned": len(ordinary)}
    return result


def result(rows, view, grid, point):
    width, starts = GRIDS[grid]
    stats, false, true, family = Counter(), set(), set(), Counter()
    for row in rows:
        positive = row["truth"] is not None
        scope = "synthetic" if positive else "genuine"
        stats[f"planned_{scope}"] += 1
        if positive:
            stats["geometrically_reachable"] += any(useful([start, start + width], row["truth"]) for start in starts)
            stats["admitted_reachable"] += any(useful([start, start + width], row["truth"]) and
                any(v["score"] is not None for v in readings) for start, readings in
                zip(starts, row["measurements"][view][grid], strict=True))
        found = best(row, view, grid)
        if found is None:
            stats[f"unassessed_{scope}"] += 1
            continue
        score, span, _ = found
        localized = useful(span, row["truth"])
        stats["useful_before_point"] += localized
        if score <= point:
            continue
        stats[f"flagged_{scope}"] += 1
        if localized:
            stats["localized"] += 1
            true.add(row["id"])
            family[row["family"]] += 1
        elif positive:
            stats["misplaced"] += 1
        else:
            false.add(row["group"])
            stats["isolated_room_flags"] += row["kind"] == "background_only"
    return dict(stats | {"false_groups": len(false)}) | {"localized_families": dict(family)}, true, false


def report(rows, selected, prior_point):
    fixed, equal = {}, {}
    original, true, false = result(rows, "mixture", "stock", prior_point)
    comparisons = {}
    for view in VIEWS:
        for grid in GRIDS:
            name = f"{view}/{grid}"
            fixed[name], found, ordinary = result(rows, view, grid, selected[name]["point"])
            comparisons[name] = {"gained": len(found - true), "lost": len(true - found),
                                 "new_false_groups": len(ordinary - false), "removed_false_groups": len(false - ordinary)}
            if view == "blind" and fixed[name].get("isolated_room_flags", 0) == 0:
                groups = {r["family"] for r in rows if r["id"] in found - true}
                comparisons[name]["qualifies_for_review"] = (len(groups) >= 2 and len(ordinary - false) <= 1 and not true - found)
            counts = {}
            for row in rows:
                item = best(row, view, grid)
                if item is not None and row["truth"] is None:
                    counts[row["group"]] = max(item[0], counts.get(row["group"], -math.inf))
            cuts = sorted(counts.values(), reverse=True)[:4]
            equal[name] = [result(rows, view, grid, cutoff)[0] for cutoff in cuts]
    return {"fixed": fixed, "relative_to_prior": comparisons, "prior_stock": original, "equal_false_groups": equal}


def scoring(args, rows, role):
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_separation_v14 import load as load_separator
    from scripts.audio_strong_traffic_view import network_free
    from scripts.prepare_audio_specialists_v15 import ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    network_free()
    model = load_voice({"weights": str(args.voice), "assets": ASSETS})
    activity = SpeechActivity(args.voice / "silero_vad.jit")
    separator = load_separator(args.separator)
    expected = [row for row in rows if (row["role"] == "assessment") == (role == "assessment")]
    capture = FrameCapture(model)
    results, cache, calls = [], {"stock": {}, "context": {}}, Counter()
    for row in expected:
        results.append(candidates(row, model, activity, separator, capture, cache, calls))
        if len(results) % 4 == 0:
            print(role, len(results), "/", len(expected), "forwards", sum(calls.values()), flush=True)
    return results, dict(calls)


def review(path):
    ledger = read(path)
    lock = ledger["freeze"]
    check(lock["schema"] == SCHEMA and lock["method"] == PROTOCOL, "Unknown frozen experiment")
    roles = {name: ledger[name] for name in ("development", "assessment")}
    check(Counter(r["role"] for r in roles["development"]) == FIT and len(roles["assessment"]) == 52 and
          all(r["role"] == "assessment" for r in roles["assessment"]), "Incomplete panel")
    check(len({r["pcm_sha256"] for part in roles.values() for r in part}) == 92, "Duplicate inputs")
    for row in [*roles["development"], *roles["assessment"]]:
        check(row["id"] == digest(row["pcm_sha256"].encode()) and set(row["measurements"]) == set(VIEWS), "Wrong case or score views")
        check((row["truth"] is not None) == (row["kind"] == "synthetic"), "Origin and interval differ")
        if row["truth"] is not None:
            check(0 <= row["truth"][0] < row["truth"][1] <= SAMPLES, "Wrong native interval")
        for grids in row["measurements"].values():
            check(grids.keys() == GRIDS.keys() and all(len(items) == len(GRIDS[k][1]) for k, items in grids.items()), "Missing planned windows")
            check(all(v["score"] is None or math.isfinite(v["score"]) for items in grids.values() for options in items for v in options), "Invalid response")
    selected = points(roles["development"])
    check(selected == ledger["points"] and report(roles["assessment"], selected, lock["prior_point"]) == ledger["summary"],
          "Frozen points or outcome counts do not replay")
    print(json.dumps(ledger["summary"], sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "development", "assessment", "review"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--voice", type=Path, default=ROOT / "models/weights/audio-upgrade")
    parser.add_argument("--separator", type=Path, default=ROOT / "models/research/separation-v14-convtasnet")
    args = parser.parse_args()
    if args.phase == "review":
        review(args.run)
        return
    lock, rows = freeze(args)
    frozen = args.run / "freeze.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700, exist_ok=True)
        write(frozen, lock)
        return
    check(read(frozen) == lock, "Frozen roster, implementation, or dependencies changed")
    if args.phase == "development":
        measurements, calls = scoring(args, rows, args.phase)
        write(args.run / "development.json", {"rows": measurements, "calls": calls, "freeze_sha256": path_sha(frozen)})
        write(args.run / "points.json", {"points": points(measurements), "development_sha256": path_sha(args.run / "development.json")})
        return
    development, selection = (read(args.run / f"{name}.json") for name in ("development", "points"))
    check(development["freeze_sha256"] == path_sha(frozen) and selection["development_sha256"] == path_sha(args.run / "development.json")
          and selection["points"] == points(development["rows"]), "Freeze points before held scoring")
    measured, calls = scoring(args, rows, args.phase)
    summary = report(measured, selection["points"], lock["prior_point"])
    write(args.run / "public.json", {"freeze": lock, "development": development["rows"], "assessment": measured,
                                    "points": selection["points"], "summary": summary, "forwards": development["calls"] | {"assessment": calls}})
    print(json.dumps(summary["relative_to_prior"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
