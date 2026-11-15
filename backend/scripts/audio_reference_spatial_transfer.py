# SPDX-License-Identifier: Apache-2.0
"""Test stereo interference reduction anchored to a matching earlier recording."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import audio_stereo_room_transfer as room

old, stereo = room.old, room.stereo
SCHEMA = "chipmunk.audio-reference-spatial-transfer/1"
RATE, SAMPLES = room.RATE, room.SAMPLES
PANELS = {
    "first": old.PANELS / "audio-stereo-room-transfer-2026-11-15.json.gz",
    "held": old.PANELS / "audio-stereo-room-held-2026-11-15.json.gz",
}
ARMS = ("downmix", "current", "projected", "fallback")
METHOD = {
    "selection": "Rebuild the 12 retained ICSI foreground groups in each earlier room comparison. Their four AMI "
                 "background recordings, native source clocks, positions, 48 kHz AAC, and original controls stay fixed.",
    "direction": "In each six-second half, require three one-second reference anchors with correlation at least .15 "
                 "in both channels and reference RMS at least 128 PCM units. Take the median gain per channel from "
                 "the three strongest anchors. Both directions must be positive and have cosine at least .985.",
    "projection": "Estimate two-channel background covariance from the quieter half of the 20 ms earlier-reference "
                  "frames after subtracting its fitted contribution. Add .05 times the average diagonal as loading. "
                  "Minimize the residual power with unity gain relative to the target's contribution in the channel "
                  "average. Require at least 15% estimated residual reduction, individual weights no greater than "
                  "2 in absolute value and an unclipped native 16 kHz result. Preserve source timing.",
    "decisions": "Use the existing clean-copy waveform check on the projected samples. Compare it alone and as "
                 "a fallback after current downmix-and-channel corroboration. Count genuine edits and the results "
                 "of matching background, unrelated and half-matching references separately.",
}


def earlier_direction(target, stereo_pcm):
    import numpy as np

    fits = []
    for half in range(2):
        records = []
        for offset in range(half * 6 * RATE, (half + 1) * 6 * RATE, RATE):
            x = target[offset:offset + RATE].astype(float)
            y = stereo_pcm[offset:offset + RATE].astype(float)
            x -= x.mean()
            y -= y.mean(axis=0)
            energy = float(x @ x)
            if energy < RATE * 128 ** 2:
                continue
            product = x @ y
            scale = np.sqrt(energy * np.sum(y ** 2, axis=0))
            scores = np.divide(product, scale, out=np.zeros(2), where=scale > 0)
            if min(scores) >= .15:
                records.append((float(min(scores)), offset, product / energy))
        records.sort(key=lambda row: (-row[0], row[1]))
        if len(records) < 3:
            return None
        fits.append(np.median([row[2] for row in records[:3]], axis=0))
    first, last = fits
    cosine = float(first @ last / (np.linalg.norm(first) * np.linalg.norm(last)))
    return np.mean(fits, axis=0) if cosine >= .985 else None


def projection(target, stereo_pcm):
    """A fixed two-channel mix that preserves the earlier voice's estimated gain."""
    import numpy as np

    if target.shape != (SAMPLES,) or stereo_pcm.shape != (SAMPLES, 2):
        raise ValueError("Expected a paired continuous twelve-second source")
    direction = earlier_direction(target, stereo_pcm)
    if direction is None:
        return "unconfirmed_direction", None, None
    x = target.astype(float).reshape(-1, 320)
    residual = (stereo_pcm.astype(float) - target[:, None] * direction[None, :]).reshape(-1, 320, 2)
    energy = np.mean(x ** 2, axis=1)
    reference_quiet = residual[energy <= np.median(energy)].reshape(-1, 2)
    reference_quiet -= reference_quiet.mean(axis=0)
    cov = reference_quiet.T @ reference_quiet / len(reference_quiet)
    load = float(np.trace(cov) * .025)
    if load <= 0:
        return "no_room_support", None, None
    solution = np.linalg.solve(cov + load * np.eye(2), direction)
    w = solution * (direction.mean() / (solution @ direction))
    base = np.full(2, .5)
    baseline = float(base @ cov @ base)
    if baseline <= 0:
        return "no_room_support", None, None
    reduction = float(1 - (w @ cov @ w) / baseline)
    proof = {"projection_weights": [round(float(x), 8) for x in w],
             "target_balance": [round(float(x), 8) for x in direction],
             "estimated_residual_reduction": round(reduction, 8)}
    if np.max(np.abs(w)) > 2. or reduction < .15:
        return "insufficient_spatial_gain", None, proof
    candidate = np.rint(stereo_pcm @ w)
    if np.max(candidate) > 32767 or np.min(candidate) < -32768:
        return "projected_clipping", None, proof
    return "projected", candidate.astype("<i2"), proof


def load(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def input_panels():
    panels = {name: load(path) for name, path in PANELS.items()}
    for receipt in panels.values():
        room.review(receipt)
    return panels


def native_cases(args, panels):
    import numpy as np

    sources = room.dry_icsi_inputs(args.icsi)
    for name, receipt in panels.items():
        tracks = room.read_rooms(args.prior_rooms if name == "first" else args.held_rooms, holdout=name == "held")
        actual = {(r["group"], r["kind"]): r for r in receipt["rows"] if r["bank"] == "icsi"}
        with TemporaryDirectory(prefix="chipmunk-spatial-audio-") as output:
            carrier = Path(output) / "capture.m4a"
            for source in receipt["plan"]["rows"]:
                if source["bank"] != "icsi":
                    continue
                start = source["start"]
                original_room = np.column_stack([channel[start:start + SAMPLES] for channel in tracks[source["meeting"]]])
                old.require(old.sha(original_room.astype("<i2").tobytes()) == source["room_pcm_sha256"],
                            "Room excerpt differs from the committed first measurement")
                decodes = {}
                for kind, (ref, mix) in room.materialize(source, tracks, sources).items():
                    original = actual[source["group"], kind]
                    mix_hash, ref_hash = old.sha(mix), old.sha(ref)
                    old.require(mix_hash == original["mixture_sha256"] and ref_hash == original["reference_sha256"],
                                "A paired PCM input changed")
                    if mix_hash not in decodes:
                        views, clock = stereo.stereo_pcm(carrier, mix, original["clock"]["encoded_sha256"])
                        old.require(clock == original["clock"] and all(old.sha(data.tobytes()) ==
                                    original["measurement"]["readings"][side]["pcm_sha256"] for side, data in views.items()),
                                    "An independently decoded native channel differs")
                        decodes[mix_hash] = views
                    yield name, original, np.frombuffer(ref, dtype="<i2"), decodes[mix_hash]


def capture(args):
    import numpy as np

    from app.evidence import audio_clean_speech_long as speech

    result = defaultdict(list)
    for name, original, earlier, views in native_cases(args, input_panels()):
        state, reduced, proof = projection(earlier, np.column_stack((views["left"], views["right"])))
        reason, measure, _ = ((state, None, None) if reduced is None else speech.locate(earlier, reduced))
        rows = result[name]
        rows.append({"group": original["group"], "kind": original["kind"], "state": state, "proof": proof,
                     "reason": reason, "span": measure["samples"] if measure else None})
        if len(rows) % 11 == 0:
            print(f"{name}: verified {len(rows)} comparisons on the native AAC source clock.", flush=True)
    record = {"schema": SCHEMA, "method": METHOD,
              "prior": {name: old.sha(path.read_bytes()) for name, path in PANELS.items()}, "samples": result}
    return record | {"results": evaluate(record)}


def evaluate(record):
    import math

    old.require(record["schema"] == SCHEMA and record["method"] == METHOD
                and record["prior"] == {name: old.sha(path.read_bytes()) for name, path in PANELS.items()}
                and set(record["samples"]) == set(PANELS), "The frozen spatial comparison differs")
    output = {}
    for name, panel in input_panels().items():
        prior = {(r["group"], r["kind"]): r for r in panel["rows"] if r["bank"] == "icsi"}
        rows = record["samples"][name]
        old.require(len(rows) == len(prior) and {(r["group"], r["kind"]) for r in rows} == set(prior),
                    "An ICSI pairing is missing or duplicated")
        counts = defaultdict(Counter)
        for row in rows:
            original = prior[row["group"], row["kind"]]
            truth, measured = original["truth"], original["measurement"]
            downmix = measured["readings"]["downmix"]["range"]
            current = downmix if room.downmix_with_channel_support(measured) else None
            projected = row["span"]
            old.require(row["state"] in {"unconfirmed_direction", "no_room_support", "insufficient_spatial_gain",
                                         "projected_clipping", "projected"}
                        and (row["reason"] in stereo.REASONS if row["state"] == "projected" else row["reason"] == row["state"])
                        and (projected is not None) == (row["reason"] == "local_change")
                        and (projected is None or (0 <= projected[0] < projected[1] <= SAMPLES
                                                  and projected[1] - projected[0] == stereo.RANGE_SAMPLES)),
                        "A projected range is outside the original source clock")
            proof = row["proof"]
            if row["state"] in {"projected", "projected_clipping", "insufficient_spatial_gain"}:
                weights, balance = proof["projection_weights"], proof["target_balance"]
                old.require(len(weights) == len(balance) == 2 and all(math.isfinite(n) for n in (*weights, *balance))
                            and min(balance) > 0 and abs(sum(x * y for x, y in zip(weights, balance, strict=True))
                                                        - sum(balance) / 2) <= .000001,
                            "The projection would distort the earlier voice's estimated contribution")
            else:
                old.require(proof is None, "An unsupported comparison has an unexpected projection")
            category = room.category(row["kind"])
            for arm, selected in zip(ARMS, (downmix, current, projected, current or projected), strict=True):
                useful = old.location(selected, truth)
                counts[f"{arm}/{category}"].update(
                    files=1, flags=bool(selected), useful=useful, gain=useful and not old.location(current, truth),
                    wrong_on_edit=bool(truth and selected and not useful),
                )
            counts[f"coverage/{category}"].update({row["state"]: 1})
        output[name] = counts
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    capture_mode = modes.add_parser("capture")
    for field in ("icsi", "prior_rooms", "held_rooms", "output"):
        capture_mode.add_argument(field, type=Path)
    portable = modes.add_parser("review")
    portable.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.mode == "capture":
        data = capture(args)
        content = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        with args.output.open("xb") as target:
            target.write(gzip.compress(content, mtime=0))
    else:
        data = load(args.receipt)
        results = evaluate(data)
        old.require(results == data["results"], "The published counts do not reproduce")
        print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
