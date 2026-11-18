# SPDX-License-Identifier: Apache-2.0
"""Test a transient speech delay against matched native-frame observations."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile

SCHEMA = "chipmunk.video-signed-lip-rescue/1"
RATE = 16_000
PICTURE_PERIODS = (Fraction(1, 28), Fraction(1, 22))
SIZES = (12, 16, 20, 24)
FLANK = 12
MIN_FLANK_COUNT = 5
MIN_ORDINARY_POINT = .060
CALIBRATION_PAD = .012
SPEAKERS = {"calibration": 6, "held": 12}
KINDS = ("original", "gain", "global_early", "global_late", "local_early", "local_late")
POINT_RULE = "At least 0.060 and the maximum on speaker-disjoint calibration controls plus 0.012"
RULE = {"minimum_useful": 4, "minimum_speakers": 3, "maximum_misplaced": 2,
        "maximum_other_false_flags": 2, "maximum_global_flags": 0}


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def save(path, value):
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as file:
        file.write(gzip.compress(payload, mtime=0) if path.suffix == ".gz" else payload)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def key(value):
    return digest(f"{SCHEMA}/{value}".encode())


def known_sources(panels):
    seen_ids, previous = set(), {}
    for path in sorted(panels.glob("video-*-sources-*.json.gz")):
        if path.name.startswith("video-lip-signed-rescue-"):
            continue
        doc = read(path)
        rows = doc.get("rows", []) if isinstance(doc, dict) else []
        identifiers = {asset["id"] for row in rows for role in ("target", "same_words", "different_words")
                       if isinstance(asset := row.get(role), dict) and isinstance(asset.get("id"), str)}
        if identifiers:
            previous[path.name] = sha(path)
            seen_ids.update(identifiers)
    training = panels / "video-mouth-patch-sources-2026-11-17.json.gz"
    trained = {row["target"]["id"].split("_")[0] for row in read(training)["rows"] if row["split"] in {"fit", "calibration"}}
    return seen_ids, trained, previous


def reserve(sources, panels, models, geometry, output):
    from app.evidence.video_lip_agreement import WEIGHT_PINS
    from app.evidence.video_reference_timing import MODEL_SHA256
    from scripts.lombard_mouth_source_intake import index_zip, publisher
    from scripts.video_neural_phone_clock import pcm

    archive = publisher(sources)
    pins = {name: sha(models / f"{name}.safetensors") for name in ("vision", "shared")}
    if any(pins[name] != WEIGHT_PINS[name] for name in pins) or sha(geometry) != MODEL_SHA256:
        raise ValueError("The existing verified mouth and sound assets are required")
    seen, trained, previous = known_sources(panels)
    pool = defaultdict(list)
    with (ZipFile(sources / "publisher-json.zip") as meta, ZipFile(sources / "alignment.zip") as labels,
          ZipFile(sources / "lombardgrid_front.zip") as video, ZipFile(sources / "lombardgrid_audio.zip") as audio):
        eyes, ears, align = index_zip(video, ".mov"), index_zip(audio, ".wav"), index_zip(labels, ".json")
        for name in index_zip(meta, ".json").values():
            for row in json.loads(meta.read(name)):
                actor, condition = row["SPKR"], row["COND"]
                uid = f"{actor}_{condition}_{row['UTTERANCE']}"
                if (row["STATUS"] == "CORRECT" and condition == "p" and uid not in seen and actor not in trained
                        and all(uid in collection for collection in (eyes, ears, align))):
                    pool[actor].append(uid)
        cases = []
        for actor in sorted(pool, key=key):
            for uid in sorted(pool[actor], key=key):
                acoustic = audio.read(ears[uid])
                nsamples = len(pcm(acoustic))
                if not 40_000 <= nsamples <= 58_000:
                    continue
                utterance = json.loads(labels.read(align[uid]))[uid]
                speech = [entry for entry in utterance if not entry["phone"].startswith("SIL")]
                speech_start = round(Fraction(str(speech[0]["offset"])) * RATE)
                speech_stop = round((Fraction(str(speech[-1]["offset"])) + Fraction(str(speech[-1]["duration"]))) * RATE)
                i = len(cases)
                width = (10_240, 14_080)[i % 2]
                shift = (3_200, 4_480)[(i // 2) % 2]
                center = (speech_start + speech_stop) // 2 + (-640 if i % 3 == 1 else 0)
                interval = [center - width // 2, center + width // 2]
                if not (speech_start + 5760 <= interval[0] and interval[1] <= speech_stop - 5760):
                    continue
                picture = video.read(eyes[uid])
                cases.append({"speaker": key(actor), "role": "calibration" if i < SPEAKERS["calibration"] else "held",
                              "audio_path": ears[uid].filename, "video_path": eyes[uid].filename,
                              "audio_sha256": digest(acoustic), "video_sha256": digest(picture), "samples": nsamples,
                              "speech_extent": [speech_start, speech_stop], "interval": interval, "shift_samples": shift})
                break
            if len(cases) == sum(SPEAKERS.values()):
                break
    if len(cases) != sum(SPEAKERS.values()):
        raise ValueError(f"Only {len(cases)} eligible novel recordings met the declared speech extent")
    save(output, {"schema": SCHEMA, "source": archive, "known_sources": previous, "model_fit_selection_speakers_excluded": len(trained),
                  "geometry_sha256": sha(geometry), "weights": pins, "script_sha256": sha(Path(__file__)), "speakers": SPEAKERS,
                  "point_rule": POINT_RULE, "acceptance": RULE, "rows": cases, "minimum_iou": .40,
                  "maximum_signed_error_samples": 1280, "carrier": "upright near-24 fps H.264 and 48 kHz mono AAC"})
    print("Frozen", len(cases), "source-disjoint recordings", flush=True)


def signed_candidate(times, evidence, samples):
    import numpy as np

    from app.detectors.video.av_local_agreement import LAGS

    lags = np.asarray(LAGS)
    outside, inside = np.nonzero((np.abs(lags[:, None]) <= 2240)
                                 & (np.abs(lags[:, None] - lags[None, :]) >= 2560)
                                 & (np.abs(lags[:, None] - lags[None, :]) <= 5440))
    outside_lag, inside_lag = lags[outside], lags[inside]
    old, new = evidence[:, outside], evidence[:, inside]
    common = np.isfinite(old) & np.isfinite(new)
    if common.sum(axis=0).max() < 2 * MIN_FLANK_COUNT + 6:
        return "coverage", None
    counts = np.vstack((np.zeros(len(outside), dtype=int), np.cumsum(common, axis=0)))
    differences = np.vstack((np.zeros(len(outside)), np.cumsum(np.where(common, new - old, 0), axis=0)))

    def window(a, b, required):
        count = counts[b] - counts[a]
        return np.where(count >= required, (differences[b] - differences[a]) / count.clip(1), np.nan)

    best = None
    for width in SIZES:
        for first in range(8, len(times) - width - 8):
            end = first + width
            start_sample, stop_sample = times[first] + inside_lag, times[end] + inside_lag
            a, b = max(0, first - FLANK), min(len(times), end + FLANK)
            left = -window(a, first, MIN_FLANK_COUNT)
            center = window(first, end, (width + 1) // 2)
            right = -window(end, b, MIN_FLANK_COUNT)
            score = np.minimum(np.minimum(left, center), right)
            score = np.where((start_sample >= 0) & (stop_sample <= samples) & np.isfinite(score), score, -np.inf)
            i = int(score.argmax())
            value = float(score[i])
            if value <= 0 or (best is not None and value <= best[0]):
                continue
            witness = (a + np.flatnonzero(common[a:first, i])[-1],
                       first + np.nanargmax(np.where(common[first:end, i], (new - old)[first:end, i], np.nan)),
                       end + np.flatnonzero(common[end:b, i])[0])
            best = value, {"score": round(value, 7), "span": [int(start_sample[i]), int(stop_sample[i])],
                           "picture_span": [first, end], "witness_frames": list(map(int, witness)),
                           "lag_samples": [int(outside_lag[i]), int(inside_lag[i])],
                           "support": [int(counts[y, i] - counts[x, i]) for x, y in ((a, first), (first, end), (end, b))],
                           "contrasts": [round(float(x[i]), 7) for x in (left, center, right)]}
    return ("measured", best[1]) if best else ("no_signed_return", None)


def edit(pcm, interval, shift, role):
    import numpy as np

    out = pcm.copy()
    if role == "original":
        return out
    a, b = interval
    if role == "gain":
        sample = pcm[a:b].astype(np.float64) * .64
    elif role.startswith("global"):
        return np.concatenate((pcm[shift:], np.zeros(shift, dtype=pcm.dtype))) if role == "global_early" else np.concatenate((np.zeros(shift, dtype=pcm.dtype), pcm[:-shift]))
    else:
        change = -shift if role == "local_early" else shift
        sample = pcm[a-change:b-change].astype(np.float64)
    ramp = np.ones(b - a)
    edge = np.linspace(0., 1., 128)
    ramp[:128], ramp[-128:] = edge, edge[::-1]
    out[a:b] = np.rint(pcm[a:b] * (1. - ramp) + sample * ramp).astype("<i2")
    assert out[:a].tobytes() == pcm[:a].tobytes() and out[b:].tobytes() == pcm[b:].tobytes() and out[a:b].tobytes() != pcm[a:b].tobytes()
    return out


def stage(plan_path, sources, models, assets, output, *, phase, calibration=None):
    import subprocess
    import sys
    from collections import Counter

    import cv2
    import numpy as np
    import torch

    from app.detectors.audio.decode import decode_audio
    from app.detectors.video import av_local_agreement as agreement
    from app.detectors.video import viseme_review, viseme_worker
    from app.detectors.video.decode import probe_frames
    from app.evidence import video_lip_agreement
    from app.evidence import video_viseme_binding as binding
    from scripts.export_video_lip_correspondence import make_mux, visual_hash
    from scripts.video_lip_timing_aac48 import encode
    from scripts.video_neural_phone_clock import pcm

    plan = read(plan_path)
    if (plan["schema"] != SCHEMA or sha(Path(__file__)) != plan["script_sha256"]
            or plan["point_rule"] != POINT_RULE or plan["acceptance"] != RULE):
        raise ValueError("The before-score plan or comparison code changed")
    if phase == "held":
        frozen = read(calibration)
        if frozen["plan_sha256"] != sha(plan_path) or frozen["phase"] != "calibration" or tally_calibration(frozen["rows"]) != frozen["point"]:
            raise ValueError("The held comparison requires the recorded calibration-only point")
    else:
        frozen = None
    geometry = assets / "face_landmarker.task"
    if sha(geometry) != plan["geometry_sha256"] or any(sha(models / f"{name}.safetensors") != pin for name, pin in plan["weights"].items()):
        raise ValueError("Model files no longer match the before-score plan")
    sys.path.insert(0, str((assets / "runtime").resolve()))
    cv2.setNumThreads(1)
    audio_events = viseme_worker.aperture_reader(assets / "checkpoint", articulation=True)[0] if phase == "held" else None
    torch.set_num_threads(1)
    if phase == "calibration":
        torch.set_num_interop_threads(1)
    eyes, ears = viseme_worker._motion_models(models)
    output.mkdir(mode=0o700)
    media = output / "media"
    media.mkdir(mode=0o700)
    rows = []
    with (ZipFile(sources / "lombardgrid_front.zip") as front, ZipFile(sources / "lombardgrid_audio.zip") as audio,
          (output / "journal.jsonl").open("x") as journal):
        for position, example in enumerate((row for row in plan["rows"] if row["role"] == phase), 1):
            blob, voice = front.read(example["video_path"]), audio.read(example["audio_path"])
            if digest(blob) != example["video_sha256"] or digest(voice) != example["audio_sha256"]:
                raise ValueError("A source archive member differs from its original commitment")
            basefile = output / "source.mov"
            basefile.write_bytes(blob)
            native = probe_frames(basefile, cap=121, timeout_s=20, max_pixels=1280 * 720)
            source = {"width": native.width, "height": native.height}
            periods = {(b - a) * native.tb for a, b in zip(native.pts, native.pts[1:])}
            if (native.pts[0] != 0 or not 34 <= len(native) <= 120 or native.truncated or native.interpolated or native.rotation
                    or len(periods) != 1 or not PICTURE_PERIODS[0] <= next(iter(periods)) <= PICTURE_PERIODS[1]):
                raise ValueError("A frozen capture has an unsupported original picture clock")
            waveform = pcm(voice)
            assert len(waveform) == example["samples"]
            packet_hash = visual_hash(basefile)
            frames, patches, valid = viseme_worker.capture_mouth_frames(basefile, native, geometry, motion=True, articulation=True)
            times = np.asarray([round(t * native.tb * RATE) for t in native.pts])
            picture = agreement.picture_windows(eyes, patches, valid)
            for role in (KINDS if phase == "held" else KINDS[:4]):
                formed = edit(waveform, example["interval"], example["shift_samples"], role)
                m4a = output / "encoded.m4a"
                encode(formed, m4a, 48_000)
                file = media / f"{position:02d}-{role}.mp4"
                make_mux(basefile, m4a, file, native, packet_hash)
                file_pin = sha(file)
                decoded = decode_audio(file, source_sha256=file_pin)
                mapping = decoded.mapping
                if (binding.presented_samples(mapping) != len(formed)
                        or not viseme_review.supported(native, mapping, [[0, len(native)]])
                        or video_lip_agreement.carrier(mapping, origin=binding.audio_origin(mapping), stereo_scope=None) != "aac48"):
                    raise ValueError("The fully muxed upload changed its admitted frame or soundtrack clock")
                values = np.rint(decoded.pcm[:len(formed)] * 32768.).astype("<i2")
                expected, actual = formed.astype(float), values.astype(float)
                best_lag = max((float(np.sum(expected[32:-32] * actual[32 + lag:len(values) - 32 + lag])), lag) for lag in range(-12, 13))[1]
                if abs(best_lag) > 1:
                    raise ValueError("AAC decoding moved the timeline by more than one analysis sample")
                readings = agreement.similarities(ears, picture, agreement.sound(values), times)
                candidate_status, candidate = signed_candidate(times, readings, len(values))
                witness = None
                if candidate is not None:
                    indices = candidate["witness_frames"]
                    selected = "+".join(f"eq(n,{i})" for i in indices)
                    output_hashes = subprocess.run([
                        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1", "-filter_threads", "1", "-i", str(file),
                        "-map", "0:v:0", "-an", "-vf", f"select='{selected}',format=rgb24", "-fps_mode", "passthrough", "-c:v", "rawvideo",
                        "-threads:v", "1", "-f", "framehash", "-hash", "sha256", "pipe:1",
                    ], check=True, timeout=20, capture_output=True, text=True).stdout
                    rgb_digests = [line.rsplit(",", 1)[1].strip() for line in output_hashes.splitlines() if line and not line.startswith("#")]
                    witness = [{k: frames[i][k] for k in ("index", "pts", "rgb_sha256", "mouth_box", "mouth_crop_sha256")} for i in indices]
                    if rgb_digests != [row["rgb_sha256"] for row in witness]:
                        raise ValueError("The exact supporting pictures did not survive muxing")
                incumbent = {"state": "not_run", "range": None, "change_samples": None}
                if audio_events:
                    _, phonetic, articulation = audio_events(values, include_events=True)
                    _, dip = agreement.localize(times, readings, len(values))
                    raw = {"frames": frames, "phones": phonetic, "articulation_phones": articulation,
                           "sound_levels": viseme_review.sound_levels(decoded.pcm, len(values))}
                    if dip is not None:
                        raw[video_lip_agreement.KEY] = video_lip_agreement.encode(dip)
                    previous = binding.measured(raw, timestamps=native.pts, time_base=str(native.tb), source=source, mapping=mapping)
                    found = previous["witness"]
                    incumbent = {"state": previous["status"], "range": found["audio_source_pts"] if found else None,
                                 "change_samples": found["change_samples"] if found else None,
                                 "basis": previous.get("measurement_basis")}
                truth = example["interval"] if role.startswith("local") else None
                shift = ((-1 if role.endswith("early") else 1) * example["shift_samples"]) if truth else None
                row = {"speaker": example["speaker"], "input": role, "file": file.name, "file_sha256": file_pin,
                       "source_video_sha256": example["video_sha256"], "submitted_video_track_sha256": packet_hash,
                       "native_presentation": {"pts": list(native.pts), "time_base": str(native.tb)},
                       "presented_samples": len(values), "pcm_sha256": digest(values.tobytes()), "audio_alignment_samples": best_lag,
                       "truth": truth, "shift_samples": shift, "candidate_status": candidate_status, "candidate": candidate,
                       "native_witness": witness, "incumbent": incumbent, "mouth_frames": int(valid.sum())}
                journal.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                journal.flush()
                rows.append(row)
            print(f"{phase}: {position} of {SPEAKERS[phase]} originals verified and measured", flush=True)
    point = frozen["point"] if frozen else tally_calibration(rows)
    receipt = {"schema": SCHEMA, "phase": phase, "plan_sha256": sha(plan_path), "rows": rows, "point": point,
               "statuses": dict(Counter(row["candidate_status"] for row in rows)),
               "outcomes": evaluate(rows, point) if frozen else None}
    if frozen:
        receipt["calibration_sha256"] = sha(calibration)
    save(output / "receipt.json.gz", receipt)
    print(receipt["outcomes"] if frozen else f"Frozen candidate point {point:.7f}", flush=True)


def tally_calibration(rows):
    maximum = max((row["candidate"]["score"] for row in rows if row["candidate"]), default=0.)
    return round(max(MIN_ORDINARY_POINT, maximum + CALIBRATION_PAD), 7)


def match(span, truth):
    if not span or not truth:
        return False
    a, b = span
    x, y = truth
    common = max(0, min(b, y) - max(a, x))
    return 5 * common >= 2 * (max(b, y) - min(a, x))


def evaluate(rows, point):
    from collections import Counter

    tally = {arm: Counter(dict.fromkeys(("edit", "ordinary", "uniform", "located", "signed_location", "misplaced",
                                        "ordinary_flags", "uniform_flags", "assessed"), 0))
             for arm in ("incumbent", "candidate", "combined")}
    useful_groups, increased_ordinary = set(), set()
    for row in rows:
        truth, shift = row["truth"], row["shift_samples"]
        prev, proposal = row["incumbent"], row["candidate"]
        new_range = proposal["span"] if proposal and proposal["score"] > point else None
        new_shift = proposal["lag_samples"][1] - proposal["lag_samples"][0] if proposal else None
        old_range, old_shift = prev["range"], prev["change_samples"]
        scope = "edit" if truth else "uniform" if row["input"].startswith("global") else "ordinary"
        new_available = row["candidate_status"] != "coverage"
        old_available = prev["state"] in ("matched", "no_match")
        for arm, span, change, available in (("incumbent", old_range, old_shift, old_available),
                                             ("candidate", new_range, new_shift, new_available),
                                             ("combined", old_range or new_range, old_shift if old_range else new_shift,
                                              old_available or new_available)):
            values = tally[arm]
            values[scope] += 1
            values["assessed"] += available
            if span is None:
                continue
            if truth:
                located = match(span, truth)
                signed = located and change is not None and change * shift > 0 and abs(change - shift) <= 1280
                values["located"] += located
                values["signed_location"] += signed
                values["misplaced"] += not located
                if arm == "combined" and old_range is None and signed:
                    useful_groups.add(row["speaker"])
            else:
                values[f"{scope}_flags"] += 1
                if arm == "combined" and old_range is None:
                    increased_ordinary.add(row["speaker"])
    base, fused = tally["incumbent"], tally["combined"]
    additions = fused["signed_location"] - base["signed_location"]
    return tally | {"new_signed_location_speakers": len(useful_groups), "new_flag_speakers": len(increased_ordinary),
                    "accepted_offline": (additions >= RULE["minimum_useful"] and len(useful_groups) >= RULE["minimum_speakers"]
                        and fused["ordinary_flags"] - base["ordinary_flags"] <= RULE["maximum_other_false_flags"]
                        and fused["uniform_flags"] - base["uniform_flags"] <= RULE["maximum_global_flags"]
                        and fused["misplaced"] - base["misplaced"] <= RULE["maximum_misplaced"])}


def review(plan, calibration, held):
    design, dev, test = (read(path) for path in (plan, calibration, held))
    if (design["schema"] != SCHEMA or test["calibration_sha256"] != sha(calibration)
            or any(report["plan_sha256"] != sha(plan) for report in (dev, test))
            or tally_calibration(dev["rows"]) != dev["point"] or dev["point"] != test["point"]):
        raise ValueError("The held results are not linked to their frozen calibration and design")
    cases = {(row["role"], row["speaker"]): row for row in design["rows"]}
    hashes = set()
    for phase, receipt, kinds in (("calibration", dev, KINDS[:4]), ("held", test, KINDS)):
        expected = {(speaker, kind) for role, speaker in cases if role == phase for kind in kinds}
        if {(row["speaker"], row["input"]) for row in receipt["rows"]} != expected or len(receipt["rows"]) != len(expected):
            raise ValueError("Some frozen inputs were omitted or counted twice")
        for row in receipt["rows"]:
            upstream = cases[phase, row["speaker"]]
            if (row["file_sha256"] in hashes or upstream["video_sha256"] != row["source_video_sha256"]
                    or upstream["samples"] != row["presented_samples"] or abs(row["audio_alignment_samples"]) > 1):
                raise ValueError("A delivery differs from its pinned source inputs")
            hashes.add(row["file_sha256"])
            native = row["native_presentation"]
            base, pts = Fraction(native["time_base"]), native["pts"]
            periods = {(b - a) * base for a, b in zip(pts, pts[1:])}
            if pts[0] != 0 or len(periods) != 1 or not PICTURE_PERIODS[0] <= next(iter(periods)) <= PICTURE_PERIODS[1]:
                raise ValueError("A verified picture clock changed")
            if candidate := row["candidate"]:
                times = [round(pts[i] * base * RATE) + candidate["lag_samples"][1] for i in candidate["picture_span"]]
                if (times != candidate["span"] or candidate["score"] != min(candidate["contrasts"])
                        or [r["index"] for r in row["native_witness"]] != candidate["witness_frames"]
                        or any(w["pts"] != pts[w["index"]] for w in row["native_witness"])):
                    raise ValueError("A proposed location differs from its native source clocks")
    result = evaluate(test["rows"], test["point"])
    if result != test["outcomes"]:
        raise ValueError("The held outcomes fail independent replay")
    print(json.dumps(result, sort_keys=True))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="mode", required=True)
    for cmd, fields in (("reserve", ("sources", "panels", "models", "geometry", "output")),
                        ("stage", ("plan_path", "sources", "models", "assets", "output")),
                        ("review", ("plan", "calibration", "held"))):
        op = commands.add_parser(cmd)
        for field in fields:
            op.add_argument(field, type=Path)
        if cmd == "stage":
            op.add_argument("--phase", choices=SPEAKERS, required=True)
            op.add_argument("--calibration", type=Path)
    args = vars(cli.parse_args())
    {"reserve": reserve, "stage": stage, "review": review}[args.pop("mode")](**args)


if __name__ == "__main__":
    main()
