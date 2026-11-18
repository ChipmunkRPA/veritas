# SPDX-License-Identifier: Apache-2.0
"""Select a within-shot signed timing check and test it on new exact sources."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile

from scripts import video_lip_signed_rescue as prior

SCHEMA = "chipmunk.video-signed-lip-contrast/1"
RATE = prior.RATE
VARIANTS = ("earlier_raw", "outside_anchor", "normalized", "guarded_normalized")
CONTROLS = ("drift_early", "drift_late")
ROLES = (*prior.KINDS, *CONTROLS)
HELD_GROUPS = 10
RULE = {"new_signed": 4, "new_speakers": 3, "new_ordinary": 2, "new_global": 0, "new_misplaced": 2}
SCORE_FLOOR = 0.06


def reserve(sources, panels, old_plan, output):
    from scripts.lombard_mouth_source_intake import index_zip, publisher
    from scripts.video_neural_phone_clock import pcm

    source_pins = publisher(sources)
    old = prior.read(old_plan)
    prior_ids, trained, history = prior.known_sources(panels)
    seen_recordings = prior_ids | {Path(row["audio_path"]).stem for row in old["rows"]}
    seen_speakers = {row["speaker"] for row in old["rows"]}
    pool = defaultdict(list)
    rows = []
    with (
        ZipFile(sources / "publisher-json.zip") as metadata,
        ZipFile(sources / "alignment.zip") as alignment,
        ZipFile(sources / "lombardgrid_audio.zip") as audio,
        ZipFile(sources / "lombardgrid_front.zip") as video,
    ):
        labels = index_zip(alignment, ".json")
        sounds, pictures = index_zip(audio, ".wav"), index_zip(video, ".mov")
        for path in index_zip(metadata, ".json").values():
            for entry in json.loads(metadata.read(path)):
                speaker, condition = entry["SPKR"], entry["COND"]
                uid = f"{speaker}_{condition}_{entry['UTTERANCE']}"
                if (
                    speaker not in trained
                    and prior.key(speaker) not in seen_speakers
                    and uid not in seen_recordings
                    and condition in {"p", "l"}
                    and entry["STATUS"] == "CORRECT"
                    and all(uid in archive for archive in (labels, sounds, pictures))
                ):
                    pool[speaker].append(uid)
        for speaker in sorted(pool, key=prior.key):
            condition_preference = "p" if len(rows) % 2 == 0 else "l"
            for uid in sorted(
                pool[speaker],
                key=lambda value: (value.split("_")[1] != condition_preference, prior.key(value)),
            ):
                sound = audio.read(sounds[uid])
                samples = len(pcm(sound))
                if not 40_000 <= samples <= 58_000:
                    continue
                phones = [
                    p
                    for p in json.loads(alignment.read(labels[uid]))[uid]
                    if not p["phone"].startswith("SIL")
                ]
                a = round(Fraction(str(phones[0]["offset"])) * RATE)
                b = round(
                    (Fraction(str(phones[-1]["offset"])) + Fraction(str(phones[-1]["duration"]))) * RATE
                )
                n = len(rows)
                half = (4800, 6720)[n % 2]
                shift = (3200, 4480)[(n // 2) % 2]
                center = (a + b) // 2 + (320 if n % 3 else -480)
                interval = [center - half, center + half]
                if not (a + 5760 <= interval[0] and interval[1] <= b - 5760):
                    continue
                picture = video.read(pictures[uid])
                rows.append(
                    {
                        "speaker": prior.key(speaker),
                        "role": "held",
                        "condition": uid.split("_")[1],
                        "audio_path": sounds[uid].filename,
                        "video_path": pictures[uid].filename,
                        "audio_sha256": prior.digest(sound),
                        "video_sha256": prior.digest(picture),
                        "samples": samples,
                        "speech_extent": [a, b],
                        "interval": interval,
                        "shift_samples": shift,
                    }
                )
                break
            if len(rows) == HELD_GROUPS:
                break
    if len(rows) != HELD_GROUPS:
        raise ValueError(f"Only {len(rows)} eligible recordings from new speakers")
    value = {
        "schema": SCHEMA,
        "prior_plan_sha256": prior.sha(old_plan),
        "source": source_pins,
        "prior_rosters": history,
        "weights": old["weights"],
        "geometry_sha256": old["geometry_sha256"],
        "script_sha256": prior.sha(Path(__file__)),
        "entries": rows,
        "roles": list(ROLES),
        "rule": RULE,
        "variants": list(VARIANTS),
        "standardized_floor": 0.20,
        "guarded_peak_gain": 0.075,
        "dev_controls": "all previous 96 frozen deliveries plus whole-track drift in both directions for every recording",
        "select": "Maximize additional correct signed locations, then new speakers, with held-style additional-control limits. Break ties with fewer added flags and fixed variant order.",
        "drift": "18ms at first audio sample to 160ms at end; continuous linear interpolation across the entire clip",
    }
    prior.save(output, value)
    print(f"Reserved {len(rows)} archived recordings from different speakers before development", flush=True)


def variant_readings(times, evidence, samples):
    import numpy as np

    from app.detectors.video.av_local_agreement import LAGS

    lags = np.asarray(LAGS)
    old_status, old = prior.signed_candidate(times, evidence, samples)
    if old_status == "coverage":
        return dict.fromkeys(VARIANTS)
    near = np.abs(lags) <= 960
    far = np.abs(lags) >= 2560
    outer, inner = np.nonzero(
        (np.abs(lags[:, None]) <= 2240)
        & (np.abs(lags[:, None] - lags[None, :]) >= 2560)
        & (np.abs(lags[:, None] - lags[None, :]) <= 5440)
    )
    near_outside = near[outer]
    common = np.isfinite(evidence[:, outer]) & np.isfinite(evidence[:, inner])
    groups = np.isfinite(evidence).sum(axis=1)
    usable = groups >= 8
    normalized = np.full_like(evidence, np.nan)
    observed = evidence[usable]
    q1, median, q3 = np.nanquantile(observed, (0.25, 0.5, 0.75), axis=1)
    normalized[usable] = np.clip(
        (observed - median[:, None]) / np.maximum(0.20, (q3 - q1)[:, None]), -2.0, 2.0
    )
    standard_pair = common & usable[:, None]
    differences = evidence[:, inner] - evidence[:, outer]
    standardized = normalized[:, inner] - normalized[:, outer]
    zero = np.zeros(len(inner))

    def prefix(array):
        return np.vstack((zero, np.cumsum(array, axis=0)))

    counts = prefix(common)
    standard_counts = prefix(standard_pair)
    total = prefix(np.where(common, differences, 0.0))
    standard_total = prefix(np.where(standard_pair, standardized, 0.0))
    strengths = np.isfinite(evidence)
    level = np.where(strengths, evidence, 0.0)
    n = len(times)
    half = n // 2
    clearly_shifted = np.zeros(len(lags), dtype=bool)
    for a, b in ((2, half), (half, n - 2)):
        near_frames = strengths[a:b, near]
        far_frames = strengths[a:b, far]
        shared = far_frames[:, :, None] & near_frames[:, None, :]
        counts_shared = shared.sum(axis=0)
        advantage = np.where(shared, level[a:b, far, None] - level[a:b, None, near], 0.0).sum(
            axis=0
        ) / counts_shared.clip(1)
        comparisons = (counts_shared >= 8) & (advantage > 0.075)
        good = comparisons.sum(axis=1) >= 4
        if a == 2:
            clearly_shifted[far] = good
        else:
            clearly_shifted[far] &= good
    compensated = bool(clearly_shifted.any())

    def summary(a, b, minimum, prefix_value, prefix_count):
        seen = prefix_count[b] - prefix_count[a]
        return np.where(seen >= minimum, (prefix_value[b] - prefix_value[a]) / seen.clip(1), -np.inf)

    best = dict.fromkeys(VARIANTS[1:])
    for width in prior.SIZES:
        for first in range(8, n - width - 8):
            end = first + width
            start_sample, end_sample = times[first] + lags[inner], times[end] + lags[inner]
            a, b = max(0, first - prior.FLANK), min(n, end + prior.FLANK)
            valid_audio = (start_sample >= 0) & (end_sample <= samples)
            for name, matrix, support, tot, cutoff, permitted in (
                ("outside_anchor", differences, common, total, counts, near_outside),
                ("normalized", standardized, standard_pair, standard_total, standard_counts, True),
                (
                    "guarded_normalized",
                    standardized,
                    standard_pair,
                    standard_total,
                    standard_counts,
                    near_outside & (not compensated),
                ),
            ):
                contrasts = [
                    summary(x, y, count, tot if sign else -tot, cutoff)
                    for x, y, count, sign in (
                        (a, first, 5, False),
                        (first, end, (width + 1) // 2, True),
                        (end, b, 5, False),
                    )
                ]
                gain = np.where(
                    valid_audio & permitted, np.minimum(np.minimum(*contrasts[:2]), contrasts[2]), -np.inf
                )
                k = int(gain.argmax())
                score = round(float(gain[k]), 7)
                if score <= 0 or (best[name] and score <= best[name]["score"]):
                    continue
                middle = np.where(support[first:end, k], matrix[first:end, k], -np.inf)
                witnesses = [
                    int(a + np.flatnonzero(support[a:first, k])[-1]),
                    int(first + middle.argmax()),
                    int(end + np.flatnonzero(support[end:b, k])[0]),
                ]
                best[name] = {
                    "score": score,
                    "span": [int(start_sample[k]), int(end_sample[k])],
                    "lag_samples": [int(lags[outer[k]]), int(lags[inner[k]])],
                    "picture_span": [first, end],
                    "witness_frames": witnesses,
                    "contrasts": [round(float(value[k]), 7) for value in contrasts],
                }
    return {"earlier_raw": old, **best}


def author(samples, spec, role):
    if not role.startswith("drift"):
        return prior.edit(samples, spec["interval"], spec["shift_samples"], role)
    import numpy as np

    indexes = np.arange(len(samples))
    sign = 1 if role.endswith("early") else -1
    return np.rint(
        np.interp(
            indexes + sign * np.linspace(288.0, 2560.0, len(samples)), indexes, samples, left=0.0, right=0.0
        )
    ).astype("<i2")


def unsupported_edit(spec, role):
    if not role.startswith("local"):
        return False
    a, b = spec["interval"]
    shift = -spec["shift_samples"] if role.endswith("early") else spec["shift_samples"]
    return not 0 <= a - shift < b - shift <= spec["samples"]


def run(
    plan_path,
    old_plan,
    old_cal,
    old_held,
    prior_media,
    sources,
    models,
    assets,
    output,
    *,
    phase,
    frozen=None,
):
    import subprocess
    import sys

    import cv2
    import numpy as np
    import torch

    from app.detectors.audio.decode import decode_audio
    from app.detectors.video import av_local_agreement as neural
    from app.detectors.video import viseme_review, viseme_worker
    from app.detectors.video.decode import probe_frames
    from app.evidence import video_lip_agreement
    from app.evidence import video_viseme_binding as binding
    from scripts.export_video_lip_correspondence import make_mux, visual_hash
    from scripts.video_lip_timing_aac48 import encode
    from scripts.video_neural_phone_clock import pcm

    plan, previous = prior.read(plan_path), prior.read(old_plan)
    if (
        plan["schema"] != SCHEMA
        or prior.sha(Path(__file__)) != plan["script_sha256"]
        or plan["prior_plan_sha256"] != prior.sha(old_plan)
        or plan["rule"] != RULE
    ):
        raise ValueError("The prospective source roster or experiment code changed")
    development = phase == "development"
    if not development:
        fitted = prior.read(frozen)
        if fitted["plan_sha256"] != prior.sha(plan_path) or select(fitted["rows"]) != fitted["selection"]:
            raise ValueError("Held measurement requires the deterministic development decision")
    landmark = assets / "face_landmarker.task"
    if prior.sha(landmark) != plan["geometry_sha256"] or any(
        prior.sha(models / f"{name}.safetensors") != sha for name, sha in plan["weights"].items()
    ):
        raise ValueError("Protected timing assets differ from the source plan")
    sys.path.insert(0, str((assets / "runtime").resolve()))
    cv2.setNumThreads(1)
    decode_phones, _ = viseme_worker.aperture_reader(assets / "checkpoint", articulation=True)
    torch.set_num_threads(1)
    eyes, ears = viseme_worker._motion_models(models)
    previous_receipts = (prior.read(old_cal), prior.read(old_held))
    if any(report["plan_sha256"] != prior.sha(old_plan) for report in previous_receipts):
        raise ValueError("The earlier comparisons are not tied to their source roster")
    former = {
        phase["phase"]: {(row["speaker"], row["input"]): row for row in phase["rows"]}
        for phase in previous_receipts
    }
    entries = previous["rows"] if development else plan["entries"]
    output.mkdir(mode=0o700)
    media = output / "media"
    media.mkdir(mode=0o700)
    rows = []
    with (
        ZipFile(sources / "lombardgrid_front.zip") as video,
        ZipFile(sources / "lombardgrid_audio.zip") as audio,
        (output / "journal.jsonl").open("x") as journal,
    ):
        for i, spec in enumerate(entries):
            raw_video, raw_audio = video.read(spec["video_path"]), audio.read(spec["audio_path"])
            if (
                prior.digest(raw_video) != spec["video_sha256"]
                or prior.digest(raw_audio) != spec["audio_sha256"]
            ):
                raise ValueError("An archive member no longer matches the frozen bytes")
            source_path = output / "source.mov"
            source_path.write_bytes(raw_video)
            table = probe_frames(source_path, cap=121, timeout_s=20, max_pixels=1280 * 720)
            periods = {(b - a) * table.tb for a, b in zip(table.pts, table.pts[1:])}
            if (
                not 34 <= len(table) <= 120
                or table.rotation
                or table.interpolated
                or table.truncated
                or table.pts[0] != 0
                or len(periods) != 1
                or not Fraction(1, 28) <= next(iter(periods)) <= Fraction(1, 22)
            ):
                raise ValueError("The fixed source clip has an unsupported mouth timing clock")
            geometry = {"width": table.width, "height": table.height}
            waveform = pcm(raw_audio)
            assert len(waveform) == spec["samples"]
            packet_hash = visual_hash(source_path)
            frames, patches, valid = viseme_worker.capture_mouth_frames(
                source_path, table, landmark, motion=True, articulation=True
            )
            times = np.asarray([round(t * table.tb * RATE) for t in table.pts])
            eye = neural.picture_windows(eyes, patches, valid)
            roles = (*prior.KINDS[:4], *CONTROLS) if development and spec["role"] == "calibration" else ROLES
            for role in roles:
                if unsupported_edit(spec, role):
                    row = {
                        "speaker": spec["speaker"],
                        "input": role,
                        "file": None,
                        "input_sha256": None,
                        "pcm_sha256": None,
                        "video_sha256": spec["video_sha256"],
                        "source_track_sha256": packet_hash,
                        "audio_alignment_samples": None,
                        "samples": len(waveform),
                        "picture_clock": {"pts": list(table.pts), "time_base": str(table.tb)},
                        "visible": int(valid.sum()),
                        "truth": spec["interval"],
                        "delay_samples": -spec["shift_samples"]
                        if role.endswith("early")
                        else spec["shift_samples"],
                        "incumbent": {"state": "not_run", "range": None, "change_samples": None},
                        "responses": {},
                        "support_images": [],
                        "unassessed": "requested audio extends beyond publisher samples",
                    }
                    journal.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    journal.flush()
                    rows.append(row)
                    continue
                transformed = author(waveform, spec, role)
                old = former[spec["role"]].get((spec["speaker"], role)) if development else None
                if old:
                    folder = "cal" if spec["role"] == "calibration" else "held"
                    path = prior_media / folder / "media" / old["file"]
                    if prior.sha(path) != old["file_sha256"]:
                        raise ValueError("A previous measured delivery changed")
                else:
                    audio_path = output / "audio.m4a"
                    encode(transformed, audio_path, 48_000)
                    path = media / f"{i:02d}-{role}.mp4"
                    make_mux(source_path, audio_path, path, table, packet_hash)
                path_pin = prior.sha(path)
                decoded = decode_audio(path, source_sha256=path_pin)
                mapping = decoded.mapping
                if (
                    binding.presented_samples(mapping) != len(waveform)
                    or not viseme_review.supported(table, mapping, [[0, len(table)]])
                    or video_lip_agreement.carrier(
                        mapping, origin=binding.audio_origin(mapping), stereo_scope=None
                    )
                    != "aac48"
                ):
                    raise ValueError("The single MP4 does not carry the original frame and audio clocks")
                sound = np.rint(decoded.pcm[: len(waveform)] * 32768.0).astype("<i2")
                sound_pin = prior.digest(sound.tobytes())
                if old and (
                    old["pcm_sha256"] != sound_pin
                    or old["submitted_video_track_sha256"] != packet_hash
                    or old["native_presentation"] != {"pts": list(table.pts), "time_base": str(table.tb)}
                ):
                    raise ValueError("The previous re-decoded input changed its sound or picture commitments")
                exact, measured = transformed.astype(float), sound.astype(float)
                aligned = max(
                    (float(np.sum(exact[32:-32] * measured[32 + shift : len(sound) - 32 + shift])), shift)
                    for shift in range(-12, 13)
                )[1]
                if abs(aligned) > 1:
                    raise ValueError("AAC readback shifts its original audio clock")
                evidence = neural.similarities(ears, eye, neural.sound(sound), times)
                candidates = variant_readings(times, evidence, len(sound))
                if old and old["incumbent"]["state"] != "not_run":
                    incumbent = old["incumbent"]
                else:
                    _, phonetic, articulation = decode_phones(sound, include_events=True)
                    _, dip = neural.localize(times, evidence, len(sound))
                    raw = {
                        "frames": frames,
                        "phones": phonetic,
                        "articulation_phones": articulation,
                        "sound_levels": viseme_review.sound_levels(decoded.pcm, len(sound)),
                    }
                    if dip is not None:
                        raw[video_lip_agreement.KEY] = video_lip_agreement.encode(dip)
                    base = binding.measured(
                        raw, timestamps=table.pts, time_base=str(table.tb), source=geometry, mapping=mapping
                    )
                    witness = base["witness"]
                    incumbent = {
                        "range": witness["audio_source_pts"] if witness else None,
                        "change_samples": witness["change_samples"] if witness else None,
                        "basis": base.get("measurement_basis"),
                        "state": base["status"],
                    }
                variant = fitted["selection"]["variant"] if not development else None
                if variant:
                    candidates = {k: v for k, v in candidates.items() if k in (variant, "earlier_raw")}
                anchors = sorted(
                    {i for value in candidates.values() if value for i in value["witness_frames"]}
                )
                exact_frames = []
                if anchors:
                    selected = "+".join(f"eq(n,{index})" for index in anchors)
                    raw = subprocess.run(
                        [
                            "ffmpeg",
                            "-nostdin",
                            "-v",
                            "error",
                            "-xerror",
                            "-threads",
                            "1",
                            "-filter_threads",
                            "1",
                            "-i",
                            str(path),
                            "-map",
                            "0:v:0",
                            "-an",
                            "-vf",
                            f"select='{selected}',format=rgb24",
                            "-fps_mode",
                            "passthrough",
                            "-c:v",
                            "rawvideo",
                            "-threads:v",
                            "1",
                            "-f",
                            "framehash",
                            "-hash",
                            "sha256",
                            "pipe:1",
                        ],
                        check=True,
                        timeout=20,
                        capture_output=True,
                        text=True,
                    ).stdout
                    digests = [
                        line.rsplit(",", 1)[1].strip()
                        for line in raw.splitlines()
                        if line and not line.startswith("#")
                    ]
                    exact_frames = [
                        {
                            k: frames[index][k]
                            for k in ("index", "pts", "rgb_sha256", "mouth_box", "mouth_crop_sha256")
                        }
                        for index in anchors
                    ]
                    if digests != [record["rgb_sha256"] for record in exact_frames]:
                        raise ValueError("The selected mouth frames differ from their submitted MP4")
                truth = spec["interval"] if role.startswith("local") else None
                delay = (-1 if role.endswith("early") else 1) * spec["shift_samples"] if truth else None
                row = {
                    "speaker": spec["speaker"],
                    "input": role,
                    "file": path.name,
                    "input_sha256": path_pin,
                    "pcm_sha256": sound_pin,
                    "video_sha256": spec["video_sha256"],
                    "source_track_sha256": packet_hash,
                    "audio_alignment_samples": aligned,
                    "samples": len(sound),
                    "picture_clock": {"pts": list(table.pts), "time_base": str(table.tb)},
                    "visible": int(valid.sum()),
                    "truth": truth,
                    "delay_samples": delay,
                    "incumbent": incumbent,
                    "responses": candidates,
                    "support_images": exact_frames,
                }
                journal.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                journal.flush()
                rows.append(row)
                if development and old is None:
                    path.unlink()
            print(f"{phase}: {i + 1}/{len(entries)} exact recordings", flush=True)
    value = {"schema": SCHEMA, "phase": phase, "plan_sha256": prior.sha(plan_path), "rows": rows}
    if development:
        value["selection"] = select(rows)
    else:
        value["development_sha256"] = prior.sha(frozen)
        picked = fitted["selection"]
        value["outcome"] = summarize(rows, picked["variant"], picked["point"])
    prior.save(output / "receipt.json.gz", value)
    print(json.dumps(value["selection"] if development else value["outcome"], sort_keys=True), flush=True)


def decision(row, name, point):
    p = row["responses"].get(name)
    if p and p["score"] > point:
        return p["span"], p["lag_samples"][1] - p["lag_samples"][0]
    return None, None


def summarize(rows, name, point):
    fields = (
        "edit",
        "located",
        "signed",
        "misplaced",
        "ordinary",
        "ordinary_flags",
        "global",
        "global_flags",
        "drift",
        "drift_flags",
    )
    tally = {arm: Counter(dict.fromkeys(fields, 0)) for arm in ("existing", "prior", "candidate", "combined")}
    added, ordinary, stable, misplaced = set(), 0, 0, 0
    for row in rows:
        truth, delta, label = row["truth"], row["delay_samples"], row["input"]
        scope = (
            "edit"
            if truth
            else "global"
            if label.startswith("global")
            else "drift"
            if label.startswith("drift")
            else "ordinary"
        )
        found, change = row["incumbent"]["range"], row["incumbent"]["change_samples"]
        old = decision(row, "earlier_raw", 0.4639570)
        new = decision(row, name, point)
        for arm, (span, shift) in (
            ("existing", (found, change)),
            ("prior", old),
            ("candidate", new),
            ("combined", (found, change) if found else new),
        ):
            count = tally[arm]
            count[scope] += 1
            if span is None:
                continue
            if truth:
                located = prior.match(span, truth)
                signed = located and shift is not None and abs(shift - delta) <= 1280
                count["located"] += located
                count["signed"] += signed
                count["misplaced"] += not located
                if arm == "combined" and not found and signed:
                    added.add(row["speaker"])
                if arm == "combined" and not found and not located:
                    misplaced += 1
            else:
                count[f"{scope}_flags"] += 1
                if arm == "combined" and not found:
                    ordinary += scope == "ordinary"
                    stable += scope in ("drift", "global")
    signed = tally["combined"]["signed"] - tally["existing"]["signed"]
    counts = {
        "new_signed": signed,
        "new_speakers": len(added),
        "new_ordinary": ordinary,
        "new_global": stable,
        "new_misplaced": misplaced,
    }
    return (
        tally
        | counts
        | {
            "unassessed_input": sum(bool(row.get("unassessed")) for row in rows),
            "accepted": signed >= RULE["new_signed"]
            and len(added) >= RULE["new_speakers"]
            and all(counts[k] <= RULE[k] for k in ("new_ordinary", "new_global", "new_misplaced")),
        }
    )


def select(rows):
    candidates = []
    summary = {}
    floor = SCORE_FLOOR
    for index, name in enumerate(VARIANTS):
        scores = {row["responses"][name]["score"] for row in rows if row["responses"].get(name)}
        eligible = []
        for point in sorted({floor, *[score for score in scores if score > floor]}):
            readout = summarize(rows, name, point)
            if any(readout[k] > RULE[k] for k in ("new_ordinary", "new_global", "new_misplaced")):
                continue
            order = (
                readout["new_signed"],
                readout["new_speakers"],
                -readout["new_ordinary"],
                -readout["new_misplaced"],
                -index,
                point,
            )
            eligible.append((order, name, point, readout))
        winner = max(eligible, key=lambda row: row[0])
        summary[name] = {"point": winner[2], "outcome": winner[3]}
        candidates.append(winner)
    _, variant, point, _ = max(candidates, key=lambda row: row[0])
    return {"variant": variant, "point": point, "comparison": summary}


def review(plan, old_plan, development, held):
    design, previous, dev, test = (prior.read(path) for path in (plan, old_plan, development, held))
    if (
        dev["plan_sha256"] != prior.sha(plan)
        or test["plan_sha256"] != prior.sha(plan)
        or test["development_sha256"] != prior.sha(development)
        or design["prior_plan_sha256"] != prior.sha(old_plan)
        or design["script_sha256"] != prior.sha(Path(__file__))
    ):
        raise ValueError("The prospective comparison does not use its committed source and scoring inputs")
    old_groups = {row["speaker"] for row in previous["rows"]}
    if old_groups & {row["speaker"] for row in design["entries"]}:
        raise ValueError("Development and held speakers overlap")
    for phase, specs, receipt in (("development", previous["rows"], dev), ("held", design["entries"], test)):
        expected = {
            (row["speaker"], role): row
            for row in specs
            for role in (
                (*prior.KINDS[:4], *CONTROLS)
                if phase == "development" and row["role"] == "calibration"
                else ROLES
            )
        }
        actual = {(row["speaker"], row["input"]): row for row in receipt["rows"]}
        if set(expected) != set(actual) or len(actual) != len(receipt["rows"]):
            raise ValueError("Submitted cases were omitted, repeated or relabeled")
        for key, row in actual.items():
            source = expected[key]
            truth = source["interval"] if row["input"].startswith("local") else None
            delay = (-1 if row["input"].endswith("early") else 1) * source["shift_samples"] if truth else None
            if (
                row["video_sha256"] != source["video_sha256"]
                or row["samples"] != source["samples"]
                or (row["truth"], row["delay_samples"]) != (truth, delay)
            ):
                raise ValueError("Frame, sound or treatment no longer matches its frozen source")
            if unsupported_edit(source, row["input"]):
                if (
                    not row.get("unassessed")
                    or row["responses"]
                    or row["incumbent"]["range"]
                    or row["support_images"]
                    or any(
                        row[key] is not None
                        for key in ("file", "input_sha256", "pcm_sha256", "audio_alignment_samples")
                    )
                ):
                    raise ValueError("An out-of-bounds treatment cannot provide a measured delivery")
            elif row.get("unassessed") or abs(row["audio_alignment_samples"]) > 1:
                raise ValueError("An admitted delivery must preserve its available native audio clock")
            picture = row["picture_clock"]
            pts, tb = picture["pts"], Fraction(picture["time_base"])
            periods = {(b - a) * tb for a, b in zip(pts, pts[1:])}
            if (
                len(periods) != 1
                or pts[0] != 0
                or not Fraction(1, 28) <= next(iter(periods)) <= Fraction(1, 22)
            ):
                raise ValueError("A picture timeline is outside the frozen bounds")
            exact = {img["index"]: img["pts"] for img in row["support_images"]}
            for selected in (value for value in row["responses"].values() if value):
                if (
                    selected["span"]
                    != [
                        round(pts[i] * tb * RATE) + selected["lag_samples"][1]
                        for i in selected["picture_span"]
                    ]
                    or selected["score"] != min(selected["contrasts"])
                    or any(exact.get(i) != pts[i] for i in selected["witness_frames"])
                ):
                    raise ValueError("A selected range is detached from its supporting original frames")
    chosen = select(dev["rows"])
    result = summarize(test["rows"], chosen["variant"], chosen["point"])
    if dev["selection"] != chosen or test["outcome"] != result:
        raise ValueError("The chosen method or held summary does not independently replay")
    print(
        json.dumps({"variant": chosen["variant"], "point": chosen["point"], "held": result}, sort_keys=True)
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="mode", required=True)
    for name, fields in (
        ("reserve", ("sources", "panels", "old_plan", "output")),
        (
            "run",
            (
                "plan_path",
                "old_plan",
                "old_cal",
                "old_held",
                "prior_media",
                "sources",
                "models",
                "assets",
                "output",
            ),
        ),
        ("review", ("plan", "old_plan", "development", "held")),
    ):
        command = commands.add_parser(name)
        for field in fields:
            command.add_argument(field, type=Path)
        if name == "run":
            command.add_argument("--phase", choices=("development", "held"), required=True)
            command.add_argument("--frozen", type=Path)
    values = vars(cli.parse_args())
    {"reserve": reserve, "run": run, "review": review}[values.pop("mode")](**values)


if __name__ == "__main__":
    main()
