# SPDX-License-Identifier: Apache-2.0
"""Freeze, construct and reobserve a fresh native Lombard audio-driven redraw panel."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import secrets
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image.faces import detect_faces, load_yunet
from scripts import av_conditioning_clock_v109 as clock
from scripts import build_lombard_neural_mouth as build
from scripts import build_lombard_paired_visual_challenge as controls
from scripts import evaluate_short_lip_release_timing as timing
from scripts import grid_musetalk_frames_v148 as prior
from scripts import review_lombard_neural_mouth as review
from scripts.video_common_sources_v89 import read, require, sha, write
from scripts.video_mouth_reconstruction_transfer import WEIGHT

SELECTED = (4, 11, 25, 29, 37, 45)
ROLES = ("original", "same_time_resample", "same_time_neural", "earlier_sound_neural", "delayed_pixels")
NEURAL = {"same_time_neural": 0, "earlier_sound_neural": -5}
SPEC = {
    "sources": "Six previously unrendered source groups, one clip per group from independently qualified original events. Fixed ordinals chosen for exact matched ±5-picture ordinary controls and representation of both speaking styles. Keep all selected roles regardless of result.",
    "interval": "Six pictures before to ten pictures after the independently observed last closed picture, half-open. Preserve every native picture PTS, original waveform sample, and all pixels outside the frozen crop or edit interval. Every edit uses exactly the permitted face-box geometry, native alpha, and half opacity on the first and last edit pictures.",
    "clock": "Full original mono waveform produces the five speech hidden layers. Select the ten 50 Hz slots around floor(50 * exact native PTS), or around the native time five pictures earlier. No rebasing or cadence conversion. Require every selected slot available. Original soundtrack remains unchanged in each delivery.",
    "observer": "Reobserve muted encoded video using opaque randomized tokens, publisher coarse phone context and the fixed later control neighborhood. Admit source-event support only for one visual event in context, within one picture of the source, inside the edit and within 80 ms of the independently measured waveform burst. A source needs same-time resample and five-picture delayed ordinary controls to qualify. Retain the earlier-conditioned neural outcome separately as a negative specificity challenge.",
    "scope": "The sole local measured event is neither full-clip synchronization nor phone identity. Prior project source exposure consists of original and ordinary timing tests only. No detector scores inform selection or admission.",
}


def seed(ordinal, index):
    payload = f"chipmunk.lombard-fresh-musetalk/1:{ordinal}:{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def bind(source, study, visual, old):
    receipt = source / "source-receipt.json"
    observation = study / "observations-held.json"
    geometry = visual / "visual-challenge.json"
    previous = old / "freeze.json"
    bank = read(receipt)
    native = read(observation)
    ordinary = read(geometry)
    require(
        bank["freeze_sha256"] == sha(source / "freeze.json")
        and native["source_receipt_sha256"] == sha(receipt)
        and ordinary["freeze_sha256"] == sha(visual / "freeze.json")
        and read(visual / "freeze.json")["observations_sha256"] == sha(observation),
        "Original source bindings differ",
    )
    observed = {r["ordinal"]: r for r in native["rows"]}
    matched = {r["ordinal"]: r for r in ordinary["cases"]}
    previous_speakers = {r["source"]["speaker"] for r in read(previous)["sources"]}
    rows = []
    for item in bank["rows"]:
        k = item["ordinal"]
        if k not in SELECTED:
            continue
        x = observed[k]
        require(
            item["speaker"] not in previous_speakers
            and x["truth"]
            and x["group"] == item["group"]
            and x["video_sha256"] == item["inputs"]["front"]["sha256"]
            and x["audio_sha256"] == item["inputs"]["audio"]["sha256"],
            "Source or previous-renderer roster differs",
        )
        require(
            all(v["qualification"] == "observed" for v in matched[k]["observations"]),
            "Selected source lacks independent matched signed control",
        )
        anchor = x["truth"]["visual_frames"][0]
        start, end = anchor - 6, anchor + 10
        require(start >= 5 and end < len(x["pts"]) - 2, "Need native flanks and earlier sound context")
        times = [p * Fraction(x["time_base"]) for p in x["pts"]]
        chunks = {
            str(shift): [
                clock.frame_plan(i, times[i + shift], 0, 0, x["audio_samples"], 0) for i in range(start, end)
            ]
            for shift in set(NEURAL.values())
        }
        require(
            all(
                all(c["retained_tick_slots"]) and c["source_sample_status"] == "supported"
                for entries in chunks.values()
                for c in entries
            ),
            "A source lacks full waveform slot coverage",
        )
        rows.append(
            {
                "source": item,
                "visual": {
                    "time_base": x["time_base"],
                    "pts": x["pts"],
                    "frames": x["truth"]["visual_frames"],
                },
                "truth": x["truth"],
                "original_audio_pcm_sha256": x["audio_pcm_sha256"],
                "original_sample_bytes": x["source_sample_bytes"],
                "audio_samples": x["audio_samples"],
                "span": [start, end],
                "conditioning": chunks,
            }
        )
    require(
        tuple(r["source"]["ordinal"] for r in rows) == SELECTED
        and len({r["source"]["group"] for r in rows}) == 6
        and {r["source"]["condition"] for r in rows} == {"p", "l"},
        "Frozen independent roster differs",
    )
    modules = [
        Path(__file__),
        Path(prior.__file__),
        Path(clock.__file__),
        Path(build.__file__),
        Path(review.__file__),
    ]
    for name in ("musetalk_benchmark_v7.py", "smoke_mouth_generator_v7.py", "prepare_mouth_edit_v4.py"):
        modules.append(Path(__file__).with_name(name))
    native_roles = {
        "same_time_neural": 0,
        "earlier_sound_neural": -5,
    }
    require(native_roles == NEURAL, "Native audio conditioning roles changed")
    return {
        "protocol": SPEC,
        "sources": rows,
        "source_receipt_sha256": sha(receipt),
        "source_timing_sha256": sha(observation),
        "ordinary_proof_sha256": sha(geometry),
        "previous_source_sha256": sha(previous),
        "renderer_asset_manifest_sha256": sha(prior.MANIFEST),
        "geometry_sha256": sha(WEIGHT),
        "restored_runtime_package_versions": prior.package_identity(),
        "code_sha256": {p.name: sha(p) for p in modules},
        "ffmpeg": build.command("ffmpeg", "-version").decode().splitlines()[0],
    }


def source_material(folder, row):
    item = row["source"]
    front = build.originals.source(folder, item, "front")
    waveform = build.originals.source(folder, item, "audio")
    raw, pcm, width = timing.source_pcm(waveform)
    require(
        build.originals.digest(raw) == row["original_audio_pcm_sha256"]
        and width == row["original_sample_bytes"]
        and len(pcm) == row["audio_samples"],
        "Frozen source audio differs",
    )
    table, step = build.table_for(front, row)
    pictures = build.read_pictures(front, table)
    return table, step, pictures, pcm


def crops_for(pictures, row, detector):
    from scripts.smoke_mouth_generator_v7 import alpha_from_face, crop_from_face

    start, stop = row["span"]
    crops = {}
    for i in range(start - 5, stop):
        rgb = pictures[i]
        faces = detect_faces(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), detector=detector, min_size=48)
        if len(faces) != 1:
            return None
        try:
            crop, matrix, geometry = crop_from_face(rgb, faces[0])
            alpha = alpha_from_face(rgb.shape, faces[0])
        except ValueError:
            return None
        crops[i] = (crop, matrix, alpha, geometry)
    return crops


def speech_states(renderer, pcm):
    import torch

    with torch.inference_mode():
        inputs = renderer.extractor(pcm, sampling_rate=16000, return_tensors="pt").input_features.to(
            renderer.device
        )
        states = renderer.encoder(inputs, output_hidden_states=True).hidden_states
        return torch.stack(states, dim=2)[0, : len(pcm) // 320].detach().to("cpu", torch.float32).numpy()


def edit(pictures, table, row, role, crops, model, hidden):
    import torch

    from scripts.prepare_mouth_edit_v4 import composite

    changed, edits = list(pictures), []
    start, stop = row["span"]
    if role == "original":
        return changed, edits
    for i in range(start, stop):
        crop, matrix, alpha, geo = crops[i]
        driver = i - 5 if role == "delayed_pixels" else i
        proof, conditioning = None, None
        if role in NEURAL:
            shift = NEURAL[role]
            route = row["conditioning"][str(shift)][i - start]
            chunk = clock.select_chunk(hidden, route)
            fixed_seed = seed(row["source"]["ordinal"], i)
            generated, proof = model.render(crop, torch.from_numpy(chunk), fixed_seed)
            conditioning = {
                "encoder_anchor": route["encoder_anchor"],
                "tick_indices": route["tick_indices"],
                "source_time_seconds": route["source_time_seconds"],
                "f32_sha256": build.originals.digest(chunk.tobytes()),
            }
        else:
            generated = crops[driver][0]
        target = cv2.warpAffine(
            generated, cv2.invertAffineTransform(matrix), (table.width, table.height), flags=cv2.INTER_LINEAR
        )
        changed[i] = composite(pictures[i], target, alpha, 0.5 if i in (start, stop - 1) else 1.0)
        mask = np.any(changed[i] != pictures[i], axis=2).astype(np.uint8) * 255
        require(
            mask.any() and not mask[alpha == 0].any(),
            "Frozen operation lacks changed mouth support or changes outside it",
        )
        edits.append(
            {
                "index": i,
                "pts": table.pts[i],
                "driver_index": driver,
                "driver_pts": table.pts[driver],
                "geometry": geo,
                "alpha_sha256": build.originals.digest(alpha.tobytes()),
                "mask_sha256": build.originals.digest(mask.tobytes()),
                "rgb_sha256": build.originals.digest(changed[i].tobytes()),
                "model": proof,
                "conditioning": conditioning,
            }
        )
    return changed, edits


def render(source, run, lock, runtime):
    cv2.setNumThreads(1)
    detector = load_yunet(WEIGHT)
    with prior.isolated_overlay():
        import torch

        from scripts.musetalk_benchmark_v7 import load

        torch.set_num_threads(1)
        require(torch.backends.mps.is_available(), "Reviewed Apple MPS runtime unavailable")
        torch.mps.set_per_process_memory_fraction(0.65)
        model = load(prior.MANIFEST, runtime / "assets", "mps")
        require(
            model.load_audit["all_three_modules_strict_loaded"],
            "Renderer did not strictly load reviewed checkpoints",
        )
        key = secrets.token_bytes(32)
        ledger, plan = [], []
        for row in lock["sources"]:
            k = row["source"]["ordinal"]
            table, step, pictures, pcm = source_material(source, row)
            crops = crops_for(pictures, row, detector)
            if crops is None:
                ledger.extend(
                    {"ordinal": k, "role": role, "status": "unavailable_geometry"} for role in ROLES
                )
                print(k, "unavailable_geometry", flush=True)
                continue
            wav = run / f"{k:02d}.wav"
            controls.float_wav(wav, pcm)
            hidden = speech_states(model, pcm)
            require(len(hidden) == len(pcm) // 320, "Complete original speech hidden prefix required")
            for role in ROLES:
                token = hmac.digest(key, f"{k}:{role}".encode(), "sha256").hex()[:24]
                path = run / "deliveries" / f"{token}.mov"
                authored, edits = edit(pictures, table, row, role, crops, model, hidden)
                build.encode(path, authored, wav, table, step)
                record = {
                    "ordinal": k,
                    "role": role,
                    "status": "authored",
                    "token": token,
                    "file_sha256": sha(path),
                    "edits": edits,
                }
                ledger.append(record)
                plan.append(
                    {
                        "token": token,
                        "file_sha256": record["file_sha256"],
                        "coarse_hint_s": row["source"]["aligned_following_phone_s"],
                    }
                )
                print(k, role, len(edits), flush=True)
            write(run / f"rendered-{k:02d}.json", [r for r in ledger if r["ordinal"] == k])
            torch.mps.empty_cache()
    write(
        run / "render.json",
        {
            "freeze_sha256": sha(run / "freeze.json"),
            "runtime_receipt_sha256": sha(runtime / "assets/verified-assets.json"),
            "cases": ledger,
        },
    )
    write(run / "blind-plan-held.json", sorted(plan, key=lambda row: row["token"]))


def audit(source, run, lock):
    path = run / "blind-observations-held.json"
    observed = read(path)
    require(observed["plan_sha256"] == sha(run / "blind-plan-held.json"), "Blind observation freeze differs")
    by_token = {r["token"]: r for r in observed["rows"]}
    ledger = read(run / "render.json")
    require(ledger["freeze_sha256"] == sha(run / "freeze.json"), "Renderer freeze differs")
    rows = []
    for row in lock["sources"]:
        k = row["source"]["ordinal"]
        table, _, source_pictures, pcm = source_material(source, row)
        start, stop = row["span"]
        times = [p * table.tb for p in table.pts]
        audio_range = [str(Fraction(i, 16000)) for i in row["truth"]["acoustic_samples"]]
        row = {
            **row,
            "event": {"audio_range_s": audio_range, "native_edit_frames": [start, stop]},
            "audio": {
                "samples": len(pcm),
                "native_pcm_sha256": build.originals.digest(pcm.astype("<f4").tobytes()),
            },
        }
        cases = [r for r in ledger["cases"] if r["ordinal"] == k]
        require([r["role"] for r in cases] == list(ROLES), "Missing frozen case roles")
        ordinary_support = None
        mapped = {}
        for case in cases:
            role = case["role"]
            if case["status"] != "authored":
                mapped[role] = {"status": case["status"]}
                continue
            native = review.exact_delivery(run, case, row, table, source_pictures)
            proof = by_token.pop(case["token"])
            require(
                proof["file_sha256"] == case["file_sha256"]
                and [e["index"] for e in case["edits"]]
                == (list(range(start, stop)) if role != "original" else []),
                "Blind proof differs or a frozen range lost edited pixels",
            )
            alpha = [e["alpha_sha256"] for e in case["edits"]]
            if role != "original":
                if ordinary_support is None:
                    ordinary_support = alpha
                require(alpha == ordinary_support, "Controls and redraws used different geometry")
            delayed = role == "delayed_pixels"
            event = proof["releases"]["later_neighborhood" if delayed else "source_neighborhood"]
            observation = review.release_status(event, row, table, delayed=delayed)
            if role == "original":
                require(
                    observation.get("frames") == row["visual"]["frames"],
                    "Original visible event changed on readback",
                )
            mapped[role] = {"status": observation["status"], "observation": observation, **native}
        qualify = (
            mapped["same_time_neural"]["status"]
            == mapped["same_time_resample"]["status"]
            == "source_event_retained"
            and mapped["delayed_pixels"]["status"] == "observed_delayed_control"
        )
        record = {
            "source_ordinal": k,
            "group": row["source"]["group"],
            "style": row["source"]["condition"],
            "edit": [str(times[start]), str(times[stop])],
            "audio_bracket": audio_range,
            "picture_step": str(times[1] - times[0]),
            "roles": mapped,
            "qualifies_local_event": qualify,
        }
        rows.append(record)
        print(k, {role: m["status"] for role, m in mapped.items()}, qualify, flush=True)
    require(not by_token, "Blinded roster includes unrelated rows")
    write(
        run / "audit.json",
        {
            "freeze_sha256": sha(run / "freeze.json"),
            "observer_sha256": sha(path),
            "render_sha256": sha(run / "render.json"),
            "rows": rows,
            "qualified": sum(r["qualifies_local_event"] for r in rows),
            "counts": dict(Counter(f"{role}:{m['status']}" for r in rows for role, m in r["roles"].items())),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "render", "observe", "audit"))
    for name in ("run", "source", "study", "visual", "previous", "runtime"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    prior.OVERLAY = args.runtime / "runtime"
    lock = bind(args.source, args.study, args.visual, args.previous)
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        (args.run / "deliveries").mkdir(mode=0o700)
        write(args.run / "freeze.json", lock)
        return
    require(lock == read(args.run / "freeze.json"), "Frozen program or source inputs differ")
    if args.phase == "render":
        render(args.source, args.run, lock, args.runtime)
    elif args.phase == "observe":
        review.observe(args.run, "held")
    else:
        audit(args.source, args.run, lock)


if __name__ == "__main__":
    main()
