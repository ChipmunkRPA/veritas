# SPDX-License-Identifier: Apache-2.0
"""Build source-bound mouth timing controls with exact rational picture clocks."""

from __future__ import annotations

import argparse
import hmac
import secrets
import struct
from collections import Counter
from contextlib import closing
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.evidence.video_reference_timing import _mesh
from scripts import build_lombard_neural_mouth as build
from scripts import evaluate_short_lip_release_timing as native
from scripts import qualify_lombard_mouth_events as original
from scripts.video_common_sources_v89 import read, require, sha, write

ROLES = {"original": 0, "resample": 0, "mouth_earlier": -5, "mouth_later": 5}
SPEC = {
    "cohort": "All prequalified fresh Lombard stop releases. Preserve every fixed role and each reason for unavailable geometry or observations.",
    "timing": "Reconstruct source mouth texture at the same time, five native pictures earlier, or five later. Keep audio unchanged. Use the same mouth alpha, registered crop and half-opacity edge geometry over [anchor-7, anchor+9).",
    "observer": "The separate stateless visual pass gets opaque case IDs, the finished RGB pictures, and the frozen observation neighborhood. It receives no role or edited-pixel mask. Rejoin cases only after recording the event, ±1 native picture tolerance.",
    "delivery": "Retain lossless native MOV on two preselected sources in distinct groups and styles. Preserve exact full native clocks, source picture hashes, float32 PCM and geometry mask proof. One 76-picture source stays in the ledger at its native length.",
}


def binding(sources, study):
    observations = study / "observations-held.json"
    source = sources / "source-receipt.json"
    doc = read(observations)
    require(doc["source_receipt_sha256"] == sha(source), "Fresh native source observations differ")
    qualified = [r for r in doc["rows"] if r["truth"]]
    require(
        len(qualified) == 20 and len({r["group"] for r in qualified}) == 9, "Expected frozen qualified roster"
    )
    return {
        "spec": SPEC,
        "observations_sha256": sha(observations),
        "source_receipt_sha256": sha(source),
        "tool_sha256": sha(Path(__file__)),
        "roles": ROLES,
        "ordinals": [r["ordinal"] for r in qualified],
        "retained_delivery_ordinals": [
            qualified[0]["ordinal"],
            next(
                r["ordinal"]
                for r in qualified
                if r["group"] != qualified[0]["group"] and r["style"] != qualified[0]["style"]
            ),
        ],
    }


def float_wav(path, pcm):
    data = pcm.astype("<f4", copy=False).tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, 16000, 64000, 4, 32)
    payload = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    with path.open("xb") as stream:
        stream.write(b"RIFF" + struct.pack("<I", len(payload)) + payload)


def geometry(crops, source, table, span, shift):
    changed = list(source)
    masks = []
    for i in range(*span):
        _, matrix, alpha = crops[i]
        driver = i - shift
        texture = cv2.resize(crops[driver][0], (512, 512), interpolation=cv2.INTER_LINEAR)
        target = cv2.warpAffine(
            texture,
            cv2.invertAffineTransform(matrix),
            (table.width, table.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        image, mask = build.portrait.composite(
            source[i], target, alpha, 0.5 if i in (span[0], span[1] - 1) else 1.0
        )
        inside = mask != 0
        require(inside.any() and not inside[alpha == 0].any(), "Changed pixels outside the declared crop")
        y, x = np.where(inside)
        masks.append(
            {
                "index": i,
                "pts": table.pts[i],
                "driver_index": driver,
                "driver_pts": table.pts[driver],
                "rgb_sha256": original.digest(image.tobytes()),
                "mask_sha256": original.digest(mask.tobytes()),
                "alpha_sha256": original.digest(alpha.tobytes()),
                "changed_pixels": int(np.count_nonzero(inside)),
                "outside_crop_changed_pixels": int(np.count_nonzero(inside & (alpha == 0))),
                "changed_xyxy": [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1],
            }
        )
        changed[i] = image
    return changed, masks


def blind_picture_observation(pictures, table, observer_task, mod, anchor, hint):
    values = [None] * len(pictures)
    observed = {}
    for i in range(max(0, anchor - 17), min(len(pictures), anchor + 18)):
        rgb = pictures[i]
        item = _mesh(observer_task, mod, rgb)
        if item is None:
            continue
        values[i] = float(np.median(item[0][:3]))
        x0, y0, x1, y1 = item[1]
        observed[i] = {
            "vector": item[0].tolist(),
            "aperture": values[i],
            "box_xyxy": item[1],
            "rgb_sha256": original.digest(rgb.tobytes()),
            "crop_sha256": original.digest(rgb[y0:y1, x0:x1].tobytes()),
        }
    times = [v * table.tb for v in table.pts]
    step = times[1] - times[0]
    events = {
        shift: original.visual_releases(values, times, hint + shift * step) for shift in set(ROLES.values())
    }
    return observed, events


def verify_encoded(path, row, source, pixels):
    table, _ = build.table_for(path, row)
    readback = build.read_pictures(path, table)
    require(
        all(np.array_equal(a, b) for a, b in zip(pixels, readback, strict=True)),
        "Lossless delivery changed pixels",
    )
    require(len(source) == len(readback), "Lossless delivery changed duration")
    return build.verified_audio(path, row)


def run(sources, study, dest):
    lock = binding(sources, study)
    require(lock == read(dest / "freeze.json"), "Frozen geometry, source or control plan differs")
    roster = {r["ordinal"]: r for r in native.sources(sources)}
    cv2.setNumThreads(1)
    mod, input_task = build.retained.mp_task()
    _, observer_task = build.retained.mp_task()
    key = secrets.token_bytes(32)
    receipt = []
    with closing(input_task), closing(observer_task):
        for obs in read(study / "observations-held.json")["rows"]:
            k = obs["ordinal"]
            if k not in lock["ordinals"]:
                continue
            item = roster[k]
            input_file = original.source(sources, item, "front")
            sound = original.source(sources, item, "audio")
            raw, pcm, width = native.source_pcm(sound)
            require(
                original.digest(raw) == obs["audio_pcm_sha256"] and width == obs["source_sample_bytes"],
                "Native soundtrack changed",
            )
            committed = {
                "visual": obs,
                "audio": {
                    "samples": len(pcm),
                    "native_pcm_sha256": original.digest(pcm.astype("<f4").tobytes()),
                },
            }
            table, step = build.table_for(input_file, committed)
            source = build.read_pictures(input_file, table)
            anchor = obs["truth"]["visual_frames"][0]
            span = [anchor - 7, anchor + 9]
            require(
                5 <= span[0] < span[1] <= len(source) - 5 and 48 <= len(source) <= 76,
                "Missing native driver or outer flanks",
            )
            crops = build.crop_geometry(source, [span[0], span[1] + 5], mod, input_task)
            timing = {
                "ordinal": k,
                "group": obs["group"],
                "style": obs["style"],
                "native_edit_frames": span,
                "native_edit_seconds": [str(table.pts[i] * table.tb) for i in span],
                "original_release_frame": anchor,
                "original_audio_range_samples": obs["truth"]["acoustic_samples"],
                "original_audio_sha256": obs["audio_sha256"],
                "original_audio_pcm_sha256": committed["audio"]["native_pcm_sha256"],
                "original_audio_samples": len(pcm),
                "original_source_sample_bytes": width,
                "original_rgb_file_sha256": sha(input_file),
                "time_base": table.time_base,
                "pts": list(table.pts),
                "width": table.width,
                "height": table.height,
                "observations": [],
            }
            if crops is None:
                timing["observations"] = [{"role": role, "status": "unavailable_geometry"} for role in ROLES]
                receipt.append(timing)
                continue
            # Sort opaque tokens to keep role order unavailable to the independent observer.
            routes = sorted(
                (hmac.digest(key, f"{k}:{role}".encode(), "sha256").hex()[:24], role, shift)
                for role, shift in ROLES.items()
            )
            hint = Fraction(item["aligned_following_phone_s"])
            retained = k in lock["retained_delivery_ordinals"]
            wav = dest / f"{k:02}-unchanged-audio.wav"
            if retained:
                float_wav(wav, pcm)
            for token, role, shift in routes:
                pictures, masks = (
                    (source, []) if role == "original" else geometry(crops, source, table, span, shift)
                )
                retained_media = None
                if retained:
                    name = f"{k:02}-{role}.mov"
                    output = dest / name
                    build.encode(output, pictures, wav, table, step)
                    exact = verify_encoded(output, committed, source, pictures)
                    pictures = build.read_pictures(output, table)
                    retained_media = {"file": name, "sha256": sha(output), "verified_audio_samples": exact}
                measured, events = blind_picture_observation(
                    pictures, table, observer_task, mod, anchor, hint
                )
                event = events[shift]
                achieved = event["frames"][0] - anchor if "frames" in event else None
                supported = achieved is not None and abs(achieved - shift) <= 1
                if role == "original":
                    require(achieved == 0, "Independent original event differs from qualified source")
                timing["observations"].append(
                    {
                        "role": role,
                        "token": token,
                        "status": "authored",
                        "intended_picture_shift": shift,
                        "observed_picture_shift": achieved,
                        "qualification": "observed"
                        if supported
                        else event["status"]
                        if achieved is None
                        else "different_observed_motion",
                        "event": event,
                        "observed_neighborhoods": events,
                        "geometry": measured,
                        "changed_frames": masks,
                        "full_delivery": retained_media,
                        "before_rgb_sha256": original.digest(pictures[span[0] - 1].tobytes()),
                        "after_rgb_sha256": original.digest(pictures[span[1]].tobytes()),
                    }
                )
                print(k, role, achieved, supported, flush=True)
            require(
                all(
                    r["before_rgb_sha256"] == timing["observations"][0]["before_rgb_sha256"]
                    and r["after_rgb_sha256"] == timing["observations"][0]["after_rgb_sha256"]
                    for r in timing["observations"]
                ),
                "Matched picture flanks differ",
            )
            receipt.append(timing)
    counts = Counter(
        (r["role"], r.get("qualification", r["status"])) for case in receipt for r in case["observations"]
    )
    write(
        dest / "visual-challenge.json",
        {
            "freeze_sha256": sha(dest / "freeze.json"),
            "cases": receipt,
            "counts": [[*k, v] for k, v in sorted(counts.items())],
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "run"))
    p.add_argument("sources", type=Path)
    p.add_argument("study", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    if a.phase == "freeze":
        a.output.mkdir(mode=0o700)
        write(a.output / "freeze.json", binding(a.sources, a.study))
    else:
        run(a.sources, a.study, a.output)


if __name__ == "__main__":
    main()
