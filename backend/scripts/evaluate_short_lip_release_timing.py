# SPDX-License-Identifier: Apache-2.0
"""Evaluate unmarked local lip timing on independently qualified short native clips."""

from __future__ import annotations

import argparse
import math
import wave
from collections import Counter
from contextlib import closing
from dataclasses import asdict
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt

from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence.video_reference_timing import _mesh
from scripts import grid_fresh_release_labels as publisher_check
from scripts import qualify_lombard_mouth_events as prior
from scripts import short_lip_release_timing as candidate
from scripts.video_common_sources_v89 import read, require, sha, write

RATE = candidate.RATE
POSITIVE_ROLES = ("local_earlier", "local_later")
CONTROL_ROLES = (
    "unaltered",
    "local_gain",
    "local_filter",
    "local_background",
    "global_earlier",
    "global_later",
    "paired_edit",
)
SPEC = {
    "development": "Previously exposed Lombard sources. Use their labels only to set the point and inspect feasibility.",
    "held": "12 new publisher speaker groups with two plain and two Lombard originals each. Independently qualify one acoustic/visible source event; no detector flags influence admission.",
    "edits": "Move sound by five or seven native picture periods (roughly ±208 or ±292 ms) locally for 640 ms around the known burst, keeping every audio sample outside the blend unchanged. Freeze roles before scoring.",
    "controls": "Original, local gain, mild equalization, deterministic pink-like background, both whole-track offsets, and synchronized delayed mouth and speech in matched prepared signals.",
    "match": "Estimate an acoustic onset and mouth reopening in their native clocks. Each must be within two native frames or 80 ms of its known event and relative signed error at most 80 ms.",
    "advance": "At least 8 useful event pairs across four groups, at least two per sign, >=35% assessed coverage, no more than one ordinary false event, zero on unchanged/global/synchronized controls.",
    "calibration": "Fixed ambiguity margin .12. Review score is max(.22, highest ordinary eligible development score + .025).",
}


def sources(folder):
    receipt = read(folder / "source-receipt.json")
    require(receipt["freeze_sha256"] == sha(folder / "freeze.json"), "Input receipt differs from source plan")
    return receipt["rows"]


def source_pcm(path):
    """Use the exact native mono waveform with lossless signed-16 to float conversion."""
    try:
        raw, pcm = prior.native_pcm(path)
    except ValueError:
        with wave.open(str(path), "rb") as sound:
            require(
                (sound.getnchannels(), sound.getframerate(), sound.getsampwidth(), sound.getcomptype())
                == (1, RATE, 2, "NONE"),
                "Unsupported source waveform representation",
            )
            n = sound.getnframes()
            raw = sound.readframes(n)
            require(20000 <= n <= 64000 and len(raw) == 2 * n, "Incomplete native signed-16 source")
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
        return raw, pcm, 2
    return raw, pcm, 4


def observe(folder, run, split):
    rows = sources(folder)
    mod, task = prior.observer.mp_task()
    cv2.setNumThreads(1)
    output = []
    with closing(task):
        for row in rows:
            k = row["ordinal"]
            front = prior.source(folder, row, "front")
            wav = prior.source(folder, row, "audio")
            table = probe_frames(front, timeout_s=15)
            raw, pcm, width = source_pcm(wav)
            times = [p * table.tb for p in table.pts]
            steps = set(b - a for a, b in pairwise(times))
            require(
                len(steps) == 1
                and times[0] == 0
                and not (table.rotation or table.interpolated or table.truncated),
                "Expected an exact zero-origin native picture clock",
            )
            step = steps.pop()
            require(Fraction(1, 25) <= step <= Fraction(2, 47), "Unsupported native short picture rate")
            require(
                max(times[-1], Fraction(len(pcm), RATE)) <= Fraction(17, 5),
                "The test is bounded to true short clips",
            )
            apertures = [None] * len(table)
            anchors = {}
            for frame in decode_frames(front, table, None, timeout_s=25):
                i = frame.index
                require(
                    frame.pts_verified and frame.pts == table.pts[i],
                    "Picture PTS differs on independent readback",
                )
                result = _mesh(task, mod, frame.rgb)
                if result is None:
                    continue
                values, box = result
                apertures[i] = float(np.median(values[:3]))
                x0, y0, x1, y1 = box
                anchors[i] = {
                    "frame": i,
                    "pts": frame.pts,
                    "full_rgb_sha256": prior.digest(frame.rgb.tobytes()),
                    "xyxy": box,
                    "crop_rgb_sha256": prior.digest(frame.rgb[y0:y1, x0:x1].tobytes()),
                }
            hint = Fraction(row["aligned_following_phone_s"])
            sound = publisher_check.audio_only(pcm.astype(np.float64) * 32768, round(hint * RATE))
            vision = prior.visual_releases(apertures, times, hint)
            truth = None
            if "range_samples" in sound and "frames" in vision:
                a, b = sound["range_samples"]
                av = Fraction(a + b, 2 * RATE)
                va = sum((times[i] for i in vision["frames"]), Fraction(0)) / 2
                if abs(av - va) <= Fraction(2, 25) and Fraction(14, 25) <= av <= Fraction(
                    len(pcm), RATE
                ) - Fraction(16, 25):
                    lo = max(3200, min(len(pcm) - 13440, (a + b) // 2 - 5120))
                    truth = {
                        "acoustic_samples": [a, b],
                        "visual_frames": vision["frames"],
                        "edit_samples": [lo, lo + 10240],
                        "acoustic_excerpt_sha256": prior.digest(raw[a * width : b * width]),
                    }
            output.append(
                {
                    "ordinal": k,
                    "group": row["group"],
                    "style": row["condition"],
                    "video_sha256": sha(front),
                    "audio_sha256": sha(wav),
                    "audio_samples": len(pcm),
                    "source_sample_bytes": width,
                    "audio_pcm_sha256": prior.digest(raw),
                    "time_base": table.time_base,
                    "pts": list(table.pts),
                    "apertures": apertures,
                    "anchors": anchors,
                    "truth": truth,
                }
            )
            print(split, k, len(anchors), "event" if truth else "control only", flush=True)
    write(
        run / f"observations-{split}.json",
        {"source_receipt_sha256": sha(folder / "source-receipt.json"), "rows": output},
    )


def shift(pcm, delta):
    out = np.zeros_like(pcm)
    if delta > 0:
        out[delta:] = pcm[:-delta]
    else:
        out[:delta] = pcm[-delta:]
    return out


def blend(pcm, replacement, span):
    a, b = span
    gain = np.ones(b - a, dtype=float)
    edge = np.arange(1, 81) / 80
    gain[:80], gain[-80:] = edge, edge[::-1]
    out = pcm.copy()
    out[a:b] = pcm[a:b] * (1 - gain) + replacement[a:b] * gain
    return out


def variants(pcm, apertures, times, obs):
    truth = obs["truth"]
    if truth:
        a, b = truth["edit_samples"]
    else:
        a = (len(pcm) - 10240) // 2
        b = a + 10240
    interval = (a, b)
    steps = 5 if obs["ordinal"] % 2 else 7
    tick = times[1] - times[0]
    offset = round(steps * tick * RATE)
    yield "unaltered", pcm, apertures, None
    yield "local_gain", blend(pcm, 0.55 * pcm, interval), apertures, None
    filtered = sosfiltfilt(butter(3, 2700, fs=RATE, output="sos"), pcm)
    yield "local_filter", blend(pcm, filtered, interval), apertures, None
    rng = np.random.default_rng(1000 + obs["ordinal"])
    noise = sosfiltfilt(
        butter(2, (100, 5500), fs=RATE, output="sos", btype="bandpass"), rng.normal(size=len(pcm))
    )
    noise *= (
        float(np.sqrt(np.mean(pcm[a:b] ** 2))) / max(1e-9, float(np.sqrt(np.mean(noise[a:b] ** 2)))) * 0.33
    )
    yield "local_background", blend(pcm, pcm + noise, interval), apertures, None
    yield "global_earlier", shift(pcm, -offset), apertures, None
    yield "global_later", shift(pcm, offset), apertures, None
    paired = list(apertures)
    for i, t in enumerate(times):
        if a <= t * RATE < b and i >= steps:
            paired[i] = apertures[i - steps]
    yield "paired_edit", blend(pcm, shift(pcm, offset), interval), paired, None
    if truth:
        av = truth["acoustic_samples"]
        event = sum(av) / (2 * RATE)
        middle = sum(times[i] for i in truth["visual_frames"]) / 2
        for role, delta in (("local_earlier", -offset), ("local_later", offset)):
            require(
                a + 80 <= av[0] + delta < av[1] + delta <= b - 80,
                "Shifted acoustic event outside edit interior",
            )
            actual = blend(pcm, shift(pcm, delta), interval)
            require(
                np.array_equal(actual[:a], pcm[:a]) and np.array_equal(actual[b:], pcm[b:]),
                "Changed sound outside the authored span",
            )
            moved = [v + delta for v in av]
            require(
                np.array_equal(actual[slice(*moved)], pcm[slice(*av)]), "Known source event waveform differs"
            )
            yield (
                role,
                actual,
                apertures,
                {
                    "mouth_s": middle,
                    "audio_s": event + delta / RATE,
                    "shift_s": delta / RATE,
                    "frame_bracket": truth["visual_frames"],
                    "audio_bracket": moved,
                    "edit_samples": interval,
                },
            )


def score(folder, run, split):
    ledger = read(run / f"observations-{split}.json")
    require(ledger["source_receipt_sha256"] == sha(folder / "source-receipt.json"), "Source roster changed")
    lock_path = run / "method-freeze.json"
    if split == "held":
        lock = read(lock_path)
        require(
            lock["spec"] == SPEC
            and lock["worker_sha256"] == sha(Path(__file__))
            and lock["candidate_sha256"] == sha(Path(candidate.__file__))
            and lock["development_sha256"] == sha(run / "development.json"),
            "Frozen detector or calibration changed",
        )
    by_id = {r["ordinal"]: r for r in sources(folder)}
    output = []
    for obs in ledger["rows"]:
        k = obs["ordinal"]
        row = by_id[k]
        raw, pcm, width = source_pcm(prior.source(folder, row, "audio"))
        require(
            prior.digest(raw) == obs["audio_pcm_sha256"] and width == obs["source_sample_bytes"],
            "Source samples or sample representation differ",
        )
        times = [float(t * Fraction(obs["time_base"])) for t in obs["pts"]]
        for role, waveform, apertures, truth in variants(pcm, obs["apertures"], times, obs):
            video, sound = candidate.visual(apertures, times), candidate.acoustic(waveform)
            result = candidate.assess(video, sound)
            flag = result["candidate"]
            correct = bool(
                flag
                and truth
                and abs(flag["event_seconds"][0] - truth["mouth_s"]) <= 0.08
                and abs(flag["event_seconds"][1] - truth["audio_s"]) <= 0.08
                and abs(flag["shift_s"] - truth["shift_s"]) <= 0.08
            )
            record = {
                "ordinal": k,
                "group": obs["group"],
                "style": obs["style"],
                "role": role,
                "truth": truth,
                "sound_pcm_sha256": prior.digest(waveform.astype("<f4").tobytes()),
                "video_events": [asdict(v) for v in video],
                "sound_events": [asdict(v) for v in sound],
                "measured": result["measured"],
                "reason": result["reason"],
                "candidate": flag,
                "correct": correct,
            }
            if split == "held":
                record["flag"] = accepted(flag, lock["point"])
            output.append(record)
        print("scored", split, k, "visual events", len(video), flush=True)
    write(
        run / f"{split}.json",
        {
            "observation_sha256": sha(run / f"observations-{split}.json"),
            "method_freeze_sha256": sha(lock_path) if split == "held" else None,
            "cases": output,
        },
    )


def accepted(flag, point):
    return bool(flag and flag["alternative_margin"] >= 0.12 and flag["score"] > point)


def calibrate(run):
    path = run / "development.json"
    data = read(path)["cases"]
    best = max(
        (
            r["candidate"]["score"]
            for r in data
            if r["role"] in CONTROL_ROLES and r["candidate"] and r["candidate"]["alternative_margin"] >= 0.12
        ),
        default=0.0,
    )
    point = max(0.22, best + 0.025)
    require(math.isfinite(point), "Unusable calibration score")
    write(
        run / "method-freeze.json",
        {
            "spec": SPEC,
            "candidate": candidate.SPEC,
            "point": point,
            "development_sha256": sha(path),
            "worker_sha256": sha(Path(__file__)),
            "candidate_sha256": sha(Path(candidate.__file__)),
        },
    )
    print(
        "point",
        point,
        "dev useful",
        sum(r["correct"] and accepted(r["candidate"], point) for r in data),
        "raw useful",
        sum(r["correct"] for r in data),
        "roles",
        Counter(r["role"] for r in data),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("observe", "score", "calibrate"))
    p.add_argument("folder", type=Path)
    p.add_argument("run", type=Path)
    p.add_argument("--split", choices=("development", "held"), default="development")
    a = p.parse_args()
    if a.phase == "calibrate":
        calibrate(a.run)
    else:
        a.run.mkdir(mode=0o700, exist_ok=True)
        (observe if a.phase == "observe" else score)(a.folder, a.run, a.split)


if __name__ == "__main__":
    main()
