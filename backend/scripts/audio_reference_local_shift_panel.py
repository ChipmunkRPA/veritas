# SPDX-License-Identifier: Apache-2.0
"""Transfer the existing matched-waveform timing calculation to standalone audio pairs."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from app.detectors.audio.decode import wav_bytes
from app.evidence.audio_reference import read_track
from app.evidence.video_reference_soundtrack import MIN_SAMPLES, measure
from scripts import qualify_lombard_mouth_events as qualification
from scripts import review_video_reference_soundtrack_panel as independent
from scripts.video_common_sources_v89 import read, require, sha, write
from scripts.video_reference_soundtrack_panel import _mix, shift

RATE = 16000
WIDTH = 10240
DELIVERIES = {"pcm": ("wav", "pcm_s16le"), "aac": ("m4a", "aac"), "flac": ("flac", "flac")}
POSITIVES = ("local_later", "local_earlier")
ORDINARY = (
    "unchanged",
    "gain",
    "equalization",
    "whole_track_shift",
    "short_offset",
    "unrelated_insert",
    "silence_local",
    "silence_all",
    "preexisting_repeat",
    "codec_flac",
    "codec_aac",
)
PROTOCOL = {
    "truth": "Two 640 ms local moves, signed shifts of 160, 200, 240, or 280 ms. 5 ms linear crossfades preserve all outside samples. Reported range must cover at least 0.60 temporal IoU of the authored submitted span, with <=16 sample signed error.",
    "schedule": "Center on the independently predeclared publisher forced-phone endpoint. Clamp to leave 400 ms of flank at each side. No detector result selects clips or edit intervals.",
    "controls": "Unchanged, local gain, local EQ, constant whole-track offset, local 40 ms offset, insertion from a different speaker, local silence, both silent, an unchanged pair containing a deliberate short repeat, and same-content cross-codec pair comparisons.",
    "delivery": "Independent PCM WAV or AAC M4A encoding for each side. Compare only equal decoded length and relative start. Check unchanged WAV to FLAC and WAV to AAC separately on each utterance.",
    "gate": "No point fitting. Per codec require at least 30/48 correct, at least 12/24 for each sign, at least five held groups, zero alerts on unchanged, repeat, or whole-track shift, at most one total ordinary alert, and >=75% measured non-silent pairs.",
    "scope": "Same-waveform movement between two supplied recordings. Independent readback of derived source clocks, samples, and timing decisions. No speech origin, authenticity, or AI inference.",
}


def pcm_hash(pcm):
    return hashlib.sha256(pcm.astype("<f4", copy=False).tobytes()).hexdigest()


def freeze(sources):
    received = read(sources / "source-receipt.json")
    require(received["freeze_sha256"] == sha(sources / "freeze.json"), "Original source roster differs")
    rows = received["rows"]
    require(
        len(rows) == 32 and len({r["group"] for r in rows if r["split"] == "held"}) == 6,
        "Missing held source groups",
    )
    return {
        "schema": "chipmunk.audio-reference-local-shift-freeze/1",
        "protocol": PROTOCOL,
        "source_receipt_sha256": sha(sources / "source-receipt.json"),
        "rows": rows,
        "worker_sha256": sha(Path(__file__)),
        "method_sha256": sha(Path(measure.__code__.co_filename)),
        "ffmpeg": command("ffmpeg", "-version").decode().splitlines()[0],
    }


def command(*args):
    return subprocess.check_output(args, timeout=15, stdin=subprocess.DEVNULL)


def load(sources, row):
    path = qualification.source(sources, row, "audio")
    _, sample = qualification.native_pcm(path)
    return np.clip(np.rint(sample.astype(np.float64) * 32768), -32768, 32767).astype("<i2")


def schedule(row, pcm, donor):
    hint = round(Fraction(row["aligned_following_phone_s"]) * RATE / 640) * 640
    left = max(6400, min(len(pcm) - 6400 - WIDTH, hint - WIDTH // 2))
    right = left + WIDTH
    span = (left, right)
    require(right + 6400 <= len(pcm), "Original lacks the frozen flank budget")
    delta = (2560, 3200, 3840, 4480)[row["ordinal"] % 4]
    silence = np.zeros_like(pcm)
    yield "unchanged", pcm, pcm, None, None
    for role, offset in (("local_later", delta), ("local_earlier", -delta)):
        yield role, pcm, _mix(pcm, shift(pcm, offset), span), span, offset
    yield "gain", pcm, _mix(pcm, pcm.astype(float) * 0.61, span), None, None
    equalized = sosfiltfilt(butter(4, 2500, fs=RATE, output="sos"), pcm)
    yield "equalization", pcm, _mix(pcm, equalized, span), None, None
    yield "whole_track_shift", pcm, shift(pcm, delta), None, None
    yield "short_offset", pcm, _mix(pcm, shift(pcm, 640), span), None, None
    yield "unrelated_insert", pcm, _mix(pcm, np.resize(donor, len(pcm)), span), None, None
    yield "silence_local", pcm, _mix(pcm, silence, span), None, None
    yield "silence_all", silence, silence, None, None
    repeated = pcm.copy()
    repeated[left + delta : left + 2 * delta] = pcm[left : left + delta]
    yield "preexisting_repeat", repeated, repeated, None, None


def stored(folder, label, codec, values):
    suffix, encoder = DELIVERIES[codec]
    path = folder / f"{label}.{suffix}"
    raw = wav_bytes(values.astype(np.float32) / 32768)
    if codec == "pcm":
        path.write_bytes(raw)
    else:
        subprocess.run(
            (
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-f",
                "wav",
                "-i",
                "pipe:0",
                "-c:a",
                encoder,
                *(("-b:a", "48k") if codec == "aac" else ()),
                str(path),
            ),
            input=raw,
            check=True,
            capture_output=True,
            timeout=15,
        )
    return path


def inspect(path):
    digest = sha(path)
    state, pcm, meta = read_track(path, {"sha256": digest}, max_seconds=12)
    raw = command(
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "s16le",
        "-",
    )
    ordinary = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    if state == "measured":
        require(np.array_equal(ordinary, pcm), "Independent PCM readback differs")
    return (
        state,
        pcm,
        meta,
        {
            "file_sha256": digest,
            "native_readback_samples": len(ordinary),
            "native_readback_sha256": pcm_hash(ordinary),
        },
    )


def compare(left, right):
    record = {
        label: {"decoder_status": v[0], "clock": v[2], **v[3]}
        for label, v in (("earlier", left), ("submitted", right))
    }
    if left[0] != right[0] or left[0] != "measured":
        return record | {"status": "unverified_clock", "support": 0, "range": None}
    keys = ("relative_start", "samples")
    if tuple(left[2][k] for k in keys) != tuple(right[2][k] for k in keys):
        return record | {"status": "different_clock", "support": 0, "range": None}
    if len(left[1]) < MIN_SAMPLES:
        return record | {"status": "too_short", "support": 0, "range": None}
    actual, support = measure(left[1], right[1])
    checked, count = independent.localize(left[1], right[1])
    expected = [actual["submitted_samples"], actual["shift_samples"]] if actual else None
    require(checked == expected and count == support, "Independent waveform arithmetic differs")
    status = "insufficient_support" if support < 8 else "local_shift" if actual else "no_local_shift"
    if status == "local_shift":
        actual["clip_pcm_sha256"] = {
            label: pcm_hash(signal[slice(*actual[f"{label}_samples"])])
            for label, signal in (("earlier", left[1]), ("submitted", right[1]))
        }
    return record | {
        "status": status,
        "support": support,
        "range": actual if status == "local_shift" else None,
    }


def score(run, sources, rows):
    cases = []
    chosen = {r["ordinal"]: load(sources, r) for r in rows}
    for row in rows:
        i = row["ordinal"]
        pcm = chosen[i]
        donor = chosen[next(r["ordinal"] for r in rows if r["group"] != row["group"])]
        local = run / f"case-{i:02}"
        local.mkdir(mode=0o700)
        direct = {}
        current = []
        for role, left, right, truth, offset in schedule(row, pcm, donor):
            for codec in ("pcm", "aac"):
                if role == "unchanged" or left is not pcm:
                    earlier = inspect(stored(local, f"{role}-{codec}-earlier", codec, left))
                    if role == "unchanged":
                        direct[codec] = earlier
                else:
                    earlier = direct[codec]
                later = inspect(stored(local, f"{role}-{codec}-submitted", codec, right))
                record = compare(earlier, later)
                current.append({"carrier": codec, "role": role, "truth": truth, "offset": offset, **record})
        for codec in ("flac", "aac"):
            later = direct["aac"] if codec == "aac" else inspect(stored(local, "flac-unchanged", "flac", pcm))
            current.append(
                {
                    "carrier": "cross",
                    "role": "codec_" + codec,
                    "truth": None,
                    "offset": None,
                    **compare(direct["pcm"], later),
                }
            )
        cases.append(
            {
                "group": row["group"],
                "ordinal": i,
                "source_sha256": row["inputs"]["audio"]["sha256"],
                "cases": current,
            }
        )
        print(
            i,
            [
                (r["carrier"], r["role"], r["status"])
                for r in current
                if r["status"] not in ("no_local_shift", "insufficient_support")
            ],
            flush=True,
        )
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "development", "held"))
    p.add_argument("run", type=Path)
    p.add_argument("sources", type=Path)
    a = p.parse_args()
    lock = freeze(a.sources)
    if a.phase == "freeze":
        a.run.mkdir(mode=0o700)
        write(a.run / "freeze.json", lock)
    else:
        require(lock == read(a.run / "freeze.json"), "Study or exact source roster differs")
        rows = [r for r in lock["rows"] if r["split"] == a.phase]
        write(
            a.run / f"{a.phase}.json",
            {"freeze_sha256": sha(a.run / "freeze.json"), "rows": score(a.run, a.sources, rows)},
        )


if __name__ == "__main__":
    main()
