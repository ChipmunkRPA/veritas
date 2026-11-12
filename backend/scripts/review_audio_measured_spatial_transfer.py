"""Recompute measured-room PCM, source support, and detection outcomes."""

from __future__ import annotations

import argparse
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np
import soundfile as sf
from review_audio_original_pcm_boundary_transfer import RATE, counts, digest, read, require, sha
from review_audio_variable_boundary_transfer import actual_overlap
from scipy.fft import next_fast_len
from scipy.io import loadmat
from scipy.signal import resample_poly

LENGTH = 6 * RATE
GUARD = 2 * RATE
WINDOWS = [[RATE, 3 * RATE], [3 * RATE, 5 * RATE]]
ARMS = ("original", "spatial")
STEMS = {"original": 1, "spatial": 1, "stereo": 2, "foreground": 2, "background": 2}


def room_responses(archive, lock):
    def pinned(path, record):
        raw = path.read_bytes()
        require(
            digest(raw) == record["sha256"] and len(raw) == record["bytes"],
            "Measured publisher asset differs",
        )
        return raw

    rooms = lock["rooms"]
    surrey = rooms["surrey"]["asset"]
    mat = pinned(archive / "audio-room-assets-v299/development" / surrey["filename"], surrey)
    values = loadmat(io.BytesIO(mat), variable_names=["brir"])["brir"]
    require(values.shape == (65536, 2, 24), "Unexpected native Surrey shape")
    source = {
        "surrey": dict(zip(("front", "left", "right"), (values[:, :, i] for i in (2, 10, 11)), strict=True)),
        "bbc": {},
    }
    for direction, info in rooms["bbc"]["positions"].items():
        asset = info["asset"]
        wav = pinned(archive / "audio-bbc-room-assets-v303" / asset["filename"], asset)
        response, rate = sf.read(io.BytesIO(wav), dtype="float64", always_2d=True)
        require(rate == 48000 and response.shape == (16384, 2), "Unexpected native BBC shape")
        source["bbc"][direction] = response

    spectra = {}
    for rig, positions in source.items():
        spectra[rig] = {}
        for side, response in positions.items():
            resampled = np.ascontiguousarray(resample_poly(response, 1, 3, axis=0), dtype="<f8")
            declared = rooms[rig]["positions"][side]
            require(
                declared["rate_hz"] == RATE
                and declared["samples"] == len(resampled) <= GUARD
                and declared["render_sha256"] == digest(resampled.tobytes())
                and declared["mean_peak_sample"] == int(np.argmax(abs(resampled.mean(axis=1)))),
                "Resampled room response or source clock differs",
            )
            spectra[rig][side] = np.fft.rfft(resampled, n=next_fast_len(LENGTH + 2 * GUARD), axis=0)
    return spectra


def apply(values, room, direction):
    n = next_fast_len(LENGTH + 2 * GUARD)
    time_guard = np.pad(values, ((GUARD, n - LENGTH - GUARD), (0, 0)))
    y_left, y_right = np.fft.rfft(time_guard, axis=0).T
    t_left, t_right = room["front"].T
    s_left, s_right = room[direction].T
    t_energy = np.maximum(abs(t_left) ** 2 + abs(t_right) ** 2, 1e-20)
    determinant = (t_right * s_left - t_left * s_right) / np.sqrt(t_energy)
    weight = (s_left + s_right) * determinant.conjugate() / 2
    weight /= abs(determinant) ** 2 + 0.04 * (abs(s_left) ** 2 + abs(s_right) ** 2) + 1e-16
    weight /= np.maximum(1, abs(weight) / 1.5)
    frequencies = np.fft.rfftfreq(n, 1 / RATE)
    low = np.sin(np.pi / 2 * np.clip((frequencies - 100) / 100, 0, 1)) ** 2
    high = np.cos(np.pi / 2 * np.clip((frequencies - 6000) / 800, 0, 1)) ** 2
    cancel = (t_right * y_left - t_left * y_right) / np.sqrt(t_energy)
    return np.fft.irfft((y_left + y_right) / 2 - low * high * weight * cancel, n=n)[GUARD : GUARD + LENGTH]


def exact_audio(run, row, stem):
    raw = (run / "pcm" / f"{row['id']}-{stem}.pcm").read_bytes()
    require(len(raw) == 2 * LENGTH * STEMS[stem] and digest(raw) == row["pcm"][stem], "A native stem changed")
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    return pcm.reshape(LENGTH, 2) if STEMS[stem] == 2 else pcm


def power_ratio(a, b):
    return 10 * math.log10(max(np.dot(a, a), 1e-16) / max(np.dot(b, b), 1e-16))


def verify_signals(run, rows, spectra, lock):
    measurements = defaultdict(list)
    captures = defaultdict(list)
    groups = defaultdict(list)
    maximum_rounding = 0.0
    for row in rows:
        data = {stem: exact_audio(run, row, stem) for stem in STEMS}
        groups[row["group"]].append((row, data["foreground"], data["background"]))
        source, stereo, spatial = data["foreground"], data["stereo"], data["spatial"]
        require(
            np.array_equal(source + data["background"], stereo)
            and np.array_equal(data["original"], np.rint(stereo.mean(axis=1))),
            "Stereo PCM does not equal its committed sources",
        )
        require(
            all(np.max(abs(data[key])) < 32768 for key in ("stereo", "original", "spatial")),
            "Rendered PCM clips",
        )
        rig, direction = row["rig"], row["position"]
        altered = apply(stereo, spectra[rig], direction)
        maximum_rounding = max(maximum_rounding, float(np.max(abs(np.rint(altered) - spatial))))
        require(np.max(abs(np.rint(altered) - spatial)) <= 1, "Spatial PCM fails the independent FFT replay")
        if row["truth"] is None:
            continue
        a, b = row["truth"]
        ref = source.mean(axis=1)
        wrong_side = "right" if direction == "left" else "left"
        wrong_rig = "bbc" if rig == "surrey" else "surrey"
        outputs = {
            "original": data["original"],
            "spatial": spatial,
            "wrong_side": apply(stereo, spectra[rig], wrong_side),
            "wrong_rig": apply(stereo, spectra[wrong_rig], direction),
        }
        for key, observed in outputs.items():
            for field, selection in (("native_snr_db", slice(None)), ("range_snr_db", slice(a, b))):
                error = (observed - ref)[selection]
                require(
                    math.isclose(
                        power_ratio(ref[selection], error), row["acoustic"][field][key], abs_tol=1e-6
                    ),
                    "Whole-clip or exact-edit reconstruction result differs",
                )
        residual = apply(source, spectra[rig], direction) - ref
        require(
            math.isclose(power_ratio(ref, residual), row["acoustic"]["target_preservation_db"], abs_tol=1e-6),
            "Measured target preservation differs",
        )
        measurements[row["panel"]].append(row["acoustic"])
    voice_ids, fragments = set(), set()
    for siblings in groups.values():
        first, foreground, background = siblings[0]
        room, start = (first["background"][key] for key in ("room", "capture_start_sample"))
        require(
            all(stop <= start or start + LENGTH <= prior for prior, stop in captures[room]),
            "Background PCM interval repeats",
        )
        captures[room].append((start, start + LENGTH))
        if first["authored_range"] is None:
            require(
                len(siblings) == 1 and first["arm"] == "background_only" and not np.any(foreground),
                "Room-only control has a foreground",
            )
            continue
        require(first["arm"] == "untouched" and len(siblings) == 4, "A matched four-view group differs")
        a, b = first["authored_range"]
        require(
            b - a in (12800, 15360) and any(start <= a < b <= end for start, end in WINDOWS),
            "Authored range differs",
        )
        identities = {row["donor"]["canonical_speaker"] for row, _, _ in siblings if row["donor"]}
        require(
            len(identities) == 3 and not identities & voice_ids, "An exact host or donor identity repeats"
        )
        voice_ids.update(identities)
        for row, matched, noise in siblings[1:]:
            donor = row["donor"]
            cut = donor["cut"]
            limit = min(LENGTH, b + lock["rooms"][row["rig"]]["positions"]["front"]["samples"] - 1)
            require(
                row["room_causal_support"] == [a, limit]
                and a <= row["change_energy_range"][0] < row["change_energy_range"][1] <= limit
                and np.array_equal(matched[:a], foreground[:a])
                and np.array_equal(matched[limit:], foreground[limit:])
                and np.array_equal(noise, background)
                and row["background"] == first["background"]
                and row["truth"] == ([a, b] if row["arm"] == "synthetic" else None),
                "Matched source or causal clock differs",
            )
            require(
                cut["time_base"] == "1/16000"
                and cut["end_sample"] - cut["start_sample"]
                == cut["source_end_pts"] - cut["source_start_pts"]
                == b - a
                and donor["pcm_sha256"] not in fragments,
                "Exact donor fragment or source clock repeats",
            )
            fragments.add(donor["pcm_sha256"])
    require(len(voice_ids) == len(fragments) == 36 and len(groups) == 16, "Source independence is incomplete")
    for panel, edits in measurements.items():
        gains = {
            field: {
                arm: median(row[field][arm] - row[field]["original"] for row in edits)
                for arm in ("spatial", "wrong_side", "wrong_rig")
            }
            for field in ("native_snr_db", "range_snr_db")
        }
        target = median(row["target_preservation_db"] for row in edits)
        expected = {
            "gain_db": gains,
            "target_preservation_db": target,
            "pass": gains["native_snr_db"]["spatial"] >= 4
            and gains["range_snr_db"]["spatial"] >= 3
            and target >= 35,
        }
        require(
            len(edits) == 6 and expected == lock["acoustics"][panel], "Frozen acoustic advancement differs"
        )
    return maximum_rounding


def verify_scores(run, roster, lock):
    points, summary = (read(run / f"{name}.json") for name in ("points", "result"))
    require(
        all(sha(run / f"{name}.json") == pin for name, pin in summary["commitments"].items())
        and points["selection_sha256"] == sha(run / "selection.json")
        and points["freeze_sha256"] == sha(run / "freeze.json"),
        "Model commitments differ",
    )
    total_windows = 0
    for panel in ("selection", "held"):
        observed = read(run / f"{panel}.json")
        require(
            observed["freeze_sha256"] == sha(run / "freeze.json"), "Native scores used a different freeze"
        )
        decisions = []
        for row, entry in zip((r for r in roster if r["panel"] == panel), observed["rows"], strict=True):
            require(row["id"] == entry["id"] and set(entry["arms"]) == set(ARMS), "Score identity differs")
            candidates = {}
            for arm, windows in entry["arms"].items():
                raw = (run / "pcm" / f"{row['id']}-{arm}.pcm").read_bytes()
                require(
                    [win["range"] for win in windows] == WINDOWS
                    and all(
                        win["sha256"] == digest(raw[2 * a : 2 * b])
                        and (win["score"] is None or math.isfinite(win["score"]))
                        for win, (a, b) in zip(windows, WINDOWS, strict=True)
                    ),
                    "Native score window differs",
                )
                candidates[arm] = [[win["score"], win["range"]] for win in windows]
                total_windows += len(windows)
            responses = {}
            for arm, entries in candidates.items():
                responses[arm] = max(
                    (entry for entry in entries if entry[0] is not None),
                    key=lambda x: x[0],
                    default=[None, None],
                )
                responses[arm + "_shared"] = max(
                    (
                        entry
                        for i, entry in enumerate(entries)
                        if all(rows[i][0] is not None for rows in candidates.values())
                    ),
                    key=lambda x: x[0],
                    default=[None, None],
                )
            decisions.append(row | {"responses": responses})
        table = summary["roles"][panel]
        for arm in (*ARMS, *(f"{key}_shared" for key in ARMS)):
            genuine = defaultdict(list)
            for row in decisions:
                if row["truth"] is None and (value := row["responses"][arm][0]) is not None:
                    genuine[row["group"]].append(value)
            ranked = sorted((max(values) for values in genuine.values()), reverse=True)
            point = points["values"][arm.removesuffix("_shared")]
            if panel == "selection" and arm in ARMS:
                require(point == ranked[0], "Frozen cutoff used an unexpected control")
            require(
                counts(decisions, arm, point) == table["fixed_matched_selection"][arm]
                and counts(decisions, arm, -math.inf)["useful_ranges"]
                == table["useful_without_threshold"][arm],
                "Reported fixed-point or unthresholded outcomes differ",
            )
            for budget, reported in table["equal_false_groups"].items():
                index = int(budget)
                require(
                    reported[arm] == (counts(decisions, arm, ranked[index]) if index < len(ranked) else None),
                    "Reported equal false-group outcome differs",
                )
        if panel == "held":
            require(
                counts(decisions, "original", lock["original_point"]) == summary["earlier_original_fixed"],
                "Earlier held original cutoff comparison differs",
            )
    return total_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("annotations", type=Path)
    args = parser.parse_args()
    lock, rows = (read(args.run / f"{key}.json") for key in ("freeze", "roster"))
    require(
        lock["roster_sha256"] == sha(args.run / "roster.json")
        and Counter(row["panel"] for row in rows) == {"selection": 26, "held": 26},
        "Frozen roster differs",
    )
    spectra = room_responses(args.archive, lock)
    maximum_rounding = verify_signals(args.run, rows, spectra, lock)
    overlap = actual_overlap([row | {"room": row["background"]["room"]} for row in rows], args.annotations)
    windows = verify_scores(args.run, rows, lock) if (args.run / "result.json").is_file() else 0
    print(
        json.dumps(
            {
                "native_views": len(rows),
                "exact_pcm_stems": len(rows) * len(STEMS),
                "max_fft_rounding_lsb": maximum_rounding,
                "edit_overlap_ms": [min(overlap), max(overlap)],
                "verified_score_windows": windows,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
