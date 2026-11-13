# SPDX-License-Identifier: Apache-2.0
"""Test frozen blind stereo separation in a separate measured room without scoring voice."""

from __future__ import annotations

import argparse
import io
import math
import time
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from statistics import median

import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags, fftconvolve, resample_poly

from scripts import audio_blind_spatial_transfer as blind
from scripts import audio_room_assets_v299 as room

SCHEMA = "chipmunk.audio-blind-huddersfield-transfer/1"
PUBLIC = Path("docs/evaluation-panels")
PRIOR_RECEIPT = PUBLIC / "audio-measured-spatial-transfer-2026-11-12.json"
SOURCE_RECEIPT = PUBLIC / "audio-remote-neighbor-replication-2026-11-12.json"
ROOM_RECEIPT = PUBLIC / "audio-room-assets-v299-holdout.json"
PATHS = ("S01", "S03", "S07")
KINDS = ("untouched", "ordinary_same_host", "ordinary_other_voice", "synthetic")
STRESSES = ("balanced_target", "lateral_target", "second_competitor")
LICENSES = {"CC-BY-4.0", "CC-BY-SA-4.0", "MIT (publisher PLN scenario declaration)"}
CONDITIONS = {
    "sources": "Eight exact interior prior remote-neighbor groups, with three expressive, three stock and two clone edits. Four previous background-only source clips serve as controls and additional feeds. All group, speaker, source, donor fragment and exact room capture intervals are disjoint from the two blind-pilot roles; underlying source banks and AMI sessions have prior project exposure.",
    "room": "Huddersfield St Paul's Hall fixed KU100 receiver, 44.1k native paired WAVs, pins v299. Use paths named S01, S03 and S07. Define roles by predeclared interaural balance and delay checks; publisher angle mapping remains unverified. Preserve complete 3s measured responses, channel order, and stored clock while converting to 16 kHz.",
    "stresses": {"balanced_target": "S01 target and one lateral masker, alternating S03 and S07.",
                 "lateral_target": "Swap target and masker paths on the same sources; assignment stays frozen.",
                 "second_competitor": "S01 target and two separately retained AMI feeds on the opposing lateral paths."},
    "mix": "Convolve six-second dry source and room inputs on a fixed 16 kHz clock. Normalize total masker energy against unedited target mean at -5 dB using a common per-group gain. Before that scaling, dual feeds each have equal mono energy. All matched edits and genuine controls share exact quantized masker samples. Include isolated background-only deliveries for each route stress.",
    "benchmark": "Compare integer mono mean, exact clean room-rendered target, both unassigned frozen AuxIVA candidates, and the unchanged lower-side-energy routing. Report oracle from best full-clip SNR, exact range SNR, clipping, and route misses. No model inference, fit, serving or confidence change.",
    "gate": {"minimum_routed_positive_groups": 8, "minimum_median_full_gain_db": 3,
             "minimum_median_edit_gain_db": 3, "maximum_clipped_routed_positive_views": 0},
}
require, read, write, sha = blind.require, blind.read, blind.write, blind.sha


def file_hash(path):
    return sha(path.read_bytes())


def bound_roster(path, receipt):
    lock = read(receipt)["commitments"]
    require(file_hash(path / "roster.json") == lock["roster"] and file_hash(path / "freeze.json") == lock["freeze"],
            "Retained roster or freeze differs from published receipt")
    return read(path / "roster.json")


def witness_tokens(rows):
    tokens = defaultdict(set)
    for row in rows:
        tokens["groups"].add(row["group"])
        for label in ("host", "donor"):
            person = row.get(label)
            if person:
                require(person["license"] in LICENSES, "Unapproved foreground source license")
                tokens["speakers"].add(person["canonical_speaker"])
                for key in ("source_sha256", "decoder_input_sha256", "pcm_sha256"):
                    if person.get(key):
                        tokens["source_bytes_or_fragments"].add(person[key])
        field = row["background"]
        tokens["background_pcm"].add(field["capture_pcm_sha256"])
    return tokens


def captures(rows):
    return {(r["background"]["room"], r["background"]["capture_start_sample"], r["background"]["samples"]) for r in rows}


def sources(args):
    parent = bound_roster(args.parent, PRIOR_RECEIPT)
    earlier = bound_roster(args.sources, SOURCE_RECEIPT)
    groups = defaultdict(list)
    controls = {}
    for row in earlier:
        if row["coverage"] == "interior":
            groups[row["group"]].append(row)
        if row["arm"] == "background_only":
            controls[row["room"]] = row
    chosen = [r for group in groups.values() for r in group] + list(controls.values())
    require(len(parent) == len(earlier) == 52 and len(groups) == 8 and len(controls) == 4 and
            all(tuple(r["arm"] for r in group) == KINDS for group in groups.values()), "Retained panel has wrong scope")
    require(Counter(r["family"] for r in chosen if r["truth"]) == {"clone": 2, "expressive": 3, "stock": 3},
            "Expected three generator families across eight groups")
    left, right = witness_tokens(parent), witness_tokens(chosen)
    for key in right:
        require(not left[key] & right[key], f"Retained sources overlap with pilot: {key}")
    old, new = captures(parent), captures(chosen)
    require(len(new) == 12 and not any(x == y and a < c + d and c < a + b for x, a, b in old for y, c, d in new),
            "Retained background intervals overlap prior pilot")
    require(not any(x == y and a < c + d and c < a + b for x, a, b in new for y, c, d in new if a != c),
            "Retained background intervals overlap in this panel")
    return groups, controls, {"selected_utterance_groups": len(groups), "selected_speaker_identities": len(right["speakers"]),
                              "selected_exact_source_and_fragment_hashes": len(right["source_bytes_or_fragments"]),
                              "distinct_capture_intervals": len(new), "prior_pilot_capture_intervals": len(old),
                              "shared_background_sessions": len({key[0] for key in old} & {key[0] for key in new})}


def impulses(archive):
    folder = archive / "audio-room-assets-v299" / "holdout"
    observed = room.check_huddersfield(folder)
    published = read(ROOM_RECEIPT)
    require(observed == published["result"], "Published Huddersfield source result differs")
    responses, records = {}, {}
    for label in PATHS:
        channels = []
        for asset in room.HUDDERSFIELD_ASSETS[label]:
            raw, _ = room.pinned_bytes(folder, asset)
            with sf.SoundFile(io.BytesIO(raw)) as handle:
                channels.append(handle.read(dtype="float64"))
        native = np.column_stack(channels)
        peaks = np.argmax(np.abs(native), axis=0)
        local = native[min(peaks)-30:max(peaks)+110]
        correlation = correlate(local[:, 0], local[:, 1])
        lag = int(correlation_lags(len(local), len(local))[np.argmax(abs(correlation))])
        balance = 10 * np.log10(np.square(local[:, 0]).sum() / np.square(local[:, 1]).sum())
        expected = {"S01": abs(balance) <= 1 and abs(lag) <= 4,
                    "S03": balance < -6 and lag > 16, "S07": balance > 6 and lag < -16}
        require(expected[label], "Role-independent early path geometry differs")
        response = np.ascontiguousarray(resample_poly(native, 160, 441, axis=0), dtype="<f8")
        require(response.shape == (48000, 2) and np.isfinite(response).all(), "Resampled room extent differs")
        responses[label] = response
        records[label] = {"native_peak_samples": peaks.tolist(), "early_left_minus_right_energy_db": round(float(balance), 5),
                          "early_correlation_delay_samples": lag, "native_rate": 44100, "render_rate": 16000,
                          "render_samples": len(response), "render_sha256": sha(response.tobytes())}
    return responses, records


def retained_pcm(path, row, part):
    suffix = "" if part == "original" else f"-{part}"
    key = "pcm_sha256" if part == "original" else f"{part}_sha256"
    body = (path / "pcm" / f"{row['id']}{suffix}.pcm").read_bytes()
    require(len(body) == blind.SAMPLES * 2 and sha(body) == row[key], "Retained audio differs from its committed source")
    return np.frombuffer(body, dtype="<i2").astype(np.float64)


def render_stem(pcm, impulse):
    return fftconvolve(pcm[:, None], impulse, axes=0)[:blind.SAMPLES]


def power(pcm):
    return float(np.square(pcm, dtype=np.float64).sum())


def bytes_of(pcm):
    require(np.max(np.abs(pcm)) < 32768 and np.all(np.isfinite(pcm)), "Rendered PCM is clipped or nonfinite")
    return np.ascontiguousarray(pcm, dtype="<i2").tobytes()


def encode_mix(dry, noise, group, responses, paths, truth):
    target, *maskers = paths
    foreground = [render_stem(pcm, responses[target]) for pcm in dry]
    rendered = [render_stem(pcm, responses[loc]) for pcm, loc in zip(noise, maskers, strict=True)]
    if len(rendered) == 2:
        rendered = [pcm / math.sqrt(power(pcm.mean(axis=1))) for pcm in rendered]
    background = np.sum(rendered, axis=0)
    if truth is not None:
        background *= math.sqrt(power(foreground[0].mean(axis=1)) * 10 ** .5 / power(background.mean(axis=1)))
    scale = 22500 / max(float(np.max(abs(signal) + abs(background))) for signal in foreground)
    background = np.rint(scale * background)
    host = np.rint(scale * foreground[0])
    result = []
    for index, values in enumerate(foreground):
        fg = np.rint(scale * values)
        stereo = fg + background
        native = np.rint(stereo.mean(axis=1))
        if truth and index:
            a, b = truth
            end = min(blind.SAMPLES, b + len(responses[target]) - 1)
            require(np.array_equal(host[:a], fg[:a]) and np.array_equal(host[end:], fg[end:]),
                    f"Authored {group} changes crossed measured room support")
        payload = dict(zip(("stereo", "foreground", "background", "original"),
                           map(bytes_of, (stereo, fg, background, native)), strict=True))
        result.append((payload, round(10 * math.log10(power(host.mean(axis=1)) / power(background.mean(axis=1))), 5) if truth else None))
    return result


def layouts(index):
    lateral = PATHS[1 + index % 2]
    opposite = PATHS[2 - index % 2]
    return {"balanced_target": ("S01", lateral), "lateral_target": (lateral, "S01"),
            "second_competitor": ("S01", lateral, opposite)}


def build(args):
    groups, controls, source_counts = sources(args)
    responses, room_records = impulses(args.archive)
    files = args.sources
    result = []
    for index, (group, rows) in enumerate(groups.items()):
        reference = rows[0]
        truth = rows[-1]["truth"]
        require(any(a <= truth[0] < truth[1] <= b for a, b in ((16000, 48000), (48000, 80000))),
                "Positive edit falls outside prior comparable windows")
        dry = [retained_pcm(files, row, "foreground") for row in rows]
        bg = retained_pcm(files, reference, "background")
        for row, values in zip(rows, dry, strict=True):
            require(row["background"] == reference["background"] and row["background_sha256"] == reference["background_sha256"] and
                    np.array_equal(values + bg, retained_pcm(files, row, "original")), "Retained matched pair does not sum exactly")
        alternate = list(controls.values())[(list(controls).index(reference["room"]) + 1) % len(controls)]
        bg_two = retained_pcm(files, alternate, "background")
        for condition, paths in layouts(index).items():
            noise = [bg, bg_two] if condition == "second_competitor" else [bg]
            rendered = encode_mix(dry, noise, group, responses, paths, truth)
            for row, (payload, snr) in zip(rows, rendered, strict=True):
                result.append(({"id": f"{condition}-{row['id']}", "group": group, "source": row["id"], "family": row["family"],
                                "kind": row["arm"], "truth": row["truth"], "condition": condition, "paths": list(paths),
                                "dry_sha256": row["foreground_sha256"], "primary_background_sha256": reference["background_sha256"],
                                "secondary_background_sha256": alternate["background_sha256"] if len(noise) == 2 else None,
                                "matched_host_input_snr_db": snr, "pcm": {name: sha(data) for name, data in payload.items()}}, payload))
    for index, row in enumerate(controls.values()):
        bg = retained_pcm(files, row, "background")
        zero = np.zeros(blind.SAMPLES)
        next_row = list(controls.values())[(index + 1) % len(controls)]
        bg_two = retained_pcm(files, next_row, "background")
        for condition, paths in layouts(index).items():
            noise = [bg, bg_two] if condition == "second_competitor" else [bg]
            (payload, _), = encode_mix([zero], noise, row["group"], responses, paths, None)
            result.append(({"id": f"{condition}-{row['id']}", "group": row["group"], "source": row["id"], "family": None,
                            "kind": row["arm"], "truth": None, "condition": condition, "paths": list(paths),
                            "dry_sha256": row["foreground_sha256"], "primary_background_sha256": row["background_sha256"],
                            "secondary_background_sha256": next_row["background_sha256"] if len(noise) == 2 else None,
                            "matched_host_input_snr_db": None, "pcm": {name: sha(data) for name, data in payload.items()}}, payload))
    require(len(result) == 108 and len({row["id"] for row, _ in result}) == len(result), "Expected 108 bounded mixtures")
    return result, source_counts, room_records


def freeze(args):
    groups, source_counts, room_records = build(args)
    folder = args.output
    folder.mkdir(mode=0o700)
    assets = folder / "pcm"
    assets.mkdir(mode=0o700)
    for row, payload in groups:
        for role, data in payload.items():
            (assets / f"{row['id']}-{role}.pcm").write_bytes(data)
    write(folder / "roster.json", [row for row, _ in groups])
    write(folder / "freeze.json", {"schema": SCHEMA, "conditions": CONDITIONS, "source_counts": source_counts, "room": room_records,
                                   "roster_sha256": file_hash(folder / "roster.json"), "runner_sha256": file_hash(Path(__file__)),
                                   "separator_sha256": file_hash(Path(blind.__file__)), "room_receipt_sha256": file_hash(ROOM_RECEIPT),
                                   "pilot_receipt_sha256": file_hash(PRIOR_RECEIPT), "source_receipt_sha256": file_hash(SOURCE_RECEIPT),
                                   "runtime": {k: version(k) for k in ("numpy", "scipy", "soundfile")}})
    print("freeze", file_hash(folder / "freeze.json"), len(groups), source_counts, flush=True)


def measure(folder, rows, *, alternate=False):
    transforms = {"fft": np.fft.rfft, "inverse": np.fft.irfft} if alternate else {}
    observations = []
    for case in rows:
        stereo, fg, bg, original = (blind.load_pcm(folder, case, name) for name in ("stereo", "foreground", "background", "original"))
        require(np.array_equal(stereo, fg + bg) and np.array_equal(original, np.rint(stereo.mean(axis=1))),
                "Frozen channel or downmix differs")
        clean = fg.mean(axis=1)
        exact = case["truth"]
        span = slice(*exact) if exact else slice(None)
        output = blind.separate(stereo, **transforms)
        signals = {"original": original}
        route, balance = None, None
        if output is not None:
            pcm, route, balance = output
            signals |= {"candidate_0": pcm[:, 0], "candidate_1": pcm[:, 1]}
        values = {}
        for name, pcm in signals.items():
            rounded = np.rint(pcm)
            values[name] = {"full_snr_db": blind.snr(clean, rounded), "edit_snr_db": blind.snr(clean[span], rounded[span]) if exact else None,
                            "clipped_samples": int(np.count_nonzero((rounded < -32768) | (rounded > 32767))),
                            "pcm_sha256": sha(np.clip(rounded, -32768, 32767).astype("<i2").tobytes()),
                            "output_mean_power": round(power(rounded) / len(rounded), 5)}
        observations.append({"id": case["id"], "group": case["group"], "condition": case["condition"], "family": case["family"],
                             "kind": case["kind"], "truth": exact, "route": route,
                             "balance_db": np.round(balance, 6).tolist() if balance is not None else None, "results": values})
        if len(observations) % 24 == 0:
            print("measure", len(observations), flush=True)
    return observations


def gains(pairs):
    return {f"median_{scope}_gain_db": round(median(a[f"{scope}_snr_db"] - b[f"{scope}_snr_db"] for a, b in pairs), 5)
            if pairs else None for scope in ("full", "edit")}


def summarize(rows):
    output = {}
    for condition in STRESSES:
        cases = [row for row in rows if row["condition"] == condition]
        synthetic = [row for row in cases if row["truth"]]
        require(len(cases) == 36 and len(synthetic) == 8, "Room stress denominator changed")
        chosen, oracle, routes = [], [], Counter()
        for row in cases:
            values = row["results"]
            index = row["route"]
            routes[row["kind"] + ("_routed" if index is not None else "_unrouted")] += 1
            if not row["truth"] or len(values) == 1:
                continue
            best = max((0, 1), key=lambda k: values[f"candidate_{k}"]["full_snr_db"])
            oracle.append((values[f"candidate_{best}"], values["original"]))
            if index is not None:
                chosen.append((values[f"candidate_{index}"], values["original"]))
                routes["synthetic_route_matches_oracle"] += int(index == best)
        selection = gains(chosen)
        clipped = sum(int(a["clipped_samples"] > 0) for a, _ in chosen)
        passed = (len(chosen) == 8 and selection["median_full_gain_db"] >= 3 and selection["median_edit_gain_db"] >= 3 and not clipped)
        output[condition] = {"views": len(cases), "synthetic": len(synthetic), "routing_counts": routes,
                             "blind": {"views": len(chosen), **selection, "clipped_views": clipped},
                             "oracle": {"views": len(oracle), **gains(oracle)}, "acoustic_gate_passes": passed}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "review"))
    for key in ("archive", "parent", "sources", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--publication", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args)
        return
    folder = args.output
    locked = read(folder / "freeze.json")
    require(locked["conditions"] == CONDITIONS and locked["roster_sha256"] == file_hash(folder / "roster.json") and
            locked["runner_sha256"] == file_hash(Path(__file__)) and locked["separator_sha256"] == file_hash(Path(blind.__file__)) and
            all(version(k) == v for k, v in locked["runtime"].items()), "Frozen source, code or runtime differs")
    rows = read(folder / "roster.json")
    began = time.monotonic()
    if args.stage == "run":
        observed = measure(folder, rows)
        write(folder / "acoustic.json", observed)
        result = locked | {"freeze_sha256": file_hash(folder / "freeze.json"), "acoustic_sha256": file_hash(folder / "acoustic.json"),
                           "model_inference_calls": 0, "seconds": round(time.monotonic() - began, 3), "summary": summarize(observed)}
        write(folder / "result.json", result)
        print(result["summary"], flush=True)
        return
    repeated, coverage, room_records = build(args)
    require([row for row, _ in repeated] == rows and coverage == locked["source_counts"] and room_records == locked["room"],
            "Source or rendered geometry changed")
    result = read(folder / "result.json")
    require(file_hash(folder / "acoustic.json") == result["acoustic_sha256"], "Acoustic journal changed")
    recorded = read(folder / "acoustic.json")
    observed = measure(folder, rows, alternate=True)
    require(observed == recorded and summarize(observed) == result["summary"], "Alternate FFT audio or outcome differs")
    result["review"] = {"rendered_pcm_bindings": sum(len(payload) for _, payload in repeated), "alternate_fft_views": len(observed),
                        "elapsed_s": round(time.monotonic() - began, 3)}
    require(args.publication is not None, "A reviewed output path is required")
    write(args.publication, result)
    print(result["review"], flush=True)


if __name__ == "__main__":
    main()
