"""Freeze and verify a private, source-separated partial-speech input panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

import numpy as np

from app.detectors.audio import decode
from scripts.audio_complete_sources_v81 import connected_roles
from scripts.audio_short_edit_controls import PEAK_TARGET, _quantize, _rms

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
SOURCE = BASE / "audio-complete-sources-v81/manifest.json"
VARIANTS = BASE / "audio-confidence-mixtures-v82/variants.json"
SOURCE_SHA256 = "8b667675a803bd03851c27251e08a63b040b2bd29474462f9f0a4e3af9108ea6"
VARIANTS_SHA256 = "8411c76325bc4dcde649db08ed05747c5eb917df13d811d1068b7395ee562d24"
ROLES = {
    "fusion_fit": {"clone": 4, "stock": 4},
    "confidence_cal": {"clone": 3, "stock": 3},
    "selection": {"clone": 3, "stock": 3},
    "confirmation": {"clone": 3, "stock": 3, "expressive": 4},
}
LICENSES = {
    "genuine": "CC-BY-4.0", "speech": "CC-BY-4.0", "stock": "Apache-2.0",
    "clone": "MIT (publisher PLN scenario declaration)", "expressive": "CC-BY-4.0",
}
CONDITIONS = {"dry": None, "speech_minus5db": -5, "speech_plus15db": 15}
ARMS = ("ordinary_same_host", "ordinary_other_voice", "synthetic")
RATE, SAMPLES = 16000, 64000
FADE = 320
DURATIONS = (10240, 12800, 15360)
INPUT_SCHEMA = "chipmunk.audio-partial-residual-inputs/1"
RECIPE = {
    "durations_samples": DURATIONS, "fade_samples": FADE, "conditions_db": CONDITIONS,
    "minimum_rms_dbfs": -38, "minimum_active_frame_dbfs": -45,
    "minimum_active_frame_fraction": 0.5, "local_level_match": "donor RMS to host target RMS",
    "window_step_samples": 1280, "room_window_step_samples": 6400,
    "foreground_guard_samples": 2560, "same_host_gap_samples": 2560,
    "background_assignment": "four role-local LibriSpeech source groups, cyclic by pair",
    "capture_scope": "Prior project sources; new controlled insertions. Background identities and chapters are separated by role; physical rooms are unknown. No AMI recordings.",
    "selection_scope": "Fixed source order and signal-level eligibility, without learned scores or transcripts in model inputs.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def rank(value):
    return digest((INPUT_SCHEMA + ":" + str(value)).encode())


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def catalogues():
    require(sha(SOURCE) == SOURCE_SHA256 and sha(VARIANTS) == VARIANTS_SHA256, "Prior source catalogue changed")
    complete, variants = read(SOURCE), read(VARIANTS)
    foregrounds = complete["foregrounds"]
    backgrounds = [row for row in complete["backgrounds"] if row["kind"] == "speech"]
    components = connected_roles([*foregrounds, *complete["backgrounds"]])
    sources = {row["raw_sha256"]: row for row in [*foregrounds, *backgrounds]}
    clean = {row["source_sha256"]: row for row in variants["variants"] if row["condition"] == "clean"}
    require(len(foregrounds) == len(clean) == 464 and len(backgrounds) == 16 and len(sources) == 480,
            "Expected the complete prior source and speech inventories")
    require(variants["source_manifest_sha256"] == SOURCE_SHA256, "Variant ancestry differs")
    for key, row in sources.items():
        require(row["role"] in ROLES and row["status"] == "ready" and row["license"] == LICENSES[row["kind"]],
                "Unqualified or unexpected source terms")
        if row["foreground"]:
            previous = clean[key]
            require((row["role"], row["study_group"], row["label"], row["kind"]) ==
                    (previous["role"], previous["group"], previous["label"], previous["kind"])
                    and previous["sample_rate"] == RATE and previous["samples"] == SAMPLES,
                    "Normalized source does not retain its exact source role")
    return sources, clean, Path(variants["source_root"]), components


def distinct_groups(rows):
    picked = {}
    for row in sorted(rows, key=lambda item: rank(item["raw_sha256"])):
        picked.setdefault(row["study_group"], row)
    return list(picked.values())


def schedule(sources):
    jobs, room_controls = [], []
    for role, counts in ROLES.items():
        current = [row for row in sources.values() if row["role"] == role]
        by_kind = {kind: distinct_groups([row for row in current if row["kind"] == kind]) for kind in LICENSES}
        positives = []
        for kind, count in counts.items():
            require(len(by_kind[kind]) >= count, "Insufficient generator groups for the fixed role")
            positives.extend(by_kind[kind][:count])
        positives.sort(key=lambda row: rank(row["raw_sha256"]))
        require(len({row["study_group"] for row in positives}) == len(positives), "Generator ancestry repeats within role")
        positive_groups = {row["study_group"] for row in positives}
        hosts = [row for row in by_kind["genuine"] if row["study_group"] not in positive_groups][:len(positives)]
        require(len(hosts) == len(positives), "Need one unique genuine host group per matched comparison")
        host_groups = {row["study_group"] for row in hosts}
        unmatched_donors = [row for row in by_kind["genuine"] if row["study_group"] not in host_groups | positive_groups]
        speech = by_kind["speech"]
        require(unmatched_donors and len(speech) == 4, "Need within-role independent donor and speech sources")
        alternate = 0
        for n, (host, positive) in enumerate(zip(hosts, positives, strict=True)):
            donor = next((row for row in by_kind["genuine"] if row["study_group"] == positive["study_group"]), None)
            if donor is None:
                donor = unmatched_donors[alternate % len(unmatched_donors)]
                alternate += 1
            require(host["canonical_speaker"] != donor["canonical_speaker"]
                    and donor["canonical_speaker"] != positive["canonical_speaker"], "Other-voice control lost its distinct speaker")
            jobs.append({"id": f"{role}-{n:02d}", "role": role, "duration": DURATIONS[n % len(DURATIONS)],
                         "host": host["raw_sha256"], "ordinary": donor["raw_sha256"],
                         "synthetic": positive["raw_sha256"], "background": speech[n % len(speech)]["raw_sha256"]})
        room_controls.extend({"role": role, "background": row["raw_sha256"]} for row in speech)
    return jobs, room_controls


def frozen(sources, components):
    jobs, room_controls = schedule(sources)
    return {"schema": INPUT_SCHEMA, "source_manifest_sha256": SOURCE_SHA256, "variants_sha256": VARIANTS_SHA256,
            "implementation_sha256": sha(Path(__file__)), "decoder_sha256": sha(Path(decode.__file__)),
            "ancestry_components": components, "sample_rate": RATE, "samples": SAMPLES,
            "recipe": RECIPE | {"durations_samples": list(DURATIONS)}, "pairs": jobs, "background_controls": room_controls,
            "planned_views": 10 * len(jobs) + len(room_controls), "new_neural_forwards": 0}


def signal_eligible(values):
    # An amplitude check screens only unusable silence, not speech identity.
    rms = _rms(values)
    if rms < 32768 * 10 ** (-38 / 20):
        return False
    frame_rms = np.sqrt(np.mean(np.square(values.reshape(-1, FADE), dtype=np.float64), axis=1))
    return 2 * int(np.count_nonzero(frame_rms >= 32768 * 10 ** (-45 / 20))) >= len(frame_rms)


def positions(samples, extent, key, *, step=1280, guard=2560):
    return sorted(range(guard, samples - extent - guard + 1, step), key=lambda offset: rank(f"{key}:{offset}"))


def choose_intervals(job, host, ordinary, synthetic, background):
    length, key = job["duration"], job["id"]
    host_windows = [s for s in positions(SAMPLES, length, key + "-host") if signal_eligible(host[s:s + length])]
    target, same = next(((s, t) for s in host_windows for t in host_windows
                         if t + length + 2560 <= s or s + length + 2560 <= t), (None, None))
    both = next((s for s in positions(SAMPLES, length, key + "-paired-donor")
                 if signal_eligible(ordinary[s:s + length]) and signal_eligible(synthetic[s:s + length])), None)
    room = next((s for s in positions(len(background), SAMPLES, key + "-background", step=6400, guard=0)
                 if signal_eligible(background[s:s + SAMPLES])), None)
    require(None not in (target, same, both, room), "Fixed comparison contains no usable source extent")
    return target, same, both, room


def splice(host, donor, target, start, length):
    carrier = host[target:target + length]
    replacement = donor[start:start + length] * (_rms(carrier) / _rms(donor[start:start + length]))
    ramp = (1 - np.cos(np.linspace(0, np.pi, FADE))) / 2
    replacement[:FADE] = carrier[:FADE] + ramp * (replacement[:FADE] - carrier[:FADE])
    replacement[-FADE:] = carrier[-FADE:] + ramp[::-1] * (replacement[-FADE:] - carrier[-FADE:])
    edited = host.copy()
    edited[target:target + length] = replacement
    return edited


def paired_mix(host, arms, noise):
    host_level, noise_level = _rms(host), _rms(noise)
    backgrounds = {condition: noise * (host_level / noise_level / 10 ** (snr / 20))
                   for condition, snr in CONDITIONS.items() if snr is not None}
    peak = max(float(np.max(np.abs(source + residual))) for source in arms.values()
               for residual in (*backgrounds.values(), 0.0))
    gain = min(1.0, PEAK_TARGET / peak)
    residuals = {key: _quantize(noise * gain) for key, noise in backgrounds.items()}
    clean = {key: _quantize(source * gain) for key, source in arms.items()}
    return gain, clean, residuals


def construct(lock, sources, clean_index, clean_root):
    @cache
    def load(key):
        row = sources[key]
        if row["foreground"]:
            entry = clean_index[key]
            encoded = (clean_root / entry["path"]).read_bytes()
            require(digest(encoded) == entry["sha256"], "Previously qualified v82 WAV bytes changed")
            pcm = decode.read_wav(encoded)
            require(len(pcm) == SAMPLES and digest(pcm.astype("<f4").tobytes()) == entry["pcm_sha256"],
                    "Previously qualified v82 sample clock or PCM changed")
        else:
            decoded = decode.decode_audio(Path(row["source_path"]), source_sha256=key)
            require(len(decoded.pcm) == row["samples"] and decoded.mapping["pcm_sha256"] == row["pcm_sha256"],
                    "Previously qualified v81 background sample clock or PCM changed")
            pcm = decoded.pcm
        return pcm.astype(np.float64) * 32768

    for job in lock["pairs"]:
        host, ordinary, synthetic, background = (load(job[key]) for key in ("host", "ordinary", "synthetic", "background"))
        target, same, both, room = choose_intervals(job, host, ordinary, synthetic, background)
        length = job["duration"]
        arms = {"ordinary_same_host": splice(host, host, target, same, length),
                "ordinary_other_voice": splice(host, ordinary, target, both, length),
                "synthetic": splice(host, synthetic, target, both, length), "untouched": host}
        gain, clean, residuals = paired_mix(host, arms, background[room:room + SAMPLES])
        family = sources[job["synthetic"]]["kind"]
        base = {"role": job["role"], "pair_id": job["id"], "family": family,
                "host_group": sources[job["host"]]["study_group"],
                "background_group": sources[job["background"]]["study_group"],
                "background_range": [room, room + SAMPLES], "gain": gain,
                "edit_range": [target, target + length], "fade_samples": FADE}
        for condition, nominal in CONDITIONS.items():
            names = (*ARMS, "untouched") if condition == "speech_minus5db" else ARMS
            residual = residuals.get(condition)
            for arm in names:
                label = int(arm == "synthetic")
                donor_key = {"synthetic": "synthetic", "ordinary_other_voice": "ordinary",
                             "ordinary_same_host": "host", "untouched": "host"}[arm]
                start = same if arm == "ordinary_same_host" else both
                samples = clean[arm] if residual is None else _quantize(clean[arm].astype(np.int32) + residual)
                metadata = base | {"condition": condition, "arm": arm, "label": label,
                                   "donor_group": sources[job[donor_key]]["study_group"],
                                   "donor_range": [start, start + length] if arm != "untouched" else None,
                                   "truth": [target, target + length] if label else None,
                                   "pure_truth": [target + FADE, target + length - FADE] if label else None,
                                   "snr_db": None if residual is None else 20 * math.log10(_rms(clean[arm]) / _rms(residual)),
                                   "nominal_snr_db": nominal,
                                   "background_pcm_sha256": None if residual is None else digest(residual.tobytes())}
                yield metadata, samples

    for control in lock["background_controls"]:
        key = control["background"]
        pcm = load(key)
        start = next((i for i in positions(len(pcm), SAMPLES, key + "-alone", step=6400, guard=0)
                      if signal_eligible(pcm[i:i + SAMPLES])), None)
        require(start is not None, "Fixed background control has no usable source extent")
        whole = pcm[start:start + SAMPLES]
        gain = min(3276.8 / _rms(whole), PEAK_TARGET / float(np.max(np.abs(whole))))
        metadata = {"role": control["role"], "pair_id": None, "family": None, "condition": "speech_minus5db",
                    "arm": "background_only", "label": 0, "host_group": None, "donor_group": None,
                    "background_group": sources[key]["study_group"], "background_range": [start, start + SAMPLES],
                    "gain": gain, "edit_range": None, "fade_samples": 0, "donor_range": None,
                    "truth": None, "pure_truth": None, "snr_db": None, "nominal_snr_db": None,
                    "background_pcm_sha256": None}
        yield metadata, _quantize(whole * gain)


def panel(lock, sources, clean_index, clean_root):
    for i, (metadata, samples) in enumerate(construct(lock, sources, clean_index, clean_root)):
        identifier = f"partial-{i:03d}"
        pcm = samples.astype("<i2", copy=False).tobytes()
        yield metadata | {"id": identifier, "path": f"pcm/{identifier}.pcm", "pcm_sha256": digest(pcm)}, pcm


def geometry_review(rows, arrays):
    by_pair = defaultdict(dict)
    owners = {}
    for row, pcm in zip(rows, arrays, strict=True):
        old = owners.setdefault(row["pcm_sha256"], row["role"])
        require(old == row["role"], "Exact PCM alias across held roles")
        require(pcm.shape == (SAMPLES,) and pcm.dtype == np.dtype("<i2"), "Invalid native PCM extent")
        if row["pair_id"] is not None:
            by_pair[row["pair_id"]][row["condition"], row["arm"]] = row, pcm.astype(np.int32)

    for pair in by_pair.values():
        require(len(pair) == 10, "Matched comparison is missing an arm or condition")
        _, original = pair["speech_minus5db", "untouched"]
        start, end = pair["dry", "synthetic"][0]["truth"]
        outside = np.r_[0:start, end:SAMPLES]
        for condition in CONDITIONS:
            residuals = []
            for arm in ARMS:
                row, observed = pair[condition, arm]
                _, dry = pair["dry", arm]
                if arm == "synthetic":
                    require(row["truth"] == [start, end] and row["pure_truth"] == [start + FADE, end - FADE],
                            "Synthetic support moved across matched conditions")
                else:
                    require(row["truth"] is row["pure_truth"] is None, "Ordinary control acquired synthetic truth")
                if condition == "speech_minus5db":
                    require(np.array_equal(observed[outside], original[outside]), "An edit leaked outside its exact support")
                    require(np.any(observed[start:end] != original[start:end]), "Matched edit contains no realized change")
                if condition != "dry":
                    residual = observed - dry
                    require(digest(residual.astype("<i2").tobytes()) == row["background_pcm_sha256"],
                            "Stored residual differs from quantized noisy minus dry samples")
                    residuals.append(residual)
            require(not residuals or all(np.array_equal(residuals[0], other) for other in residuals[1:]),
                    "Background waveform differs between matched arms")
    return len(by_pair)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "build", "verify"))
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run
    sources, clean, clean_root, components = catalogues()
    lock = frozen(sources, components)
    freeze_path = run / "inputs-freeze.json"
    if args.phase == "freeze":
        run.mkdir(mode=0o700)
        write(freeze_path, lock)
        print(json.dumps({"pairs": len(lock["pairs"]), "planned_views": lock["planned_views"],
                          "freeze_sha256": sha(freeze_path)}), flush=True)
        return
    require(read(freeze_path) == lock, "Exact frozen source, schedule and preparation code required")
    built = args.phase == "build"
    if built:
        (run / "pcm").mkdir(mode=0o700)
    expected, native = [], []
    for row, pcm in panel(lock, sources, clean, clean_root):
        path = run / row["path"]
        if built:
            with path.open("xb") as stream:
                stream.write(pcm)
        observed = path.read_bytes()
        require(len(observed) == 2 * SAMPLES and observed == pcm, "An output cannot be rebuilt from its frozen source")
        native.append(np.frombuffer(observed, dtype="<i2"))
        expected.append(row)
    require(len(expected) == lock["planned_views"], "Every frozen view is required")
    pairs = geometry_review(expected, native)
    record = {"schema": INPUT_SCHEMA, "freeze_sha256": sha(freeze_path), "sample_rate": RATE,
              "samples": SAMPLES, "rows": expected}
    manifest_path = run / "inputs.json"
    if built:
        write(manifest_path, record)
    else:
        require(read(manifest_path) == record, "Stored source, role, timing or waveform evidence differs")
        result = {"schema": INPUT_SCHEMA, "phase": "verify", "inputs_sha256": sha(manifest_path),
                  "rows": len(expected), "matched_pairs": pairs, "new_neural_forwards": 0,
                  "roles": dict(Counter(row["role"] for row in expected))}
        write(run / "inputs-verified.json", result)
    print(json.dumps({"phase": args.phase, "rows": len(expected), "matched_pairs": pairs,
                      "roles": dict(Counter(row["role"] for row in expected)), "inputs_sha256": sha(manifest_path)}), flush=True)


if __name__ == "__main__":
    main()
