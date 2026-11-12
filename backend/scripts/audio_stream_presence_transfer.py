"""Compare separated voice presence under aligned context shifts and ordinary overlap."""
from __future__ import annotations

import argparse
import itertools
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts import audio_independent_overlap_count as voice
from scripts import audio_measured_speaker_routing as receipt
from scripts import audio_separation_third_voice_pilot as sep

VIEWS = ((0, 0), (320, 80), (720, 160))
MIN_RELATIVE_RMS = .04
COACTIVE_SAMPLES = 1280
VAD_LENGTH = 48000
SAMPLES = 24000
ARMS = ("activity", "aligned_stability", "stable_independent")
METHOD = {
    "source_roles": "Eight earlier separator-pilot triplets for point selection, all 13 remaining VoxPopuli triplets for constructed confirmation, and three separately downloaded first-party natural AMI rooms with annotation-controlled two- and three-voice intervals.",
    "views_at_8_khz": [list(item) for item in VIEWS],
    "activity": "Measure 16 kHz activity on the three base outputs. Require at least 160 ms of shared activity with each output at or above 4% of mixture RMS.",
    "stability": "Permute the two shifted output sets to match the base output set by absolute sample-aligned correlation. Use the weakest aligned output across both shifts.",
    "independence": "Use the weakest base stream's unexplained energy after fitting the other two streams by least squares at exact alignment. Compound stability is the minimum of aligned stability and independence.",
    "comparison": "Fit each score at the second-highest maximum over two-voice controls from the eight development source groups. A result above its own strict point supports three voices. Also replay the raw 10% RMS rule.",
    "advance": "A compound cue must flag at least half the three-voice inputs and at most one in six two-voice controls in both confirmation panels. The result carries zero authenticity weight.",
}


def aligned_correlations(reference, trial):
    def unit(rows):
        centered = rows - rows.mean(axis=-1, keepdims=True)
        return centered / np.maximum(np.linalg.norm(centered, axis=-1, keepdims=True), 1e-12)

    corr = np.clip(np.abs(unit(reference) @ unit(trial).T), 0, 1)
    assignment = max(itertools.permutations(range(3)), key=lambda order: sum(corr[i, j] for i, j in enumerate(order)))
    return [float(corr[i, j]) for i, j in enumerate(assignment)]


def explained_energy(output, references) -> float:
    centered = output - output.mean()
    matrix = np.asarray([row - row.mean() for row in references]).T
    energy = float(np.dot(centered, centered))
    if energy <= 1e-12:
        return 1.
    weights = np.linalg.lstsq(matrix, centered, rcond=1e-6)[0]
    residual = centered - matrix @ weights
    return float(np.clip(1 - np.dot(residual, residual) / energy, 0, 1))


def assess(mixture, outputs, active):
    base = outputs[0]
    if len(outputs) != 3 or any(arr.shape != (3, SAMPLES) or not np.isfinite(arr).all() for arr in outputs):
        raise ValueError("Every separation view must contain three finite exact-clock outputs")
    if len(active) != 3 or any(mask.shape != (VAD_LENGTH,) or mask.dtype != bool for mask in active):
        raise ValueError("Each base output needs its exact 16 kHz activity mask")
    magnitude = np.linalg.norm(base, axis=1) / max(np.linalg.norm(mixture), 1e-12)
    common = int(np.logical_and.reduce(active).sum())
    independence = min(1 - explained_energy(output, np.delete(base, index, axis=0)) for index, output in enumerate(base))
    stability = min(value for repeat in outputs[1:] for value in aligned_correlations(base, repeat))
    admitted = common >= 2 * COACTIVE_SAMPLES and min(magnitude) >= MIN_RELATIVE_RMS
    return {"raw_energy": bool(min(magnitude) >= .10), "activity_samples": common, "qualified": bool(admitted),
            "scores": {"activity": common / VAD_LENGTH if admitted else None,
                       "aligned_stability": stability if admitted else None,
                       "stable_independent": min(independence, stability) if admitted else None},
            "minimum_relative_rms": float(min(magnitude)), "minimum_stability": stability,
            "minimum_unexplained_energy": independence}


def known_source_audit(mixture, base, references):
    if len(references) == 3:
        quality, _ = sep.matches(mixture, references, base)
        return {"third_recovered": bool(quality[2]["si_sdr_db"] >= 0 and quality[2]["gain_db"] >= 3)}
    match = max(itertools.permutations(range(3), 2),
                key=lambda order: sum(sep.signal_quality(base[j], references[i]) for i, j in enumerate(order)))
    extra = base[next(i for i in range(3) if i not in match)]
    share = np.linalg.norm(extra) / max(np.linalg.norm(mixture), 1e-12)
    if share < .10:
        kind = "low_energy"
    elif max(explained_energy(extra, [target]) for target in references) >= .5:
        kind = "largely_one_known_voice"
    elif explained_energy(extra, references) >= .5:
        kind = "largely_both_known_voices"
    else:
        kind = "unexplained_by_linear_reference_mix"
    return {"extra_output_category": kind}


def fit_points(rows):
    by_arm = {arm: defaultdict(list) for arm in ARMS}
    for row in rows:
        if row["expected_voices"] != 2:
            continue
        for arm, value in row["observation"]["scores"].items():
            by_arm[arm][row["group"]].append(value if value is not None else 0.)
    if any(len(values) != 8 for values in by_arm.values()):
        raise ValueError("All eight development two-voice source groups are required")
    return {arm: sorted((max(values) for values in groups.values()), reverse=True)[1] for arm, groups in by_arm.items()}


def summarize(rows, points):
    totals = defaultdict(Counter)
    affected = defaultdict(set)
    for row in rows:
        label = "two" if row["expected_voices"] == 2 else "three"
        observation = row["observation"]
        values = {"raw_energy": observation["raw_energy"], **{arm: value is not None and value > points[arm]
                  for arm, value in observation["scores"].items()}}
        for key in (row["panel"], f"{row['panel']}/{row['delivery']}"):
            tally = totals[key]
            tally[f"planned_{label}"] += 1
            tally[f"activity_qualified_{label}"] += int(observation["qualified"])
            for arm, flagged in values.items():
                tally[f"{arm}_{label}_flag"] += int(flagged)
                if flagged:
                    affected[key, arm, label].add(row["group"])
            if ref := row.get("reference_audit"):
                if label == "two":
                    tally[f"unmatched_{ref['extra_output_category']}"] += 1
                else:
                    tally["reference_third_recovered"] += int(ref["third_recovered"])
                    tally["reference_third_recovered_and_compound_flagged"] += int(ref["third_recovered"] and values["stable_independent"])
    def passes(panel):
        return (2 * totals[panel]["stable_independent_three_flag"] >= totals[panel]["planned_three"] > 0
                and 6 * totals[panel]["stable_independent_two_flag"] <= totals[panel]["planned_two"] and totals[panel]["planned_two"] > 0)

    return {"totals": totals, "affected_groups": {"/".join(key): len(groups) for key, groups in affected.items()},
            "advance": passes("constructed_confirmation") and passes("natural_confirmation")}


def input_plan(folder: Path, parent: Path, assets: Path, vad: Path):
    if receipt.sha(parent / "manifest.json") != sep.PARENT_MANIFEST or receipt.sha(vad) != voice.VAD_SHA:
        raise ValueError("Exact prior source and activity-model commitments required")
    if {n: receipt.sha(assets / n) for n in sep.ASSETS} != sep.ASSETS:
        raise ValueError("Exact Apache-licensed separator checkpoint required")
    hard = receipt.read(parent / "manifest.json")
    development = set()
    for lang, quota in sep.QUOTAS.items():
        groups = [r["group"] for r in hard["groups"] if r["language"] == lang]
        development.update(sorted(groups, key=lambda value: sep.overlap.digest(f"separation/1:{value}".encode()))[:quota])
    if len(development) != 8 or len(hard["groups"]) != 21:
        raise ValueError("Expected eight development and 13 source-disjoint constructed groups")
    rows = []
    for source in hard["rows"]:
        if source["role_rotation"] or source["condition"] not in sep.TEST_CASES:
            continue
        panel = "development" if source["group"] in development else "constructed_confirmation"
        rows.append({**source, "panel": panel, "key": f"c{source['id']:03d}"})
    natural_path = folder / "natural-source-plan.json"
    natural = receipt.read(natural_path)
    if len(natural["rooms"]) != 3 or len({room["meeting"] for room in natural["rooms"]}) != 3:
        raise ValueError("Expected three separately planned natural sources")
    sources = []
    for index, room in enumerate(natural["rooms"]):
        source = folder / "natural-raw" / f"{room['meeting']}.wav"
        sources.append({"group": index, "sha256": receipt.sha(source)})
        with wave.open(str(source), "rb") as stream:
            if stream.getparams()[:3] != (1, 2, 16000):
                raise ValueError("Natural input must be native signed mono 16 kHz WAV")
            for ordinal, case in enumerate(room["cases"]):
                start = case["begin_centiseconds"] * 160
                stream.setpos(start)
                data = stream.readframes(VAD_LENGTH)
                if len(data) != VAD_LENGTH * 2:
                    raise ValueError("Planned natural interval falls outside its source")
                key = f"n{index}-{ordinal:02d}"
                rows.append({"key": key, "group": index, "panel": "natural_confirmation", "delivery": "native_mono",
                             "expected_voices": case["expected_voices"], "start_sample": start, "pcm_sha256": sep.overlap.digest(data)})
    lock = {"schema": "chipmunk.stream-presence-transfer/1", "method": METHOD, "implementation_sha256": receipt.sha(Path(__file__)),
            "parent_manifest_sha256": sep.PARENT_MANIFEST, "separator_revision": sep.REVISION, "separator_assets": sep.ASSETS,
            "activity_sha256": voice.VAD_SHA, "natural_selection_sha256": receipt.sha(natural_path), "native_sources": sources,
            "cases": dict(Counter(r["panel"] for r in rows)), "input_roster_sha256": sep.overlap.digest(str([(r["key"], r["pcm_sha256"]) for r in rows]).encode())}
    return rows, lock, natural


def infer(folder, assets, parent, vad, rows, natural):
    import torch
    from scipy.signal import resample_poly

    from app.detectors.audio.upgraded import SpeechActivity

    torch.set_num_threads(2)
    model = sep.load_separator(assets)
    activity = SpeechActivity(vad)
    for index, row in enumerate(rows):
        if row["panel"] == "natural_confirmation":
            room = natural["rooms"][row["group"]]["meeting"]
            with wave.open(str(folder / "natural-raw" / f"{room}.wav"), "rb") as stream:
                stream.setpos(row["start_sample"])
                pcm = stream.readframes(VAD_LENGTH)
        else:
            pcm = receipt.pcm(parent, row)
        if len(pcm) != VAD_LENGTH * 2 or sep.overlap.digest(pcm) != row["pcm_sha256"]:
            raise ValueError("Audio differs from the exact frozen input")
        mixture = resample_poly(voice.decode(pcm), 1, 2)
        outputs = []
        for lead, tail in VIEWS:
            padded = np.pad(mixture.astype("float32"), (lead, tail))
            with torch.inference_mode():
                encoded = model["encoder"](torch.from_numpy(padded)[None, ...])
                masks = model["masknet"](encoded)
                decoded = torch.stack([model["decoder"](mask * encoded)[0] for mask in masks]).numpy()
            aligned = np.zeros((3, len(padded)), dtype="float64")
            width = min(aligned.shape[1], decoded.shape[1])
            aligned[:, :width] = decoded[:, :width]
            outputs.append(aligned[:, lead:lead + SAMPLES])
        measured_activity = []
        for stream in outputs[0]:
            wave16 = resample_poly(stream, 2, 1)
            peak = float(np.max(abs(wave16)))
            mask = np.zeros(VAD_LENGTH, dtype=bool)
            if peak > 1e-6:
                for begin, end in activity.spans((wave16 * (.7 / peak)).astype("float32")):
                    mask[round(begin * 16000):round(end * 16000)] = True
            measured_activity.append(mask)
        row = {**row, "observation": assess(mixture, outputs, measured_activity)}
        if row["panel"] != "natural_confirmation":
            refs = [resample_poly(x, 1, 2) for x in sep.sources(parent, row)]
            row["reference_audit"] = known_source_audit(mixture, outputs[0], refs)
        if (index + 1) % 24 == 0:
            print(f"Measured {index + 1}/{len(rows)} exact inputs", flush=True)
        yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "fit", "confirm"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("parent", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("vad", type=Path)
    args = parser.parse_args()
    rows, plan, natural = input_plan(args.folder, args.parent, args.assets, args.vad)
    path = args.folder / "stream-presence-freeze.json"
    if args.action == "freeze":
        return receipt.write(path, plan)
    if receipt.read(path) != plan:
        raise ValueError("Exact frozen comparison required")
    development = args.action == "fit"
    selected = [r for r in rows if (r["panel"] == "development") == development]
    points_path = args.folder / "points.json"
    if not development:
        points_record = receipt.read(points_path)
        if (points_record["freeze_sha256"] != receipt.sha(path)
                or points_record["development_sha256"] != receipt.sha(args.folder / "fit-private.json")):
            raise ValueError("Only the frozen development controls may set the points")
    started = time.monotonic()
    observed = list(infer(args.folder, args.assets, args.parent, args.vad, selected, natural))
    outcome_path = args.folder / f"{args.action}-private.json"
    receipt.write(outcome_path, observed)
    if development:
        points_record = {"points": fit_points(observed), "freeze_sha256": receipt.sha(path), "development_sha256": receipt.sha(outcome_path)}
        receipt.write(points_path, points_record)
    summary = summarize(observed, points_record["points"])
    receipt.write(args.folder / f"{args.action}-aggregate.json", {"freeze_sha256": receipt.sha(path), "points_sha256": receipt.sha(points_path),
        "private_response_sha256": receipt.sha(outcome_path), "forwards": len(observed) * len(VIEWS), "seconds": time.monotonic() - started, **summary})


if __name__ == "__main__":
    main()
