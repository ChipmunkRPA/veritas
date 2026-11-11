"""Materialize the frozen matched Turkish source and background panel."""

from __future__ import annotations

import argparse
import io
import json
import wave
from collections import Counter, defaultdict
from itertools import combinations, cycle, islice, zip_longest
from pathlib import Path

from app.detectors.audio.decode import read_wav
from scripts import audio_alania_followup_v236 as alania
from scripts import audio_commons_background_v234 as background
from scripts import audio_fresh_shared_noise_v217 as prior
from scripts import audio_okur_turkish_followup_v245 as okur_followup
from scripts import audio_okur_turkish_v240 as okur
from scripts import audio_rueg_genuine_v246 as genuine
from scripts import audio_shared_noise_v216 as shared
from scripts.audio_voxforge_tr_transfer_v201 import verify_data_guard
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = prior.ROOT, prior.BASE, prior.PANEL
MODULE = "scripts.audio_matched_background_panel_v248"
TRIAL = BASE / "audio-matched-background-panel-v248"
STAGE, OUTPUT = TRIAL / "bind-00", TRIAL / "bind-00/output"
PUBLIC = PANEL / "audio-matched-background-panel-freeze-v248.json"
PACKET = BASE / "audio-commons-slice-screen-v247/authored/output/private-admitted-slices.json"
RESULTS = {
    "previous_panel": PANEL / "audio-fresh-shared-noise-complete-v217.json",
    "background": PANEL / "audio-commons-background-complete-v234.json",
    "alania": PANEL / "audio-alania-followup-complete-v236.json",
    "okur_pilot": PANEL / "audio-okur-turkish-complete-v240.json",
    "okur_followup": PANEL / "audio-okur-turkish-followup-complete-v245.json",
    "genuine": PANEL / "audio-rueg-genuine-complete-v246.json",
    "screen": PANEL / "audio-commons-slice-screen-complete-v247.json",
}
PINNED_RESULTS = {
    "previous_panel": "5fb45d9333ec42aee1221905f48d2e613f6c67a3dca277ad0959a80d35c270d2",
    "background": "3d5a702131268cf65217a163404ddc6095eefb6b199d1ed5ea97487df0a98149",
    "alania": "eef1ffacc9de432d59b1336df16b3ffe8876664f87b20e56ef2c4836893ce155",
    "okur_pilot": "37e697d698ff4be034aa8631505c081e83185aa2a39e1ed32f8cf9ae1e11b6f8",
    "okur_followup": "33a41facabc915464c7d16ea435a9463b02e56c02fb30ed163bc2792a6f9abe8",
}
ARMS = ("genuine", "alania", "okur")
CONDITIONS = prior.CONDITIONS
TAPES = shared.TAPES
TRIPLETS = 24
TRIPLET_IDS = tuple(f"triplet-{index:02d}" for index in range(TRIPLETS))
SAMPLES = shared.SAMPLES
PROVISIONAL = "machine_screened_provisional"
CODE = (
    "backend/scripts/audio_matched_background_panel_v248.py",
    "backend/tests/test_audio_matched_background_panel_v248.py",
    "docs/audio-matched-background-panel-v248-method.md",
    "backend/scripts/audio_fresh_shared_noise_v217.py",
    "backend/scripts/audio_shared_noise_v216.py",
    "backend/scripts/audio_short_edit_controls.py",
    "backend/scripts/video_appearance_runtime_v108.py",
    "backend/scripts/video_appearance_corrected_v112.py",
    "backend/scripts/audio_voxforge_tr_transfer_v201.py",
)
read, write, sha, require = prior.read, prior.write, prior.sha, prior.require
digest = okur.digest
mix_shared_noise = scoped(shared.mix_shared_noise, {"ARMS": ARMS})
read_alania_pcm = scoped(alania.prior.transfer.read_pcm, {"prepared_dir": alania.prepared_dir})
read_okur_followup_pcm = scoped(okur.read_prepared_pcm, {"OUTPUT": okur_followup.OUTPUT})


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def result_set():
    hashes = {key: sha(path) for key, path in RESULTS.items()}
    require(
        all(hashes[k] == expected for k, expected in PINNED_RESULTS.items()),
        "Previously published source and panel receipts differ",
    )
    return {key: read(path) for key, path in RESULTS.items()}, hashes


def source_record(cohort, slot, group, row, pcm):
    prefix = prior.signed_prefix(pcm) if pcm is not None else None
    return {
        "cohort": cohort,
        "source_slot": slot,
        "group": group,
        "source_status": "qualified" if prefix is not None else "unavailable",
        "source_wav_sha256": row["wav_sha256"] if prefix is not None else None,
        "source_prefix_sha256": digest(prefix) if prefix is not None else None,
    }, prefix


def foregrounds(results):
    current = genuine.readout()
    require(
        current == results["genuine"]
        and current["planned"] == TRIPLETS
        and all(current["terminal_stages"].values())
        and current["model_calls"] == 0,
        "Exact published terminal RUEG preparation required",
    )
    _, members = genuine.verify()
    sources = {arm: [] for arm in ARMS}
    for slot, member in enumerate(members, 1):
        receipt = genuine.stage_receipt("qualify", slot)
        row = read(receipt) if receipt.is_file() else {}
        pcm = None
        if row.get("status") == "qualified_pcm":
            wav = (genuine.CACHE / f"slot-{slot:02d}.crop.wav").read_bytes()
            pcm = read_wav(wav)
            require(
                digest(wav) == row["wav_sha256"] and len(pcm) == row["samples"],
                "Qualified genuine WAV differs",
            )
        sources["genuine"].append(source_record("v246", slot, member.participant, row, pcm))

    rows, receipt = alania.verified_preparation()
    require(
        {**receipt, "schema": "chipmunk.audio-alania-followup-preparation/236"} == results["alania"],
        "Exact published Alania preparation required",
    )
    plan = alania.verify()
    for row, item in zip(rows[:TRIPLETS], plan[:TRIPLETS], strict=True):
        require(row["slot"] == item["slot"], "Alania source order differs")
        pcm = read_alania_pcm(row) if row["status"] == "qualified" else None
        sources["alania"].append(source_record("v236", row["slot"], item["voice_id"], row, pcm))

    for cohort, module, reader, key in (
        ("v240", okur, okur.read_prepared_pcm, "okur_pilot"),
        ("v245", okur_followup, read_okur_followup_pcm, "okur_followup"),
    ):
        require(module.readout() == results[key], "Exact published Okur preparation required")
        for row in json_lines(module.OUTPUT / "private-cases.jsonl"):
            if row["status"] == "qualified":
                sources["okur"].append(
                    source_record(cohort, row["slot"], "okur_stock_voice", row, reader(row))
                )
    require(
        all(len(rows) == TRIPLETS for rows in sources.values())
        and all(len({row["group"] for row, _ in sources[arm]}) == TRIPLETS for arm in ARMS[:2])
        and {row["group"] for row, _ in sources["okur"]} == {"okur_stock_voice"},
        "Expected 24 participants, 24 designed voices and 24 utterances of one stock voice",
    )
    return sources


def check_exposure(sources, old_rows):
    for key in ("source_prefix_sha256", "source_wav_sha256"):
        old = {row[key] for row in old_rows if row["status"] == "prepared"}
        new = [row[key] for arm in ARMS for row, pcm in sources[arm] if pcm is not None]
        require(
            len(set(new)) == len(new) and set(new).isdisjoint(old),
            "A planned source repeats a current or v217 waveform fingerprint",
        )


def evidence_packet(results, result_hashes):
    screen = results["screen"]
    require(
        screen["schema"] == "chipmunk.audio-commons-slice-screen-result/247"
        and screen["status"] == "complete"
        and screen["background_result_sha256"] == result_hashes["background"]
        and screen["human_listening_reviewed"] is False
        and PACKET.is_file()
        and not PACKET.is_symlink()
        and sha(PACKET) == screen["private_admitted_slices_sha256"],
        "Exact provisional v247 evidence commitment required",
    )
    packet = read(PACKET)
    require(
        packet["schema"] == "chipmunk.audio-commons-slice-evidence/247"
        and packet["background_result_sha256"] == result_hashes["background"]
        and packet["evidence_type"] == PROVISIONAL
        and packet["human_listening_reviewed"] is False,
        "Use only machine-screened provisional slice evidence",
    )
    return packet["excerpts"]


def checked_excerpts(results, result_hashes):
    require(background.readout() == results["background"], "Exact published v234 preparation required")
    rows, tapes = evidence_packet(results, result_hashes), background.load_sources()
    available, cache, seen = {name: [] for name in TAPES}, {}, set()
    for row in rows:
        slot, start, environment = row["slot"], row["start_sample"], row["environment"]
        require(
            type(slot) is int
            and 1 <= slot <= len(tapes)
            and environment in TAPES
            and type(start) is int
            and start >= 0
            and row["samples"] == SAMPLES
            and (slot, start) not in seen
            and tapes[slot - 1].environment == environment,
            "Exact unique approved slice coordinate and environment required",
        )
        seen.add((slot, start))
        if slot not in cache:
            receipt = background.TRIAL / f"qualification-{slot:02d}.json"
            qualification = read(receipt)
            _, wav = background.asset_paths(slot, tapes[slot - 1])
            with wave.open(str(wav), "rb") as reader:
                require(reader.getnframes() == qualification["samples"], "Background PCM extent differs")
                body = reader.readframes(reader.getnframes())
            require(
                qualification["status"] == "verified_pcm" and digest(body) == qualification["pcm_sha256"],
                "Exact qualified background samples required",
            )
            cache[slot] = (sha(receipt), body)
        qualification_hash, body = cache[slot]
        excerpt = body[start * 2 : (start + SAMPLES) * 2]
        require(
            qualification_hash == row["qualification_receipt_sha256"]
            and len(excerpt) == SAMPLES * 2
            and digest(excerpt) == row["pcm_sha256"],
            "Provisional excerpt differs from its v234 samples",
        )
        available[environment].append((row, excerpt))
    return available


def allocate(entries):
    """Use reviewed cuts from each recording in turn before cycling any cut."""
    require(entries, "At least one provisional slice in each band required for the matched study")
    groups = defaultdict(list)
    for entry in sorted(entries, key=lambda item: (item[0]["slot"], item[0]["start_sample"])):
        groups[entry[0]["slot"]].append(entry)
    ordered = [entry for turn in zip_longest(*groups.values()) for entry in turn if entry is not None]
    return list(islice(cycle(ordered), TRIPLETS))


def source_coverage(sources):
    return {
        arm: {
            "planned_utterances": len(rows),
            "available_utterances": sum(pcm is not None for _, pcm in rows),
            "known_planned_voice_groups": len({row["group"] for row, _ in rows}),
            "known_available_voice_groups": len({row["group"] for row, pcm in rows if pcm is not None}),
            "maximum_utterances_per_group": max(Counter(row["group"] for row, _ in rows).values()),
            "source_cohort_counts": dict(Counter(row["cohort"] for row, _ in rows)),
        }
        for arm, rows in sources.items()
    }


def tape_coverage(available, assigned):
    return {
        name: {
            "provisional_excerpts_available": len(available[name]),
            "assigned_excerpts": len(entries),
            "assigned_distinct_recordings": len({row["slot"] for row, _ in entries}),
            "assigned_distinct_cuts": len({(row["slot"], row["start_sample"]) for row, _ in entries}),
            "assigned_distinct_excerpt_pcm": len({row["pcm_sha256"] for row, _ in entries}),
            "maximum_assignments_per_recording": max(Counter(row["slot"] for row, _ in entries).values()),
        }
        for name, entries in assigned.items()
    }


def context():
    prior.model_free()
    results, result_hashes = result_set()
    sources = foregrounds(results)
    previous = prior.OUTPUT / "private-cases.jsonl"
    require(
        sha(previous) == results["previous_panel"]["private_cases_sha256"],
        "Exact known v217 exposure journal required",
    )
    check_exposure(sources, json_lines(previous))
    available = checked_excerpts(results, result_hashes)
    assigned = {band: allocate(entries) for band, entries in available.items()}
    plan = {
        "foregrounds": {arm: [row for row, _ in rows] for arm, rows in sources.items()},
        "backgrounds": {band: [row for row, _ in entries] for band, entries in assigned.items()},
    }
    identity = {
        "schema": "chipmunk.audio-matched-background-panel-freeze/248",
        "planned_triplets": TRIPLETS,
        "planned_cases": TRIPLETS * len(ARMS) * len(CONDITIONS),
        "arms": list(ARMS),
        "conditions": list(CONDITIONS),
        "sample_rate": shared.SAMPLE_RATE,
        "source_prefix_samples": SAMPLES,
        "maximum_quantized_snr_error_db": shared.MAX_SNR_ERROR_DB,
        "source_results_sha256": result_hashes,
        "known_exposure_journal_sha256": sha(previous),
        "provisional_slice_packet_sha256": sha(PACKET),
        "foreground_coverage": source_coverage(sources),
        "background_coverage": tape_coverage(available, assigned),
        "background_evidence_type": PROVISIONAL,
        "human_listening_reviewed": False,
        "genuine_age_scope": "unverified",
        "detector_calls_permitted": False,
        "fitting_permitted": False,
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
    }
    prior.model_free()
    return identity, plan, sources, assigned


def preflight():
    missing = [key for key, path in {**RESULTS, "slice_evidence": PACKET}.items() if not path.is_file()]
    return {"status": "awaiting_inputs", "missing": missing} if missing else context()[0]


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve every earlier matched-panel attempt")
    identity, plan, _, _ = context()
    TRIAL.mkdir(mode=0o700)
    private = TRIAL / "private-plan.json"
    write(private, plan)
    identity["private_plan_sha256"] = sha(private)
    write(TRIAL / "freeze.json", identity)
    write(PUBLIC, identity)
    return identity


def verify():
    identity, plan, sources, assigned = context()
    private = TRIAL / "private-plan.json"
    identity["private_plan_sha256"] = sha(private)
    require(
        read(PUBLIC) == read(TRIAL / "freeze.json") == identity and read(private) == plan,
        "Exact frozen matched source and background plan required",
    )
    return identity, sources, assigned


def cases(sources, assigned):
    for index, triplet in enumerate(TRIPLET_IDS):
        raw = {arm: sources[arm][index][1] for arm in ARMS if sources[arm][index][1] is not None}
        metadata = {name: assigned[name][index][0] for name in TAPES}
        excerpts = {name: assigned[name][index][1] for name in TAPES}
        treatment, reason = None, "source_unavailable"
        if raw:
            try:
                treatment = mix_shared_noise(raw, excerpts)
            except shared.UnsupportedControl as error:
                reason = f"mix_unavailable:{error}"
        for arm in ARMS:
            source, _ = sources[arm][index]
            for condition in CONDITIONS:
                base = {
                    "case": f"{triplet}-{arm}-{condition}",
                    "triplet": triplet,
                    "arm": arm,
                    "condition": condition,
                    **source,
                }
                if arm not in raw or treatment is None:
                    yield (
                        {
                            **base,
                            "status": "unavailable",
                            "reason": reason if arm in raw else "source_unavailable",
                        },
                        None,
                    )
                    continue
                variant = treatment.variants[arm]
                row = {
                    **base,
                    "status": "prepared",
                    "samples": SAMPLES,
                    "common_gain": treatment.common_gain,
                    "level_match_gain": variant.level_match_gain,
                    "reference_arm": treatment.reference_arm,
                }
                if condition == "matched_clean":
                    payload = variant.clean_s16le
                    row["background"] = None
                else:
                    tape, db = condition.rsplit("_", 1)
                    key = (tape, int(db.removesuffix("db")))
                    payload = variant.mixed_s16le[key]
                    row["background"] = {
                        **metadata[tape],
                        "evidence_type": PROVISIONAL,
                        "quantized_residual_sha256": digest(treatment.noise_s16le[key]),
                    }
                    row["measured_snr_db"] = variant.measured_snr_db[key]
                yield {**row, "pcm_sha256": digest(payload)}, payload


def summarize(rows):
    expected = [
        (triplet, arm, condition) for triplet in TRIPLET_IDS for arm in ARMS for condition in CONDITIONS
    ]
    require(
        [(row["triplet"], row["arm"], row["condition"]) for row in rows] == expected
        and all(row["status"] in {"prepared", "unavailable"} for row in rows),
        "Every ordered planned triplet, arm and condition is required",
    )
    lookup = {(row["triplet"], row["arm"], row["condition"]): row["status"] == "prepared" for row in rows}
    outcomes, pairs = {}, {}
    for condition in CONDITIONS:
        complete = [[lookup[triplet, arm, condition] for arm in ARMS] for triplet in TRIPLET_IDS]
        outcomes[condition] = {
            arm: {
                "prepared": sum(flags[j] for flags in complete),
                "unavailable": sum(not flags[j] for flags in complete),
            }
            for j, arm in enumerate(ARMS)
        }
        pairs[condition] = {
            "planned_triplets": TRIPLETS,
            "complete_triplets": sum(all(flags) for flags in complete),
            **{
                f"{ARMS[a]}_{ARMS[b]}": sum(flags[a] and flags[b] for flags in complete)
                for a, b in combinations(range(len(ARMS)), 2)
            },
        }
    return {
        "by_condition_and_arm": outcomes,
        "pair_coverage": pairs,
        "unavailable_reasons": dict(Counter(row["reason"] for row in rows if row["status"] == "unavailable")),
    }


def result(identity, rows, journal):
    return {
        "schema": "chipmunk.audio-matched-background-panel-result/248",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "private_cases_sha256": sha(journal),
        "planned_cases": len(rows),
        "foreground_coverage": identity["foreground_coverage"],
        "background_coverage": identity["background_coverage"],
        "background_evidence_type": PROVISIONAL,
        "human_listening_reviewed": False,
        **summarize(rows),
        "new_neural_calls": 0,
        "detector_calls": 0,
        "holdout_scored": False,
    }


def worker():
    identity, sources, assigned = verify()
    require(
        (STAGE / "resources.jsonl").is_file()
        and (STAGE / "worker.log").is_file()
        and not (STAGE / "profile.json").exists(),
        "Use the owned data-only v108 guard",
    )
    OUTPUT.mkdir(mode=0o700)
    prepared = OUTPUT / "prepared"
    prepared.mkdir(mode=0o700)
    rows, journal = [], OUTPUT / "private-cases.jsonl"
    with journal.open("x") as stream:
        for row, pcm in cases(sources, assigned):
            if pcm is not None:
                row["wav_sha256"] = prior.save_wave(prepared / f"{row['case']}.wav", pcm)
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
    prior.model_free()
    write(OUTPUT / "result.json", result(identity, rows, journal))


run = scoped(
    prior.run, {"TRIAL": TRIAL, "OUTPUT": OUTPUT, "MODULE": MODULE, "verify": verify, "__file__": __file__}
)


def prepared_waveform(row):
    """Return the verified WAV path and exact little-endian signed16 samples."""
    triplet, arm, condition = row["triplet"], row["arm"], row["condition"]
    require(
        row["status"] == "prepared"
        and row["samples"] == SAMPLES
        and triplet in TRIPLET_IDS
        and arm in ARMS
        and condition in CONDITIONS
        and row["case"] == f"{triplet}-{arm}-{condition}",
        "Exact prepared waveform row required",
    )
    path = OUTPUT / "prepared" / f"{row['case']}.wav"
    require(
        path.is_file() and not path.is_symlink() and path.stat().st_size <= 2 * SAMPLES + 1024,
        "Direct bounded matched WAV required",
    )
    body = path.read_bytes()
    with wave.open(io.BytesIO(body), "rb") as reader:
        require(
            (
                reader.getnchannels(),
                reader.getsampwidth(),
                reader.getframerate(),
                reader.getnframes(),
                reader.getcomptype(),
            )
            == (1, 2, shared.SAMPLE_RATE, SAMPLES, "NONE"),
            "Exact matched mono signed16 WAV required",
        )
        pcm = reader.readframes(SAMPLES)
    require(
        len(pcm) == 2 * SAMPLES and digest(body) == row["wav_sha256"] and digest(pcm) == row["pcm_sha256"],
        "Saved matched PCM differs from its row commitment",
    )
    return path, pcm


def readout():
    identity, sources, assigned = verify()
    journal, result_path, profile = (
        OUTPUT / "private-cases.jsonl",
        OUTPUT / "result.json",
        STAGE / "profile.json",
    )
    verify_data_guard(read(profile), STAGE, result_path, sha(Path(__file__)))
    rows = json_lines(journal)
    for (expected, pcm), row in zip(cases(sources, assigned), rows, strict=True):
        if pcm is not None:
            _, observed_pcm = prepared_waveform(row)
            require(observed_pcm == pcm, "Saved mixed samples differ from exact reconstruction")
            expected["wav_sha256"] = row["wav_sha256"]
        require(row == expected, "Reconstructed source or background case differs")
    observed = read(result_path)
    require(observed == result(identity, rows, journal), "Exact complete matched-panel outcome required")
    prior.model_free()
    return {
        **observed,
        "profile_sha256": sha(profile),
        "scope": "One stock synthetic voice and reused background recordings limit independence. "
        "The background screen is provisional and no authenticity detector ran.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = {"preflight": preflight, "freeze": freeze, "run": run, "readout": readout, "_worker": worker}
    parser.add_argument("command", choices=commands)
    result = commands[parser.parse_args().command]()
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
