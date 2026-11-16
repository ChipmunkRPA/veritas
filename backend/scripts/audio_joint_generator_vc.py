# SPDX-License-Identifier: MIT
"""Check frozen short voice-edit models on new declared readers and voice conversion."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from scripts import audio_joint_generator_transfer as experiment

SOURCE, base, prior = experiment.SOURCE, experiment.base, experiment.prior
ROOT = Path(__file__).resolve().parents[2]
PANELS = ROOT / "docs/evaluation-panels"
PUBLISHER = PANELS / "audio-phonemedf-release-19522979.json"
HISTORICAL = PANELS / "audio-phoneme-short-edit-intake-2026-11-13.json"
SCHEMA = "chipmunk.audio-joint-generator-vc/1"
LEVELS = (None, -5, -10)
PREFIX_BYTES = 448 * 1024**2
EXTENDED_BYTES = 640 * 1024**2
EXTENSION_CHUNK = 16 * 1024**2
METHOD = {
    "selection": "Select one six-second crop per new declared reader, up to 16, from the declared bounded "
                 "PhonemeDF audio prefixes. Use hash order, published word alignment and relative timing only. Exclude "
                 "the fitting, calibration and held PhonemeDF readers. Check exact audio against the earlier same-word panel.",
    "arms": "Keep the six-second original and reinsert the same words from either the publisher's TTS or voice "
            "conversion recording. Use one matched genuine round-trip resampling for each insert, same fades "
            "and RMS, and exact original samples outside the annotated words.",
    "mix": "Use the same recorded room excerpt for matched inputs. Require the room to be at least as loud during "
           "the ordinary edit interval in the nominal -5 dB mixture. Also deliver at -10 dB and dry, including "
           "the exact isolated backgrounds at both levels.",
}


def rank(key):
    return SOURCE.digest(f"{SCHEMA}/{key}".encode())


def extension(folder, origin, size):
    for offset in range(PREFIX_BYTES, EXTENDED_BYTES, EXTENSION_CHUNK):
        path = folder / origin / f"{offset}.range"
        expected = f"content-range: bytes {offset}-{offset + EXTENSION_CHUNK - 1}/{size}"
        assert path.stat().st_size == EXTENSION_CHUNK and expected in path.with_suffix(".headers").read_text().casefold().splitlines()
        yield path


def collect(args, known_readers):
    import numpy as np

    from scripts import audio_phoneme_short_edit_intake as intake

    release = prior.read(PUBLISHER)
    assert release["id"] == intake.prior.SOURCE_RECORD and release["metadata"]["license"]["id"] == "cc-by-4.0"
    sizes = {item["key"]: item["size"] for item in release["files"]}
    payloads, pins = {}, {}
    historical = prior.read(HISTORICAL)["bounded_wav_ranges"]
    for origin in intake.ORIGINS:
        old = args.old_cache if origin != "ChatterboxVC" else None
        size = sizes[f"PhonemeDF_{origin}.tar.gz"]
        chunks = intake.archive(args.cache, origin, size, PREFIX_BYTES, old)
        if args.extension:
            chunks.extend(extension(args.extension, origin, size))
        hashes = [SOURCE.path_sha(path) for path in chunks]
        assert hashes[:len(historical[origin])] == historical[origin]
        pins[origin] = hashes
        directory = "real/wav" if origin == "real" else origin
        pattern = re.compile(rf"{directory}/((\d+)-\d+-\d+)\.wav")
        archive = {}
        for name, body in intake.prior.archive_members(chunks):
            match = pattern.fullmatch(name)
            if match and base.sources.order(match[2]) not in known_readers:
                key = match[1]
                pcm = intake.prior.pcm_bytes(body)
                assert key not in archive
                archive[key] = (SOURCE.digest(body), SOURCE.digest(pcm), np.frombuffer(pcm, "<i2"))
        payloads[origin] = archive
    return payloads, set.intersection(*(set(archive) for archive in payloads.values())), pins


def prepare(args):
    import numpy as np

    from scripts import audio_phoneme_short_edit_intake as intake
    from scripts.audio_edit_direction_fresh import mix

    frozen, digest = base.load_plan(args.data)
    training_and_held = {row["reader_sha256"] for row in frozen["sources"]}
    history = prior.read(HISTORICAL)
    old = {row["pcm_sha256"] for row in history["deliveries"]}
    payloads, keys, pins = collect(args, training_and_held)
    print(f"Found {len(keys)} matched utterances outside the earlier reader groups.", flush=True)
    aligned = intake.grid_inputs(args, keys, payloads)
    rooms = prior.read(args.background / "freeze.json")["room_plan"]
    backgrounds = {}
    for room in rooms:
        data = (args.background / "background" / f"{room['room']}.pcm").read_bytes()
        assert SOURCE.digest(data) == room["pcm_sha256"] and len(data) == 2 * base.SAMPLES
        backgrounds[room["room"]] = np.frombuffer(data, "<i2").astype(np.float64)

    sources, chosen = [], set()
    output = args.output
    output.mkdir(mode=0o700)
    (output / "pcm").mkdir()
    rows = []
    for key in sorted(keys, key=rank):
        reader = key.split("-")[0]
        if reader in chosen or set(aligned[key]) != set(intake.ORIGINS):
            continue
        options = intake.possibilities(aligned[key], payloads, key)
        if not options:
            continue
        crop, a, b, spans, _, words = min(options, key=lambda x: rank(f"{key}/{x[1]}/{x[5]}"))
        host = payloads["real"][key][2][crop:crop + base.SAMPLES].astype(np.float64)
        segment = host[a:b]
        possible = [name for name, wave in backgrounds.items()
                    if np.linalg.norm(wave[a:b]) / np.linalg.norm(wave) * np.linalg.norm(host) * 10**.25
                    >= np.linalg.norm(segment)]
        if not possible:
            continue
        room = min(possible, key=lambda name: rank(f"{key}/{name}"))
        group = len(sources)
        sources.append({"group": group, "reader_sha256": base.sources.order(reader), "source_key": key, "room": room,
                        "word_count": words, "crop": [crop, crop + base.SAMPLES], "truth": [a, b],
                        "source_bindings": {kind: {"wav_sha256": payloads[kind][key][0],
                                                   "word_tier_sha256": aligned[key][kind][1], "word_span": spans[kind]}
                                            for kind in intake.ORIGINS}})
        chosen.add(reader)
        versions = {"untouched": host}
        for origin in intake.ORIGINS[1:]:
            left, right = spans[origin]
            donor = payloads[origin][key][2][left:right].astype(np.float64)
            edited = intake.resample(donor, len(segment))
            ordinary = intake.resample(intake.resample(segment, len(donor)), len(segment))
            for kind, phrase in ((f"edited/{origin}", edited), (f"resampled/{origin}", ordinary)):
                phrase *= np.linalg.norm(segment) / np.linalg.norm(phrase)
                versions[kind] = intake.replace(host, phrase, a, b)

        for level in LEVELS:
            delivered, isolated = mix(np.stack(list(versions.values())), backgrounds[room], level)
            panel = "vc_dry" if level is None else f"vc_{level}dB"
            values = dict(zip(versions, delivered, strict=True))
            local_db = background_sha = None
            if isolated is not None:
                values["background"] = isolated
                background_sha = SOURCE.digest(isolated.tobytes())
                ordinary = delivered[0, a:b].astype(np.float64) - isolated[a:b]
                local_db = round(float(20 * np.log10(np.linalg.norm(ordinary) / np.linalg.norm(isolated[a:b]))), 6)
                assert local_db <= .02
            for name, pcm in values.items():
                if name != "background":
                    assert np.array_equal(pcm[:a], delivered[0, :a]) and np.array_equal(pcm[b:], delivered[0, b:])
                digest_pcm = SOURCE.digest(pcm.tobytes())
                assert digest_pcm not in old, "An earlier same-word delivery was reselected"
                (output / "pcm" / f"{digest_pcm}.pcm").write_bytes(pcm.tobytes())
                rows.append({"panel": panel, "room": room, "group": group, "pcm_sha256": digest_pcm,
                             "kind": name, "truth": [[a, b]] if name.startswith("edited/") else [],
                             "local_foreground_to_background_db": local_db, "isolated_pcm_sha256": background_sha})
        if len(sources) == 16:
            break
    assert sources, "No new declared-reader pairs survived the frozen selection"
    packet = {"schema": SCHEMA, "method": METHOD, "publisher_sha256": SOURCE.path_sha(PUBLISHER),
              "prefix_bytes": EXTENDED_BYTES if args.extension else PREFIX_BYTES,
              "earlier_inputs_sha256": SOURCE.path_sha(HISTORICAL), "fitting_sources_sha256": digest,
              "room_plan": rooms, "archive_hashes": pins, "sources": sources, "cases": rows,
              "code_sha256": SOURCE.path_sha(Path(__file__)), "head_ledger_sha256": SOURCE.path_sha(args.fit / "heads.json")}
    prior.save(output / "freeze.json", packet)
    print(f"Froze {len(sources)} new source readers and {len(rows)} deliveries.", flush=True)


def score(args):
    from app.detectors.audio.short_voice_review import scan
    from scripts.run_audio_specialists_v15 import load_voice

    protocol = prior.read(args.media / "freeze.json")
    assert protocol["schema"] == SCHEMA and protocol["method"] == METHOD
    assert protocol["code_sha256"] == SOURCE.path_sha(Path(__file__))
    assert protocol["head_ledger_sha256"] == SOURCE.path_sha(args.fit / "heads.json")
    activity = prior.runtime(args.weights / "silero_vad.jit")
    encoder = experiment.features.frozen_encoder(args.encoder)
    assets = prior.read(experiment.panels.PRIMARY)["incumbent"]["assets"]
    incumbent = load_voice({"weights": args.weights, "assets": assets})
    heads = prior.read(args.fit / "heads.json")
    rows = protocol["cases"]
    cache = {}

    class Deadline:
        def check(self, _stage):
            pass

    for row in rows:
        key = row["pcm_sha256"]
        if key not in cache:
            pcm = base.pcm(args.media, row)
            reading = scan(pcm, activity.spans(pcm), incumbent, Deadline(), codec="pcm_s16le")
            result = reading["selected"]
            cache[key] = (encoder(pcm), {"assessed": bool(reading["responses"]),
                                        "range": [result["start_sample"], result["end_sample"]] if result else None})
    observed = [row | {"current": cache[row["pcm_sha256"]][1], "candidates": {}} for row in rows]
    for head in heads:
        name = head["arm"]
        model, norm = experiment.features.load_head(args.fit, {"trained_family": name, "weights_sha256": head["sha256"]})
        measured = base.score_rows(model, norm, args.media, rows, None, activity,
                                   feature_of=lambda _, row: cache[row["pcm_sha256"]][0])
        for row, result in zip(observed, measured, strict=True):
            row["candidates"][name] = result["best"]
    record = {"freeze": prior.read(args.fit / "freeze.json"), "heads": heads, "rows": observed, "source_manifest": protocol}
    record["results"] = experiment.review(record)
    assert Counter(row["panel"] for row in observed) == {
        f"vc_{level}dB" if level is not None else "vc_dry": len(protocol["sources"]) * (5 if level is None else 6)
        for level in LEVELS}
    prior.save(args.output, record)


def export(args):
    from app.bench import independent
    from app.bench.independent_locations import TRUTH_SCHEMA, count_events

    receipt = prior.read(args.receipt)
    assert receipt["results"] == experiment.review(receipt)
    roster = receipt["source_manifest"]
    assert roster["schema"] == SCHEMA and roster["cases"] == [
        {key: value for key, value in row.items() if key not in {"current", "candidates"}} for row in receipt["rows"]
    ]
    rooms = {row["room"]: row["pcm_sha256"] for row in roster["room_plan"]}
    sources = {row["group"]: row for row in roster["sources"]}
    cases, exact = [], []
    for row in receipt["rows"]:
        background = row["kind"] == "background"
        source = sources[row["group"]]
        binding = source["source_bindings"]
        origin = row["kind"].split("/", 1)[1] if row["truth"] else "real"
        ident = f"{row['panel']}/{row['group']}/{row['kind']}"
        reference = rooms[row["room"]] if background else binding[origin]["wav_sha256"]
        group = f"declared-reader/{source['reader_sha256']}"
        condition = row["panel"] + ("/background" if background else "")
        cases.append({"id": ident, "modality": "audio", "split": "confirmation", "group": group,
                      "label": int(bool(row["truth"])), "family": origin, "condition": condition,
                      "source_sha256": reference, "input_sha256": row["pcm_sha256"],
                      "lineage": [group, rooms[row["room"]]] if row["isolated_pcm_sha256"] else [group],
                      "license": {"id": "CC BY 4.0 (PhonemeDF and AMI)", "evaluation_permitted": True,
                                  "redistribution_permitted": True},
                      "localization_truth": {"unit": "samples", "source_units": base.SAMPLES,
                                             "target_units": sum(b-a for a, b in row["truth"])}})
        exact.append({"id": ident, "input_sha256": row["pcm_sha256"], "extent": [base.SAMPLES], "regions": row["truth"]})
    manifest = {"schema": independent.SCHEMA, "sample_rate_hz": base.RATE, "cases": cases}
    digest = independent.audit_manifest(manifest)["manifest_sha256"]
    common = {"manifest_sha256": digest, "split": "confirmation"}
    truth = common | {"schema": TRUTH_SCHEMA, "cases": exact}
    args.output.mkdir(mode=0o700)
    prior.save(args.output / "manifest.json", manifest)
    prior.save(args.output / "truth.json", truth)
    arms = {"current": None} | {head["arm"]: head for head in receipt["heads"]}
    baseline = prior.read(experiment.panels.PRIMARY)["incumbent"]["assets"]
    service = {key: SOURCE.path_sha(ROOT / key) for key in (
        "backend/app/detectors/audio/short_voice_review.py", "backend/app/evidence/audio_short_review_binding.py",
    )}
    for name, head in arms.items():
        point = head["points"]["strict"] if head else None
        predictions = []
        for case, row in zip(cases, receipt["rows"], strict=True):
            assessed, selected = (row["current"][key] for key in ("assessed", "range"))
            if point is not None and selected is None:
                candidate = row["candidates"][name]
                assessed = candidate is not None
                if assessed and candidate["score"] > point:
                    selected = candidate["range"]
            outcome = {"id": case["id"], "input_sha256": row["pcm_sha256"]}
            if assessed:
                outcome.update(status="ok", response=int(selected is not None),
                               localization={"status": "ok", "regions": [selected] if selected is not None else []})
            else:
                reason = "No eligible speech window"
                outcome.update(status="not_applicable", response=None, reason=reason,
                               localization={"status": "not_applicable", "reason": reason})
            predictions.append(outcome)
        weights = [baseline["pellav2_detector.pt"], baseline["silero_vad.jit"]]
        if head:
            weights.extend([receipt["freeze"]["encoder"]["assets_sha256"]["pytorch_model.bin"], head["sha256"]])
        response = common | {"system": name, "implementation_sha256": independent.digest(
            {"service": service, "candidate": receipt["freeze"]["code_sha256"] if head else None, "point": point}),
            "method_kind": "weighted", "weight_sha256": weights, "score_kind": "raw",
            "response_units": "Frozen interval flag (0 or 1)", "predictions": predictions}
        counts = count_events(manifest, truth, response, minimum_iou=.30, minimum_recall=.50)
        for panel, study in receipt["results"].items():
            expected = study["current"] if name == "current" else study[name]["strict"]["fallback"]
            matched = [row for row in counts["strata"] if row["condition"] == panel and row["label"] == 1]
            assert sum(row["complete_positive_inputs"] for row in matched) == expected["useful"]
            assert sum(row["reported_events"] for row in counts["strata"]
                       if row["condition"] == panel and row["label"] == 0) == expected["ordinary_alerts"]
            assert sum(row["reported_events"] for row in counts["strata"]
                       if row["condition"] == panel + "/background") == expected["isolated_alerts"]
        prior.save(args.output / f"{name}.json", response)
        prior.save(args.output / f"{name}-event-counts.json", counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    for name, action, paths in (("prepare", prepare, ("data", "old_cache", "cache", "background", "fit", "output")),
                                ("score", score, ("media", "fit", "weights", "encoder", "output")),
                                ("export", export, ("receipt", "output"))):
        cmd = commands.add_parser(name)
        for path in paths:
            cmd.add_argument(path, type=Path)
        if name == "prepare":
            cmd.add_argument("--extension", type=Path)
        cmd.set_defaults(action=action)
    args = parser.parse_args()
    args.action(args)
