"""Compare an independent CPU voice detector under loud competing backgrounds."""

from __future__ import annotations

import argparse
import math
import resource
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np

from scripts import audio_strong_traffic_view as incumbent
from scripts.prepare_natural_overlap_transfer import ANNOTATION_SHA256, track_words

ROOT, RATE, COUNT, PANEL = incumbent.ROOT, incumbent.RATE, incumbent.COUNT, incumbent.PANEL
read, sha, write_new = incumbent.read, incumbent.sha, incumbent.write_new
CHECKPOINT = "spectra-aasist3-int8-dynamic.onnx"
ASSETS = {CHECKPOINT: "444f832d306a2be4f823119f84e698e8821db6a1aab248593d4b05b7a9a48108",
          "official-README.md": "160d8cf926c19dff4faad87d9f170f77455ed1d2dfa4656d07ced220e29231c7",
          "quantized-README.md": "7f550379e313fcc59beba7c754dabf941b3d0a409b1262cefb03995493279258",
          "quantization_metadata.json": "8e23249f993a9cd30c4ba3c7da9c514343f143af4a1882d85ddc4d7216c9ed0b"}
SHAPE = 64_600
ROOMS = {"mostly_single_interferer": ("ES2012b", 0, 0), "overlapping_interferers": ("IS1001b", 180, 300)}
RULE = {"additional_genuine_max": 1, "additional_clean_genuine_max": 0, "isolated_genuine_max": 0,
        "minimum_new_noisy_synthetic_groups": 3, "minimum_new_noisy_synthetic_families": 2,
        "minimum_genuine_assessed_each_prospective_scene": 6, "minimum_prospective_scenes_with_gain": 2,
        "minimum_point_genuine_groups": 8, "point": "strict excess of the maximum eligible genuine score"}


def pinned_assets(folder):
    result = {name: sha(folder / name) for name in ASSETS}
    if result != ASSETS:
        raise ValueError("Pinned CPU checkpoint or its publisher evidence changed")
    return result


def exact(folder, row):
    raw = (folder / "pcm" / f"{row['id']:03}.pcm").read_bytes()
    if len(raw) != COUNT * 2 or sha(raw) != row["pcm_sha256"]:
        raise ValueError("Frozen source PCM differs")
    return raw


def validate_parent(folder, published, score_path, baseline_key, roster_key="roster_sha256"):
    lock, scores = read(folder / "freeze.json"), read(folder / score_path)
    if (sha(folder / "freeze.json") != published["freeze_sha256"] or lock[roster_key] != sha(folder / "roster.json")
            or scores["freeze_sha256"] != published["freeze_sha256"] or sha(folder / score_path) != published[baseline_key]):
        raise ValueError("Pre-existing source or baseline binding differs")
    result = []
    for row, score in zip(read(folder / "roster.json"), scores["rows"], strict=True):
        if (row["id"], row["pcm_sha256"]) != (score["id"], score["pcm_sha256"]):
            raise ValueError("Pre-existing baseline has a different source")
        result.append((row, score.get("current", score)))
    return lock, result


def room_cuts(folder, annotations):
    bind = read(PANEL / "audio-channel-speech-gap-transfer-2026-11-11.json")["participant_disjoint_confirmation"]["freeze"]
    if sha(annotations) != ANNOTATION_SHA256 or sha(folder / "roster.json") != bind["roster_sha256"]:
        raise ValueError("Earlier native room sources changed")
    old_cuts = read(folder / "roster.json")
    excluded = {room: [row["source_start_sample"] for row in old_cuts
                       if row["panel"] == "native" and row["group"] == f"meeting:{index}"]
                for index, (room, _, _) in enumerate(ROOMS.values(), 1)}
    return verified_room_cuts(folder / "native", annotations, bind["sources"], excluded)


def verified_room_cuts(native, annotations, source_hashes, old_cuts):
    result = {}
    with ZipFile(annotations) as corpus:
        for condition, (room, min_overlap, max_overlap) in ROOMS.items():
            path = native / f"{room}.Array1-01.wav"
            media_hash = sha(path)
            if media_hash != source_hashes[path.name]:
                raise ValueError("Physical AMI channel differs from the publisher receipt")
            excluded = old_cuts[room]
            tracks = [track_words(corpus.read(f"words/{room}.{speaker}.words.xml")) for speaker in "ABCD"]
            active = np.zeros((4, 105_000), dtype=bool)
            for person, words in enumerate(tracks):
                for a, b in words:
                    active[person, a:b] = True
            candidates = []
            for start in range(300, active.shape[1] - 299, 300):
                source_sample = start * 160
                if any(abs(source_sample - cut) < COUNT for cut in excluded):
                    continue
                busy = active[:, start:start + 300].sum(axis=0)
                overlaps = int((busy >= 2).sum())
                if int((busy >= 1).sum()) >= 290 and min_overlap <= overlaps <= max_overlap:
                    candidates.append((sha(f"chipmunk/spectra-background/1:{room}:{start}".encode()), source_sample, overlaps))
            _, source_sample, overlap = min(candidates)
            with wave.open(str(path), "rb") as audio:
                if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, RATE):
                    raise ValueError("Native AMI format differs")
                audio.setpos(source_sample)
                raw = audio.readframes(COUNT)
            if len(raw) != COUNT * 2:
                raise ValueError("Native AMI cut is truncated")
            result[condition] = (raw, {"meeting": room, "archive_sha256": media_hash, "pcm_sha256": sha(raw),
                                       "original_start_sample": source_sample, "annotated_overlap_ms": overlap * 10})
    return result


def mix(clean, background):
    clean, background = (np.frombuffer(raw, "<i2").astype(np.float64) for raw in (clean, background))
    coefficient = np.linalg.norm(clean) / np.linalg.norm(background) * 10**.25
    body = clean + background * coefficient
    gain = min(1., 31_129 / np.max(np.abs(body)))
    return np.rint(body * gain).astype("<i2").tobytes(), float(coefficient), float(gain)


def prepare(folder, assets, point, older, rooms, annotations):
    assets_verified = pinned_assets(assets)
    point_public = read(PANEL / "audio-strong-traffic-view-transfer-2026-11-11.json")["commitments"]
    older_public = read(PANEL / "audio-traffic-enhancement-transfer-2026-11-11.json")
    point_lock, point_rows = validate_parent(point, point_public, "baseline.json", "baseline_sha256")
    older_lock, older_rows = validate_parent(older, older_public, "scores.json", "score_sha256")
    if point_lock["assets"] != older_lock["current_assets"]:
        raise ValueError("Incumbent checkpoints differ across previous panels")
    cuts = room_cuts(rooms, annotations)
    point_clean = [r for r, _ in point_rows if r["condition"] == "clean"]
    held_clean = [(r, score) for r, score in older_rows if r["condition"] == "clean"]
    for key in ("group", "speaker", "source_sha256", "source_pcm_sha256"):
        if {r[key] for r in point_clean} & {r[key] for r, _ in held_clean}:
            raise ValueError(f"Point and confirmation share a foreground {key}")
    (folder / "pcm").mkdir()
    rows = []

    def save(raw, role, condition, reference, score=None, **evidence):
        index = len(rows)
        (folder / "pcm" / f"{index:03}.pcm").write_bytes(raw)
        rows.append({"id": index, "role": role, "condition": condition, "family": reference["family"],
                     "group": reference["group"], "pcm_sha256": sha(raw), "baseline": score, **evidence})

    for row, score in point_rows:
        save(exact(point, row), "point", row["condition"], row, score, tape=row.get("tape"))
    for row, score in held_clean:
        clean = exact(older, row)
        save(clean, "prospective", "clean", row, score, source_sha256=row["source_sha256"])
        for condition, (raw, source) in cuts.items():
            noisy, scale, gain = mix(clean, raw)
            save(noisy, "prospective", condition, row, source_sha256=row["source_sha256"], clean_pcm_sha256=sha(clean),
                 background=source, scale=scale, common_gain=gain)
    for condition, (raw, source) in cuts.items():
        save(raw, "prospective", f"isolated_{condition}", {"family": "background", "group": condition}, background=source)
    for row, score in older_rows:
        if row.get("tape") == "new_sunday":
            save(exact(older, row), "historical_stress", f"traffic_{row['condition'].lstrip('-')}", row, score,
                 noise_pcm_sha256=row["noise_sha256"])
    write_new(folder / "roster.json", rows)
    binding = {"schema": "chipmunk.audio-independent-background-comparison/1", "runner_sha256": sha(Path(__file__)),
               "roster_sha256": sha(folder / "roster.json"), "rule": RULE, "candidate_assets": assets_verified,
               "incumbent_assets": point_lock["assets"], "incumbent_calibration_sha256": sha(ROOT / "backend/app/detectors/audio/calibrations/local-speech-v3.json"),
               "point_parent_freeze_sha256": point_public["freeze_sha256"], "point_parent_baseline_sha256": point_public["baseline_sha256"],
               "held_parent_freeze_sha256": older_public["freeze_sha256"], "held_parent_baseline_sha256": older_public["score_sha256"],
               "ami_annotation_sha256": ANNOTATION_SHA256, "prospective_backgrounds": [details for _, details in cuts.values()],
               "planned": dict(Counter(row["role"] for row in rows)), "input_samples": COUNT, "candidate_samples": SHAPE}
    write_new(folder / "freeze.json", binding)
    print(binding["planned"], flush=True)


def frozen(folder):
    lock, rows = read(folder / "freeze.json"), read(folder / "roster.json")
    if (lock["runner_sha256"] != sha(Path(__file__)) or lock["rule"] != RULE or lock["roster_sha256"] != sha(folder / "roster.json")
            or [r["id"] for r in rows] != list(range(len(rows)))):
        raise ValueError("Source manifest, ordering, or runner changed")
    return lock, rows


def start_session(assets):
    import onnxruntime as ort

    pinned_assets(assets)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.enable_mem_pattern = False
    session = ort.InferenceSession(str(assets / CHECKPOINT), sess_options=options, providers=["CPUExecutionProvider"])
    inp, = session.get_inputs()
    out, = session.get_outputs()
    if (inp.name, inp.type, inp.shape[-1], out.name, out.type, out.shape[-1]) != ("wav", "tensor(float)", SHAPE, "logits", "tensor(float)", 2):
        raise ValueError("Pinned candidate input or output contract differs")
    return session


def candidate_score(session, raw):
    original = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    emphasized = original.copy()
    emphasized[1:] -= np.float32(.97) * original[:-1]
    graph_input = np.tile(emphasized, 2)[:SHAPE][None, :]
    logits, = session.run(["logits"], {"wav": graph_input})
    if logits.shape != (1, 2) or not np.isfinite(logits).all():
        raise ValueError("Invalid independent detection logits")
    return -float(logits[0, 1]), sha(graph_input.astype("<f4").tobytes())


def baseline(folder):
    from app.detectors.audio import calibration
    from app.detectors.audio.upgraded import SpeechActivity
    from app.detectors.video.adapters.audio import PellaVoiceDetector
    from app.detectors.video.adapters.common import AudioTrack
    from app.engine.boundary import Deadline
    from scripts.audio_noise_speech_gate_v256 import CountedVAD
    from scripts.run_audio_specialists_v15 import load_voice

    lock, rows = frozen(folder)
    if sha(ROOT / "backend/app/detectors/audio/calibrations/local-speech-v3.json") != lock["incumbent_calibration_sha256"]:
        raise ValueError("Incumbent calibration bytes changed")
    incumbent.network_free()
    voice = load_voice({"weights": str(incumbent.WEIGHTS), "assets": lock["incumbent_assets"]})
    activity = SpeechActivity(incumbent.WEIGHTS / "silero_vad.jit")
    activity.model = gate = CountedVAD(activity.model)
    detector, decisions = PellaVoiceDetector(), []
    forwards = 0

    def count(*_):
        nonlocal forwards
        forwards += 1

    hook = voice.register_forward_pre_hook(count)
    try:
        for row in rows:
            if row["baseline"] is not None:
                continue
            wave = np.frombuffer(exact(folder, row), "<i2").astype(np.float32) / 32768
            ctx = SimpleNamespace(profile="full", lease=SimpleNamespace(get=lambda _: SimpleNamespace(obj=activity)),
                                  outputs=SimpleNamespace(audio=AudioTrack("analysed", wave, COUNT)))
            selected = detector._windows(ctx)
            outcome, raw = "unassessed", None
            if selected:
                if len(selected) != 1 or selected[0][0] != 0 or not np.array_equal(selected[0][1], wave):
                    raise ValueError("Incumbent admitted a different waveform")
                raw, = voice.predict([wave])
                assessed = calibration.evaluate(raw, wave, selected, voice, Deadline(120))
                if not assessed["scope_supported"]:
                    raise ValueError("Waveform is outside the frozen incumbent calibration")
                outcome = assessed["decision"]
            decisions.append({"id": row["id"], "pcm_sha256": row["pcm_sha256"], "decision": outcome, "raw": raw})
            if len(decisions) % 12 == 0:
                print(f"Fresh incumbent decisions {len(decisions)} ({forwards} model forwards)", flush=True)
    finally:
        hook.remove()
    write_new(folder / "baseline.json", {"freeze_sha256": sha(folder / "freeze.json"), "rows": decisions,
                                         "classifier_forwards": forwards, "gate_forwards": gate.calls,
                                         "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def point_value(rows):
    eligible = [r for r in rows if r["family"] in {"genuine", "background"} and r["decision"] == "inconclusive"]
    groups = {r["group"] for r in eligible if r["family"] == "genuine"}
    if len(groups) < RULE["minimum_point_genuine_groups"]:
        raise ValueError("Too few point-setting genuine groups with eligible speech")
    return {"candidate": max(r["candidate"] for r in eligible), "raw": max(r["raw"] for r in eligible),
            "eligible_views": len(eligible), "genuine_groups": len(groups)}


def score(folder, assets, mode):
    _, rows = frozen(folder)
    freeze_sha = sha(folder / "freeze.json")
    if mode == "point":
        recorded_baseline, point_sha = {}, None
    else:
        stored = read(folder / "baseline.json")
        point_stage = read(folder / "point.json")
        point_sha = sha(folder / "point.json")
        if stored["freeze_sha256"] != freeze_sha or point_stage["freeze_sha256"] != freeze_sha:
            raise ValueError("Baseline and preconfirmation point must be frozen on the exact roster")
        recorded_baseline = {r["id"]: r for r in stored["rows"]}
    incumbent.network_free()
    session = start_session(assets)
    outcomes, forwards, repeat_ok = [], 0, False
    start = time.monotonic()
    for row in rows:
        if (mode == "point") != (row["role"] == "point"):
            continue
        ref = row["baseline"] or recorded_baseline.pop(row["id"])
        if "pcm_sha256" in ref and ref["pcm_sha256"] != row["pcm_sha256"]:
            raise ValueError("Baseline response differs from exact candidate input")
        baseline_raw = ref.get("raw")
        if (baseline_raw is None) != (ref["decision"] == "unassessed"):
            raise ValueError("Incumbent admission and recorded score disagree")
        raw = exact(folder, row)
        score_value, graph = None, None
        if ref["decision"] != "unassessed":
            score_value, graph = candidate_score(session, raw)
            forwards += 1
            if not repeat_ok:
                if candidate_score(session, raw) != (score_value, graph):
                    raise ValueError("First CPU input failed exact deterministic repeat")
                forwards += 1
                repeat_ok = True
        outcomes.append({"id": row["id"], "group": row["group"], "family": row["family"], "pcm_sha256": row["pcm_sha256"],
                         "raw": baseline_raw, "decision": ref["decision"], "candidate": score_value, "model_input_sha256": graph})
        if len(outcomes) % 20 == 0:
            print(f"Independent {mode} decisions {len(outcomes)} ({forwards} model forwards)", flush=True)
    if recorded_baseline:
        raise ValueError("Unused fresh incumbent scores")
    extra = {"points": point_value(outcomes)} if mode == "point" else {"points_sha256": point_sha, "baseline_sha256": sha(folder / "baseline.json")}
    write_new(folder / f"{mode}.json", {**extra, "freeze_sha256": freeze_sha, "rows": outcomes, "forwards": forwards,
                                        "elapsed_seconds": time.monotonic() - start, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def auc(values, key):
    scores = [r[key] for r in values if r[key] is not None]
    positive = [r[key] for r in values if r[key] is not None and r["family"] not in {"genuine", "background"}]
    negative = [r[key] for r in values if r[key] is not None and r["family"] == "genuine"]
    if not positive or not negative or not all(math.isfinite(s) for s in scores):
        return None
    return sum((p > n) + .5 * (p == n) for p in positive for n in negative) / (len(positive) * len(negative))


def readout(folder):
    lock, rows = frozen(folder)
    fit, confirm = (read(folder / f"{stage}.json") for stage in ("point", "confirm"))
    freeze_sha = sha(folder / "freeze.json")
    if (fit["freeze_sha256"] != freeze_sha or confirm["freeze_sha256"] != freeze_sha or fit["points"] != point_value(fit["rows"])
            or confirm["points_sha256"] != sha(folder / "point.json") or confirm["baseline_sha256"] != sha(folder / "baseline.json")):
        raise ValueError("The model results do not bind the fixed points and incumbent")
    measured = {r["id"]: r for r in fit["rows"] + confirm["rows"]}
    cells, ranked, gains, evidence = defaultdict(Counter), defaultdict(list), [], []
    false = Counter()
    for row in rows:
        got = measured.pop(row["id"])
        exact(folder, row)
        if any(row[key] != got[key] for key in ("pcm_sha256", "group", "family")):
            raise ValueError("Result differs from the source roster")
        if row["role"] == "point":
            continue
        role, condition, family = (row[key] for key in ("role", "condition", "family"))
        decision = got["decision"]
        assessed = decision != "unassessed"
        if assessed != (got["raw"] is not None and got["candidate"] is not None):
            raise ValueError("Shared admission must be exact in both arms")
        alternate = assessed and got["candidate"] > fit["points"]["candidate"]
        added = alternate and decision == "inconclusive"
        raw_added = decision == "inconclusive" and got["raw"] > fit["points"]["raw"]
        cells[(role, condition, family)].update(planned=1, assessed=assessed, incumbent_ai=decision == "ai", incumbent_original=decision == "original",
                                                incumbent_inconclusive=decision == "inconclusive", unassessed=not assessed, candidate_alone_ai=alternate,
                                                combined_ai=decision == "ai" or added, raw_point_combined_ai=decision == "ai" or raw_added)
        ranked[(role, condition)].append(got)
        if added:
            evidence.append({"id": row["id"], "pcm_sha256": row["pcm_sha256"], "first_sample": 0, "end_sample": COUNT,
                             "time_base": "1/16000", "family": family, "role": role, "condition": condition,
                             "candidate_input_sha256": got["model_input_sha256"]})
        if not added or role != "prospective":
            continue
        if family in {"genuine", "background"}:
            false[condition] += 1
        elif condition in ROOMS:
            gains.append((condition, got["group"], family))
    if measured:
        raise ValueError("Unused observations")
    checks = {"genuine": sum(false.values()) <= RULE["additional_genuine_max"], "clean": false["clean"] <= RULE["additional_clean_genuine_max"],
              "isolated": sum(v for k, v in false.items() if k.startswith("isolated_")) <= RULE["isolated_genuine_max"],
              "new_groups": len({g for _, g, _ in gains}) >= RULE["minimum_new_noisy_synthetic_groups"],
              "new_families": len({f for _, _, f in gains}) >= RULE["minimum_new_noisy_synthetic_families"],
              "scenes_with_gain": len({c for c, _, _ in gains}) >= RULE["minimum_prospective_scenes_with_gain"],
              "genuine_assessed": all(cells[("prospective", condition, "genuine")]["assessed"] >= RULE["minimum_genuine_assessed_each_prospective_scene"] for condition in ROOMS)}
    write_new(folder / "evidence-private.json", evidence)
    aggregate = {"schema": "chipmunk.audio-independent-background-comparison-evaluation/1", "freeze_sha256": freeze_sha, "freeze": lock,
                 "point_sha256": sha(folder / "point.json"), "confirmation_sha256": sha(folder / "confirm.json"),
                 "evidence_sha256": sha(folder / "evidence-private.json"), "baseline_sha256": confirm["baseline_sha256"],
                 "checks": checks, "advance": all(checks.values()), "counts": [{"role": r, "condition": c, "family": f, **counts} for (r, c, f), counts in sorted(cells.items())],
                 "point_eligibility": {k: fit["points"][k] for k in ("eligible_views", "genuine_groups")},
                 "resources": {s: {k: stage[k] for k in ("forwards", "elapsed_seconds", "peak_rss_bytes")} for s, stage in (("point", fit), ("confirmation", confirm))},
                 "ranking": [{"role": r, "condition": c, "raw_auc": auc(items, "raw"), "candidate_auc": auc(items, "candidate")} for (r, c), items in sorted(ranked.items())],
                 "prospective_gained_groups": len({g for _, g, _ in gains}), "prospective_gained_families": len({f for _, _, f in gains}),
                 "source_evidence_intervals": len(evidence)}
    write_new(folder / "aggregate.json", aggregate)
    print({"checks": checks, "advance": aggregate["advance"], "source_evidence_intervals": len(evidence)}, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("freeze", "point", "baseline", "confirm", "readout"))
    cli.add_argument("folder", type=Path)
    for key in ("assets", "point", "older", "rooms", "annotations"):
        cli.add_argument(f"--{key}", type=Path)
    args = cli.parse_args()
    if args.stage == "freeze":
        prepare(args.folder, args.assets, args.point, args.older, args.rooms, args.annotations)
    elif args.stage == "baseline":
        baseline(args.folder)
    elif args.stage == "readout":
        readout(args.folder)
    else:
        score(args.folder, args.assets, args.stage)
