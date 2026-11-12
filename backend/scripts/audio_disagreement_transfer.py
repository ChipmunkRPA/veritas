"""Compare a selective second voice opinion under matched recorded overlap."""

from __future__ import annotations

import argparse
import math
import resource
import time
import wave
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from scipy.special import expit, logit

from app.detectors.audio import calibration
from scripts import audio_paired_nuisance_transfer as paired
from scripts import audio_spectra_background_transfer as spectra
from scripts import prepare_natural_overlap_transfer as ami
from scripts.audio_short_edit_cases_v172 import exact_source, sources
from scripts.prepare_audio_specialists_v15 import ASSETS

RATE, SAMPLES = 16000, 48000
PANELS = ("calibration", "confirmation")
ARMS = ("current", "paired", "independent", "disagreement")
ROOM_RECEIPT = paired.PANEL / "audio-stream-presence-transfer-2026-11-11.json"
MATERIAL = paired.PANEL / "audio-short-edit-materialization-complete-v169c.json"
EARLIER = paired.BASE / "spectra-recovery-2026-11-11/comparison"
COUNTS = ({"genuine": 7, "clone": 2, "expressive": 2, "stock": 2},
          {"genuine": 8, "clone": 3, "expressive": 3, "stock": 3})
CAL_PATH = paired.ROOT / "backend/app/detectors/audio/calibrations/local-speech-v3.json"
METHOD = {
    "room_cuts": "One deterministic unused annotated two-voice and three-voice cut per room. First room calibrates.",
    "mixture": "Use exact -5 dB recorded background mixes and one same-gain clean control per foreground.",
    "paired": "The existing frozen nuisance correction, with its original operating point.",
    "independent": "The independent checkpoint's score must strictly exceed the maximum genuine calibration score. "
                   "Preserve original and unassessed directions.",
    "disagreement": "Only restore a current AI flag withheld by paired correction. Require the independent score to "
                    "exceed every disputed genuine or isolated background score in calibration. Use the standalone point "
                    "if no disputed genuine score is available. Do not alter the admitted span.",
    "advance": "Restore a current synthetic AI flag lost by the correction in the three-voice interference condition, "
               "lose no current synthetic AI flags in that condition, and make no additional genuine false flags versus "
               "current there or versus paired on clean and isolated controls. Overall noisy false flags must remain "
               "below current without a net synthetic loss. Any positive result is local evidence only.",
}
read, write, sha, require = paired.read, paired.write, paired.digest, paired.require


def foregrounds(role, quotas, previous):
    path = sources.MEDIA / role / "private-manifest.json"
    require(sha(path) == read(MATERIAL)["roles"][role]["private_manifest_sha256"], "Foreground receipt differs")
    selected = []
    for family, count in quotas.items():
        eligible = [row for row in read(path)["foregrounds"] if row["lane"] == family and row["status"] == "qualified"
                    and (family == "genuine" or row["group_id"] not in previous)]
        groups = set()
        for row in sorted(eligible, key=lambda item: ami.salt(f"voice-disagreement/1:{item['slot']}")):
            if row["group_id"] not in groups:
                groups.add(row["group_id"])
                selected.append((exact_source(row, role), {k: row[k] for k in
                                 ("group_id", "voice", "lane", "original_sha256", "prefix_pcm_sha256")}))
                if len(groups) == count:
                    break
        require(len(groups) == count, f"Insufficient distinct {family} foreground groups in {role}")
    return selected, sha(path)


def room_cuts(annotations, rooms):
    plan = ami.make_plan(annotations)
    receipt = read(ROOM_RECEIPT)["freeze"]
    require(sha(annotations) == ami.ANNOTATION_SHA256, "Native annotations changed")
    with ZipFile(annotations) as corpus:
        for index, room in enumerate(plan["rooms"]):
            name = room["meeting"]
            path = rooms / f"{name}.wav"
            digest = sha(path)
            require(receipt["native_sources"][index] == {"group": index, "sha256": digest}, "Room recording changed")
            tracks = [ami.track_words(corpus.read(f"words/{name}.{key}.words.xml")) for key in "ABCD"]
            candidates = ami.candidate_windows(tracks)
            excluded = {row["begin_centiseconds"] for row in room["cases"]}
            with wave.open(str(path), "rb") as stream:
                require(stream.getparams()[:3] == (1, 2, RATE), "Room sample clock changed")
                for voices, possible in candidates.items():
                    times = [start for start in possible if start not in excluded]
                    require(bool(times), "Expected a new cut outside the earlier overlap test")
                    start = min(times, key=lambda t: ami.salt(f"voice-disagreement/1:{name}:{t}")) * 160
                    stream.setpos(start)
                    raw = stream.readframes(SAMPLES)
                    require(len(raw) == SAMPLES * 2, "Room recording was truncated")
                    yield raw, {"group": index, "meeting": name, "start_sample": start, "annotated_voices": voices,
                                "source_sha256": digest, "pcm_sha256": spectra.sha(raw)}


def encoded(values):
    return np.rint(values).astype("<i2").tobytes()


def views(front, backgrounds):
    voice = np.frombuffer(front, "<i2").astype(np.float64)
    audio = [np.frombuffer(raw, "<i2").astype(np.float64) for raw in backgrounds]
    mixed = [voice + values * (np.linalg.norm(voice) / np.linalg.norm(values)) * 10**.25 for values in audio]
    gain = min(1., 31129 / max(float(np.abs(values).max()) for values in [voice, *mixed]))
    return [encoded(values * gain) for values in [voice, *mixed]]


def identity(rows, key):
    return {item[key] for _, item in rows}


def inputs(args):
    earlier_public = paired.read(paired.PANEL / "audio-spectra-recovery-comparison-2026-11-11.json")
    earlier_lock = read(EARLIER / "freeze.json")
    require(sha(EARLIER / "freeze.json") == earlier_public["freeze_sha256"]
            and sha(EARLIER / "roster.json") == earlier_lock["roster_sha256"], "Earlier comparison receipts changed")
    earlier = {row["group"] for row in read(EARLIER / "roster.json") if row["role"] == "prospective"}
    left, left_sha = foregrounds("development", COUNTS[0], earlier)
    right, right_sha = foregrounds("holdout", COUNTS[1], earlier)
    for key in ("group_id", "voice", "original_sha256", "prefix_pcm_sha256"):
        require(not identity(left, key) & identity(right, key), f"Cross-role foreground overlap in {key}")
    previous = paired.paths()
    v84 = read(previous["variants"])["variants"]
    historical = {row[k] for row in v84 for k in ("group", "source_sha256", "pcm_sha256")}
    require(not any(identity(left + right, key) & historical for key in
                    ("group_id", "original_sha256", "prefix_pcm_sha256")), "Source overlaps paired model fit archive")
    cuts = list(room_cuts(args.annotations, args.rooms))
    return (left, right), cuts, {"development": left_sha, "holdout": right_sha, "v84_variants": sha(previous["variants"]),
                                 "earlier_roster": sha(EARLIER / "roster.json")}


def protocol(args):
    paired.checked(args.paired)
    point, fitted = args.paired / "points.json", args.paired / "fitted-private.json"
    require(read(point)["fitted_sha256"] == sha(fitted) and read(fitted)["freeze_sha256"] == sha(args.paired / "freeze.json"),
            "Frozen correction and point differ")
    require(sha(CAL_PATH) == calibration.SHA256, "Serving calibration changed")
    return {"method": METHOD, "code_sha256": sha(Path(__file__)), "paired_points_sha256": sha(point),
            "paired_fitted_sha256": sha(fitted), "incumbent_assets": ASSETS, "calibration_sha256": sha(CAL_PATH),
            "secondary_assets": spectra.pinned_assets(args.assets), "room_receipt_sha256": sha(ROOM_RECEIPT),
            "room_annotation_sha256": sha(args.annotations)}


def freeze(args):
    lock = protocol(args)
    foregrounds_by_role, cuts, commitments = inputs(args)
    args.output.mkdir(mode=0o700)
    folder = args.output / "pcm"
    folder.mkdir()
    manifest = []
    for panel, clean_rows in zip(PANELS, foregrounds_by_role, strict=True):
        take = [(raw, row) for raw, row in cuts if (row["group"] == 0) == (panel == "calibration")]
        conditions = [{"condition": "clean", "group": None, "meeting": None, "annotated_voices": 0}] + [
            {"condition": f"{row['annotated_voices']}_voices", **row} for _, row in take]
        planned = [(item, ref, payload, False) for raw, ref in clean_rows for item, payload in
                   zip(conditions, views(raw, [blob for blob, _ in take]), strict=True)]
        planned.extend((item, {"lane": "background", "group_id": item["meeting"]}, payload, True)
                       for item, (payload, _) in zip(conditions[1:], take, strict=True))
        for item, ref, payload, isolated in planned:
            ordinal = len(manifest)
            (folder / f"{ordinal:03d}.pcm").write_bytes(payload)
            manifest.append({**ref, **item, "id": ordinal, "panel": panel, "isolated": isolated,
                             "source_room_pcm_sha256": item.get("pcm_sha256"), "pcm_sha256": spectra.sha(payload)})
    write(args.output / "roster.json", manifest)
    write(args.output / "freeze.json", {"schema": "chipmunk.audio-disagreement-transfer/1", **lock,
          "source_receipts": commitments, "rooms": [row for _, row in cuts],
          "roster_sha256": sha(args.output / "roster.json"), "views": dict(Counter(row["panel"] for row in manifest))})


def checked(args):
    lock = read(args.output / "freeze.json")
    require(all(lock[k] == value for k, value in protocol(args).items()) and sha(args.output / "roster.json") == lock["roster_sha256"],
            "Frozen policy, checkpoint or view roster changed")
    return lock, read(args.output / "roster.json")


def voice_scores(args, rows):
    from app.detectors.audio.upgraded import SpeechActivity
    from app.detectors.video.adapters.audio import PellaVoiceDetector
    from app.detectors.video.adapters.common import AudioTrack
    from app.engine.boundary import Deadline
    from scripts.audio_noise_speech_gate_v256 import CountedVAD
    from scripts.run_audio_specialists_v15 import load_voice

    voice = load_voice({"weights": str(args.weights), "assets": ASSETS})
    activity = SpeechActivity(args.weights / "silero_vad.jit")
    activity.model = gate = CountedVAD(activity.model)
    detector = PellaVoiceDetector()
    fitted = read(args.paired / "fitted-private.json")
    center, correction = np.asarray(fitted["center"]), np.asarray(fitted["correction"])
    _, _, weight, bias = paired.checked(args.paired)
    require(voice.head.weight.detach().numpy()[0].tolist() == weight.tolist()
            and voice.head.bias.detach().item() == bias, "Voice head differs from paired fit")
    point = read(args.paired / "points.json")["points"]["paired_correction"]["point"]
    capture = []

    def trace(_, inputs):
        capture.extend(inputs[0].detach().numpy().astype(np.float64))

    hook = voice.head.register_forward_pre_hook(trace)
    observed, forwards = [], 0
    try:
        for row in rows:
            audio = np.frombuffer(spectra.exact(args.output, row), "<i2").astype(np.float32) / 32768
            ctx = SimpleNamespace(profile="full", lease=SimpleNamespace(get=lambda _: SimpleNamespace(obj=activity)),
                                  outputs=SimpleNamespace(audio=AudioTrack("analysed", audio, SAMPLES)))
            admitted = detector._windows(ctx)
            old, raw, corrected = "unassessed", None, None
            if admitted:
                require(len(admitted) == 1 and admitted[0][0] == 0 and np.array_equal(admitted[0][1], audio), "Admitted audio changed")
                response, = voice.predict([audio])
                embedding, = capture
                require(abs(response - float(expit(embedding @ weight + bias))) < 1e-6, "Raw response mismatch")
                raw = float(logit(np.clip(response, 1e-6, 1 - 1e-6)))
                corrected = raw - float((embedding - center) @ correction)
                verdict = calibration.evaluate(response, audio, admitted, voice, Deadline(120))
                require(verdict["scope_supported"], "Frozen calibration rejected admitted scope")
                old = verdict["decision"]
                forwards += len(capture)
                capture.clear()
            observed.append({"id": row["id"], "pcm_sha256": row["pcm_sha256"], "current": old,
                             "raw": raw, "corrected": corrected, "paired": paired.decision(old, corrected, point)})
            if len(observed) % 16 == 0:
                print(len(observed), "views with incumbent and correction", flush=True)
    finally:
        hook.remove()
    return observed, {"incumbent_forwards": forwards, "activity_forwards": gate.calls}


def secondary_scores(args, rows, voice):
    session = spectra.start_session(args.assets)
    observed = []
    for row, baseline in zip(rows, voice, strict=True):
        require((row["id"], row["pcm_sha256"]) == (baseline["id"], baseline["pcm_sha256"]), "Voice row binding differs")
        value, digest = None, None
        if baseline["current"] != "unassessed":
            value, digest = spectra.candidate_score(session, spectra.exact(args.output, row))
        observed.append({"id": row["id"], "pcm_sha256": row["pcm_sha256"], "secondary_score": value, "input_sha256": digest})
        if len(observed) % 16 == 0:
            print(len(observed), "views with independent decisions", flush=True)
    return observed, {"independent_forwards": sum(row["secondary_score"] is not None for row in observed)}


def ledger(folder, panel):
    voice, secondary = (read(folder / f"{panel}-{mode}.json")["rows"] for mode in ("voice", "secondary"))
    return [{**a, **b} for a, b in zip(voice, secondary, strict=True)]


def fit(rows):
    controls = [row for row in rows if row["lane"] in ("genuine", "background") and row["secondary_score"] is not None]
    require(len({row["group_id"] for row in controls if row["lane"] == "genuine"}) >= 6, "Too few genuine calibration groups")
    scores = [row["secondary_score"] for row in controls]
    disputes = [row for row in controls if row["current"] == "ai" and row["paired"] == "inconclusive"]
    return {"independent": max(scores), "disagreement": max((row["secondary_score"] for row in disputes), default=max(scores)),
            "genuine_or_background_views": len(scores), "disputed_control_views": len(disputes),
            "disputed_control_groups": len({row["group_id"] for row in disputes})}


def outcomes(row, points):
    current, projected, score = (row[key] for key in ("current", "paired", "secondary_score"))
    supported = current == "ai" and projected == "inconclusive" and score > points["disagreement"]
    return {"current": current, "paired": projected, "independent": paired.decision(current, score, points["independent"]),
            "disagreement": "ai" if supported else projected}


def joint(rows, scored):
    require(len(rows) == len(scored) and all((a["id"], a["pcm_sha256"]) == (b["id"], b["pcm_sha256"])
            for a, b in zip(rows, scored, strict=True)), "Roster and scores differ")
    return [{**row, **score} for row, score in zip(rows, scored, strict=True)]


def summarize(rows, points):
    cells = {}
    for row in rows:
        kind = row["lane"] if row["lane"] in ("genuine", "background") else "synthetic"
        cell = "isolated" if row["isolated"] else row["condition"]
        section = cells.setdefault(cell, {arm: Counter() for arm in ARMS})
        for arm, value in outcomes(row, points).items():
            section[arm][f"{kind}_{value}"] += 1
    return {cell: {arm: dict(values) for arm, values in section.items()} for cell, section in cells.items()}


def advance(rows, points, result):
    changed = [(row, outcomes(row, points)) for row in rows]
    synth_three = [v for row, v in changed if row["condition"] == "3_voices" and row["lane"] not in ("genuine", "background")]
    def count(cell, arm, kind):
        return result[cell][arm].get(f"{kind}_ai", 0)
    def noisy_total(arm, kind):
        return sum(count(cell, arm, kind) for cell in ("2_voices", "3_voices"))
    return {"three_voice_restoration": any(v["current"] == v["disagreement"] == "ai" and v["paired"] != "ai" for v in synth_three),
            "three_voice_no_lost_current": not any(v["current"] == "ai" and v["disagreement"] != "ai" for v in synth_three),
            "three_voice_no_false_increase_vs_current": count("3_voices", "disagreement", "genuine") <= count("3_voices", "current", "genuine"),
            "no_added_clean_or_background_false_flags": all(count(cell, "disagreement", kind) <= count(cell, "paired", kind)
                                                        for cell, kind in (("clean", "genuine"), ("isolated", "background"))),
            "noisy_false_reduction_vs_current": noisy_total("disagreement", "genuine") < noisy_total("current", "genuine"),
            "no_noisy_net_synthetic_loss_vs_current": noisy_total("disagreement", "synthetic") >= noisy_total("current", "synthetic")}


def run(args):
    lock, rows = checked(args)
    mode, panel = args.stage, args.panel
    if mode == "fit":
        part = joint([r for r in rows if r["panel"] == "calibration"], ledger(args.output, "calibration"))
        write(args.output / "points.json", {"freeze_sha256": sha(args.output / "freeze.json"), "points": fit(part),
              "calibration_ledgers_sha256": {mode: sha(args.output / f"calibration-{mode}.json") for mode in ("voice", "secondary")}})
        return
    points_path = args.output / "points.json"
    points = read(points_path) if mode == "report" or panel == "confirmation" else None
    require(points is None or points["freeze_sha256"] == sha(args.output / "freeze.json"), "Held scoring requires frozen calibration")
    if mode == "report":
        result, digests = {}, {}
        for name in PANELS:
            part = joint([r for r in rows if r["panel"] == name], ledger(args.output, name))
            result[name] = summarize(part, points["points"])
            digests[name] = {stage: sha(args.output / f"{name}-{stage}.json") for stage in ("voice", "secondary")}
        advancement = advance(part, points["points"], result["confirmation"])
        write(args.output / "result.json", {"views": lock["views"], "score_ledgers_sha256": digests,
              "results": result, "advancement": advancement, "advances": all(advancement.values()),
              "points_sha256": sha(points_path), "freeze_sha256": sha(args.output / "freeze.json")})
        return
    start = time.monotonic()
    selected = [row for row in rows if row["panel"] == panel]
    if mode == "voice":
        observed, runtime = voice_scores(args, selected)
    else:
        voice = read(args.output / f"{panel}-voice.json")["rows"]
        observed, runtime = secondary_scores(args, selected, voice)
    require(len(observed) == lock["views"][panel] and all(row.get("raw") is None or math.isfinite(row["raw"]) for row in observed),
            "Incomplete panel or nonfinite response")
    write(args.output / f"{panel}-{mode}.json", {"rows": observed, "runtime": runtime | {
          "elapsed_s": time.monotonic() - start, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "voice", "secondary", "fit", "report"))
    for field in ("output", "paired", "annotations", "rooms", "assets", "weights"):
        parser.add_argument(field, type=Path)
    parser.add_argument("--panel", choices=PANELS, default="calibration")
    args = parser.parse_args()
    (freeze if args.stage == "freeze" else run)(args)
