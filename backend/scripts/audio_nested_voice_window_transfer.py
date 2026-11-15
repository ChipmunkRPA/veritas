# SPDX-License-Identifier: MIT
"""Test a narrower location inside an existing speech alert on unseen deliveries."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import wave
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from scripts import audio_input_contribution_transfer as signal
from scripts import audio_shared_mel_context as earlier

ROOT = earlier.source.ROOT
PANELS = earlier.PANELS
SOURCE = earlier.source
RATE = 16000
SAMPLES = 6 * RATE
BASE_POINT = 6.822523597593649
DENSE_POINT = 5.3355751
ROLES = ("real", "MeloTTS", "ChatterboxTTS")
ROOMS = {"ES2012b": "dd6f2caa7dc2ca8357964efa4d252a025151e11de4ce496b3b12d89bfd6464ba",
         "IS1001b": "164dda162e5e8e851cb56b629116e916835d571b6b4cdf3d164337297b0a853d"}
LEVELS = (0, -5)
GROUPS = 8
SCHEMA = "chipmunk.audio-nested-voice-window/1"
INPUT_PARENT = PANELS / "audio-input-contribution-transfer-2026-11-14.json"
METHOD = {
    "source": "Select eight distinct PhonemeDF prompt readers by hashed order from the existing qualified publisher "
              "prefixes. Match the whole publisher word tier across real speech, MeloTTS and Chatterbox TTS, "
              "exclude exact named historical source inputs, and select one or two words in two length strata. "
              "Publish prior reader exposure separately. Each reader supplies one native six-second host crop.",
    "controls": "Replace the same words with the publisher-rendered words or a duration-matched round-trip resampling "
                "of the original words. Use a shared gain, 20 ms fades, exact samples outside the changed interval, "
                "matched unedited files and isolated real meeting audio.",
    "background": "Use disjoint six-second excerpts from two local native AMI captures. Require at least 1.2 seconds "
                  "of overlapping annotated speakers overall and overlap for at least half of the edit. "
                  "Mix the same background at 0 and -5 dB, measured over the six-second host clip.",
    "comparison": "Retain the highest flagged two-second result of the current five-window PCM check. "
                  "Only consider one-second candidates wholly inside that range. Reuse the frozen dense-response "
                  "point, and choose the largest dense response, with earlier ties. Require half of the edit "
                  "covered and at least 0.30 IoU. The nested rule can change the shown range only.",
}


def _rank(value):
    return SOURCE.digest(f"{SCHEMA}/{value}".encode())


def _inputs(args):
    from scripts import audio_phoneme_short_edit_intake as intake
    from scripts import review_audio_melo_short_sources as melo

    intake_receipt = SOURCE.read(melo.PARENT)
    census = SOURCE.read(PANELS / "audio-melo-short-source-census-2026-11-13.json")
    arrays, pins = {}, {}
    for key, total in (("real", melo.GENUINE_ARCHIVE), ("MeloTTS", melo.MELO_ARCHIVE), ("ChatterboxTTS", 7848724770)):
        if key == "MeloTTS":
            paths = melo.parts(args.melo, total)
            expected = census["melo_range_sha256"]
        else:
            paths = intake.archive(args.aligned, key, total, intake.PREFIX, args.earlier)
            expected = intake_receipt["bounded_wav_ranges"][key]
        pins[key] = [SOURCE.path_sha(path) for path in paths]
        SOURCE.check(pins[key] == expected, "Publisher range commitment differs")
        carrier = "real/wav" if key == "real" else key
        arrays[key] = melo.wavs(paths, rf"{carrier}/(\d+-\d+-\d+)\.wav")
    shared = set.intersection(*(set(bank) for bank in arrays.values()))
    grids = intake.archive(args.aligned, "textgrids", intake.TEXTGRID_BYTES, intake.TEXTGRID_BYTES,
                           complete_md5=intake.TEXTGRID_MD5)
    labels = defaultdict(dict)
    for path, body in intake.prior.archive_members(grids):
        matched = re.fullmatch(r"textgrids/(real|MeloTTS|ChatterboxTTS)/(\d+-\d+-\d+)\.TextGrid", path)
        if matched and matched[2] in shared:
            origin, key = matched.groups()
            if spans := intake.words(body, len(arrays[origin][key][2])):
                labels[key][origin] = spans, SOURCE.digest(body)
    return arrays, labels, pins


def _ranges(labels, native, key):
    import numpy as np

    if set(labels) != set(ROLES) or len(native["real"][key][2]) < SAMPLES:
        return
    real = labels["real"][0]
    if any([word for _, _, word in labels[origin][0]] != [word for _, _, word in real] for origin in ROLES[1:]):
        return
    for word, count in itertools.product(range(1, len(real) - 2), (1, 2)):
        spans = {origin: [tier[word][0], tier[word + count - 1][1]] for origin, (tier, _) in labels.items()}
        start, end = spans["real"]
        width = end - start
        if not 4800 <= width <= 15360 or any(not .8 <= (b - a) / width <= 1.25 for a, b in spans.values()):
            continue
        if any(np.linalg.norm(native[origin][key][2][a:b].astype(np.float64)) / math.sqrt(b-a) < 140
               for origin, (a, b) in spans.items()):
            continue
        lower = max(0, end + RATE - SAMPLES)
        upper = min(len(native["real"][key][2]) - SAMPLES, start - RATE)
        crops = range(((lower + 1599) // 1600) * 1600, upper + 1, 1600)
        crop = min(crops, key=lambda p: _rank((key, word, count, p)), default=None)
        if crop is not None:
            yield {"crop": crop, "truth": [start-crop, end-crop], "native": spans, "word_count": count}


def _select(arrays, labels, history):
    from scripts import review_audio_melo_short_sources as melo

    historical = melo.known_speakers(history)
    trained = set.union(*historical.values())
    past = SOURCE.read(history / melo.HISTORY["v9"][0])["sources"]
    blocked = {value.removeprefix("librispeech:") for row in past for field in ("source_key", "source_group", "id")
               if (value := row.get(field) or "").startswith("librispeech:") or re.fullmatch(r"\d+-\d+-\d+", value)}
    blocked.update(row["id"].removeprefix("real:") for row in past if row.get("id", "").startswith("real:"))
    eligible = defaultdict(list)
    for key, tiers in labels.items():
        if key not in blocked:
            for option in _ranges(tiers, arrays, key):
                a, b = option["truth"]
                eligible[b-a >= 9600].append((_rank((key, option)), key, option))
    used, chosen = set(), []
    for width_group in (False, True):
        count = 0
        for _, key, option in sorted(eligible[width_group]):
            reader = key.split("-")[0]
            if reader in used:
                continue
            used.add(reader)
            chosen.append({**option, "group": len(chosen), "key": key, "historical_reader": reader in trained,
                           "bindings": {origin: {"wav_sha256": arrays[origin][key][0], "pcm_sha256": arrays[origin][key][1],
                                                "grid_sha256": labels[key][origin][1]} for origin in ROLES}})
            count += 1
            if count == GROUPS // 2:
                break
        SOURCE.check(count == GROUPS // 2, "Not enough disjoint readers in the selected width stratum")
    return chosen


def _room_inputs(history, roster):
    import numpy as np

    from scripts.audio_harmonic_overlap_transfer import room_activity
    from scripts.audio_locate_competing_speech import ANNOTATIONS, PANEL

    old = SOURCE.read(history / "audio-variable-boundary-2026-11-12/roster.json")
    notes = SOURCE.read(PANEL)["freeze"]["annotations_sha256"]
    SOURCE.check(SOURCE.path_sha(ANNOTATIONS) == notes, "Expected AMI word timing release")
    with ZipFile(ANNOTATIONS) as text:
        for room, capture_sha in ROOMS.items():
            activity = room_activity(text, room)
            path = ANNOTATIONS.parent / f"{room}.Array1-01.wav"
            SOURCE.check(SOURCE.path_sha(path) == capture_sha, "Different AMI capture")
            used = {r["background"]["capture_start_sample"] for r in old if r.get("room") == room and "background" in r}
            with wave.open(str(path), "rb") as wav:
                SOURCE.check(wav.getparams()[:3] == (1, 2, RATE), "Unexpected AMI PCM profile")
                for row in roster:
                    if tuple(ROOMS)[row["group"] % len(ROOMS)] != room:
                        continue
                    a, b = row["truth"]
                    options = []
                    for tick in range(900, min(len(activity), wav.getnframes() // 160) - 600, 100):
                        start = tick * 160
                        if any(abs(start - old_start) < SAMPLES for old_start in used):
                            continue
                        scope = activity[tick:tick + 600]
                        support = 160 * int(np.count_nonzero(scope[(a+159)//160:b//160] >= 2))
                        if support * 2 >= b-a and np.count_nonzero(scope >= 2) >= 120 and np.count_nonzero(scope) >= 450:
                            options.append((_rank((row["key"], room, tick)), start, support))
                    _, start, support = min(options)
                    wav.setpos(start)
                    raw = wav.readframes(SAMPLES)
                    SOURCE.check(len(raw) == 2*SAMPLES, "AMI range is truncated")
                    used.add(start)
                    yield row["group"], raw, {"room": room, "capture_sha256": capture_sha, "start_sample": start,
                                              "edit_overlap_samples": support, "pcm_sha256": SOURCE.digest(raw)}


def _materialize(args):
    import numpy as np

    from scripts import audio_phoneme_short_edit_intake as intake
    from scripts.audio_odss_matched_room import rms

    arrays, labels, archive = _inputs(args)
    roster = _select(arrays, labels, args.history)
    backgrounds = {group: (raw, bind) for group, raw, bind in _room_inputs(args.history, roster)}
    rows, bank = [], {}
    seen = {row["pcm_sha256"] for panel in SOURCE.read(INPUT_PARENT)["panels"].values() for row in panel}
    for source in roster:
        index, key = source["group"], source["key"]
        left = source["crop"]
        a, b = source["truth"]
        host = arrays["real"][key][2][left:left+SAMPLES].astype(np.float64)
        word = host[a:b]
        variants = {"untouched": host}
        for family in ROLES[1:]:
            x, y = source["native"][family]
            donor = arrays[family][key][2][x:y].astype(np.float64)
            ordinary = intake.resample(intake.resample(word, len(donor)), len(word))
            synthetic = intake.resample(donor, len(word))
            for kind, material in (("ordinary", ordinary), ("edited", synthetic)):
                variants[f"{kind}/{family}"] = intake.replace(host, material * rms(word) / rms(material), a, b)
        raw, background = backgrounds[index]
        masker = np.frombuffer(raw, "<i2").astype(np.float64)
        masker *= rms(host) / rms(masker)
        peak = max(np.max(abs(value) + abs(masker) * 10 ** .25) for value in variants.values())
        gain = 26000 / peak
        for db in LEVELS:
            mix = np.rint(masker * gain * 10 ** (-db / 20)).astype(np.int32)
            prepared = {kind: np.rint(value * gain).astype(np.int32) + mix for kind, value in variants.items()}
            if db == -5:
                prepared["background"] = mix
            for kind, pcm in prepared.items():
                if kind != "background":
                    control = prepared["untouched"]
                    SOURCE.check(np.array_equal(pcm[:a], control[:a]) and np.array_equal(pcm[b:], control[b:]),
                                 "Prepared change escaped its verified edit range")
                SOURCE.check(np.max(abs(pcm)) <= 26001, "Prepared recording clips")
                data = pcm.astype("<i2").tobytes()
                digest = SOURCE.digest(data)
                SOURCE.check(digest not in seen, "New delivery repeats an earlier exact input")
                bank[digest] = data
                rows.append({"group": index, "condition_db": db, "kind": kind, "pcm_sha256": digest,
                             "truth": [a, b] if kind.startswith("edited/") else None})
        source["background"] = background
        source.pop("key")
    return {"archive": archive, "sources": roster, "clips": rows}, bank


def _dense(model, pcm, spans, before):
    import torch

    from scripts.audio_trimmed_mel_normalization import capture_windows

    if before is None:
        return [None] * len(earlier.STARTS)
    with torch.inference_mode():
        output = capture_windows(pcm, spans, lambda _, x: signal._frames(model, signal._normalized_input(x)[1]).numpy())
    return [round(value, 8) if value is not None and before[0] <= a and a + earlier.WIDTH <= before[1] else None
            for a, value in zip(earlier.STARTS, output, strict=True)]


def _readout(row):
    scored = [(value, -start) for start, value in zip(earlier.STARTS, row["nested"], strict=True)
              if value is not None and value > DENSE_POINT]
    if not scored:
        return row["current"]
    _, negative = max(scored)
    start = -negative
    return [start, start + earlier.WIDTH]


def review(record):
    lock = record["freeze"]
    SOURCE.check(lock["schema"] == SCHEMA and lock["method"] == METHOD and lock["points"] == [BASE_POINT, DENSE_POINT]
                 and lock["exploration_sha256"] == SOURCE.path_sha(INPUT_PARENT), "Method or points differ")
    stats = defaultdict(Counter)
    for clip, scores in zip(lock["plan"]["clips"], record["measurements"], strict=True):
        SOURCE.check(scores["pcm_sha256"] == clip["pcm_sha256"], "A measured source differs from the frozen plan")
        current, candidate = scores["current"], _readout(scores)
        SOURCE.check((current is None) == (candidate is None) and (current is None or
                     current[0] <= candidate[0] < candidate[1] <= current[1]), "Nested range changed the alert or escaped its scope")
        truth = clip["truth"]
        family = clip["kind"].split("/", 1)[0]
        tally = stats[f"{clip['condition_db']}/{family}"]
        tally["files"] += 1
        tally["unassessed"] += not scores["speech_admitted"]
        tally["alerts"] += current is not None
        tally["narrowed"] += current != candidate
        if truth:
            useful_before = bool(current and earlier.stock.useful(current, truth))
            useful_after = bool(candidate and earlier.stock.useful(candidate, truth))
            tally["useful_current"] += useful_before
            tally["useful_nested"] += useful_after
            tally["useful_gains"] += useful_after and not useful_before
            tally["useful_losses"] += useful_before and not useful_after
    return {key: dict(value) for key, value in sorted(stats.items())}


def _run(args):
    import numpy as np
    import torch

    from app.detectors.audio import short_voice_review
    from app.detectors.audio.upgraded import SpeechActivity
    from app.evidence.audio_short_review_binding import REVIEW_POINT
    from scripts.audio_strong_traffic_view import network_free
    from scripts.prepare_audio_specialists_v15 import ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    SOURCE.check(REVIEW_POINT == BASE_POINT and SOURCE.read(INPUT_PARENT)["points"]["dense"] == DENSE_POINT,
                 "Recorded serving or candidate points have changed")
    args.path.mkdir(mode=0o700)
    plan, bank = _materialize(args)
    lock = {"schema": SCHEMA, "method": METHOD, "points": [BASE_POINT, DENSE_POINT], "plan": plan,
            "runner_sha256": SOURCE.path_sha(Path(__file__)), "assets": ASSETS,
            "exploration_sha256": SOURCE.path_sha(INPUT_PARENT)}
    SOURCE.write(args.path / "freeze.json", lock)
    print(f"Froze {len(plan['sources'])} readers and {len(plan['clips'])} deliveries before inference", flush=True)
    network_free()
    model = load_voice({"weights": args.weights, "assets": ASSETS})
    torch.set_num_threads(2)
    speech = SpeechActivity(args.weights / "silero_vad.jit")

    class Deadline:
        def check(self, _):
            pass  # Archive evaluation measures all planned inputs outside the worker queue.

    rows, cache = [], {}
    with (args.path / "scores.jsonl").open("x") as journal:
        for clip in plan["clips"]:
            digest = clip["pcm_sha256"]
            if digest not in cache:
                pcm = np.frombuffer(bank[digest], "<i2").astype(np.float32) / 32768
                spans = speech.spans(pcm)
                result = short_voice_review.scan(pcm, spans, model, Deadline(), codec="pcm_s16le")
                selected = result["selected"]
                current = [selected["start_sample"], selected["end_sample"]] if selected is not None else None
                cache[digest] = {"pcm_sha256": digest, "current": current, "speech_admitted": bool(result["responses"]),
                                 "nested": _dense(model, pcm, spans, current)}
            row = cache[digest]
            rows.append(row)
            journal.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            journal.flush()
            if len(rows) % 8 == 0 or len(rows) == len(plan["clips"]):
                print(f"Measured {len(rows)}/{len(plan['clips'])}", flush=True)
    result = {"freeze": lock, "measurements": rows}
    result["outcomes"] = review(result)
    SOURCE.write(args.path / "receipt.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("measure", "review"))
    parser.add_argument("path", type=Path)
    for field in ("melo", "aligned", "earlier"):
        parser.add_argument(f"--{field}", type=Path)
    parser.add_argument("--history", type=Path, default=ROOT / "backend/data/independent-evaluation")
    parser.add_argument("--weights", type=Path, default=ROOT / "models/weights/audio-upgrade")
    args = parser.parse_args()
    if args.phase == "measure" and any(getattr(args, field) is None for field in ("melo", "aligned", "earlier")):
        parser.error("Native measurement needs the three qualified publisher caches")
    receipt = _run(args) if args.phase == "measure" else SOURCE.read(args.path)
    outcomes = review(receipt)
    SOURCE.check(receipt["outcomes"] == outcomes, "Stored outcomes differ from replay")
    print(json.dumps(outcomes, sort_keys=True))
