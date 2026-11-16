# SPDX-License-Identifier: MIT
"""Test genuine-only local speech profiles against short replacements under competing voices."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from app.evidence.audio_short_review_binding import GRID as CURRENT_GRID
from scripts import audio_short_edit_detector_training as base
from scripts import audio_short_edit_ssl_transfer as existing
from scripts import audio_short_edit_union_holdout as union

SOURCE = base.SOURCE
RATE, SAMPLES = base.RATE, base.SAMPLES
PANELS = Path(__file__).resolve().parents[2] / "docs/evaluation-panels"
PRIMARY = PANELS / "audio-short-edit-detector-training-2026-11-14.json"
ENCODER = PANELS / "audio-short-edit-speech-layers-2026-11-14.json"
SINE = PANELS / "audio-short-edit-union-holdout-2026-11-15.json.gz"
STEREO = PANELS / "audio-stereo-channel-replication-2026-11-15.json.gz"
SCHEMA = "chipmunk.audio-local-speech-normality/1"
LAYERS = (3, 9, 12)
PROJECTED_WIDTH = 20
WINDOWS = (6400, 9600, 12800, 16000)
STEP, GUARD = 1600, 1600
SHRINKAGE = .15
FIT_KINDS = {"untouched", "genuine_splice"}
ARMS = ("early", "averaged_layers", "local_contrast")
POINTS = {"tighter": .90, "wider": .75}
FRESH_READERS = 24
METHOD = {
    "fit": "Use only the pinned genuine fitting readers' dry and loud-competing versions, including genuine "
           "splices. Read three frozen encoder layers at native frame centers. Project each layer with fixed "
           "signed orthogonal columns. Fit duration-specific means and shrunk correlations on pooled "
           "per-window means and standard deviations for dry and mixed speech separately.",
    "inference": "Use a single uploaded waveform. A region's response is the smaller of its standardized "
                 "distances from the dry and mixed genuine profiles, without a noise label or any original. "
                 "Compare the early layer, the equal average of all three layers, and the early response "
                 "above both equal-width nearby speech intervals. Keep the largest eligible response and "
                 "withhold context contrast without both flanks.",
    "clock": "Use 0.40, 0.60, 0.80 and 1.00 second intervals every 100 ms. Require half detected speech "
             "in each interval. Flanks have a 100 ms guard. Use only complete original frames.",
    "calibration": "Set cutoffs from separate ordinary file maxima at fixed 90th and 75th percentile ranks, "
                   "both bounded below by the highest isolated-background reading. Calibrate each readout "
                   "independently. Use scores strictly above the point.",
    "held": "Recount exact time overlap, missed edits, ordinary and isolated background flags and matched "
            "controls that flag at the same authored interval. Compare the short serving check and "
            "genuine-only rules separately, then as a nonreplacing fallback. No AI confidence is fitted.",
}
BACKGROUND_PROFILE = (
    "Add a third genuine profile for isolated conversation from the fitting capture only. Keep the dry and "
    "mixed profiles unchanged. Use the smallest standardized distance over all three genuine profiles and "
    "recalculate ordinary operating points with the same separate calibration readers and room controls."
)


def contract(source_hash, *, isolated_profile=False):
    recorded = SOURCE.read(ENCODER)["freeze"]["encoder"]
    encoder = {key: recorded[key] for key in ("revision", "assets_sha256", "weight_terms")}
    freeze = {"schema": SCHEMA, "method": METHOD, "fit_sources_sha256": source_hash, "encoder": encoder,
            "layers": list(LAYERS), "projected_width": PROJECTED_WIDTH, "windows": list(WINDOWS),
            "shrinkage": SHRINKAGE, "speech_sha256": existing.SPEECH,
            "baseline_sha256": SOURCE.path_sha(PRIMARY), "retained_sine_sha256": SOURCE.path_sha(SINE)}
    if isolated_profile:
        freeze["isolated_profile"] = BACKGROUND_PROFILE
    return freeze


def sequence(weights):
    import numpy as np
    import torch

    from scripts import audio_wavlm_features_v13 as encoder

    model = encoder.load(weights)
    rng = np.random.default_rng(611416)
    projections = [torch.from_numpy(np.linalg.qr(rng.choice([-1., 1.], size=(encoder.WIDTH, PROJECTED_WIDTH)))[0].astype(np.float32))
                   for _ in LAYERS]

    @torch.inference_mode()
    def encode(audio):
        waveform = torch.from_numpy(audio)[None]
        response = model(waveform, attention_mask=torch.ones_like(waveform, dtype=torch.long),
                         output_hidden_states=True, return_dict=True)
        SOURCE.check(response.last_hidden_state.shape == (1, existing.FRAMES, encoder.WIDTH), "Unexpected hidden-state clock")
        return torch.stack([response.hidden_states[layer][0] @ projection
                            for layer, projection in zip(LAYERS, projections, strict=True)]).numpy()

    return encode


def runtime(encoder, vad):
    import torch

    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free

    SOURCE.check(SOURCE.path_sha(vad) == existing.SPEECH, "Speech activity weights changed")
    network_free()
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True)
    return sequence(encoder), SpeechActivity(vad)


def available_windows(audio_spans, features):
    import numpy as np

    from app.detectors.audio.content import window_in_speech

    frame_clock = 200 + 320 * np.arange(existing.FRAMES)
    zeros = np.zeros((len(LAYERS), 1, PROJECTED_WIDTH))
    first = np.concatenate((zeros, np.cumsum(features, axis=1, dtype=np.float64)), axis=1)
    second = np.concatenate((zeros, np.cumsum(features.astype(np.float64) ** 2, axis=1)), axis=1)
    result = {}
    for width in WINDOWS:
        ranges = [(a, a + width) for a in range(0, SAMPLES - width + 1, STEP)
                  if window_in_speech(audio_spans, a / RATE, (a + width) / RATE, fraction=.5)]
        indices = np.searchsorted(frame_clock, np.asarray(ranges, dtype=np.int64).reshape(-1, 2))
        a, b = indices.T
        average = (first[:, b] - first[:, a]) / (b - a)[None, :, None]
        variance = (second[:, b] - second[:, a]) / (b - a)[None, :, None] - average ** 2
        result[width] = (ranges, np.concatenate((average, np.sqrt(np.maximum(variance, 0.))), axis=-1))
    return result


def distances(vectors, center, scale, rotation):
    import numpy as np

    whitened = np.matmul((vectors - center[:, None, :]) / scale[:, None, :], rotation)
    return np.sqrt(np.mean(whitened ** 2, axis=-1))


def fit_profiles(eligible, domains):
    import numpy as np

    models = {}
    for (condition, width), batches in eligible.items():
        vectors = np.concatenate(batches, axis=1).astype(np.float64)
        center = vectors.mean(axis=1)
        scale = np.maximum(1e-4, vectors.std(axis=1))
        standardized = (vectors - center[:, None, :]) / scale[:, None, :]
        d = vectors.shape[-1]
        covariance = np.matmul(standardized.transpose(0, 2, 1), standardized) / vectors.shape[1]
        eigenvalues, vectors_base = np.linalg.eigh((1-SHRINKAGE) * covariance + SHRINKAGE * np.eye(d))
        whitening = vectors_base / np.sqrt(eigenvalues[:, None, :])
        training_scores = distances(vectors, center, scale, whitening)
        midpoint = np.median(training_scores, axis=1)
        spread = np.maximum(.01, 1.4826 * np.median(abs(training_scores - midpoint[:, None]), axis=1))
        models[condition, width] = (center, scale, whitening, midpoint, spread)
    SOURCE.check(set(models) == {(name, width) for name in domains for width in WINDOWS},
                 "Missing genuine profiles")
    return models


def readout(available, profiles):
    import numpy as np

    proposals = defaultdict(list)
    domains = sorted({domain for domain, _ in profiles})
    for width, (ranges, vectors) in available.items():
        if not ranges:
            continue
        normalized = []
        for condition in domains:
            center, scale, rotation, midpoint, spread = profiles[condition, width]
            distance = distances(vectors, center, scale, rotation)
            normalized.append((distance - midpoint[:, None]) / spread[:, None])
        compatible = np.minimum.reduce(normalized)
        early = dict(zip(ranges, compatible[0].tolist(), strict=True))
        for (start, end), average in zip(ranges, compatible.mean(axis=0), strict=True):
            score = early[start, end]
            proposals["early"].append((score, -start, -end))
            proposals["averaged_layers"].append((float(average), -start, -end))
            before = early.get((start - GUARD - width, start - GUARD))
            after = early.get((end + GUARD, end + GUARD + width))
            if before is not None and after is not None:
                proposals["local_contrast"].append((score - max(before, after), -start, -end))
    report = {}
    for arm in ARMS:
        results = proposals[arm]
        if results:
            score, start, end = max(results)
            report[arm] = {"score": round(score, 8), "range": [-start, -end]}
        else:
            report[arm] = None
    return report


def points(rows, *, protect_background=True):
    result = {}
    for arm in ARMS:
        foreground, background = [], []
        for row in rows:
            SOURCE.check(row["truth"] is None, "An edited file entered operating-point selection")
            value = row["readings"][arm]
            if value is not None:
                (background if row["kind"] == "background" else foreground).append(value["score"])
        ladder = sorted(foreground)
        SOURCE.check(ladder, "A readout has no ordinary calibration")
        floor = max(background, default=-math.inf) if protect_background else -math.inf
        result[arm] = {key: max(ladder[math.ceil(len(ladder) * q) - 1], floor) for key, q in POINTS.items()}
    return result


def save_profiles(location, profiles):
    import numpy as np

    for (domain, width), arrays in sorted(profiles.items()):
        np.savez(location / f"{domain}-{width}.npz", *[np.asarray(value, dtype="<f8") for value in arrays])
    return {path.name: SOURCE.path_sha(path) for path in sorted(location.glob("*.npz"))}


def load_profiles(folder, sealed):
    import numpy as np

    profiles = {}
    for name, digest in sealed["profiles"].items():
        path = folder / name
        SOURCE.check(SOURCE.path_sha(path) == digest, "Frozen genuine profile changed")
        domain, width = path.stem.rsplit("-", 1)
        with np.load(path, allow_pickle=False) as saved:
            profiles[domain, int(width)] = tuple(saved[f"arr_{i}"] for i in range(5))
    return profiles


def fit(args):
    plan, digest = base.load_plan(args.data)
    args.output.mkdir(mode=0o700)
    encode, activity = runtime(args.encoder, args.vad)
    freeze = contract(digest, isolated_profile=args.extend is not None)
    SOURCE.write(args.output / "freeze.json", freeze)
    eligible = defaultdict(list)
    initial = {}
    if args.extend is not None:
        prior = SOURCE.read(args.extend / "sealed.json")
        SOURCE.check(prior["freeze"] == contract(digest), "The reference fit differs")
        initial = load_profiles(args.extend, prior)
    kinds = {"background"} if initial else FIT_KINDS
    domains = ("isolated",) if initial else ("dry", "competing_speech")
    fitting = [row for row in plan["rows"] if row["role"] == "fit" and row["kind"] in kinds]
    for i, row in enumerate(fitting, 1):
        audio = base.pcm(args.data, row)
        domain = "isolated" if initial else row["condition"]
        for width, (_, array) in available_windows(activity.spans(audio), encode(audio)).items():
            eligible[domain, width].append(array)
        if i == len(fitting) or i % 24 == 0:
            print(f"Read {i}/{len(fitting)} genuine fit waveforms", flush=True)
    fitted = initial | fit_profiles(eligible, domains)
    profile_hashes = save_profiles(args.output, fitted)
    calibration = [row for row in plan["rows"] if row["role"] == "calibration" and row["truth"] is None]
    scores = []
    for i, row in enumerate(calibration, 1):
        audio = base.pcm(args.data, row)
        scores.append(row | {"readings": readout(available_windows(activity.spans(audio), encode(audio)), fitted)})
        if i == len(calibration) or i % 24 == 0:
            print(f"Read {i}/{len(calibration)} separate ordinary and background waveforms", flush=True)
    sealed = {"freeze": freeze, "fit_count": len(fitting), "profiles": profile_hashes, "calibration": scores, "points": points(scores)}
    if initial:
        sealed["earlier_fit_sha256"] = SOURCE.path_sha(args.extend / "sealed.json")
    SOURCE.write(args.output / "sealed.json", sealed)
    print("Froze the genuine profiles and both operating points", flush=True)


def fresh_inputs(args):
    prior = union.duration.read_gz(SINE)
    retained = prior["frozen"]
    union.validate_frozen(retained)
    SOURCE.check(SOURCE.read(args.retained / "freeze.json") == retained, "The earlier SINE delivery plan changed")
    excluded = union.known_readers(union.parent()) | {row["reader"] for row in retained["pairs"]}
    listing = union.SINE.source_listing(args.cache)
    candidates, shards = {}, []
    for shard in union.SHARDS:
        offers, pin = union.duration.previous.select_shard(
            args.cache, shard, listing[f"data/split-{shard}.tar.gz"], excluded)
        for reader, offer in offers.items():
            if reader not in candidates or offer[0] < candidates[reader][0]:
                candidates[reader] = offer
        shards.append(pin)
        print(f"SINE prefix {shard} supplies {len(offers)} unused reader candidates", flush=True)
    picked = sorted(candidates.values(), key=lambda item: item[0])[:FRESH_READERS]
    SOURCE.check(len(picked) >= 12 and shards == retained["shards"], "Too few unused readers or changed publisher prefixes")
    args.output.mkdir(mode=0o700)
    for name in ("sine", "background", "pcm"):
        (args.output / name).mkdir()
    pairs, rooms = [], retained["rooms"]
    for i, (_, row, bodies) in enumerate(picked):
        pairs.append(row | {"id": i, "room": rooms[i % len(rooms)]["room"]})
        for name, data in bodies.items():
            (args.output / "sine" / f"{i:03}-{name}.wav").write_bytes(data)
    for room in rooms:
        path = args.retained / "background" / f"{room['room']}.pcm"
        raw = path.read_bytes()
        SOURCE.check(SOURCE.digest(raw) == room["pcm_sha256"], "The reused independent room changed")
        (args.output / "background" / path.name).write_bytes(raw)
    cases = []
    for item, signal in union.duration.previous.deliveries(args.output, {"pairs": pairs, "backgrounds": rooms}):
        if item["condition"] != "competing":
            continue
        (args.output / "pcm" / f"{item['pcm_sha256']}.pcm").write_bytes(signal.tobytes())
        cases.append(item)
    freeze = {"schema": SCHEMA, "retained_sha256": SOURCE.path_sha(SINE), "pairs": pairs,
              "rooms": rooms, "shards": shards, "cases": cases, "excluded_readers": len(excluded)}
    validate_fresh(freeze, retained)
    SOURCE.write(args.output / "freeze.json", freeze)
    print(f"Froze {len(pairs)} additional reader pairs with {len(rooms)} measured room controls", flush=True)


def validate_fresh(freeze, retained):
    pairs, rooms, cases = (freeze[key] for key in ("pairs", "rooms", "cases"))
    excluded = union.known_readers(union.parent()) | {row["reader"] for row in retained["pairs"]}
    SOURCE.check(freeze["schema"] == SCHEMA and freeze["retained_sha256"] == SOURCE.path_sha(SINE)
                 and rooms == retained["rooms"] and freeze["shards"] == retained["shards"]
                 and len(excluded) == freeze["excluded_readers"]
                 and 12 <= len(pairs) == len({row["reader"] for row in pairs}) <= FRESH_READERS
                 and not excluded.intersection(row["reader"] for row in pairs), "Fresh source identities overlap")
    primary = SOURCE.read(PRIMARY)["sources"]["sources"]
    readers = {row["reader_sha256"] for row in primary}
    for i, row in enumerate(pairs):
        stamps, _ = union.SINE.sine.label_clock(row["publisher_cut_paste_clock"], row["files"]["variant"]["samples"], "cut_paste")
        start, stop, _ = stamps[1]
        duration = min(item["samples"] for item in row["files"].values())
        SOURCE.check(row["id"] == i and row["room"] == rooms[i % len(rooms)]["room"]
                     and row["reader"] == row["stem"].split("-", 1)[0] and base.sources.order(row["reader"]) not in readers
                     and row["shard"] in union.SHARDS and row["native_start"] % 2400 == 0
                     and 0 <= row["native_start"] <= duration - 144000
                     and row["truth"] == [(edge - row["native_start"]) * 2 // 3 for edge in (start, stop)]
                     and 4000 <= row["truth"][0] < row["truth"][1] <= 92000,
                     "The newly selected speaker or publisher edit clock changed")
    old_pcm = {row["pcm_sha256"] for row in retained["cases"] if row["kind"] != "background"}
    expected = [(row["id"], label, truth) for row in pairs for label, truth in
                (("untouched", None), ("edited/sine", row["truth"]))]
    expected.extend((i, "background", None) for i in range(len(rooms)))
    SOURCE.check([(row["group"], row["kind"], row["truth"]) for row in cases] == expected
                 and len({row["pcm_sha256"] for row in cases}) == len(cases)
                 and not old_pcm.intersection(row["pcm_sha256"] for row in cases), "Fresh PCM or stage roles overlap")


def overlap(ranges, truth):
    if not ranges:
        return False
    a, b = ranges
    for lo, hi in truth:
        intersection = max(0, min(b, hi) - max(a, lo))
        union_length = b - a + hi - lo - intersection
        if 2 * intersection >= hi - lo and 10 * intersection >= 3 * union_length:
            return True
    return False


def outcomes(sealed, rows, *, foreground_only=False):
    SOURCE.check(sealed["freeze"] == contract(sealed["freeze"]["fit_sources_sha256"],
                                            isolated_profile="isolated_profile" in sealed["freeze"])
                 and sealed["points"] == points(sealed["calibration"]), "Operating points or method changed")
    frozen = points(sealed["calibration"], protect_background=False) if foreground_only else sealed["points"]
    choices = {}
    measured = defaultdict(Counter)
    seen = set()
    for row in rows:
        panel, kind, group, key = (row[field] for field in ("panel", "kind", "group", "pcm_sha256"))
        SOURCE.check((panel, group, kind) not in seen and set(row["readings"]) == set(ARMS),
                     "Repeated delivery role or missing result")
        seen.add((panel, group, kind))
        spans = row["truth"]
        SOURCE.check((not spans or kind.startswith("edited")) and all(0 <= a < b <= SAMPLES for a, b in spans),
                     "Invalid source edit clock")
        current = row["current"]
        SOURCE.check(type(current["assessed"]) is bool and (current["range"] is None
                     or current["assessed"] and tuple(current["range"]) in CURRENT_GRID), "The serving range differs")
        predictions = {"current": current}
        for arm, values in row["readings"].items():
            if values:
                a, b = values["range"]
                SOURCE.check(0 <= a < b <= SAMPLES and a % STEP == 0 and b - a in WINDOWS
                             and math.isfinite(values["score"]), "Invalid measured short window")
            for level, point in frozen[arm].items():
                chosen = values["range"] if values and values["score"] > point else None
                rule = f"{arm}/{level}"
                predictions[rule] = {"assessed": values is not None, "range": chosen}
                predictions[f"fallback/{rule}"] = {"assessed": current["assessed"] or values is not None,
                                                    "range": current["range"] or chosen}
        SOURCE.check((panel, key) not in choices or choices[panel, key] == predictions,
                     "The same waveform has different readings")
        choices[panel, key] = predictions
        for name, result in predictions.items():
            selected = result["range"]
            hit = overlap(selected, spans)
            base_hit = overlap(current["range"], spans)
            state = "edited" if spans else "background" if kind == "background" else "ordinary"
            bucket = measured[panel, name, state]
            if spans:
                measured[panel, name, kind].update(files=1, useful=hit, flagged=bool(selected))
            bucket.update(files=1, assessed=result["assessed"], flags=bool(selected), useful=hit,
                          misplaced=bool(spans and selected and not hit),
                          new_locations=bool(hit and not base_hit), lost_locations=bool(base_hit and not hit),
                          new_ordinary=bool(state == "ordinary" and selected and not current["range"]))
    controls = defaultdict(list)
    for row in rows:
        if not row["truth"]:
            controls[row["panel"], row["group"]].append(row["pcm_sha256"])
            if row["kind"] == "background" and row["panel"].startswith("sine_"):
                controls[row["panel"], "room", row["room"]].append(row["pcm_sha256"])
    for row in rows:
        if not (truth := row["truth"]):
            continue
        panel = row["panel"]
        matching = controls[panel, row["group"]] + controls[panel, "room", row["room"]]
        for name, selected in choices[panel, row["pcm_sha256"]].items():
            if overlap(selected["range"], truth):
                collision = any(overlap(choices[panel, key][name]["range"], truth) for key in matching)
                measured[panel, name, "edited"].update(clear_locations=not collision, same_clock_alerts=collision)
    return {"/".join(key): dict(value) for key, value in sorted(measured.items())}


def primary_inputs(plan, primary):
    baseline = {item["pcm_sha256"]: {"assessed": item["assessed"], "range": item["selected"]}
                for item in primary["incumbent"]["results"]}
    subjects = {row["group"]: row["background"]["room"] for row in plan["sources"]}
    held = [row for row in plan["rows"] if row["role"] == "held" and row["condition"] == "competing_speech"]
    SOURCE.check(len(held) == len(primary["incumbent"]["results"])
                 and Counter(row["pcm_sha256"] for row in held)
                 == Counter(row["pcm_sha256"] for row in primary["incumbent"]["results"]),
                 "Some serving comparison roles were missing")
    for row in held:
        yield (row | {"truth": [row["truth"]] if row["truth"] else [], "panel": "phoneme",
                       "room": subjects[row["group"]]}), baseline[row["pcm_sha256"]]


def inputs(args, plan, primary, retained):
    for row, reading in primary_inputs(plan, primary):
        yield row, base.pcm(args.data, row), reading
    frozen = retained["frozen"]
    for path, label, packet in ((args.retained, "sine_retained", frozen), (args.fresh, "sine_new",
                                                                          SOURCE.read(args.fresh / "freeze.json"))):
        if label == "sine_retained":
            SOURCE.check(SOURCE.read(path / "freeze.json") == frozen, "Retained SINE sources differ")
            saved = {item["pcm_sha256"]: item["reading"] for item in retained["current"]["rows"]}
        else:
            validate_fresh(packet, frozen)
            saved = {}
        for row in packet["cases"]:
            is_background = row["kind"] == "background"
            room = packet["rooms"][row["group"]]["room"] if is_background else packet["pairs"][row["group"]]["room"]
            remap = row | {"truth": [row["truth"]] if row["truth"] else [], "panel": label, "room": room,
                           "group": -row["group"] - 1 if is_background else row["group"]}
            yield remap, base.pcm(path, row), saved.get(row["pcm_sha256"])
    if args.stereo:
        from scripts import audio_stereo_channel_replication as stereo

        packet = union.duration.read_gz(STEREO)
        calibration = packet["plan"]["contract"]
        SOURCE.check(packet["plan"] == SOURCE.read(args.stereo / "freeze.json")
                     and packet["results"] == stereo.review(packet), "Stereo source labels or serving replay changed")
        for row, prior in zip(packet["plan"]["cases"], packet["observed"], strict=True):
            pcm = stereo.prior.arithmetic_mono(stereo.voices.load_stereo(args.stereo, row["pcm_sha256"]))
            key = stereo.prior.audio_hash(pcm)
            SOURCE.check(key == prior["mono"]["pcm_sha256"], "Stereo downmix changed")
            selected, assessed = stereo.decisions(prior, calibration)["mono"]
            remap = {"pcm_sha256": key, "panel": f"stereo_{row['room']}", "room": row["room"],
                     "group": row["scene"], "kind": f"edited/{row['kind']}" if row["labels"] else row["kind"],
                     "truth": list(row["labels"].values())}
            yield remap, pcm, {"assessed": assessed, "range": selected}


def score(args):
    from app.detectors.audio.short_voice_review import scan
    from scripts.run_audio_specialists_v15 import load_voice

    plan, digest = base.load_plan(args.data)
    primary, retained = SOURCE.read(PRIMARY), union.duration.read_gz(SINE)
    union.validate_frozen(retained["frozen"])
    SOURCE.check(all(plan[key] == value for key, value in primary["sources"].items()), "The primary roster changed")
    sealed = SOURCE.read(args.fit / "sealed.json")
    SOURCE.check(sealed["freeze"] == contract(digest, isolated_profile="isolated_profile" in sealed["freeze"])
                 and sealed["points"] == points(sealed["calibration"]),
                 "Genuine fit or calibration changed")
    profiles = load_profiles(args.fit, sealed)
    previously_measured = {}
    if args.prior_readout:
        previous = json.loads(gzip.decompress(args.prior_readout.read_bytes()))
        SOURCE.check(previous["outcomes"] == outcomes(previous["sealed"], previous["rows"])
                     and previous["fresh"] == SOURCE.read(args.fresh / "freeze.json"), "The original serving comparison changed")
        previously_measured = {(row["panel"], row["pcm_sha256"]): row["current"] for row in previous["rows"]}
    current = None if previously_measured else load_voice({"weights": args.weights, "assets": primary["incumbent"]["assets"]})
    encode, activity = runtime(args.encoder, args.weights / "silero_vad.jit")

    class Deadline:
        def check(self, _stage):
            pass

    rows, cache = [], {}
    for source, pcm, incumbent in inputs(args, plan, primary, retained):
        key = source["pcm_sha256"]
        if key not in cache:
            spans = activity.spans(pcm)
            cache[key] = readout(available_windows(spans, encode(pcm)), profiles), spans
        readings, spans = cache[key]
        if incumbent is None:
            if previously_measured:
                incumbent = previously_measured[source["panel"], source["pcm_sha256"]]
            else:
                ready = scan(pcm, spans, current, Deadline(), codec="pcm_s16le")
                chosen = ready["selected"]
                incumbent = {"assessed": bool(ready["responses"]),
                             "range": [chosen["start_sample"], chosen["end_sample"]] if chosen else None}
        rows.append(source | {"current": incumbent, "readings": readings})
        if len(rows) % 20 == 0:
            print(f"Measured {len(rows)} held waveforms, through {source['panel']}", flush=True)
    packet = {"sealed": sealed, "fresh": SOURCE.read(args.fresh / "freeze.json"),
              "calibration_sha256": SOURCE.path_sha(args.fit / "sealed.json"), "rows": rows,
              "stereo_sha256": SOURCE.path_sha(STEREO) if args.stereo else None,
              "outcomes": outcomes(sealed, rows)}
    raw = json.dumps(packet, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    args.output.write_bytes(gzip.compress(raw, mtime=0))
    report(packet)


def report(packet, *, foreground_only=False):
    summary = outcomes(packet["sealed"], packet["rows"])
    SOURCE.check(summary == packet["outcomes"], "Replayed locations differ")
    union_data = union.duration.read_gz(SINE)
    calibrated = packet["sealed"]["calibration"]
    SOURCE.check(len(calibrated) == 144 and len({row["group"] for row in calibrated}) == 16
                 and all(row["role"] == "calibration" for row in calibrated)
                 and SOURCE.digest((json.dumps(packet["sealed"], sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
                 == packet["calibration_sha256"], "The ordinary fit and cutoffs have a different commitment")
    primary = SOURCE.read(PRIMARY)
    source = {row["group"]: row for row in primary["sources"]["sources"] if row["role"] == "held"}
    previous = {row["pcm_sha256"]: {"assessed": row["assessed"], "range": row["selected"]}
                for row in primary["incumbent"]["results"]}
    compared = [row for row in packet["rows"] if row["panel"] == "phoneme"]
    SOURCE.check(Counter(row["pcm_sha256"] for row in compared)
                 == Counter(row["pcm_sha256"] for row in primary["incumbent"]["results"])
                 and all(row["current"] == previous[row["pcm_sha256"]]
                         and row["group"] in source and row["room"] == source[row["group"]]["background"]["room"]
                         and row["truth"] == ([source[row["group"]]["truth"]] if row["kind"].startswith("edited/") else [])
                         for row in compared)
                 and {group for group, count in Counter(row["group"] for row in compared).items() if count == 7} == set(source),
                 "The primary paired roles or timing comparison changed")
    stereo = [row for row in packet["rows"] if row["panel"].startswith("stereo_")]
    if stereo:
        from scripts import audio_stereo_channel_replication as replica

        SOURCE.check(packet["stereo_sha256"] == SOURCE.path_sha(STEREO), "The stereo comparison ledger changed")
        readouts = union.duration.read_gz(STEREO)
        for row, native, previous in zip(stereo, readouts["plan"]["cases"], readouts["observed"], strict=True):
            selected, assessed = replica.decisions(previous, readouts["plan"]["contract"])["mono"]
            SOURCE.check(row["pcm_sha256"] == previous["mono"]["pcm_sha256"]
                         and row["truth"] == list(native["labels"].values())
                         and row["panel"] == f"stereo_{native['room']}" and row["group"] == native["scene"]
                         and row["current"] == {"assessed": assessed, "range": selected},
                         "Changed stereo case or baseline reading")
    validate_fresh(packet["fresh"], union_data["frozen"])
    for name, rows in (("sine_new", packet["fresh"]["cases"]), ("sine_retained", union_data["frozen"]["cases"])):
        measured = [row for row in packet["rows"] if row["panel"] == name]
        SOURCE.check([(r["pcm_sha256"], r["truth"]) for r in measured]
                     == [(r["pcm_sha256"], [r["truth"]] if r["truth"] else []) for r in rows], "SINE score pairing differs")
        if name == "sine_retained":
            SOURCE.check([row["current"] for row in measured] == [row["reading"] for row in union_data["current"]["rows"]],
                         "The retained serving reference changed")
    if foreground_only:
        alternative = {"points": points(packet["sealed"]["calibration"], protect_background=False),
                       "outcomes": outcomes(packet["sealed"], packet["rows"], foreground_only=True)}
        print(json.dumps(alternative, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    fitting = actions.add_parser("fit")
    for key in ("data", "encoder", "vad", "output"):
        fitting.add_argument(key, type=Path)
    fitting.add_argument("--extend", type=Path)
    fresh = actions.add_parser("fresh")
    for key in ("cache", "retained", "output"):
        fresh.add_argument(key, type=Path)
    measured = actions.add_parser("score")
    for key in ("data", "retained", "fresh", "fit", "encoder", "weights", "output"):
        measured.add_argument(key, type=Path)
    measured.add_argument("--stereo", type=Path)
    measured.add_argument("--prior-readout", type=Path)
    replay = actions.add_parser("review")
    replay.add_argument("receipt", type=Path)
    replay.add_argument("--foreground-only-points", action="store_true",
                        help="Recount the diagnostic without the isolated-conversation floor")
    args = parser.parse_args()
    if args.command == "fit":
        fit(args)
    elif args.command == "fresh":
        fresh_inputs(args)
    elif args.command == "score":
        score(args)
    else:
        report(json.loads(gzip.decompress(args.receipt.read_bytes())), foreground_only=args.foreground_only_points)


if __name__ == "__main__":
    main()
