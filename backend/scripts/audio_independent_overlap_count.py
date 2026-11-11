"""Measure an independent speaker-count veto on exact retained voice inputs."""
from __future__ import annotations

import argparse
import hashlib
import math
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_measured_speaker_routing as prior

UPSTREAM = "ae15e1ac096862667a7bfdedf0b67a70a7543edd"
ASSETS = {
    "CRNN.h5": "00f67eb2aa2749bae9d45f3642b9d031b16f372aff929f74699b5a04a266c98d",
    "scaler.npz": "90ec0061d27ef224687598e715562933a6ba0849dc9aab447cafad6f0259a7a7",
    "LICENSE": "ff823d0ff3dadd8e59572a9c825c598993673e3bba34ae7f78181cae56e45ca5",
    "5_speakers.wav": "d9ba22bb9cfa3493b8e0cbe85174d0a859558cee828f239433cc5688f8014e65",
}
PREVIOUS_FREEZE = "799fa698f3b20b8d0916092faa960778641b1ec6e05ffdc92206bc612d803b43"
PREVIOUS_MANIFEST = "509034819fb11ba344d5616173d97f6cae0d9413d3647276b9921e312da0e95c"
PREVIOUS_READOUT = "11118e315499846f715e348f7f7a83cd198ee789c065d914b21369271cc25638"
READER_ROSTER = "aeb1723512015d8dca40790a4d0e64da360822cf243fa9d3646634edd8331e35"
VAD_SHA = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"
POSITIONS = {"left": 8000, "center": 16000, "right": 24000}
PADDED_SAMPLES = 5 * prior.RATE
CONFIRM = "separate_room_confirmation_replay"
FRESH = "disjoint_reader_count_controls"
METHOD = {
    "candidate": "Author-released CountNet CRNN, read original HDF5 weights as arrays; explicit CPU inference",
    "features": "5 s mono 16 kHz, 400-sample periodic Hann, 160-sample hop and reflect STFT boundary, first 500 magnitude frames, exact published scaler, mean frame L2 normalization",
    "views": "One unchanged 3-second PCM input surrounded by digital silence in each of three positions",
    "disjoint_reader_controls": "Salted groups from older English and Turkish source banks absent from the prior routing panels. Use three-second center cuts and disjoint mixed-language triads with at least 400 ms of joint separate-source speech. Include one, two, and two third-voice levels at fixed minus 5 dB foreground/background SNR.",
    "left_pad_samples": POSITIONS,
    "overlap": "Argmax count of at least three; center-only and at least two of three views tested separately",
    "route": "Withhold the existing corroborated two-voice route on a counted overlap. Never promote a route.",
    "advance": "Eliminate controlled three-voice leaks on both the retained and disjoint-reader panels, retain at least half of fresh two-voice controls, and keep at least five of seven added true flags with zero added false flags in separate-room confirmation.",
}


def padded(source, left: int):
    import numpy as np

    if source.shape != (prior.SAMPLES,) or left not in POSITIONS.values():
        raise ValueError("Exact three-second source and a frozen input position required")
    return np.pad(source, (left, PADDED_SAMPLES - len(source) - left))


def features(audio, mean, scale):
    import numpy as np

    if audio.shape != (PADDED_SAMPLES,) or not np.isfinite(audio).all():
        raise ValueError("Exactly five finite seconds of mono PCM required")
    frame_length, hop = 400, 160
    hann = np.sin(np.pi * np.arange(frame_length) / frame_length) ** 2
    reflected = np.pad(audio, frame_length // 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(reflected, frame_length)[::hop][:500]
    magnitudes = np.abs(np.fft.rfft(frames * hann, axis=1))
    standardized = (magnitudes - mean) / scale
    normalized = standardized / (np.linalg.norm(standardized, axis=1).mean() + np.finfo(np.float64).eps)
    return normalized.astype("float32")


class CountNet:
    def __init__(self, folder: Path):
        import h5py
        import numpy as np
        import torch

        self.torch = torch
        with np.load(folder / "scaler.npz", allow_pickle=False) as scaler:
            self.mean, self.scale = scaler["arr_0"], scaler["arr_1"]
        if self.mean.shape != (201,) or self.scale.shape != (201,) or (self.scale <= 0).any():
            raise ValueError("Unexpected CountNet feature scaler")
        with h5py.File(folder / "CRNN.h5", "r") as checkpoint:
            weights = checkpoint["model_weights"]

            def read_weight(layer, part):
                return torch.from_numpy(weights[layer][f"{layer}_{part}"][:].astype("float32"))

            self.conv = [(read_weight(layer, "W").flip((-2, -1)), read_weight(layer, "b"))
                         for layer in ("conv1", "conv2", "conv3", "conv4")]
            self.lstm = [torch.cat([read_weight("lstm_1", f"{part}_{gate}") for gate in "ifco"], dim=-1)
                         for part in ("W", "U", "b")]
            self.dense = read_weight("dense_1", "W"), read_weight("dense_1", "b")

    def score(self, audio) -> list[float]:
        torch, layer = self.torch, self.torch.nn.functional
        with torch.inference_mode():
            value = torch.from_numpy(features(audio, self.mean, self.scale))[None, None, ...]
            for index, (kernel, bias) in enumerate(self.conv):
                value = layer.conv2d(value, kernel, bias).relu()
                if index in (1, 3):
                    value = layer.max_pool2d(value, 3)
            sequence = value.permute(0, 2, 1, 3).reshape(53, 1280)
            input_weight, recurrence, bias = self.lstm
            hidden = torch.zeros(40)
            memory = torch.zeros_like(hidden)
            hidden_rows = []
            for affine in sequence @ input_weight + bias:
                ingate, forget, update, outgate = (affine + hidden @ recurrence).chunk(4)
                memory = (forget * 0.2 + 0.5).clamp(0, 1) * memory + (ingate * 0.2 + 0.5).clamp(0, 1) * update.tanh()
                hidden = (outgate * 0.2 + 0.5).clamp(0, 1) * memory.tanh()
                hidden_rows.append(hidden)
            pooled = torch.stack(hidden_rows)[:52].reshape(26, 2, 40).amax(dim=1).flatten()
            weight, offset = self.dense
            return (pooled @ weight + offset).softmax(dim=0).tolist()


def decode(blob):
    import numpy as np

    return np.frombuffer(blob, "<i2").astype("float64") / 32768


def count(probabilities: list[float]) -> int:
    if (len(probabilities) != 11 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in probabilities)
            or abs(sum(probabilities) - 1) > 1e-6):
        raise ValueError("Invalid CountNet distribution")
    return max(range(11), key=probabilities.__getitem__)


def route_arms(measurements, counts):
    base = prior.routes(measurements)["corroborated"]
    if set(counts) != set(POSITIONS) or any(type(n) is not int or not 0 <= n <= 10 for n in counts.values()):
        raise ValueError("Missing or invalid independently measured count")
    return {"diarization": base,
            "center_veto": "overlap_withheld" if base == "single_interferer" and counts["center"] >= 3 else base,
            "repeat_veto": "overlap_withheld" if base == "single_interferer" and sum(n >= 3 for n in counts.values()) >= 2 else base}


def load(folder: Path):
    bindings = {name: prior.sha(folder / name) for name in ASSETS}
    if bindings != ASSETS:
        raise ValueError("Exact author checkpoint, scaler, license and demonstration required")
    pinned = {"previous-freeze.json": PREVIOUS_FREEZE, "previous-manifest.json": PREVIOUS_MANIFEST,
              "previous-measured.json": PREVIOUS_READOUT}
    if any(prior.sha(folder / name) != digest for name, digest in pinned.items()):
        raise ValueError("Exact previously measured routing panel required")
    retained = prior.read(folder / "previous-manifest.json")["rows"]
    fresh = prior.read(folder / "fresh-manifest.json")["rows"]
    rows = retained + fresh
    if len(retained) != 146 or len(fresh) < 20 or [r["id"] for r in rows] != list(range(len(rows))):
        raise ValueError("Missing complete ordered count panel")
    if any(r["panel"] != FRESH for r in fresh):
        raise ValueError("Incorrect panel role")
    old = prior.read(folder / "previous-measured.json")["rows"]
    freeze = {"schema": "chipmunk.independent-overlap-count-freeze/1", "method": METHOD,
              "upstream_revision": UPSTREAM, "assets": ASSETS, "prior": pinned,
              "code_sha256": prior.sha(Path(__file__)), "fresh_manifest_sha256": prior.sha(folder / "fresh-manifest.json"),
              "planned_rows": len(rows)}
    return rows, old, freeze


def prepare_fresh(folder: Path, source: Path, vad: Path):
    import numpy as np
    import torch

    from app.detectors.audio.upgraded import SpeechActivity

    roster = source / "roster.json"
    if prior.sha(roster) != READER_ROSTER or prior.sha(vad) != VAD_SHA:
        raise ValueError("Pinned existing reader roster and speech-activity weights required")
    retained_path = folder / "previous-manifest.json"
    if prior.sha(retained_path) != PREVIOUS_MANIFEST:
        raise ValueError("Missing retained routing manifest")
    retained = prior.read(retained_path)["rows"]
    retained_groups = {r["group"] for r in retained}
    retained_hashes = {r["pcm_sha256"] for r in retained}
    rows = [r for r in prior.read(roster) if r["view"] == "clean" and r["lane"] in {"english_genuine", "turkish_genuine"}]
    if len(rows) != len({r["group"] for r in rows}) or retained_groups.intersection(r["group"] for r in rows):
        raise ValueError("Declared reader groups must be distinct from the original panel")
    torch.set_num_threads(2)
    activity = SpeechActivity(vad)
    pools = defaultdict(list)
    for row in sorted(rows, key=lambda r: hashlib.sha256(f"overlap-count/1:{r['group']}".encode()).hexdigest()):
        track = (source / "pcm" / f"{row['id']}.pcm").read_bytes()
        if len(track) != 112000 or hashlib.sha256(track).hexdigest() != row["track_sha256"]:
            raise ValueError("The exact gain-matched clean track is required")
        blob = track[8000:104000]  # 250 ms into the retained 3.5-second track.
        digest = hashlib.sha256(blob).hexdigest()
        if digest in retained_hashes:
            raise ValueError("Clean reader excerpt duplicates the retained panel")
        audio = decode(blob)
        active = np.zeros(prior.SAMPLES, dtype=bool)
        for a, b in activity.spans(audio.astype("float32")):
            active[round(a * prior.RATE):round(b * prior.RATE)] = True
        pools[row["lane"]].append((audio / np.linalg.norm(audio), active, digest))
    english, turkish = pools["english_genuine"], pools["turkish_genuine"]
    output, roster_counts = [], Counter()
    for group in range(9):
        major, minor = (english, turkish) if group % 2 == 0 else (turkish, english)
        sources = [major.pop(0), minor.pop(0), major.pop(0)]
        joint = int(np.logical_and.reduce([item[1] for item in sources]).sum())
        roster_counts["planned_triads"] += 1
        if joint < 6400:
            roster_counts["unassessed_insufficient_joint_speech"] += 1
            continue
        a, b, c = (item[0] for item in sources)
        noise_gain = 10 ** .25
        variants = {"one_source_control": a, "two_source_control": a + noise_gain * b}
        for condition, third in (("three_balanced_source_control", 1.), ("three_quiet_third_source_control", 10 ** -.3)):
            interference = b + third * c
            variants[condition] = a + noise_gain * interference / np.linalg.norm(interference)
        maximum = max(abs(x).max() for x in variants.values())
        roster_counts["qualified_triads"] += 1
        for condition, audio in variants.items():
            idx = len(retained) + len(output)
            blob = np.rint(audio * (.92 * 32768 / maximum)).astype("<i2").tobytes()
            with (folder / "pcm" / f"{idx:03}.pcm").open("xb") as stream:
                stream.write(blob)
            output.append({"id": idx, "panel": FRESH, "condition": condition, "label": 0, "decision": None,
                           "crossed_room_point": {}, "group": f"independent-triad:{group}",
                           "source_pcm_sha256": [item[2] for item in sources], "source_joint_samples": joint,
                           "pcm_sha256": hashlib.sha256(blob).hexdigest()})
    prior.write(folder / "fresh-manifest.json", {"source_roster_sha256": READER_ROSTER, "vad_sha256": VAD_SHA,
                "roster_counts": roster_counts, "rows": output})


def evaluate(rows, measurements, raw):
    routes = defaultdict(lambda: defaultdict(Counter))
    count_results = defaultdict(lambda: defaultdict(Counter))
    effects = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    gained_groups = defaultdict(set)
    for row, measured, result in zip(rows, measurements, raw, strict=True):
        if row["id"] != measured["id"] or row["id"] != result["id"] or row["pcm_sha256"] != result["pcm_sha256"]:
            raise ValueError("Independent count is not bound to the diarization and PCM record")
        counts = {name: count(values) for name, values in result["probabilities"].items()}
        arms = route_arms(measured["measurements"], counts)
        panel, condition = row["panel"], row["condition"]
        count_results[panel][condition].update({f"center_count_{counts['center']}": 1,
                                                "repeated_count_ge_3": int(sum(n >= 3 for n in counts.values()) >= 2)})
        for arm, route in arms.items():
            routes[panel][condition][f"{arm}:{route}"] += 1
            if route != "single_interferer" or row["decision"] != "inconclusive":
                continue
            for score, crossed in row["crossed_room_point"].items():
                if crossed:
                    truth = "synthetic" if row["label"] else "genuine"
                    effects[panel][condition][arm][f"{score}:added_{truth}"] += 1
                    if row["label"]:
                        gained_groups[panel, arm, score].add(row["group"])
    return {"routes": routes, "count_results": count_results, "effects": effects,
            "synthetic_groups_added": {":".join(key): len(groups) for key, groups in gained_groups.items()}}


def fresh_measured(folder: Path, rows):
    root = folder / "fresh-diarization"
    freeze_path = root / "freeze.json"
    manifest_path = root / "manifest.json"
    frozen = prior.read(freeze_path)
    data = prior.read(root / "measured.json")
    local_rows = prior.read(manifest_path)["rows"]
    if (frozen["manifest_sha256"] != prior.sha(manifest_path) or data["freeze_sha256"] != prior.sha(freeze_path)
            or frozen["model_sha256"] != prior.MODEL_SHA or frozen["runtime_archive_sha256"] != prior.RUNTIME_ARCHIVE_SHA
            or data["calls"] != len(rows) * len(prior.POINTS) or frozen["method"] != prior.ROUTING):
        raise ValueError("Missing frozen matching diarization replay for disjoint readers")
    results = []
    for index, (row, local, measured) in enumerate(zip(rows, local_rows, data["rows"], strict=True)):
        if local != {**row, "id": index} or measured["id"] != index:
            raise ValueError("Independent-reader diarization does not match the frozen roster")
        prior.pcm(root, local)
        for name in prior.POINTS:
            saved = root / name / f"{index:03}.json"
            replay = {**prior.measurement(prior.read(saved)["segments"]), "segments_sha256": prior.sha(saved)}
            if replay != measured["measurements"][name]:
                raise ValueError("Fresh diarization interval measurements changed since scoring")
        results.append({**measured, "id": row["id"]})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "freeze", "run", "readout"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--vad", type=Path)
    args = parser.parse_args()
    folder = args.folder.resolve()
    if args.action == "prepare":
        if args.source is None or args.vad is None:
            parser.error("prepare requires the pinned reader source folder and --vad")
        prepare_fresh(folder, args.source, args.vad)
        return
    rows, old, freeze = load(folder)
    freeze_path = folder / "freeze.json"
    if args.action == "freeze":
        for row in rows:
            prior.pcm(folder, row)
        prior.write(freeze_path, freeze)
        return
    if prior.read(freeze_path) != freeze:
        raise ValueError("Use the originally frozen inputs and implementation")
    if args.action == "readout":
        raw_path = folder / "counts-private.json"
        raw = prior.read(raw_path)
        if raw["freeze_sha256"] != prior.sha(freeze_path):
            raise ValueError("Independent count output has the wrong freeze")
        report = evaluate(rows, old + fresh_measured(folder, rows[len(old):]), raw["rows"])
        prior.write(folder / "aggregate.json", {**report, "frozen_inputs": len(rows), "forwards": len(rows) * len(POSITIONS),
                    "freeze_sha256": raw["freeze_sha256"], "response_commitment": prior.sha(raw_path),
                    "qualification_demonstration_count": raw["qualification_demonstration_count"], "elapsed_s": raw["elapsed_s"]})
        return
    model = CountNet(folder)
    model.torch.set_num_threads(2)
    with wave.open(str(folder / "5_speakers.wav"), "rb") as demo:
        if demo.getparams()[:4] != (1, 2, prior.RATE, PADDED_SAMPLES):
            raise ValueError("Unexpected publisher demonstration geometry")
        demonstration = count(model.score(decode(demo.readframes(PADDED_SAMPLES))))
    if demonstration != 5:
        raise ValueError("The reference audio did not reproduce the publisher's five-speaker example")
    output = []
    started = time.monotonic()
    for row in rows:
        source = decode(prior.pcm(folder, row))
        scores = {name: model.score(padded(source, left)) for name, left in POSITIONS.items()}
        output.append({"id": row["id"], "pcm_sha256": row["pcm_sha256"], "probabilities": scores})
    prior.write(folder / "counts-private.json", {"rows": output, "freeze_sha256": prior.sha(freeze_path),
                "qualification_demonstration_count": demonstration, "elapsed_s": time.monotonic() - started})


if __name__ == "__main__":
    main()
