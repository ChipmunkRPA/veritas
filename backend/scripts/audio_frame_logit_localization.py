# SPDX-License-Identifier: Apache-2.0
"""Test short speech locations from native frame responses of the existing detector."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

RATE = 16000
STRIDE = 320
FIRST_CENTER = 280
CONTEXT = 3 * RATE
HOP = RATE
EDGE = 4
FLANK = 12
WIDTHS = (12, 18, 26, 36)
SCOPE = "overlap_-10dB"
SCHEMA = "chipmunk.audio-context-frame-localization/1"


def sha(body):
    return hashlib.sha256(body).hexdigest()


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())


def write(path, payload):
    data = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)


def choose(archive, other, *, development):
    source = read(archive / "freeze.json")
    cases = [row for row in source["cases"] if row["condition"] == SCOPE]
    if not development:
        return cases
    exposed = {row["group"] for row in read(other / "freeze.json")["origins"]}
    return [row for row in cases if row["group"] not in exposed]


def pcm_snapshot(archive, row):
    data = (archive / "pcm" / f"{row['pcm_sha256']}.pcm").read_bytes()
    if len(data) != 2 * row["samples"] or sha(data) != row["pcm_sha256"]:
        raise ValueError("Native PCM differs from the source commitment")
    return data


def plan(archive, other, development):
    cases = choose(archive, other, development=development)
    if not cases:
        raise ValueError("Empty source roster")
    for row in cases:
        pcm_snapshot(archive, row)
    from app.detectors.audio import upgraded

    return {"schema": SCHEMA, "role": "development" if development else "transfer",
            "input": sha((archive / "freeze.json").read_bytes()),
            "other_input": sha((other / "freeze.json").read_bytes()),
            "script": sha(Path(__file__).read_bytes()), "backbone_runtime": sha(Path(upgraded.__file__).read_bytes()),
            "geometry": {"context_samples": CONTEXT, "context_hop": HOP, "frame_step": STRIDE,
                         "first_center": FIRST_CENTER, "widths": list(WIDTHS), "edge": EDGE, "flank": FLANK},
            "cases": cases}


class FrameReadout:
    def __init__(self, voice):
        self.voice = voice
        self.frames = None

    def project(self, module, args, kwargs, result):
        import torch

        weights = torch.softmax(self.voice.layer_weights, dim=0)
        hidden = torch.stack(result.hidden_states).mul(weights[:, None, None, None]).sum(0)
        mask = kwargs["attention_mask"][0].bool()
        self.frames = self.voice.head(hidden[0, mask]).flatten()

    def score(self, pcm):
        import torch

        from app.detectors.audio.upgraded import prepare_voice_window

        with self.voice.backbone.register_forward_hook(self.project, with_kwargs=True), torch.inference_mode():
            self.frames = None
            logit = float(self.voice(torch.from_numpy(prepare_voice_window(pcm))[None]).item())
            frames = self.frames
            self.frames = None
            if frames is None or not torch.isfinite(frames).all():
                raise ValueError("A measured context has no valid frame readings")
            error = abs(float(frames.mean()) - logit)
            if error > 3e-5:
                raise ValueError("Frame average does not reproduce the original raw logit")
            return frames.numpy(), error


def contexts(length):
    last_start = (length - CONTEXT) // STRIDE * STRIDE
    if last_start < HOP:
        return []
    starts = list(range(0, last_start + 1, HOP))
    if last_start - starts[-1] < HOP // 2:
        starts[-1] = last_start
    else:
        starts.append(last_start)
    return starts


def candidates(pcm, readout, activity):
    import numpy as np

    from app.evidence.audio_short_review_binding import speech_counts

    views, errors = {}, []
    for start in contexts(len(pcm)):
        frames, delta = readout.score(pcm[start:start+CONTEXT])
        views[start // STRIDE] = np.r_[0., np.cumsum(frames, dtype=np.float64)]
        errors.append(delta)
    windows = [(start, start + width) for width in WIDTHS
               for start in range(0, len(pcm) // STRIDE - width, 4)]
    # Each frame's native timestamp is the midpoint of its two input feature frames.
    ranges = [(a * STRIDE + FIRST_CENTER - STRIDE // 2, b * STRIDE + FIRST_CENTER - STRIDE // 2) for a, b in windows]
    speech = speech_counts(activity.spans(pcm), ranges)
    rows = []
    for (start, end), (a, b), spoken in zip(windows, ranges, speech, strict=True):
        if 2 * spoken < b - a or b > len(pcm):
            continue
        level, contrast = [], []
        for offset, summed in views.items():
            left, right = start - offset, end - offset
            if left < EDGE or right > len(summed) - 1 - EDGE:
                continue
            average = (summed[right] - summed[left]) / (right - left)
            level.append(float(average))
            if left - FLANK >= EDGE and right + FLANK <= len(summed) - 1 - EDGE:
                before = (summed[left] - summed[left - FLANK]) / FLANK
                after = (summed[right + FLANK] - summed[right]) / FLANK
                contrast.append(float(average - max(before, after)))
        if len(level) >= 2:
            rows.append({"range": [a, b], "speech_samples": spoken, "context_support": len(level),
                         "level": round(sorted(level)[-2], 6),
                         "contrast": round(sorted(contrast)[-2], 6) if len(contrast) >= 2 else None})
    return rows, max(errors, default=0.)


def score(frozen, archive, other, weights, sink):
    import numpy as np

    from scripts.audio_joint_generator_transfer import panels, prior
    from scripts.run_audio_specialists_v15 import load_voice

    wanted = read(frozen)
    if wanted != plan(archive, other, wanted["role"] == "development"):
        raise ValueError("Replay plan or native inputs changed")
    assets = prior.read(panels.PRIMARY)["incumbent"]["assets"]
    activity = prior.runtime(weights / "silero_vad.jit")
    readout = FrameReadout(load_voice({"weights": weights, "assets": assets}))
    with sink.open("x") as output:
        output.write(json.dumps({"schema": SCHEMA, "plan": sha(frozen.read_bytes()), "weights": assets}) + "\n")
        for index, source in enumerate(wanted["cases"], 1):
            pcm = np.frombuffer(pcm_snapshot(archive, source), dtype="<i2").astype(np.float32) / 32768.
            rows, error = candidates(pcm, readout, activity)
            output.write(json.dumps(source | {"readings": rows, "raw_logit_reconstruction_error": error}, allow_nan=False) + "\n")
            output.flush()
            if index % 4 == 0 or index == len(wanted["cases"]):
                print(f"Measured {index}/{len(wanted['cases'])} native recordings.", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("prepare", "score"))
    cli.add_argument("archive", type=Path)
    cli.add_argument("other", type=Path)
    cli.add_argument("plan", type=Path)
    cli.add_argument("--development", action="store_true")
    cli.add_argument("--output", type=Path)
    cli.add_argument("--weights", type=Path, default=Path("models/weights/audio-upgrade"))
    args = cli.parse_args()
    if args.stage == "prepare":
        write(args.plan, plan(args.archive, args.other, args.development))
    else:
        if args.output is None:
            cli.error("scoring needs an output path")
        score(args.plan, args.archive, args.other, args.weights, args.output)


if __name__ == "__main__":
    main()
