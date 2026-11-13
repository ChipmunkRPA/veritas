# SPDX-License-Identifier: Apache-2.0
"""Compare source-linked enhanced speech with its partly blended counterpart."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import sys
from collections import defaultdict
from pathlib import Path

from scripts import audio_transient_reduction_transfer as window

sources, io, clock = window.sources, window.io, window.clock
ROOT = Path(__file__).resolve().parents[2]
PRIOR = io.PUBLIC / "audio-reduced-background-localization-2026-11-12.json"
PANEL = io.PUBLIC / "audio-focused-window-transfer-2026-11-12.json"
SCHEMA = "chipmunk.audio-enhancement-blend-transfer/1"
ARMS = ("enhanced", "supported")
METHOD = {
    "scope": "Use the focused panel and revisit the same original inconclusive or unassessed dense windows.",
    "enhanced": "At the existing separately calibrated enhancement point, with speech activity rechecked.",
    "supported": "Require the same point and speech activity on the enhanced window and its 50% original PCM blend.",
    "agreement": "Two adjacent AI windows, at least one eligible rescue, original ranges retained.",
}
require = window.require


def freeze():
    previous = io.read(PRIOR)["freeze"]
    return {"schema": SCHEMA, "method": METHOD, "point": previous["enhanced_point"].hex(),
            "runtime": previous["runtime"], "enhancer_assets": previous["enhancer_assets"],
            "files": {path.name: io.sha(path) for path in (PRIOR, PANEL, Path(__file__), Path(window.__file__))}}


def roster():
    data = io.read(PANEL)
    return list(zip(data["roster"], (row["windows"] for row in data["responses"]), strict=True)), data["freeze"]["voice_assets"]


def signal(body):
    from scripts.audio_gtcrn_traffic import float_pcm

    return float_pcm(body)


def codec(values):
    import numpy as np

    return np.clip(np.rint(values * 32768), -32768, 32767).astype("<i2").tobytes()


def blend(native, reduced):
    return codec((signal(native) + signal(reduced)) * .5)


def assets(folder, overlay, locked):
    installed = {d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions(path=[str(overlay)])}
    require(all(installed.get(name) == v for name, v in locked["runtime"].items())
            and all(io.sha(folder / name) == pin for name, pin in locked["enhancer_assets"].items()), "Enhancer assets differ")
    sys.path.insert(0, str(overlay))
    import mlx.core as mx
    from mlx_audio.sts.utils import load_model

    mx.set_cache_limit(256 * 1024**2)
    return load_model(str(folder))


def enhance(folder, held, model, overlay):
    import numpy as np

    folder.mkdir(mode=0o700)
    locked = freeze()
    io.save(folder / "freeze.json", locked)
    (folder / "processed").mkdir(mode=0o700)
    worker = assets(model, overlay, locked)
    output = []
    for row, native in roster()[0]:
        raw = sources.native(held, row)
        windows = []
        for a, b in sources.eligible(row, native):
            values = signal(raw[2*a:2*b])
            processed = np.asarray(worker.generate(values, sample_rate=clock.RATE, chunk_seconds=None).audio, dtype=np.float32)
            require(processed.shape == values.shape and np.isfinite(processed).all(), "Enhanced PCM changed its source clock")
            body = codec(processed)
            (folder / "processed" / f"{row['case']:03}-{a}.pcm").write_bytes(body)
            windows.append({"span": [a,b], "enhanced_sha256": io.digest(body)})
        output.append({"id": row["case"], "windows": windows})
        if len(output) % 10 == 0:
            print("enhanced", len(output), flush=True)
    io.save(folder / "enhanced.json", output)


def check_clock(row, native, recorded):
    require(row["case"] == recorded["id"] and [item["span"] for item in recorded["windows"]] == sources.eligible(row, native),
            "Enhanced windows differ from the frozen schedule")


def score(folder, held):
    import torch

    from app.detectors.audio.calibration import logit
    from app.detectors.audio.content import window_in_speech
    from scripts.audio_reduced_background_localization import assets_for_scores

    locked = io.read(folder / "freeze.json")
    require(locked == freeze(), "Enhancement policy or data changed")
    rows, installed = roster()
    point = float.fromhex(locked["point"])
    torch.set_num_threads(2)
    voice, activity = assets_for_scores({"voice_assets": installed})

    def measure(body):
        clip = signal(body)
        return logit(voice.predict([clip])[0]) if window_in_speech(activity.spans(clip), 0., 3.) else None

    responses = []
    for (row, native), recorded in zip(rows, io.read(folder / "enhanced.json"), strict=True):
        check_clock(row, native, recorded)
        raw = sources.native(held, row)
        measurements = []
        for item in recorded["windows"]:
            start, end = item["span"]
            body = (folder / "processed" / f"{row['case']:03}-{start}.pcm").read_bytes()
            require(io.digest(body) == item["enhanced_sha256"] and len(body) == (end-start)*2, "Enhanced samples differ")
            candidate = measure(body)
            second, second_pin = None, None
            if candidate is not None and candidate > point:
                mixed = blend(raw[2*start:2*end], body)
                second, second_pin = measure(mixed), io.digest(mixed)
            measurements.append(item | {"enhanced": candidate.hex() if candidate is not None else None,
                                         "blend": second.hex() if second is not None else None, "blend_sha256": second_pin})
        responses.append({"id": row["case"], "windows": measurements})
        if len(responses) % 10 == 0:
            print("scored", len(responses), flush=True)
    results = {arm: calculate(rows, responses, point, arm) for arm in ARMS}
    io.save(folder / "receipt.json", {"freeze": locked, "responses": responses, "results": results})


def calculate(rows, records, point, arm, held=None, folder=None):
    # Adapt the existing localization comparison, whose internal threshold applies to both arms.
    translated = []
    coverage = defaultdict(int)
    for (row, native), record in zip(rows, records, strict=True):
        check_clock(row, native, record)
        raw = sources.native(held, row) if held else None
        readings = []
        for item in record["windows"]:
            start, end = item["span"]
            reduced = float.fromhex(item["enhanced"]) if item["enhanced"] is not None else None
            supported = float.fromhex(item["blend"]) if item["blend"] is not None else None
            pinned = item["blend_sha256"]
            require(all(v is None or math.isfinite(v) for v in (reduced, supported))
                    and bool(pinned) == (reduced is not None and reduced > point)
                    and (pinned is not None or supported is None)
                    and (pinned is None or clock.SHA.fullmatch(pinned)), "Invalid separate response scope")
            coverage["enhanced_admitted"] += reduced is not None
            coverage["blend_measured"] += pinned is not None
            coverage["blend_admitted"] += supported is not None
            if folder:
                body = (folder / "processed" / f"{row['case']:03}-{start}.pcm").read_bytes()
                require(len(body) == (end-start)*2 and io.digest(body) == item["enhanced_sha256"], "Enhanced PCM changed")
                if pinned:
                    require(io.digest(blend(raw[2*start:2*end], body)) == pinned, "Original blend samples changed")
            value = reduced if arm == "enhanced" else supported
            readings.append({"span": item["span"], "sha256": item["enhanced_sha256"],
                             "logit": value.hex() if value is not None else None})
        translated.append({"id": row["case"], "windows": readings})
    return window.evaluate(rows, translated, point) | {"coverage": dict(coverage)}


def review(path, held, folder):
    data = io.read(path)
    locked = freeze()
    require(data["freeze"] == locked and bool(held) == bool(folder), "Freeze or native replay arguments differ")
    rows, _ = roster()
    point = float.fromhex(locked["point"])
    results = {arm: calculate(rows, data["responses"], point, arm, held, folder) for arm in ARMS}
    require(results == data["results"], "Reported enhanced or blend decisions differ")
    print(io.encoded(results).decode(), end="")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("enhance", "score", "review"))
    cli.add_argument("path", type=Path)
    for name in ("held", "model", "overlay", "native-run"):
        cli.add_argument("--" + name, type=Path)
    args = cli.parse_args()
    if args.phase == "enhance":
        if not all((args.held, args.model, args.overlay)):
            cli.error("enhance needs --held, --model and --overlay")
        enhance(args.path, args.held, args.model, args.overlay)
    elif args.phase == "score":
        if not args.held:
            cli.error("score needs --held")
        score(args.path, args.held)
    else:
        review(args.path, args.held, args.native_run)
