# SPDX-License-Identifier: MIT
"""Replay the frozen speech heads on a separate measured two-voice room."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts import audio_clean_teacher_fresh as fresh
from scripts import audio_clean_teacher_transfer as experiment
from scripts import audio_two_voice_components as components

SOURCE = experiment.SOURCE
PARENT = experiment.panels.PANELS / "audio-two-voice-components-2026-11-15.json.gz"
ROLES = {"untouched": (), "neither": (), "foreground": (0,), "background": (1,), "both": (0, 1)}


def build(args):
    import numpy as np

    parent = experiment.read(PARENT)
    assert experiment.read(args.stereo / "freeze.json") == parent["plan"]
    fitting, pcm_hash = experiment.base.load_plan(args.phoneme)
    excluded = {row["reader_sha256"] for row in fitting["sources"] if row["role"] != "held"}
    existing = {(row["scene"], row["condition"]): row["pcm_sha256"] for row in parent["current"]["rows"]}
    frozen_fit = experiment.read(args.heads / "freeze.json")
    assert frozen_fit["source_sha256"] == pcm_hash
    args.output.mkdir(mode=0o700)
    (args.output / "pcm").mkdir()
    cases, blocked, kept = [], [], []
    for group, scene in enumerate(parent["plan"]["scenes"]):
        cohort = scene["cohort"]
        if cohort == "development":
            continue
        if excluded.intersection(voice["speaker"] for voice in scene["voices"]):
            blocked.append(group)
            continue
        kept.append(group)
        for condition, edited in ROLES.items():
            original = scene["mixes"][condition]["pcm_sha256"]
            assert original == existing[group, condition]
            native = components.load_stereo(args.stereo, original)
            mono = np.rint(native.mean(axis=1)).astype("<i2").tobytes()
            digest = SOURCE.digest(mono)
            (args.output / "pcm" / f"{digest}.pcm").write_bytes(mono)
            cases.append({"panel": f"bbc_{cohort}", "room": "bbc", "group": group,
                          "kind": f"edited/{condition}" if edited else condition, "pcm_sha256": digest,
                          "stereo_pcm_sha256": original,
                          "truth": [scene["voices"][i]["span"] for i in edited]})
    document = {"heads": frozen_fit, "runner_sha256": SOURCE.path_sha(Path(fresh.__file__)),
                "builder_sha256": SOURCE.path_sha(Path(__file__)), "cases": cases,
                "source_receipt_sha256": SOURCE.path_sha(PARENT), "kept": kept, "excluded_scene_ids": blocked}
    experiment.save(args.output / "freeze.json", document)
    print(f"Committed {len(cases)} room deliveries from {len(kept)} reader scenes.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("phoneme", "stereo", "heads", "output"):
        parser.add_argument(name, type=Path)
    build(parser.parse_args())
