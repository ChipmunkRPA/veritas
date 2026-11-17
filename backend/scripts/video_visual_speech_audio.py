"""Reconstruct the audible speech controls and read their native sound events."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from scripts import video_lip_articulation_transfer as prior
from scripts import video_lip_correspondence as receipt
from scripts import video_visual_speech_train as training

ROLES = ("original", "gain", "same_words", "different_words", "global_early", "global_late")
CARRIERS = ("pcm", "aac48")


def sound_reader(checkpoint):
    from app.detectors.audio.phone_events import collapse, phone_posteriors
    from app.evidence.video_viseme_articulation import PHONES

    predict, vocabulary, device = phone_posteriors(checkpoint)
    labels = {vocabulary[token]: group for group, tokens in PHONES.items() for token in tokens}

    def read(pcm):
        return collapse(predict(pcm), labels)

    return read, device


def read_captures(plan, source, checkpoint, capture_files, output):
    from app.detectors.audio.phone_events import WEIGHT_PIN
    from scripts.video_lip_delivery_transfer import qualified_48k
    from scripts.video_lip_sequence_panel import make_soundtracks

    frozen = receipt.read(plan)
    assert frozen["schema"] == training.SCHEMA
    parents = {phase: {r["picture_sha256"]: r for r in receipt.read(path)["rows"]} for phase, path in capture_files.items()}
    read, device = sound_reader(checkpoint)
    cases = []
    records = [r for r in frozen["rows"] if r["split"] in parents]
    with ZipFile(source / "lombardgrid_audio.zip") as archive, TemporaryDirectory() as work:
        carrier = Path(work) / "sound.m4a"
        for index, row in enumerate(records, 1):
            target = row["target"]
            saved = parents[row["split"]][target["front_sha256"]]
            assert (saved["speaker"], saved["samples"], saved["truth"]) == (row["speaker"], target["samples"], target["span"])
            altered = make_soundtracks(archive, row)
            observed = {}
            for name in ROLES:
                pcm = altered[name]
                for delivery in CARRIERS:
                    if delivery == "aac48":
                        pcm, _, metadata = qualified_48k(pcm, carrier)
                    pin = prior.phone.digest(pcm.tobytes())
                    incumbent = saved["cases"][delivery][name]
                    assert pin == incumbent["source_pcm_sha256"]
                    if delivery == "aac48":
                        assert metadata["file_sha256"] == incumbent["carrier_sha256"]
                    observed.setdefault(delivery, {})[name] = {
                        "events": read(pcm), "audio_sha256": pin, "carrier_sha256": incumbent["carrier_sha256"],
                        "baseline": incumbent["current_span"], "baseline_status": incumbent["current_status"],
                    }
            cases.append({key: saved[key] for key in ("speaker", "condition", "picture_sha256", "pts", "time_base", "truth", "samples")}
                         | {"split": row["split"], "cases": observed})
            if index % 4 == 0:
                print(f"Speech sounds {index}/{len(records)} source clips", flush=True)
    receipt.save(output, {"schema": training.SCHEMA, "plan": receipt.sha(plan), "device": device, "weights": WEIGHT_PIN,
                          "incumbent_sources": [receipt.sha(path) for path in capture_files.values()], "rows": cases})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "source", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("captures", nargs="+", help="Reader groups in phase=path form")
    args = parser.parse_args()
    capture_files = dict(item.split("=", 1) for item in args.captures)
    read_captures(args.plan, args.source, args.checkpoint, {k: Path(v) for k, v in capture_files.items()}, args.output)
