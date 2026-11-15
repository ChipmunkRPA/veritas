# SPDX-License-Identifier: MIT
"""Measure lip timing on paired 16 kHz and 48 kHz AAC deliveries."""

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from app.evidence import video_viseme_binding as binding
from app.evidence.video_viseme_timing import RATE, SCORE_UNITS
from scripts import video_lip_viseme_transfer as study


def encode(pcm, path, source_rate):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "s16le", "-ar", str(RATE),
                    "-ac", "1", "-i", "pipe:0", "-ar", str(source_rate), "-threads", "1", "-c:a", "aac",
                    "-b:a", "96k" if source_rate == 48_000 else "48k", str(path)],
                   input=pcm.tobytes(), check=True, timeout=15)


def run(receipt, sources, checkpoint, output):
    import numpy as np

    from app.detectors.audio.decode import decode_audio

    panel = study.load(receipt)
    assert panel["points"]["all_visible"]["tight"] == binding.POINT
    assert {name: {"median": row["median"]} for name, row in panel["references"].items()} == binding.REFERENCE
    acoustic, device = study.decoder(checkpoint)
    rows, counts = [], {str(rate): Counter() for rate in (16_000, 48_000)}
    with (ZipFile(sources / "lombardgrid_audio.zip") as audio,
          TemporaryDirectory(prefix="chipmunk-lip-aac48-") as directory):
        path = Path(directory) / "delivery.m4a"
        for row in panel["rows"]:
            waveform = audio.read(row["members"]["audio"]["name"])
            assert study.phone.digest(waveform) == row["members"]["audio"]["sha256"]
            frames = [{key: value for key, value in frame.items() if key not in {"gap", "sample"}}
                      | {"gap_units": None if frame["gap"] is None else round(frame["gap"] * SCORE_UNITS)}
                      for frame in row["frames"]]
            for role, pcm in study.parent.treatments(study.phone.pcm(waveform), row["edit"]).items():
                assert study.phone.digest(pcm.tobytes()) == row["cases"]["pcm"][role]["sha256"]
                shift = study.parent.EDITS.get(role)
                kind = "local" if shift else "constant" if role.startswith("global_") else "ordinary"
                comparison = {"source": row["source"], "role": role, "kind": kind, "carriers": {}}
                for rate in (16_000, 48_000):
                    encode(pcm, path, rate)
                    decoded = decode_audio(path, source_sha256=study.phone.sha(path))
                    mapping = decoded.mapping
                    extent = binding.presented_samples(mapping)
                    assert binding.audio_origin(mapping) == 0 and extent is not None
                    assert rate != 48_000 or extent == len(pcm)
                    values = np.rint(decoded.pcm * 32768.).astype("<i2")
                    if rate == 48_000:
                        received, expected = values[:extent].astype("float64"), pcm.astype("float64")
                        alignment = range(-16, 17)
                        best = max(alignment, key=lambda lag: np.dot(expected[64:-64], received[64+lag:extent-64+lag]))
                        assert abs(best) <= 1
                    measured = binding.measured({"frames": frames, "phones": acoustic(values[:extent])},
                                                timestamps=[frame["pts"] for frame in frames], time_base=row["time_base"],
                                                source={"width": 1280, "height": 720}, mapping=mapping)
                    match = measured["witness"]
                    useful = bool(shift and match and study.parent.useful(
                        match["audio_source_pts"], match["change_samples"], row["edit"], shift))
                    counter = counts[str(rate)]
                    counter[f"{kind}_files"] += 1
                    counter[f"{kind}_flags"] += bool(match)
                    counter[f"{kind}_unresolved"] += measured["status"] not in {"matched", "no_match"}
                    if shift:
                        counter["local_useful"] += useful
                    comparison["carriers"][str(rate)] = {"status": measured["status"], "useful": useful,
                        "decoded_samples": len(values), "checked_samples": extent,
                        "audio_source_pts": match["audio_source_pts"] if match else None,
                        "change_samples": match["change_samples"] if match else None}
                rows.append(comparison)
            print(f"Checked {row['source']} at both rates", flush=True)
    output.write_text(json.dumps({"source_receipt_sha256": study.phone.sha(receipt), "device": device,
                                  "point": binding.POINT, "counts": counts, "rows": rows}, indent=2) + "\n")
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("receipt", "sources", "checkpoint", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.receipt, args.sources, args.checkpoint, args.output)
