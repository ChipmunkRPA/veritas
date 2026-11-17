# SPDX-License-Identifier: Apache-2.0
"""Reconstruct the exact measured mute and control videos without model inference."""

from __future__ import annotations

import argparse
import hashlib
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from app.bench import independent
from app.bench.independent_inputs import INDEX_SCHEMA, verify_inputs
from app.bench.independent_locations import TRUTH_SCHEMA, count_events
from app.detectors.audio.decode import wav_bytes
from app.detectors.video.decode import probe_frames
from scripts import export_video_lip_correspondence as exporter
from scripts import video_silent_mouth_panel as panel
from scripts.review_video_speech_mismatch import silent_mouth
from scripts.video_lip_correspondence import read, save, sha
from scripts.video_lip_delivery_transfer import qualified_48k


def records(roster, report, sources, folder):
    trial = {(row["group"], row["delivery"], row["role"]): row for row in report["rows"]}
    for name in ("clips", "source"):
        (folder / name).mkdir()
    with (TemporaryDirectory(prefix="quiet-replay-") as temp,
          ZipFile(sources / "lombardgrid_front.zip") as eyes, ZipFile(sources / "lombardgrid_audio.zip") as ears):
        duration_clock = Path(temp)
        wav, aac = duration_clock / "audio.wav", duration_clock / "audio.m4a"
        for index, work in enumerate(roster["rows"]):
            item = work["target"]
            identity = item["front_sha256"]
            original = folder / "source" / f"{index:02d}.mov"
            original.write_bytes(eyes.read(item["front"]))
            if sha(original) != identity:
                raise ValueError("Publisher picture differs from the frozen source")
            table = probe_frames(original, cap=120, timeout_s=20)
            compressed_picture = exporter.visual_hash(original)
            times = [Fraction(pts) * table.tb * panel.RATE for pts in table.pts]
            altered, truth = panel.render_tracks(ears, work)
            for role, audio in sorted(altered.items()):
                wav.write_bytes(wav_bytes(audio.astype("float32") / 32768.))
                decoded, _, codec = qualified_48k(audio, aac)
                for carrier, track, values, committed in (("pcm", wav, audio, sha(wav)),
                                                          ("aac48", aac, decoded, codec["file_sha256"])):
                    row = trial[identity, carrier, role]
                    if (row["pts"] != list(table.pts) or row["picture_timescale"] != table.time_base
                            or row["audio_carrier_sha256"] != committed or row["truth"] != (truth if role not in panel.CONTROLS else None)
                            or row["pcm_sha256"] != hashlib.sha256(values.tobytes()).hexdigest()):
                        raise ValueError("A reconstructed source clock or waveform differs")
                    clip = folder / "clips" / f"{index:02d}-{role}-{carrier}.mov"
                    exporter.make_mux(original, track, clip, table, compressed_picture)
                    exporter.verify_sound(clip, carrier, row["pcm_sha256"], len(audio))
                    if row["playable_sha256"] is not None and sha(clip) != row["playable_sha256"]:
                        raise ValueError("A previously rendered file differs")
                    yield original, clip, times, row
            print(f"Rebuilt {index + 1}/{len(roster['rows'])} source clips", flush=True)


def prepare(plan, receipt, sources, output):
    from app.detectors.audio.phone_events import WEIGHT_PIN
    from app.evidence.video_reference_timing import MODEL_SHA256

    roster, report = read(plan), read(receipt)
    silent_mouth(report)
    if sha(plan) != report["source_plan_sha256"]:
        raise ValueError("The source plan differs from the receipt")
    output.mkdir(mode=0o700)
    manifest, truth, index, predictions = [], [], [], {"current": [], "combined": []}
    for original, clip, times, row in records(roster, report, sources, output):
        case = {"id": clip.stem, "input_sha256": sha(clip)}
        group = "lombard-grid/" + row["speaker"]
        mark = [exporter.frame_range(row["truth"], times)] if row["truth"] else []
        manifest.append(case | {"split": "confirmation", "modality": "video", "group": group,
            "label": int(bool(mark)), "source_sha256": row["group"], "family": row["role"],
            "condition": f'{row["delivery"]}/{row["role"]}', "lineage": [group],
            "license": {"id": "CC BY 4.0 (Lombard Grid)", "evaluation_permitted": True, "redistribution_permitted": True},
            "localization_truth": {"unit": "frames", "source_units": len(times),
                                   "target_units": mark[0][1] - mark[0][0] if mark else 0}})
        truth.append(case | {"extent": [len(times)], "regions": mark})
        index.append({"id": case["id"], "input": clip.relative_to(output).as_posix(),
                      "source": original.relative_to(output).as_posix()})
        reading = row["reading"]
        for key, rows in predictions.items():
            span = reading["baseline"] or (reading["added"] if key == "combined" else None)
            if reading["status"] == "unsupported":
                reason = "The presented sound and picture clocks are outside the method scope."
                rows.append(case | {"status": "not_applicable", "response": None, "reason": reason,
                                    "localization": {"status": "not_applicable", "reason": reason}})
            else:
                rows.append(case | {"status": "ok", "response": 1 if span else 0,
                                    "localization": {"status": "ok", "regions": [exporter.frame_range(span, times)] if span else []}})
    if len(manifest) != len(report["rows"]):
        raise ValueError("Some measured deliveries could not be reconstructed")
    plan_doc = {"schema": independent.SCHEMA, "purpose": "Locate staged mute intervals on a native picture clock.",
                "publisher": "https://zenodo.org/records/3736465", "cases": manifest}
    references = {"manifest_sha256": independent.audit_manifest(plan_doc)["manifest_sha256"], "split": "confirmation"}
    truth_doc = references | {"schema": TRUTH_SCHEMA, "cases": truth}
    file_index = {"schema": INDEX_SCHEMA, "manifest_sha256": references["manifest_sha256"], "cases": index}
    verified = verify_inputs(plan_doc, file_index, output)
    result = {"manifest": plan_doc, "truth": truth_doc, "file-index": file_index, "verified": verified}
    for key, rows in predictions.items():
        response = references | {"system": key, "method_kind": "weighted", "score_kind": "raw",
            "implementation_sha256": independent.digest([sha(receipt), key]), "weight_sha256": [WEIGHT_PIN, MODEL_SHA256],
            "response_units": "Selected interval (1 or 0)", "predictions": rows}
        result[key] = response
        result[f"{key}-events"] = count_events(plan_doc, truth_doc, response, minimum_iou=.40, minimum_recall=.40)
    for key, data in result.items():
        exporter.json_file(output / f"{key}.json", data)
    save(output / "portable.json.gz", result)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("plan", "receipt", "sources", "output"):
        cli.add_argument(arg, type=Path)
    prepare(**vars(cli.parse_args()))
