"""Rebuild the exact held media and portable event counts for the speech-motion study."""
from __future__ import annotations

import argparse
import wave
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from app.bench import independent
from app.bench.independent_inputs import INDEX_SCHEMA, verify_inputs
from app.bench.independent_locations import TRUTH_SCHEMA, count_events
from scripts import export_video_lip_correspondence as export
from scripts import video_lip_correspondence as data
from scripts import video_lip_sequence_panel as study

RATE = 16_000


def receipt_plan(plan, result):
    plan_data = data.read(plan)
    if plan_data["schema"] != study.SCHEMA or result["freeze"]["plan_sha256"] != data.sha(plan):
        raise ValueError("The media roster differs from the measured source plan")
    selected = [row for row in plan_data["rows"] if row["split"] == "held"]
    if {row["target"]["front_sha256"] for row in selected} != {row["picture_sha256"] for row in result["rows"]}:
        raise ValueError("The held media roster is incomplete")
    return selected


def _readbacks(selected, measured, sources, folder):
    from app.detectors.video.decode import probe_frames
    from scripts.video_lip_delivery_transfer import qualified_48k

    for name in ("clips", "source"):
        (folder / name).mkdir()
    with (TemporaryDirectory(prefix="lip-sequence-audio-", dir=folder) as temp,
          ZipFile(sources / "lombardgrid_front.zip") as video, ZipFile(sources / "lombardgrid_audio.zip") as audio):
        wav, aac = Path(temp) / "track.wav", Path(temp) / "track.m4a"
        for i, case in enumerate(selected, 1):
            target = case["target"]
            original = folder / "source" / f"source-{i:02d}.mov"
            original.write_bytes(video.read(target["front"]))
            if data.sha(original) != target["front_sha256"]:
                raise ValueError("Publisher picture differs from the held source")
            table = probe_frames(original, cap=120, timeout_s=20, max_pixels=1280 * 720)
            if table.truncated or table.interpolated or table.rotation or not table.pts:
                raise ValueError("The source picture clock is out of scope")
            times = [Fraction(pts - table.pts[0]) * table.tb * RATE for pts in table.pts]
            original_visual = export.visual_hash(original)
            soundtracks = export.make_soundtracks(audio, case)
            for role, track in soundtracks.items():
                with wave.open(str(wav), "wb") as handle:
                    handle.setparams((1, 2, RATE, 0, "NONE", ""))
                    handle.writeframes(track.tobytes())
                _, _, compressed = qualified_48k(track, aac)
                for profile, audio_path in (("pcm", wav), ("aac48", aac)):
                    row = measured[target["front_sha256"], profile, role]
                    if (row["pts"] != list(table.pts) or row["time_base"] != table.time_base or row["truth"] != target["span"]
                            or (profile == "aac48" and compressed["file_sha256"] != row["carrier_sha256"])):
                        raise ValueError("The measured timestamps or sound carrier differ from the original")
                    submitted = folder / "clips" / f"seq-{i:02d}-{profile}-{role}.mov"
                    export.make_mux(original, audio_path, submitted, table, original_visual)
                    export.verify_sound(submitted, profile, row["source_pcm_sha256"], len(track))
                    yield submitted, original, times, row
            print(f"Verified source {i}/{len(selected)} on both native deliveries", flush=True)


def _prediction(case, row, times, selection, *, append):
    proposal = row["candidate"][selection["arm"]] if append else None
    adopted = proposal["span"] if proposal is not None and proposal["score"] > selection["point"] else None
    span = row["current_span"] or adopted
    if span or proposal is not None or row["current_status"] in {"measured", "no_supported_change", "no_match"}:
        regions = [export.frame_range(span, times)] if span else []
        return case | {"status": "ok", "response": int(bool(regions)), "localization": {"status": "ok", "regions": regions}}
    reason = "The compared audio and mouth observations did not support a local review"
    return case | {"status": "not_applicable", "response": None, "reason": reason,
                   "localization": {"status": "not_applicable", "reason": reason}}


def prepare(plan, receipt, sources, output):
    from app.detectors.audio.phone_events import WEIGHT_PIN
    from app.evidence.video_reference_timing import MODEL_SHA256

    result = data.read(receipt)
    study.verify(result)
    selected = receipt_plan(plan, result)
    measured = {(row["picture_sha256"], row["delivery"], row["role"]): row for row in result["rows"]}
    if len(measured) != len(selected) * len(study.DELIVERIES) * len(data.ROLES):
        raise ValueError("Source deliveries do not match the full paired roster")
    output.mkdir()
    manifest, truth, index, current, combined = [], [], [], [], []
    choice = result["freeze"]["selected"]
    for submitted, source, times, row in _readbacks(selected, measured, sources, output):
        case = {"id": submitted.stem, "input_sha256": data.sha(submitted)}
        group = f"lombard-grid/{row['speaker']}"
        scope = data.ROLES[row["role"]]
        region = export.frame_range(row["truth"], times) if scope == "content" else []
        manifest.append(case | {"split": "confirmation", "modality": "video", "group": group,
            "label": int(bool(region)), "family": scope, "condition": f"{row['delivery']}/{scope}",
            "source_sha256": row["picture_sha256"], "lineage": [group],
            "license": {"id": "CC BY 4.0 (Lombard Grid)", "evaluation_permitted": True, "redistribution_permitted": True},
            "localization_truth": {"unit": "frames", "source_units": len(times),
                                   "target_units": region[1] - region[0] if region else 0}})
        truth.append(case | {"extent": [len(times)], "regions": [region] if region else []})
        index.append({"id": case["id"], "input": submitted.relative_to(output).as_posix(), "source": source.relative_to(output).as_posix()})
        current.append(_prediction(case, row, times, choice, append=False))
        combined.append(_prediction(case, row, times, choice, append=True))
    roster = {"schema": independent.SCHEMA, "purpose": "Locate short same-speaker word replacements in two native audio deliveries.",
              "publisher": "https://zenodo.org/records/3736465", "cases": manifest}
    binding = {"manifest_sha256": independent.audit_manifest(roster)["manifest_sha256"], "split": "confirmation"}
    truth_doc = binding | {"schema": TRUTH_SCHEMA, "cases": truth}
    files = {"schema": INDEX_SCHEMA, "manifest_sha256": binding["manifest_sha256"], "cases": index}
    checked = verify_inputs(roster, files, output)
    for name, doc in (("manifest", roster), ("truth", truth_doc), ("file-index", files), ("input-verification", checked)):
        export.json_file(output / f"{name}.json", doc)
    weights = [WEIGHT_PIN, MODEL_SHA256]
    fit = independent.digest(result["freeze"]["fits"][choice["arm"]])
    for name, predictions in (("current", current), ("combined", combined)):
        response = binding | {"system": name, "method_kind": "weighted", "score_kind": "raw",
            "implementation_sha256": independent.digest([name, result["freeze_sha256"]]),
            "weight_sha256": weights if name == "current" else weights + [fit],
            "response_units": "Selected local mismatch (1 or 0)", "predictions": predictions}
        summary = count_events(roster, truth_doc, response, minimum_iou=.40, minimum_recall=.40)
        export.json_file(output / f"{name}.json", response)
        export.json_file(output / f"{name}-events.json", summary)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("plan", "receipt", "sources", "output"):
        cli.add_argument(arg, type=Path)
    prepare(**vars(cli.parse_args()))
