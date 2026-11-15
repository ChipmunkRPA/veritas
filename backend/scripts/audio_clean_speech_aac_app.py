# SPDX-License-Identifier: Apache-2.0
"""Check the AAC earlier-copy path through paid analysis and both downloads."""

import argparse
import base64
import gzip
import json
import os
from pathlib import Path

from scripts.audio_clean_speech_aac_transfer import (
    CARRIERS48,
    ami_cases,
    encode,
    encode48,
    icsi_cases,
    read,
    review,
    sha,
)
from scripts.audio_focused_window_app import write_wav
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, counts

ROOT = Path(__file__).resolve().parents[2]


def stereo_cases(args):
    from scripts import audio_stereo_room_transfer as room

    receipt = json.loads(gzip.decompress(args.receipt.read_bytes()))
    room.review(receipt)
    assert receipt["plan"]["parameters"] == room.HOLDOUT_METHOD
    rows = {(row["bank"], row["group"], row["kind"]): row for row in receipt["rows"]}
    for (bank, group, kind), row in rows.items():
        measurement = row["measurement"]
        span = measurement["readings"]["downmix"]["range"]
        wrong = rows[bank, group, "background_reference"]["measurement"]
        if (row["half"] == "late" and room.category(kind) == "synthetic_edit"
                and room.old.location(span, row["truth"]) and room.downmix_with_channel_support(measurement)
                and wrong["readings"]["downmix"]["range"] and not room.downmix_with_channel_support(wrong)):
            break
    else:
        raise ValueError("The held bank has no matching positive and background-reference challenge")
    inputs = room.dry_inputs(args.icsi, args.sources)
    rooms = room.read_rooms(args.rooms, holdout=True)
    placement = next(item for item in receipt["plan"]["rows"] if (item["bank"], item["group"]) == (bank, group))
    pairs = room.materialize(placement, rooms, inputs)
    for role, case in (("edited", kind), ("untouched", "untouched"), ("unrelated", "background_reference")):
        reference, mixture = pairs[case]
        witness = rows[bank, group, case]
        assert sha(reference) == witness["reference_sha256"] and sha(mixture) == witness["mixture_sha256"]
        entry = witness["measurement"]
        base = entry["readings"]["downmix"]
        supported = room.channel_check.supporting_channels(entry["readings"], entry["support"], entry["selected"])
        reason = ("stereo_range_not_corroborated" if not supported and any(view["range"] for view in entry["readings"].values())
                  else base["reason"])
        yield role, reference, mixture, witness["clock"]["encoded_sha256"], {
            **witness["clock"], "reason": reason, "range": base["range"] if supported else None,
            "witness_channels": supported,
        }


def cases(args):
    if args.rooms:
        yield from stereo_cases(args)
        return
    if args.long:
        from scripts import audio_clean_speech_aac_long_transfer as long

        receipt = json.loads(gzip.decompress(args.receipt.read_bytes()))
        long.review(receipt)
        measured = next(row for row in receipt["rows"] if row["bank"] == args.bank and row["half"] == "late"
                        and long.category(row["kind"]) == "synthetic_edit"
                        and long.previous.location(row["readings"][args.carrier]["range"], row["truth"]))
        group = measured["group"]
        expected = {row["kind"]: row for row in receipt["rows"] if row["bank"] == args.bank and row["group"] == group}
        fixtures = {row["kind"]: pair for row, pair in long.pairs(args.bank, args.sources) if row["group"] == group}
        for role, kind in (("edited", measured["kind"]), ("untouched", "untouched"), ("unrelated", "half_matched")):
            earlier, submitted = fixtures[kind]
            pin = expected[kind]
            assert sha(earlier) == pin["earlier_sha256"] and sha(submitted) == pin["submitted_sha256"]
            row = pin["readings"][args.carrier]
            yield role, earlier, submitted, row["encoded_sha256"], row
        return
    receipt = read(args.receipt)
    review(receipt)
    if receipt["bank"] == "ami":
        group_id, correction, source = 0, "piper", ami_cases
    else:
        source = icsi_cases
        selected = next(row for row in receipt["rows"] if row["kind"].startswith("edited/")
                        and row["outcomes"][args.carrier]["range"])
        group_id, correction = selected["group"], selected["kind"]
    expected = {row["kind"]: row for row in receipt["rows"] if row["group"] == group_id}
    fixtures = {kind: (dry, noisy) for group, kind, _, dry, noisy in source(args.sources) if group == group_id}
    edit_role = "piper" if receipt["bank"] == "ami" else "edited"
    for role, kind in ((edit_role, correction), ("untouched", "untouched"), ("unrelated", "unrelated")):
        dry, noisy = fixtures[kind]
        pin = expected[kind]
        assert sha(dry) == pin["earlier_pcm_sha256"] and sha(noisy) == pin["submitted_pcm_sha256"]
        yield role, dry, noisy, pin["encoded_sha256"][args.carrier], pin["outcomes"][args.carrier]


def run(args):
    configure(args.output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_AUDIO_REFERENCE_COMPARISON_ENABLED="true",
                      VERITAS_AUDIO_REFERENCE_LOCAL_TIMING="true", VERITAS_AUDIO_REFERENCE_AAC_TIMING="true",
                      VERITAS_AUDIO_REFERENCE_CLEAN_SPEECH="true", VERITAS_AUDIO_REFERENCE_ALIGNED_SPEECH="true",
                      VERITAS_AUDIO_REFERENCE_MATCHING_WAVEFORM="true")
    import pymupdf
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.evidence.audio_reference import (
        SCHEMA_WITH_AAC_CLEAN_SPEECH,
        SCHEMA_WITH_LONG_AAC_CLEAN_SPEECH,
        SCHEMA_WITH_MATCHING_WAVEFORM,
        SCHEMA_WITH_STEREO_AAC_CLEAN_SPEECH,
    )
    from app.main import app
    from app.storage.db import Analysis, get_session
    from scripts.compare_audio_references import run as compare_replay
    from scripts.verify_audio_clean_speech_export import verify_export
    from scripts.verify_multimedia_exports import scan

    carrier = args.carrier
    seconds = 12 if args.long else 6
    stereo_room = args.rooms is not None
    schema = (SCHEMA_WITH_STEREO_AAC_CLEAN_SPEECH if stereo_room else
              SCHEMA_WITH_LONG_AAC_CLEAN_SPEECH if args.long else SCHEMA_WITH_AAC_CLEAN_SPEECH)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = PaymentStub()
    service._stripe = stub
    results, analyses = {}, {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-aac-speech-compare", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()
        for role, dry, noisy, encoded_sha, measured in cases(args):
            earlier = args.output / f"{role}-earlier.wav"
            write_wav(earlier, dry)
            upload = "untouched" if role == "unrelated" else role
            submitted = args.output / f"{upload}-submitted.m4a"
            if upload not in analyses:
                if stereo_room:
                    from scripts.audio_clean_speech_stereo_long import stereo_pcm

                    stereo_pcm(submitted, noisy, expected_hash=encoded_sha)
                elif carrier in CARRIERS48:
                    encode48(noisy, submitted, carrier)
                else:
                    encode(noisy, submitted, 32)
                analyses[upload] = checkout(client, stub, submitted, kind="audio", media_type="audio/mp4", filename=submitted.name)
            assert sha(submitted.read_bytes()) == encoded_sha
            aid = analyses[upload]
            with get_session() as db:
                frozen = db.get(Analysis, aid).result_hash
            endpoint = f"/api/v2/analyses/{aid}/audio-reference-comparison"
            files = {"earlier": (earlier.name, earlier.read_bytes(), "audio/wav")}
            form = {"include_pdf": "true", "earlier_has_clean_speech": "true"}
            before = counts()
            settings.audio_reference_clean_speech_aac = args.long
            settings.audio_reference_long_clean_speech = stereo_room
            settings.audio_reference_clean_speech_stereo_aac = False
            availability = body(client.get(endpoint))
            assert bool(availability.get("clean_speech_aac_available")) is args.long
            assert bool(availability.get("long_clean_speech_available")) is stereo_room
            assert "clean_speech_stereo_aac_available" not in availability
            baseline = body(client.post(endpoint, data=form, files=files))["comparison"]["clean_speech"]
            assert baseline["status"] == "unassessed" and "range" not in baseline
            settings.audio_reference_clean_speech_aac = True
            settings.audio_reference_long_clean_speech = args.long
            settings.audio_reference_clean_speech_stereo_aac = stereo_room
            available = body(client.get(endpoint))
            assert available["clean_speech_aac_available"] is True
            assert bool(available.get("clean_speech_stereo_aac_available")) is stereo_room
            result = body(client.post(endpoint, data=form, files=files))
            comparison = result["comparison"]
            check = comparison["clean_speech"]
            assert comparison["schema"] == SCHEMA_WITH_MATCHING_WAVEFORM
            assert comparison["comparison_schema"] == schema
            span = measured["range"]
            assert check["reason"] == measured["reason"]
            aac = check["aac_presentation"]
            if args.long:
                assert aac["native_clock"] == measured["native_clock"] and aac["decoded_samples"] == measured["decoded_samples"]
                assert aac["presentation_end_exclusive_pts"] == 192000
            else:
                assert aac == measured["presentation"]
            assert check.get("range", {}).get("samples") == span and check["affects_overall_confidence"] is False
            if stereo_room:
                assert check["stereo_support"]["witness_channels"] == measured["witness_channels"]
            exported = args.output / f"{role}.json"
            exported.write_text(json.dumps({name: result[name] for name in ("comparison", "assets")}))
            # The base64 waveform may contain coincidental substrings of model names.
            scan(json.dumps({"comparison": comparison, "asset_names": list(result["assets"])}))
            assert compare_replay(earlier, submitted, exported, verify=True) == comparison
            proof = verify_export(earlier, submitted, exported)
            assert proof["excerpts_verified"] == (2 if span else 0)
            pdf = base64.b64decode(result["pdf"], validate=True)
            (args.output / f"{role}.pdf").write_bytes(pdf)
            with pymupdf.open(stream=pdf, filetype="pdf") as document:
                page = next(page for page in document if "Local speech comparison" in page.get_text())
                words = " ".join(page.get_text().split())
                scan(words)
                assert f"AAC duration {seconds}.000 s" in words and check["summary"] in words
                if carrier in CARRIERS48:
                    channels = CARRIERS48[carrier][1]
                    assert f"48 kHz {'stereo' if channels == 2 else 'mono'} source" in words
                if span:
                    assert f"Source ticks [{span[0]}, {span[1]})" in words and len(page.get_drawings()) >= 4
                    if stereo_room:
                        assert "Analyzed recording · mono mix" in words
                        for side in measured["witness_channels"]:
                            assert f"{side.capitalize()} channel supported" in words
            with get_session() as db:
                assert db.get(Analysis, aid).result_hash == frozen
            assert counts() == before
            results[role] = {"earlier_pcm_sha256": sha(dry), "submitted_m4a_sha256": sha(submitted.read_bytes()),
                             "baseline_status": baseline["status"], "presentation": check["aac_presentation"],
                             "pdf_sha256": sha(pdf), "source_replay": proof}
            if stereo_room:
                results[role]["stereo_support"] = check["stereo_support"]
    (args.output / "receipt.json").write_text(json.dumps({"cases": results, "paid_analyses": len(analyses),
        "usd_cents": 999, "jobs_before_payment": 0, "saved_assessments_preserved": True}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--carrier", choices=("aac32", *CARRIERS48), default="aac32")
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--bank", choices=("ami", "icsi"), default="ami")
    parser.add_argument("--rooms", type=Path, help="Held stereo AMI microphone files")
    parser.add_argument("--icsi", type=Path, help="Dry ICSI sources for held stereo replay")
    args = parser.parse_args()
    if args.rooms and not (args.long and args.carrier == "aac48_stereo" and args.icsi):
        parser.error("The held stereo replay requires --long, --carrier aac48_stereo and --icsi.")
    if args.long and not args.rooms and args.carrier != "aac48_mono":
        parser.error("The twelve-second replay requires a 48 kHz mono carrier.")
    run(args)
