"""Exercise the prepaid earlier-copy soundtrack comparison in the local app."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

from scripts import video_lip_correspondence as data
from scripts import video_short_return_encoded_app as paid

ROOT = Path(__file__).resolve().parents[2]


def fixture(clips, short_receipt):
    if short_receipt is None:
        videos = [clips / name for name in ("clip-01-pcm-00.mov", "clip-01-aac48-03.mov", "clip-01-aac48-00.mov")]
        return *videos, 43200, [22400, 27520]

    from scripts.video_visual_speech_evaluation import overlaps

    rows = data.read(short_receipt)["rows"]
    target = next(row for row in rows if row["cohort"] == "fresh" and row["pair"] == "pcm_aac" and row["role"] == "different_words"
                  and row["short"] and overlaps(row["short"]["samples"], row["truth"]))
    manifest = json.loads((clips.parent / "manifest.json").read_bytes())
    videos = {case["id"].split("-", 2)[2]: clips / (case["id"] + ".mov") for case in manifest["cases"]
              if case["source_sha256"] == target["picture_sha256"]}
    return videos["pcm-original"], videos["aac48-different_words"], videos["aac48-gain"], target["size"], target["short"]["samples"]


def run(clips, output, browser, resume, short_receipt=None):
    paid.configure(output, ROOT, resume=resume)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_LOG_LEVEL="WARNING", VERITAS_WEBHOOKS_ENABLED="false",
        VERITAS_VIDEO_REFERENCE_COMPARISON_ENABLED="true", VERITAS_VIDEO_REFERENCE_SOUND_DIFFERENCE="true",
        VERITAS_VIDEO_REFERENCE_SHORT_NATIVE_SOUND_DIFFERENCE="true", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        VERITAS_VIDEO_REFERENCE_SHORT_PHRASE_DIFFERENCE=str(short_receipt is not None).lower(), TOKENIZERS_PARALLELISM="false")

    import pymupdf
    from fastapi.testclient import TestClient
    from sqlalchemy import select

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.evidence import video_reference
    from app.evidence.video_reference_soundtrack import NATIVE_DIFFERENCE_BASIS
    from app.main import app
    from app.storage.db import Analysis, BillingOrderRow, PrincipalRow, get_session
    from scripts.compare_video_references import run as verify
    from scripts.verify_multimedia_exports import scan

    service._stripe = stub = paid.PaymentStub()
    earlier, edited, ordinary, sample_count, expected_span = fixture(clips, short_receipt)
    expected_seconds = [f"{sample / 16000:.3f}" for sample in expected_span]
    results = {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            name = "local-short-native-soundtrack"
            owner = db.scalar(select(PrincipalRow).where(PrincipalRow.username == name)) if resume else None
            if owner is None:
                owner = auth.new_principal(name, "examiner")
                db.add(owner)
            db.flush()
            owner_id = owner.id
            key, _ = auth.issue_api_key(db, owner)
            client.headers["X-API-Key"] = key
            db.commit()

        def compare(aid, *, pdf=True):
            endpoint = f"/api/v2/analyses/{aid}/video-reference-comparison"
            before = paid.counts()
            result = paid.body(client.post(endpoint, data={"include_sound_difference": "true", "include_pdf": str(pdf).lower()},
                                           files={"earlier": (earlier.name, earlier.read_bytes(), "video/quicktime")}))
            assert paid.counts() == before
            scan(json.dumps(result["comparison"]))
            return result

        def analyze(path):
            if resume:
                with get_session() as db:
                    aid = db.scalar(select(Analysis.id).join(BillingOrderRow, BillingOrderRow.analysis_id == Analysis.id).where(
                        Analysis.sha256 == data.sha(path), Analysis.submitted_by == owner_id, Analysis.status == "done",
                        BillingOrderRow.status == "fulfilled", BillingOrderRow.amount_minor == 999,
                    ))
                if aid:
                    return aid
            return paid.checkout(client, stub, path, kind="video", media_type="video/quicktime", filename=path.name)

        edited_id = analyze(edited)
        assert paid.body(client.get(f"/api/v2/analyses/{edited_id}/video-reference-comparison"))["short_native_sound_difference_available"]
        settings.video_reference_short_native_sound_difference = False
        before = compare(edited_id, pdf=False)["comparison"]["sound_difference"]
        assert before["reason"] == "unsupported_picture_profile"
        settings.video_reference_short_native_sound_difference = True
        if short_receipt:
            settings.video_reference_short_phrase_difference = False
            baseline = compare(edited_id, pdf=False)
            assert baseline["comparison"]["schema"] == video_reference.SCHEMA_WITH_NATIVE_SOUND_DIFFERENCE
            assert "range" not in baseline["comparison"]["sound_difference"]
            prior = output / "short-window-disabled.json"
            prior.write_text(json.dumps(baseline))
            assert verify(earlier, edited, prior, verify=True) == baseline["comparison"]
            settings.video_reference_short_phrase_difference = True
        plain_id = analyze(ordinary)
        for role, aid, submitted in (("replacement", edited_id, edited), ("ordinary", plain_id, ordinary)):
            response = compare(aid)
            record = response["comparison"]
            difference = record["sound_difference"]
            assert record["schema"] == (video_reference.SCHEMA_WITH_SHORT_PHRASE_DIFFERENCE if short_receipt
                                       else video_reference.SCHEMA_WITH_NATIVE_SOUND_DIFFERENCE)
            assert difference["basis"] == NATIVE_DIFFERENCE_BASIS and difference["affects_overall_confidence"] is False
            assert record["picture"]["changed_frames"] == 0 and record["audio"]["status"] == "audio_timelines_differ"
            presented = difference["audio_presentation"]
            assert presented["earlier"]["presented_samples"] == presented["submitted"]["presented_samples"] == sample_count
            assert presented["submitted"]["decoded_samples"] >= sample_count
            if short_receipt is None:
                assert presented["submitted"]["decoded_samples"] > sample_count
            assert presented["submitted"]["aac48"]["presentation_end_exclusive_pts"] == 3 * sample_count
            span = difference.get("range")
            if role == "replacement":
                assert span["samples"] == expected_span
                if short_receipt:
                    assert span["measurement_basis"] == "short_phrase_correlation/1" and len(span["bands"]) >= 2
                assert len(span["paired_pictures"]) == 3 and all(
                    pair["earlier"]["rgb_sha256"] == pair["submitted"]["rgb_sha256"] for pair in span["paired_pictures"].values())
            else:
                assert span is None and difference["status"] in {"no_local_range", "unassessed"}
                if short_receipt is None:
                    assert difference["status"] == "no_local_range"
            pdf = base64.b64decode(response.pop("pdf"), validate=True)
            report = output / f"{role}.pdf"
            report.write_bytes(pdf)
            exported = output / f"{role}.json"
            exported.write_text(json.dumps(response, sort_keys=True))
            assert verify(earlier, submitted, exported, verify=True) == record
            with pymupdf.open(stream=pdf, filetype="pdf") as pages:
                text = " ".join(" ".join(page.get_text().split()) for page in pages)
                scan(text)
                if span:
                    assert f"{' to '.join(expected_seconds)} s" in text
                    assert f"Common presented audio duration {sample_count / 16000:.3f} s." in text
                    if short_receipt:
                        assert "Sound matched before and after the interval" in text
                    else:
                        assert "23.976 fps" in text
                    assert sum(len(page.get_images()) for page in pages) >= 6
            results[role] = {"source_sha256": data.sha(submitted), "reference_sha256": data.sha(earlier),
                             "report_sha256": data.sha(report), "presentations": presented,
                             "status": difference["status"], "samples": span["samples"] if span else None}
        receipt = {"cases": results, "trial_checkouts_usd_9_99": len(results), "replayed_checkout_records": len(results) - len(stub.sessions),
                   "unpaid_jobs": 0, "comparison_jobs": 0, "earlier_profile_result": before["reason"]}
        if browser:
            from scripts.image_jpeg_turned_copy_app import browser_check

            browser_case = {"analysis_id": edited_id, "reference": str(earlier)}
            if short_receipt:
                browser_case.update(range_s=expected_seconds, presented_s=f"{sample_count / 16000:.3f}")
            receipt["browser"] = browser_check(output, app, key, [browser_case], scenario="short-native-soundtrack-replay.mjs")
            if short_receipt:
                exported = verify(earlier, edited, output / "browser-comparison.json", verify=True)
                assert exported["sound_difference"]["range"]["samples"] == expected_span
                with pymupdf.open(output / "browser-comparison.pdf") as report:
                    text = " ".join(" ".join(page.get_text().split()) for page in report)
                    scan(text)
                    assert f"{' to '.join(expected_seconds)} s" in text
                    assert "Sound matched before and after the interval" in text
                receipt["browser"].update(json_source_replay=True, pdf_interval_verified=True)
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("clips", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--browser", action="store_true")
    cli.add_argument("--resume", action="store_true", help="Reuse completed paid jobs from this isolated replay directory")
    cli.add_argument("--short-receipt", type=Path, help="Check a short range and a gain-only control from the fresh comparison")
    run(**vars(cli.parse_args()))
