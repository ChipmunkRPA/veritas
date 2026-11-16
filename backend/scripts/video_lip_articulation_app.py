# SPDX-License-Identifier: MIT
"""Verify the licensed local mouth-opening fallback after simulated checkout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from zipfile import ZipFile

from scripts import video_lip_articulation_transfer as panel
from scripts.video_lip_event_alignment import aac_clocked
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

ROOT = Path(__file__).resolve().parents[2]
ARM = "aperture_classes"


def select_48k(plan_path, receipt):
    from scripts import video_lip_delivery_transfer as transfer

    packet = panel.read(receipt)
    assert packet["study"] == "aac48" and packet["parents"]["plan"] == panel.phone.sha(plan_path)
    assert transfer.outcomes(packet) == packet["summary"]
    for row in packet["rows"]:
        cases = row["cases"]
        changed = cases["different_words"]["readings"]
        candidate = changed["articulation"]
        witness = candidate["witness"]
        if (not changed["timing"]["witness"] and witness and candidate["basis"] == "local_articulation"
                and panel.parent.overlap(row["truth"], witness["audio_source_pts"])
                and all(cases[role]["readings"]["articulation"]["witness"] is None for role in ("original", "same_words"))):
            return row, witness
    raise ValueError("No held 48 kHz clip qualified for app replay")


def select_16k(plan_path, receipt):
    packet = panel.read(receipt)
    panel.review(packet)
    assert packet["held_codec"] == "aac-lc-m4a-48k-16k-mono" and packet["plan_sha256"] == panel.phone.sha(plan_path)
    points = packet["points"]
    select = panel.parent.choose
    examples = []
    for row in packet["cohort"]:
        changed = row["readings"]["different_words"]
        old = select(changed, "existing", points, "tight")
        new = select(changed, ARM, points, "tight")
        if (not old and new and panel.parent.overlap(row["truth"], new["span"])
                and all(select(row["readings"][name], ARM, points, "tight") is None for name in ("original", "same_words"))):
            examples.append((row, new))
    assert examples
    return examples[0]


def fixtures(output, sources, plan_path, receipt, *, aac48=False):
    frozen = json.loads(plan_path.read_bytes())["rows"]
    if aac48:
        from scripts.video_lip_delivery_transfer import qualified_48k

        delivery, witness = select_48k(plan_path, receipt)
        proof, chosen = {"picture_sha256": delivery["source"]}, {"span": witness["audio_source_pts"], "witness": witness}
    else:
        proof, chosen = select_16k(plan_path, receipt)
    row = next(item for item in frozen if item["target"]["front_sha256"] == proof["picture_sha256"])
    with (ZipFile(sources / "lombardgrid_front.zip") as front, ZipFile(sources / "lombardgrid_audio.zip") as audio):
        picture = front.read(row["target"]["front"])
        assert panel.phone.digest(picture) == proof["picture_sha256"]
        path = output / "source.mov"
        path.write_bytes(picture)
        sounds = {}
        for role in ("target", "same_words", "different_words"):
            data = audio.read(row[role]["audio"])
            assert panel.phone.digest(data) == row[role]["audio_sha256"]
            if not aac48:
                assert row[role]["audio_sha256"] == proof["source_audio_sha256"][role]
            sounds[role] = panel.phone.pcm(data)
    source_pcm = sounds.pop("target")
    variants = {role: panel.parent.splice(source_pcm, row["target"]["span"], donor, row[role]["span"])
                for role, donor in sounds.items()}
    variants["original"] = source_pcm
    for name, pcm in variants.items():
        aac = output / f"{name}.m4a"
        if aac48:
            _, _, carrier = qualified_48k(pcm, aac)
            assert carrier == delivery["cases"][name]["carrier"]
        else:
            readback = aac_clocked(pcm, aac)
            assert panel.phone.digest(readback.tobytes()) == proof["pcm"][name]
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-i", str(aac),
                        "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", str(output / f"{name}.mp4")], check=True, timeout=20)
    return proof, chosen


def stereo_fixtures(args):
    from scripts import video_lip_stereo_transfer as test

    result = panel.read(args.receipt)
    test.verify(result)
    assert result["freeze_sha256"] == panel.phone.sha(args.plan)
    # Prefer a separate speaker with panned background sound for the app
    # replay. The numeric transfer still includes every planned delivery.
    choices = []
    for row in result["rows"]:
        lookup = {(case["scene"], case["role"]): case for case in row["cases"]}
        for index, scene in enumerate(("panned_environment", "centered")):
            candidate = lookup[scene, "different_words"]
            witness = candidate["witness"]
            if (candidate["guard_passed"] and candidate["basis"] == "local_articulation" and witness
                    and panel.parent.overlap(row["truth"], witness["audio_source_pts"])
                    and all(lookup[scene, role]["guard_passed"] and lookup[scene, role]["witness"] is None
                            for role in ("same_words", "original"))
                    and not lookup["opposed", "original"]["guard_passed"]):
                choices.append(((index, row["split"] != "new_speaker", row["source"]), row, scene, lookup))
    _, chosen, scene, lookup = min(choices, key=lambda item: item[0])
    plan = json.loads(args.plan.read_bytes())
    row = next(item for item in plan["rows"] if item["target"]["front_sha256"] == chosen["source"])
    path = args.output / "source.mov"
    with ZipFile(args.sources / "lombardgrid_front.zip") as front:
        path.write_bytes(front.read(row["target"]["front"]))
        assert panel.phone.sha(path) == chosen["source"]
    with ZipFile(args.sources / "lombardgrid_audio.zip") as audio:
        for profile, role, values in test.cases(row, audio, args.stereo_environment / "environment", args.stereo_rooms):
            if profile != scene and profile != "opposed":
                continue
            name = role if profile == scene else "opposed"
            encoded = args.output / f"{name}.m4a"
            case = lookup[profile, role]
            assert panel.phone.digest(values.tobytes()) == case["input_sha256"]
            test.encode(values, encoded)
            assert panel.phone.sha(encoded) == case["audio_container_sha256"]
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-i", str(encoded), "-map", "0:v:0", "-map",
                            "1:a:0", "-c", "copy", "-movie_timescale", "48000", str(args.output / f"{name}.mp4")],
                           check=True, timeout=20)
    measured = lookup[scene, "different_words"]
    return {"picture_sha256": chosen["source"], "scene": scene}, {
        "span": measured["witness"]["audio_source_pts"], "witness": measured["witness"], "stereo_scope": measured["scope"],
    }


def verify(client, aid, source, output, evidence, comparison):
    import pymupdf

    from app.evidence import video_viseme_binding as binding
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import pixels, scan, sha

    review = evidence[binding.KEY]
    witness = review["witness"]
    [packet] = [item for item in evidence["packets"] if item["id"] == binding.PACKET_ID]
    schema = binding.SCHEMA_WITH_STEREO if "stereo_scope" in comparison else binding.SCHEMA_WITH_ARTICULATION
    assert review["schema"] == schema and review["measurement_basis"] == "local_articulation"
    assert review["affects_overall_confidence"] is packet["affects_overall_confidence"] is False
    assert witness == packet[binding.KEY]
    # The carrier may include end padding. Its native clock still has to
    # reproduce the original window and the specific conflicting picture.
    assert comparison["span"] == review["measurement"]["span"], (comparison["span"], review["measurement"]["span"])
    if expected := comparison.get("witness"):
        soundtrack = evidence["audio"]["source"]
        assert soundtrack["source_sample_rate"] == 48_000 and soundtrack["source_codec"] == "aac"
        assert review["inputs"]["samples"] == binding.presented_samples(soundtrack)
        assert witness["audio_source_pts"] == expected["audio_source_pts"]
        assert witness["picture_frames"] == expected["picture_frames"]
        assert witness["articulation"]["audio_sample"] == expected["articulation"]["audio_sample"]
    if scope := comparison.get("stereo_scope"):
        assert evidence["audio"]["source"]["source_channels"] == 2
        assert review["stereo_scope"] == scope
        assert "mix of both soundtrack channels" in packet["observation"]
    during = packet["frames"][1]["anchor"]
    assert witness["articulation"]["picture_frame"] == during["frame_index"]
    assert witness["articulation"]["picture_pts"] == during["pts"] and witness["change_samples"] == 0
    assets = {role: fetch(client, asset["url"]) for role, asset in packet["artifacts"].items()}
    assert all(sha(data) == packet["artifacts"][role]["sha256"] for role, data in assets.items())
    report = body(client.post(f"/api/v2/analyses/{aid}/reports", json={"profile": "standard"}))
    wait(client, f"/api/v2/reports/{report['id']}")
    pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
    (output / "mismatch.pdf").write_bytes(pdf)
    with pymupdf.open(stream=pdf, filetype="pdf") as document:
        text = " ".join(word for page in document for word in page.get_text().split())
        scan(text)
        a, b = (sample / 16000 for sample in witness["audio_source_pts"])
        assert all(phrase in text for phrase in ("Local speech and mouth mismatch", "Before mismatch range",
                   "During mismatch range", "After mismatch range", f"{a:.3f}–{b:.3f} s"))
        if comparison.get("stereo_scope"):
            assert "mix of both soundtrack channels" in text
        embedded = {pixels(document.extract_image(ref)["image"])
                    for ref in {image[0] for page in document for image in page.get_images(full=True)}}
        assert {pixels(data) for role, data in assets.items() if role != binding.CLIP} <= embedded
    package = body(client.post(f"/api/v2/analyses/{aid}/packages", json={
        "report_id": report["id"], "kind": "draft", "include_original": False}))
    wait(client, f"/api/v2/packages/{package['id']}")
    archive = output / "mismatch.zip"
    archive.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
    public = body(client.get("/api/v2/instance/public-key"))["public_key_pem"].encode()
    checked = verify_package(archive, public_key_pem=public, source_video=source)
    assert checked.ok, checked.as_dict()
    replay = next(check.detail for check in checked.checks if check.name == "original video frames")
    assert "timing review's soundtrack" in replay
    with ZipFile(archive) as bundle:
        assert json.loads(bundle.read("evidence.json")) == evidence
        assert bundle.read(f"assets/{packet['artifacts'][binding.CLIP]['sha256']}.wav") == assets[binding.CLIP]
    return {"witness": witness, "verified_assets": len(assets), "source_replay": replay, "pdf_sha256": sha(pdf),
            "sound_sha256": packet["artifacts"][binding.CLIP]["sha256"]}


def run(args):
    configure(args.output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_VIDEO_LIP_TIMING_REVIEW="true", VERITAS_VIDEO_LIP_TIMING_RANKED="true",
                      VERITAS_VIDEO_LIP_TIMING_MODEL_DIR=str((args.assets / "checkpoint").resolve()),
                      VERITAS_VIDEO_LIP_TIMING_FACE_ASSET=str((args.assets / "face_landmarker.task").resolve()),
                      VERITAS_VIDEO_LIP_TIMING_RUNTIME=str((args.assets / "runtime").resolve()), VERITAS_LOG_LEVEL="ERROR")
    if args.standard_checks:
        os.environ.update(VERITAS_DEVICE="auto", VERITAS_JOB_TIMEOUT_S="480")
    proof, comparison = (stereo_fixtures(args) if args.stereo_environment else
                         fixtures(args.output, args.sources, args.plan, args.receipt, aac48=args.aac48))

    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.engine.registry import get_registry
    from app.evidence import video_viseme_binding as binding
    from app.main import app
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import scan

    def decision(aid):
        with get_session() as db:
            data = json.loads(default_store().read_bytes(db.get(Analysis, aid).result_hash))
        return data["decision"], data.get("customer_assessment")

    if not args.standard_checks:
        registry = get_registry()
        enabled = registry.enabled
        registry.enabled = lambda modality, profile: [key for key in enabled(modality, profile)
                                                       if registry.detector_family(key) == "provenance"]
    stub = PaymentStub()
    service._stripe = stub
    checks = []
    browser_examples = []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-articulation-smoke", "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            client.headers["X-API-Key"] = key
            db.commit()
        variants = [(False, "different_words"), (True, "different_words"), (True, "same_words"), (True, "original")]
        if args.stereo_environment:
            variants.append((True, "opposed"))
        for active, name in variants:
            settings.video_lip_articulation_review = active or args.aac48 or args.stereo_environment is not None
            settings.video_lip_aac48_articulation_review = bool(args.stereo_environment or (active and args.aac48))
            settings.video_lip_stereo_aac_review = bool(active and args.stereo_environment)
            source = args.output / f"{name}.mp4"
            aid = checkout(client, stub, source, filename=source.name)
            evidence = body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            exported = body(client.get(f"/api/analyses/{aid}/export.json"))
            assert evidence["status"] == "ready" and exported["evidence"] == evidence
            scan(json.dumps(exported))
            review = evidence[binding.KEY]
            print(name, active, review["status"], review.get("measurement_basis"), flush=True)
            case = {"delivery": name, "enabled": active, "status": review["status"]}
            if not active:
                if args.stereo_environment:
                    assert review == binding.skipped("unsupported")
                else:
                    assert "articulation_phones" not in review["inputs"] and review["witness"] is None
                baseline = decision(aid)
            elif name == "different_words":
                assert decision(aid) == baseline
                case.update(verify(client, aid, source, args.output, evidence, comparison))
                browser_examples.append({"analysis_id": aid, "witness": case["witness"], "sound_sha256": case["sound_sha256"]})
                (args.output / "mismatch.json").write_text(json.dumps(exported, indent=2) + "\n")
            elif name == "opposed":
                assert review == binding.skipped("stereo_unresolved")
                assert all(packet["id"] != binding.PACKET_ID for packet in evidence["packets"])
            else:
                schema = binding.SCHEMA_WITH_STEREO if args.stereo_environment else binding.SCHEMA_WITH_ARTICULATION
                assert review["witness"] is None and review["schema"] == schema
            checks.append(case)
        browser = None
        if args.browser:
            from scripts.image_jpeg_turned_copy_app import browser_check

            browser = browser_check(args.output, app, key, browser_examples, scenario="lip-articulation-replay.mjs")
    record = {"paid_analyses": len(stub.sessions), "unpaid_jobs": 0, "price_usd": 9.99, "same_assessment": True,
              "standard_checks": args.standard_checks, "picture_sha256": proof["picture_sha256"], "checks": checks, "browser": browser}
    if args.aac48 or args.stereo_environment:
        record["source_sample_rate"] = 48_000
    if args.stereo_environment:
        record.update(source_channels=2, source_delivery=proof["scene"])
    panel.phone.write(args.output / "receipt.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for target in ("output", "sources", "plan", "receipt", "assets"):
        parser.add_argument(target, type=Path)
    parser.add_argument("--standard-checks", action="store_true")
    parser.add_argument("--browser", action="store_true")
    parser.add_argument("--aac48", action="store_true")
    parser.add_argument("--stereo-environment", type=Path)
    parser.add_argument("--stereo-rooms", type=Path)
    args = parser.parse_args()
    if bool(args.stereo_environment) != bool(args.stereo_rooms) or (args.stereo_environment and args.aac48):
        parser.error("Stereo replay requires both stereo source folders and cannot use the mono replay option.")
    run(args)
