# SPDX-License-Identifier: MIT
"""Test reversed-motion evidence in lossy deliveries and ordinary controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.video-reverse-motion-transfer/1"
ROLES = ("reverse", "original", "normal_cut", "hold", "forward", "changing_spacing")
PROFILES = {"sd": (320, 180, 20), "larger": (640, 360, 25)}


def raster(path, width, height, rate):
    return subprocess.check_output(["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file", "-i", f"file:{path}",
                                    "-map", "0:v:0", "-an", "-vf", f"fps={rate},scale={width}:{height}:flags=area",
                                    "-frames:v", str(9*rate), "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"], timeout=30)


def edit(raw, shape, codec, path, role, early, late, length):
    import numpy as np

    from scripts import video_lossy_sequence_transfer as parent

    width, height, rate = shape
    size = width * height * 3
    frames = [memoryview(raw)[i*size:(i+1)*size] for i in range(9*rate)]
    first = frames[early:early+length]
    if role in {"reverse", "changing_spacing"}:
        frames[late:late+length] = first[::-1]
    elif role == "forward":
        frames[late:late+length] = first
    elif role == "hold":
        frames[late:late+length] = [frames[late]] * length
    elif role == "normal_cut":
        frames[late:late+length] = [np.frombuffer(frame, np.uint8).reshape(height, width, 3)[:, ::-1].tobytes()
                                    for frame in frames[late:late+length]]
    clock = f"settb=expr=1/1000,setpts=N*{1000//rate}"
    if role == "changing_spacing":
        clock += rf"+10*mod(N\,2)*between(N\,{late}\,{late+length-1})"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size",
                    f"{width}x{height}", "-framerate", str(rate), "-i", "pipe:0", "-filter_threads", "1",
                    "-vf", clock, "-fps_mode", "passthrough", "-enc_time_base", "1:1000", "-threads", "1",
                    "-pix_fmt", "yuv420p", *parent.CODECS[codec], str(path)], input=b"".join(frames), check=True, timeout=40)


def bounds(review):
    pair = review["pair"]
    return [pair[side][f"{edge}_frame_index"] for side in ("first", "second") for edge in ("start", "end")] if pair else None


def measure(path, codec):
    from app.detectors.video import picture_motion
    from app.detectors.video.decode import decode_frames, probe_frames
    from app.evidence import picture_sequence as bind

    table = probe_frames(path)
    observations, values = [], []
    for frame in decode_frames(path, table, range(len(table)), timeout_s=15):
        if frame.index != len(observations):
            raise ValueError("Decoded frame order differs from the source table")
        observations.append((frame.pts if frame.pts_verified else None, hashlib.sha256(frame.rgb).hexdigest()))
        values.append(picture_motion.descriptor(frame.rgb))
    encoding = (codec, "yuv420p")
    profile = picture_motion.profile(table, encoding, schema=bind.MOTION_REVERSE_SCHEMA)
    if profile is None or profile[1] != len(observations):
        raise ValueError("Unsupported delivery or incomplete native decode")
    exact = bind.scan(observations, table.tb, reverse=True)
    observed = picture_motion.scan(observations, values, table.tb, encoding, exact["pair"], source_profile=profile,
                                   schema=bind.MOTION_REVERSE_SCHEMA)
    return {"exact": bounds(exact), "motion": bounds(observed), "status": observed["status"], "pair": observed["pair"]}


def overlap(first, second):
    a, b = first
    c, d = second
    return 2 * max(0, min(b, d)-max(a, c)+1) >= max(b, d)-min(a, c)+1


def review(receipt):
    from scripts import video_lossy_sequence_transfer as parent

    rows = receipt["rows"]
    split = receipt["split"]
    if receipt["schema"] != SCHEMA or split not in {"development", "held"}:
        raise ValueError("Unexpected video comparison receipt")
    by_clip = defaultdict(set)
    sources = defaultdict(set)
    counts = defaultdict(Counter)
    for row in rows:
        profile, slot, codec, role = (row[k] for k in ("profile", "slot", "codec", "role"))
        if role not in ROLES or role in by_clip[profile, slot] or profile not in PROFILES:
            raise ValueError("Unexpected source variant")
        by_clip[profile, slot].add(role)
        sources[slot].add(row["source_sha256"])
        if not all(re.fullmatch(r"[a-f0-9]{64}", row[key]) for key in ("source_sha256", "delivery_sha256")):
            raise ValueError("A source or delivery hash is invalid")
        truth, found, pair = row["truth"], row["motion"], row["pair"]
        rate = PROFILES[profile][2]
        a, b, width = (round(n*rate/20) for n in parent.geometry(split, slot))
        if truth != ([a, a+width-1, b, b+width-1] if role == "reverse" else None):
            raise ValueError("Authored reverse boundaries differ from the source plan")
        if row["status"] not in {"matched", "no_match", "unavailable"}:
            raise ValueError("Unexpected reverse review status")
        if bool(found) != (row["status"] == "matched") or (found and bounds({"pair": pair}) != found):
            raise ValueError("Compared range differs from the source-bound pair")
        if pair and (pair["order"] != "reverse" or found[1] >= found[2]
                     or len(pair["supporting_edge_offsets"]) < 8):
            raise ValueError("Invalid reversed motion support")
        good = bool(truth and found and all(overlap(truth[i:i+2], found[i:i+2]) for i in (0, 2)))
        counts[profile, codec, role].update(files=1, exact=bool(row["exact"]), flags=bool(found),
                                           useful=good, exact_interval=bool(truth and found == truth),
                                           assessed=row["status"] != "unavailable")
    sample = parent.DEVELOPMENT if split == "development" else parent.HELD
    wanted = ("sd",) if split == "development" else tuple(PROFILES)
    if (set(by_clip) != {(profile, slot) for profile in wanted for slot in sample}
            or any(items != set(ROLES) for items in by_clip.values())
            or any(len(digests) != 1 for digests in sources.values())):
        raise ValueError("Incomplete separate source groups")
    result = {"/".join(keys): dict(count) for keys, count in sorted(counts.items())}
    if "outcomes" in receipt and result != receipt["outcomes"]:
        raise ValueError("Reported reverse sequence outcomes do not reproduce")
    return result


def native_input(folder, row):
    from scripts import video_lossy_sequence_transfer as parent

    path = folder / f"{row['profile']}-{row['slot']}-{row['role']}.mkv"
    if parent.hash_file(path) != row["delivery_sha256"]:
        raise ValueError("The delivered video differs from its recorded bytes")
    return path


def replay(args, receipt):
    review(receipt)
    for row in receipt["rows"]:
        path = native_input(args.archive, row)
        attrs = measure(path, row["codec"])
        if any(row[key] != result for key, result in attrs.items()):
            raise ValueError("The recorded motion ranges or source pictures differ")
    print(f"Reproduced native timing and motion on {len(receipt['rows'])} encoded clips.")


def app_test(args, receipt):
    import os
    from zipfile import ZipFile

    from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure, fetch, wait

    configure(args.output, Path(__file__).resolve().parents[2])
    os.environ["VERITAS_ENVIRONMENT"] = "dev"
    import pymupdf
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.engine.registry import get_registry
    from app.evidence import picture_sequence as binding
    from app.main import app
    from app.reports.verify import verify_package
    from app.storage.db import get_session
    from scripts import video_lossy_sequence_transfer as parent
    from scripts.verify_multimedia_exports import pixels, scan, sha

    review(receipt)
    chosen = [next(row for row in receipt["rows"] if row["codec"] == codec and row["profile"] == profile
                   and row["role"] == "reverse" and row["motion"] and not row["exact"])
              for codec, profile in (("vp9", "sd"), ("h264", "larger"))]
    for profile, role in (("sd", "original"), ("larger", "changing_spacing")):
        chosen.append(next(row for row in receipt["rows"] if row["profile"] == profile and row["role"] == role))
    registry = get_registry()
    enabled = registry.enabled
    # Leave the production payment and evidence stages in place. Other video inference is outside this check.
    registry.enabled = lambda modality, profile: [key for key in enabled(modality, profile)
                                                   if registry.detector_family(key) == "provenance"]
    stub = PaymentStub()
    service._stripe = stub
    outcomes = []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("reverse-motion-smoke", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()
        for row in chosen:
            source = native_input(args.archive, row)
            aid = checkout(client, stub, source, media_type="video/x-matroska", filename=source.name)
            evidence = body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            export = body(client.get(f"/api/analyses/{aid}/export.json"))
            assert evidence["status"] == "ready" and evidence == export["evidence"]
            scan(json.dumps(export))
            observed = evidence["picture_hold"][binding.MOTION_REVERSE_KEY]
            assert observed["status"] == row["status"] and observed["pair"] == row["pair"]
            matching = [packet for packet in evidence["packets"] if packet["id"] == binding.MOTION_REVERSE_PACKET_ID]
            assert bool(matching) == (row["status"] == "matched")
            proof = {"profile": row["profile"], "codec": row["codec"], "role": row["role"],
                     "source_sha256": row["delivery_sha256"], "status": row["status"], "frames": row["motion"]}
            if matching:
                [packet] = matching
                assert packet["affects_overall_confidence"] is False and packet["picture_sequence"] == row["pair"]
                assert [f["anchor"]["frame_index"] for f in packet["frames"]] == row["motion"]
                images = {asset["sha256"]: fetch(client, asset["url"]) for asset in packet["artifacts"].values()}
                assert all(sha(data) == digest for digest, data in images.items())
                report = body(client.post(f"/api/v2/analyses/{aid}/reports", json={"profile": "standard"}))
                wait(client, f"/api/v2/reports/{report['id']}")
                pdf = fetch(client, f"/api/v2/reports/{report['id']}/file")
                (args.output / f"{row['profile']}.pdf").write_bytes(pdf)
                with pymupdf.open(stream=pdf, filetype="pdf") as document:
                    words = " ".join(" ".join(page.get_text().split()) for page in document)
                    scan(words)
                    assert "Similar moving pictures in reverse order" in words and "transitions support the reversed motion" in words
                    assert all(f"{row['pair'][side][f'{edge}_stream_s']:.3f}" in words
                               for side in ("first", "second") for edge in ("start", "end"))
                    embedded = {pixels(document.extract_image(ref)["image"])
                                for ref in {image[0] for page in document for image in page.get_images(full=True)}}
                    assert {pixels(data) for data in images.values()} <= embedded
                package = body(client.post(f"/api/v2/analyses/{aid}/packages",
                                           json={"report_id": report["id"], "kind": "draft", "include_original": False}))
                wait(client, f"/api/v2/packages/{package['id']}")
                archive = args.output / f"{row['profile']}.zip"
                archive.write_bytes(fetch(client, f"/api/v2/packages/{package['id']}/file"))
                public = body(client.get("/api/v2/instance/public-key"))["public_key_pem"].encode()
                checked = verify_package(archive, public_key_pem=public, source_video=source)
                assert checked.ok, checked.as_dict()
                with ZipFile(archive) as bundle:
                    assert json.loads(bundle.read("evidence.json")) == evidence
                proof.update(source_replay=True, pdf_sha256=sha(pdf), verified_images=len(images))
            outcomes.append(proof)
    parent.write(args.output / "receipt.json", {"paid_analyses": len(stub.sessions), "unpaid_jobs": 0, "outcomes": outcomes})


def run(args):
    from scripts import video_lossy_sequence_transfer as parent

    args.output.mkdir(mode=0o700)
    results = []
    slots = parent.DEVELOPMENT if args.phase == "development" else parent.HELD
    for name, shape in PROFILES.items():
        if args.phase == "development" and name != "sd":
            continue
        for slot in slots:
            path, sha, _ = parent.locate(args.archive, args.phase, slot)
            width, height, rate = shape
            raw = raster(path, width, height, rate)
            if len(raw) != 9*rate*width*height*3:
                raise ValueError("The source does not span nine full seconds")
            start, other, count = (round(value*rate/20) for value in parent.geometry(args.phase, slot))
            known = [start, start+count-1, other, other+count-1]
            codec = "h264" if (slot % 2 == 0) == (name == "sd") else "vp9"
            for role in ROLES:
                output = args.output / f"{name}-{slot}-{role}.mkv"
                edit(raw, shape, codec, output, role, start, other, count)
                result = measure(output, codec)
                results.append({"profile": name, "slot": slot, "codec": codec, "role": role,
                                "source_sha256": sha, "delivery_sha256": parent.hash_file(output),
                                "truth": known if role == "reverse" else None, **result})
                if not args.keep:
                    output.unlink()
            print(name, slot, "measured", flush=True)
    report = {"schema": SCHEMA, "split": args.phase, "rows": results}
    report["outcomes"] = review(report)
    parent.write(args.output / "receipt.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("development", "held", "review", "replay", "app"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "review":
        print(json.dumps(review(json.loads(args.output.read_bytes())), sort_keys=True))
    elif args.phase in {"replay", "app"}:
        if args.archive is None or args.receipt is None:
            parser.error("Encoded replay requires --archive for deliveries and --receipt for the original comparisons")
        operation = replay if args.phase == "replay" else app_test
        operation(args, json.loads(args.receipt.read_bytes()))
    else:
        if args.archive is None:
            parser.error("A qualified source archive is required")
        run(args)
