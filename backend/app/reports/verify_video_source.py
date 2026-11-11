"""Verify public video image evidence against the original video."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image

from app.core.canonical import canonical_hash
from app.detectors.video.decode import DecodedFrame, FrameTable, decode_frames, probe_frames
from app.detectors.video.lipsync import source_mouth_patch
from app.evidence.face_boundary_binding import validate_packet as validate_face_boundary
from app.evidence.source_geometry import pixel_bounds
from app.reports.public_verify import SCHEMA
from app.reports.verify_image_source import _matches

MAX_FRAME_PIXELS = 16 * 1024 ** 2


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _frame_anchor(frame: DecodedFrame, expected_pts: int, expected_sha: str, label: str, decoded_sha: str) -> None:
    if not frame.pts_verified or frame.pts != expected_pts:
        raise ValueError(f"{label} does not match a verified native presentation timestamp")
    if decoded_sha != expected_sha:
        raise ValueError(f"{label} differs from the decoded source pixels")


def _verify_clip_clock(packet: dict, table: FrameTable, interpolated: set[int]) -> None:
    clip = packet.get("clip") or {}
    if clip.get("status") != "ready":
        return
    start, end = clip["start_frame_index"], clip["end_frame_index_exclusive"]
    if (not 0 <= start < end <= len(table) or clip["source_pts"] != list(table.pts[start:end])
            or interpolated.intersection(range(start, end))):
        raise ValueError(f"{packet['id']} context clip differs from the source frame clock")
    if end < len(table) and clip["last_frame_duration"]["ticks"] != (table.pts[end] - table.pts[end - 1]) * table.tb.numerator:
        raise ValueError(f"{packet['id']} context clip ends at a different source time")


def _verify_images(archive: ZipFile, image: Image.Image, packet: dict, item: dict) -> int:
    refs, assets = item["artifact_refs"], packet["artifacts"]
    images = [("original", (0, 0, image.width, image.height))]
    if "crop" in refs:
        images.append(("crop", pixel_bounds(item["context_region"], image.width, image.height)))
    for role, bounds in images:
        asset = assets[refs[role]]
        with archive.open(f"assets/{asset['sha256']}.png") as stream, Image.open(stream) as plate:
            if not _matches(image, bounds, plate):
                raise ValueError(f"{item['id']} {role} differs from pixels at its recorded source coordinates")
    return len(images)


def _verify_mouth(archive: ZipFile, rgb: np.ndarray, packet: dict, row: dict) -> None:
    height, width = rgb.shape[:2]
    crop = source_mouth_patch(rgb, pixel_bounds(row["mouth_region"], width, height))
    asset = packet["artifacts"][row["artifact_ref"]]
    if hashlib.sha256(crop.tobytes()).hexdigest() != row["mouth_patch_sha256"]:
        raise ValueError(f"{packet['id']} mouth patch differs from the native mouth rectangle")
    with archive.open(f"assets/{asset['sha256']}.png") as stream, Image.open(stream) as plate:
        expected = np.repeat(crop[:, :, None], 3, axis=2) if plate.mode == "RGB" else crop
        if plate.format != "PNG" or plate.mode not in {"L", "RGB"} or not np.array_equal(np.asarray(plate), expected):
            raise ValueError(f"{packet['id']} published mouth patch differs from its measured input")


def _verify_aligned_face(rgb: np.ndarray, row: dict) -> None:
    side = row["alignment_side"]
    if type(side) is not int or side <= 0 or side * side > MAX_FRAME_PIXELS:
        raise ValueError("A recorded face alignment exceeds native verification limits")
    affine = np.asarray([[float.fromhex(value) for value in vector] for vector in row["alignment_matrix_hex"]])
    aligned = cv2.resize(cv2.warpAffine(rgb, affine, (side, side), flags=cv2.INTER_LINEAR),
                         (112, 112), interpolation=cv2.INTER_AREA)
    if hashlib.sha256(aligned.tobytes()).hexdigest() != row["aligned_rgb_sha256"]:
        raise ValueError("A recorded aligned face differs from the native source pixels")


def check(archive: ZipFile, bundle: dict, original: Path) -> str:
    if bundle.get("schema") != SCHEMA or bundle.get("analysis", {}).get("kind") != "video":
        raise ValueError("Original-video verification requires a public video evidence package")
    digest = bundle["source_commitment"]["sha256"]
    if _sha(original) != digest:
        raise ValueError("The supplied video SHA-256 differs from the committed original")

    evidence = json.loads(archive.read("evidence.json"))
    source = evidence["source"]
    table = probe_frames(original)
    if table.width * table.height > MAX_FRAME_PIXELS:
        raise ValueError("The original video exceeds the native verification limit of 16 megapixels per frame")
    if (source.get("stream_selector") != "v:0" or source.get("coordinate_space") != "display-oriented-native-frame"
            or (table.width, table.height, table.time_base, len(table)) !=
               (source["width"], source["height"], source["time_base"], source["frame_count"])
            or canonical_hash(table.to_json()) != source["frame_table_sha256"]):
        raise ValueError("The decoded frame table differs from its commitment. Check the local FFmpeg version.")

    pictures, mouths, faces = defaultdict(list), defaultdict(list), defaultdict(list)
    interpolated = set(table.interpolated)
    packets = [*evidence["packets"], *(evidence.get("av") or {}).get("packets", [])]
    for packet in packets:
        _verify_clip_clock(packet, table, interpolated)
        validate_face_boundary(packet, source, evidence["version"], table_pts=table.pts, interpolated=interpolated)
        for item in (*packet["frames"], *packet.get("face_boundary_frames", ())):
            pictures[item["anchor"]["frame_index"]].append((packet, item))
        if packet.get("semantics") == "mouth_audio_correlation":
            for row in packet["window"]["frame_inputs"]:
                mouths[row["frame_index"]].append((packet, row))
        for key in ("face_sequence", "face_boundary"):
            for row in (packet.get(key) or {}).get("frame_inputs", []):
                if row not in faces[row["frame_index"]]:
                    faces[row["frame_index"]].append(row)
    indices = set(pictures) | set(mouths) | set(faces)
    found = set()
    images = patches = alignments = 0
    with closing(decode_frames(original, table, sorted(indices))) as stream:
        for frame in stream:
            index = frame.index
            found.add(index)
            rgb_sha = hashlib.sha256(frame.rgb).hexdigest()
            if index in pictures:
                with Image.fromarray(frame.rgb) as image:
                    for packet, item in pictures[index]:
                        anchor = item["anchor"]
                        if (anchor["original_sha256"], anchor["stream_selector"], anchor["time_base"]) != (digest, "v:0", table.time_base):
                            raise ValueError(f"{item['id']} is bound to a different video source or stream")
                        _frame_anchor(frame, anchor["pts"], anchor["decoded_rgb_sha256"], item["id"], rgb_sha)
                        images += _verify_images(archive, image, packet, item)
            for packet, row in mouths[index]:
                _frame_anchor(frame, row["frame_pts"], row["decoded_rgb_sha256"], packet["id"], rgb_sha)
                _verify_mouth(archive, frame.rgb, packet, row)
                patches += 1
            for row in faces[index]:
                _frame_anchor(frame, row["pts"], row["decoded_rgb_sha256"], "Aligned face", rgb_sha)
                _verify_aligned_face(frame.rgb, row)
                alignments += 1
    if found != indices:
        raise ValueError("The decoder did not return every referenced native frame")
    if _sha(original) != digest:
        raise ValueError("The supplied original changed during decoding")
    return (f"Original SHA-256 and the source frame clock matched. Verified native frames ({len(found)}), "
            f"original images or crops ({images}), mouth patches ({patches}), and aligned face inputs ({alignments}).")
