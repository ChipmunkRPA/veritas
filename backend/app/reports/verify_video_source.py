"""Verify public video image evidence against the original video."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from contextlib import closing
from itertools import groupby, pairwise
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image

from app.core.canonical import canonical_hash
from app.detectors.image.faces import Face
from app.detectors.video import (
    face_paired_edge,
    face_return,
    face_return_fine_scene,
    face_short_proposal,
    face_spatial_cut,
    picture_motion,
)
from app.detectors.video.decode import DecodedFrame, FrameTable, decode_frames, probe_frames
from app.detectors.video.frames import probe as probe_container
from app.detectors.video.lipsync import source_mouth_patch
from app.evidence import (
    face_boundary_binding,
    face_paired_edge_binding,
    face_short_carrier_binding,
    face_short_proposal_binding,
    face_short_window_binding,
    face_spatial_cut_binding,
    mouth_local_hold_binding,
    picture_hold_binding,
    picture_sequence,
)
from app.evidence.face_boundary_binding import validate_packet as validate_face_boundary
from app.evidence.face_return_binding import DENSE_ENCODING, FINE_SCENE_METHOD, GUARDED_METHODS
from app.evidence.face_return_binding import validate_packet as validate_face_return
from app.evidence.source_geometry import pixel_bounds
from app.evidence.video_packet_versions import (
    DENSE_FACE_VERSIONS,
    GUARDED_720P30_FACE_VERSION,
    GUARDED_CODEC_FACE_VERSION,
    PAIRED_EDGE_VERSION,
    SECOND_HD_FACE_VERSION,
    SECOND_SHORT_FACE_VERSION,
    SHORT_CARRIER_VERSION,
    SHORT_PROPOSAL_VERSION,
    SHORT_WINDOW_VERSION,
    SPATIAL_CUT_VERSION,
)
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


def _scene_sample(row, frames):
    rgb = frames[row["frame_index"]]
    box = row["face_box"]
    face = None if box is None else Face(*box, 0.)
    return face_return.Sample(face, None, face_return.scene(rgb, face))


def _verify_scene_support(record, frames) -> int:
    count = 0
    rows = record["scene_frame_inputs"]
    pairs = ((rows[0], rows[1]), (rows[2], rows[3]))
    for pair, expected, correlation in zip(pairs, record["fine_scene_edges"], record["scene_correlations_hex"], strict=True):
        samples = [_scene_sample(row, frames) for row in pair]
        value = face_return.scene_correlation(*samples)
        if expected is not None:
            fine = [face_return_fine_scene.frame(frames[row["frame_index"]], row["face_box"]) for row in pair]
            if value is not None:
                raise ValueError("Fine scene support replaced an available coarse background measurement")
            coarse_tiles = int((samples[0].scene[1] & samples[1].scene[1]).sum())
            value, support = face_return_fine_scene.edge(*fine, coarse_tiles)
            if support is None or any(
                (not math.isclose(float.fromhex(measurement), float.fromhex(expected[key]), abs_tol=1e-6, rel_tol=0)
                 if key.endswith("_hex") else measurement != expected[key])
                for key, measurement in support.items()
            ):
                raise ValueError("Fine scene background measurements differ from the original video")
            count += 1
        if value is None or not math.isclose(value, float.fromhex(correlation), abs_tol=1e-6, rel_tol=0):
            raise ValueError("Scene boundary correlation differs from the original video")
    return count


def _verify_guarded_scene(record, frames) -> None:
    rows = record["scene_frame_inputs"]
    pairs = ((rows[0], rows[1]), (rows[2], rows[3]))
    for pair, median, quartile in zip(pairs, record["scene_correlations_hex"], record["scene_lower_quartiles_hex"], strict=True):
        samples = [_scene_sample(row, frames) for row in pair]
        for observed, recorded in ((face_return.scene_correlation(*samples), median),
                                   (face_return.scene_lower_quartile(*samples), quartile)):
            if observed is None or not math.isclose(observed, float.fromhex(recorded), abs_tol=1e-6, rel_tol=0):
                raise ValueError("Guarded background support differs from the original video")


def _verify_paired_motion(record, reduced, *, proposal=False) -> None:
    boxes = {row["frame_index"]:row["bbox"] for row in record["frame_inputs"]}
    planned = set(reduced) if proposal else set(record["scheduled_frame_indices"])
    if proposal and face_short_proposal.shortlist(reduced, [boxes[i] for i in record["shortlist_anchor_indices"]], lambda _:None) != record["shortlist_edges"]:
        raise ValueError("The original video does not reproduce its bounded face shortlist")
    for label, edge in (("entry",record["start_frame_index"]),("exit",record["end_frame_index"])):
        inputs = face_paired_edge_binding.motion_inputs(edge,planned)
        observed = face_paired_edge.motion(edge,{i:reduced[i] for i in inputs},boxes[edge-1])
        committed = record["edges"][label]
        if observed is None or observed["sector_ids"] != committed["sector_ids"] or any(
            not math.isclose(float.fromhex(a),float.fromhex(b),abs_tol=.000001,rel_tol=0)
            for a,b in zip([observed["central_hex"],*observed["sectors_hex"]],
                           [committed["central_hex"],*committed["sectors_hex"]],strict=True)
        ):
            raise ValueError("Paired face boundary motion differs from the original video")


def _verify_picture_hold(evidence, table, observations) -> None:
    """Recompute the scoped longest equal-picture run from the supplied source."""
    coverage = evidence["picture_hold"]
    ticks = [point for point, _ in observations]
    broken = [i for i, (first, second) in enumerate(pairwise(ticks), 1)
              if first is None or second is None or not 0 < (second - first) * table.tb <= picture_hold_binding.MAX_STEP]
    if len(broken) != coverage["continuity_breaks"] or ticks[-1] != coverage["last_pts"]:
        raise ValueError("The original video differs from the recorded repeated-picture coverage")
    duration = picture_hold_binding.MIN_SPAN
    eligible = False
    holds = []
    for start, stop in pairwise([0, *broken, len(ticks)]):
        if ticks[start] is None or (ticks[stop - 1] - ticks[start]) * table.tb < duration:
            continue
        eligible = True
        index = start
        for digest, run in groupby(observations[start:stop], key=lambda row: row[1]):
            count = sum(1 for _ in run)
            end = index + count - 1
            span = ticks[end] - ticks[index]
            if span * table.tb >= duration:
                holds.append((span, index, end, digest))
            index = end + 1
    if not holds:
        if coverage["status"] != ("no_match" if eligible else "unavailable"):
            raise ValueError("The original video does not reproduce the recorded repeated-picture outcome")
        return
    _, start, end, digest = max(holds, key=lambda row: row[0])
    match = next((packet.get("picture_hold") for packet in evidence["packets"] if packet["id"] == picture_hold_binding.PACKET_ID), None)
    if match is None or (match["start_frame_index"], match["end_frame_index"], match["decoded_rgb_sha256"]) != (start, end, digest):
        raise ValueError("The longest repeated picture differs from the original video")


def check(archive: ZipFile, bundle: dict, original: Path) -> str:
    if bundle.get("schema") != SCHEMA or bundle.get("analysis", {}).get("kind") != "video":
        raise ValueError("Original-video verification requires a public video evidence package")
    digest = bundle["source_commitment"]["sha256"]
    if _sha(original) != digest:
        raise ValueError("The supplied video SHA-256 differs from the committed original")

    evidence = json.loads(archive.read("evidence.json"))
    source = evidence["source"]
    encoding = None
    delivery = None
    guarded_codec = evidence["version"] in {
        GUARDED_CODEC_FACE_VERSION, GUARDED_720P30_FACE_VERSION, SECOND_SHORT_FACE_VERSION, SECOND_HD_FACE_VERSION,
    }
    proposal = evidence["version"] == SHORT_PROPOSAL_VERSION
    windowed = evidence["version"] == SHORT_WINDOW_VERSION
    carrier_windows = evidence["version"] == SHORT_CARRIER_VERSION
    paired = evidence["version"] in {PAIRED_EDGE_VERSION, SHORT_PROPOSAL_VERSION, SPATIAL_CUT_VERSION}
    if evidence["version"] in DENSE_FACE_VERSIONS:
        delivery = probe_container(original)
        encoding = delivery.get("codec"), delivery.get("pix_fmt")
        if not guarded_codec and encoding != DENSE_ENCODING:
            raise ValueError("The original video encoding differs from the dense short face-change scope")
    table = probe_frames(original)
    if table.width * table.height > MAX_FRAME_PIXELS:
        raise ValueError("The original video exceeds the native verification limit of 16 megapixels per frame")
    if (source.get("stream_selector") != "v:0" or source.get("coordinate_space") != "display-oriented-native-frame"
            or (table.width, table.height, table.time_base, len(table)) !=
               (source["width"], source["height"], source["time_base"], source["frame_count"])
            or canonical_hash(table.to_json()) != source["frame_table_sha256"]):
        raise ValueError("The decoded frame table differs from its commitment. Check the local FFmpeg version.")
    if paired:
        delivery = probe_container(original, include_codec_tag=True)
        if not face_paired_edge_binding.supported(table, (delivery.get("codec"),delivery.get("pix_fmt"),delivery.get("codec_tag")),
                                                  face_return.source_carrier(original)):
            raise ValueError("The original video differs from its brief face-review source profile")
    if windowed:
        delivery = probe_container(original, include_codec_tag=True)
        if not face_short_window_binding.supported(table, (delivery.get("codec"), delivery.get("pix_fmt"), delivery.get("codec_tag")),
                                                    face_return.source_carrier(original)):
            raise ValueError("The original video differs from its overlapping face-window source profile")
    if carrier_windows:
        delivery = probe_container(original, include_codec_tag=True)
        encoding = (delivery.get("codec"), delivery.get("pix_fmt"), delivery.get("codec_tag"))
        carrier = face_return.source_carrier(original)
        if (not face_short_carrier_binding.supported(table, encoding, carrier)
                or any(evidence["coverage"][face_short_window_binding.KEY][key] != value
                       for key, value in face_short_carrier_binding.source_metadata(table, encoding).items())):
            raise ValueError("The original video differs from its measured face-window carrier or cadence")

    picture_hold_binding.validate(evidence)
    hold = evidence.get("picture_hold") or {}
    sequence = hold.get("sequence_review")
    motion = hold.get("motion_sequence_review")
    local_mouth = hold.get("mouth_local_hold")
    mouth_limit = 0
    if local_mouth is not None:
        delivery = delivery if delivery is not None else probe_container(original)
        if (not mouth_local_hold_binding.eligible(table, (delivery.get("codec"), delivery.get("pix_fmt")), schema=local_mouth["schema"])
                or (local_mouth["schema"] == mouth_local_hold_binding.HD_SCHEMA and face_return.source_carrier(original) != "mp4")):
            raise ValueError("The original video differs from the mouth-region review profile")
        mouth_limit = local_mouth["frames_scanned"]
    motion_encoding = None
    motion_profile = None
    motion_limit = 0
    if motion is not None:
        delivery = delivery if delivery is not None else probe_container(original)
        motion_encoding = delivery.get("codec"), delivery.get("pix_fmt")
        motion_profile = picture_motion.profile(table, motion_encoding, schema=motion["schema"])
        if motion_profile is None or motion["encoding"] != list(motion_encoding):
            raise ValueError("The original video differs from its motion comparison source profile")
        motion_limit = motion_profile[1]
    extended = "continuity_breaks" in hold
    hold_frames = 0
    if extended or sequence is not None or local_mouth is not None:
        picture_hold_binding.validate_source_clock(evidence, table.pts, table.interpolated)
        hold_frames = hold["frames_scanned"]
    pictures, mouths, faces, scenes = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    fine_records, guarded_records, paired_records, background_indices, background_frames = [], [], [], set(), {}
    proposal_records, paired_indices, paired_pixels = [], set(), {}
    window_records = []
    cut_records, cut_indices, cut_pixels = [], set(), {}
    interpolated = set(table.interpolated)
    packets = [*evidence["packets"], *(evidence.get("av") or {}).get("packets", [])]
    for packet in packets:
        _verify_clip_clock(packet, table, interpolated)
        validate_face_boundary(packet, source, evidence["version"], table_pts=table.pts, interpolated=interpolated)
        face_spatial_cut_binding.validate_packet(packet, source, evidence["version"], table_pts=table.pts, interpolated=interpolated)
        validate_face_return(packet, source, table_pts=table.pts, interpolated=interpolated,
                             rotation=table.rotation, truncated=table.truncated)
        for item in (*packet["frames"], *packet.get("face_boundary_frames", ()), *packet.get("face_spatial_cut_frames", ())):
            pictures[item["anchor"]["frame_index"]].append((packet, item))
        if cut := packet.get("face_spatial_cut"):
            cut_records.append(cut)
            for row in cut["frame_inputs"]:
                scenes[row["frame_index"]].append(row)
                cut_indices.add(row["frame_index"])
        if packet.get("semantics") == "mouth_audio_correlation":
            for row in packet["window"]["frame_inputs"]:
                mouths[row["frame_index"]].append((packet, row))
        for key in ("face_sequence", "face_boundary", "face_return"):
            for row in (packet.get(key) or {}).get("frame_inputs", []):
                if row not in faces[row["frame_index"]]:
                    faces[row["frame_index"]].append(row)
        if face_boundary_binding.is_paired(boundary := packet.get("face_boundary")):
            paired_records.append(boundary)
            for row in boundary["scene_frame_inputs"]:
                scenes[row["frame_index"]].append(row)
                paired_indices.add(row["frame_index"])
        record = packet.get("face_return") or {}
        for row in record.get("scene_frame_inputs", []):
            scenes[row["frame_index"]].append(row)
        if record.get("method") == face_short_proposal_binding.METHOD:
            proposal_records.append(record)
            paired_indices.update(row["frame_index"] for row in record["scene_frame_inputs"])
        if record.get("method") in {face_short_window_binding.METHOD, face_short_carrier_binding.METHOD}:
            window_records.append(record)
            paired_indices.update(row["frame_index"] for row in record["scene_frame_inputs"])
        if record.get("method") == FINE_SCENE_METHOD:
            fine_records.append(record)
            background_indices.update(row["frame_index"] for row in record["scene_frame_inputs"])
        elif record.get("method") in GUARDED_METHODS:
            guarded_records.append(record)
            background_indices.update(row["frame_index"] for row in record["scene_frame_inputs"])
    if guarded_codec:
        expected = 2 if evidence["version"] == SECOND_HD_FACE_VERSION else 1
        carrier = face_return.source_carrier(original)
        if (len(guarded_records) != expected or any(encoding != tuple(record["source_encoding"])
                or carrier != record["source_carrier"] for record in guarded_records)):
            raise ValueError("The original video's codec or carrier differs from its guarded face-review scope")
    elif guarded_records:
        raise ValueError("Guarded face evidence requires its dedicated version")
    indices = set(pictures) | set(mouths) | set(faces) | set(scenes) | set(range(hold_frames))
    hold_observations = []
    motion_descriptors = []
    mouth_pictures = []
    found = set()
    images = patches = alignments = 0
    with closing(decode_frames(original, table, sorted(indices))) as stream:
        for frame in stream:
            index = frame.index
            found.add(index)
            rgb_sha = hashlib.sha256(frame.rgb).hexdigest()
            if index < hold_frames:
                hold_observations.append((frame.pts if frame.pts_verified else None, rgb_sha))
            if index < motion_limit:
                motion_descriptors.append(picture_motion.descriptor(frame.rgb))
            if index < mouth_limit:
                _frame_anchor(frame, local_mouth["pts"][index], local_mouth["frame_hashes"][index], "Mouth-region input", rgb_sha)
                mouth_pictures.append(frame.rgb)
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
            for row in scenes[index]:
                _frame_anchor(frame, row["pts"], row["decoded_rgb_sha256"], "Scene boundary", rgb_sha)
            if index in background_indices:
                background_frames[index] = frame.rgb
            if index in paired_indices:
                paired_pixels[index] = face_paired_edge.reduced_rgb(frame.rgb)
            if index in cut_indices:
                cut_pixels[index] = face_spatial_cut.reduce_picture(frame.rgb)
    if found != indices:
        raise ValueError("The decoder did not return every referenced native frame")
    if _sha(original) != digest:
        raise ValueError("The supplied original changed during decoding")
    if extended:
        _verify_picture_hold(evidence, table, hold_observations)
    if sequence is not None and canonical_hash(picture_sequence.scan(hold_observations, table.tb)) != canonical_hash(sequence):
        raise ValueError("The original video does not reproduce its repeated moving-picture sequences")
    if motion is not None:
        if motion["schema"] == picture_sequence.MOTION_EXTENDED_SCHEMA:
            opening_profile = picture_motion.profile(table, motion_encoding)
            opening_limit = opening_profile[1]
            opening = picture_motion.scan(hold_observations[:opening_limit], motion_descriptors[:opening_limit],
                                          table.tb, motion_encoding, sequence["pair"], source_profile=opening_profile)
            if opening["status"] != "no_match":
                raise ValueError("The extended motion comparison displaced an opening result")
        reproduced = picture_motion.scan(hold_observations[:motion_limit], motion_descriptors, table.tb, motion_encoding,
                                         sequence["pair"], source_profile=motion_profile, schema=motion["schema"])
        if canonical_hash(reproduced) != canonical_hash(motion):
            raise ValueError("The original video does not reproduce its similar moving-picture sequences")
    if local_mouth is not None:
        from app.detectors.video.mouth_local_hold import verify_measurements

        if len(mouth_pictures) != mouth_limit:
            raise ValueError("The source is missing a verified mouth-region input")
        verify_measurements(mouth_pictures, local_mouth["match"], hd=local_mouth["schema"] == mouth_local_hold_binding.HD_SCHEMA)
    fine_edges = sum(_verify_scene_support(record, background_frames) for record in fine_records)
    for record in guarded_records:
        _verify_guarded_scene(record, background_frames)
    for record in paired_records:
        _verify_paired_motion(record, {row["frame_index"]:paired_pixels[row["frame_index"]] for row in record["scene_frame_inputs"]})
    if len(proposal_records) != int(proposal):
        raise ValueError("The original video evidence version disagrees with its brief face proposal")
    for record in proposal_records:
        _verify_paired_motion(record, paired_pixels, proposal=True)
    if window_records and not (windowed or carrier_windows):
        raise ValueError("The original video evidence version disagrees with its overlapping face windows")
    for record in window_records:
        offset = record["window_start_frame_index"]
        binding = face_short_carrier_binding if carrier_windows else face_short_window_binding
        _verify_paired_motion(binding.local_record(record),
                              {i:paired_pixels[offset+i] for i in range(face_short_window_binding.WINDOW)}, proposal=True)
    if len(cut_records) != int(evidence["version"]==SPATIAL_CUT_VERSION):
        raise ValueError("The source evidence version disagrees with its picture-cut context")
    for cut in cut_records:
        coarse = (table.pts.index(cut["coarse_start_pts"]),table.pts.index(cut["coarse_end_pts"]))
        observed,tiles = face_spatial_cut.paired_edges(cut_pixels,face_spatial_cut.face_box(cut["face_inputs"]),coarse)
        if (not observed or observed["interval"] != [cut["start_frame_index"],cut["end_frame_index"]] or cut["tiles"] != tiles
                or any(observed[key]!=cut[key] for key in ("shared_tiles","entry_tiles","exit_tiles","sectors"))
                or not math.isclose(observed["score"],float.fromhex(cut["score_hex"]),rel_tol=0,abs_tol=.000001)):
            raise ValueError("The original video does not reproduce its surrounding-picture cut context")
    return (f"Original SHA-256 and the source frame clock matched. Verified native frames ({len(found)}), "
            f"original images or crops ({images}), mouth patches ({patches}), and aligned face inputs ({alignments})."
            + (f" Recomputed fine background boundaries ({fine_edges})." if fine_edges else "")
            + (" Recomputed 4 guarded background boundaries and verified the source carrier." if len(guarded_records) == 2 else
               " Recomputed both guarded background boundaries and verified the source carrier." if guarded_codec else "")
            + (" Recomputed both paired face boundary motion measurements and verified the source carrier." if paired_records else "")
            + (" Recomputed the bounded face shortlist and both near-head motion boundaries." if proposal_records else "")
            + (" Recomputed each carrier-qualified window shortlist and both native face boundaries." if carrier_windows and window_records else "")
            + (" Recomputed each overlapping window shortlist and both native face boundaries." if windowed and window_records else "")
            + (" Recomputed the surrounding-picture cut context and both native boundaries." if cut_records else "")
            + (f" Replayed every frame in the repeated-picture inspection ({hold_frames})." if hold_frames else "")
            + (" Recomputed the held mouth-region pixel measurements." if local_mouth and local_mouth["match"] else "")
            + (" Recomputed the exact moving-sequence comparison." if sequence is not None else "")
            + (" Replayed the motion-aware moving-sequence check." if motion is not None else ""))
