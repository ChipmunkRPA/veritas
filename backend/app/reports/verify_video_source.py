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
    face_single_context,
    face_spatial_cut,
    picture_motion,
    scene_return_carrier,
)
from app.detectors.video.decode import DecodedFrame, FrameTable, decode_frames, decode_stream, probe_frames
from app.detectors.video.frames import probe as probe_container
from app.detectors.video.lipsync import source_mouth_patch
from app.detectors.video.shots import ShotDetector
from app.evidence import (
    eye_local_hold_binding,
    face_boundary_binding,
    face_local_return_binding,
    face_paired_edge_binding,
    face_portrait_return_binding,
    face_short_carrier_binding,
    face_short_proposal_binding,
    face_short_window_binding,
    face_single_boundary_binding,
    face_spatial_cut_binding,
    mouth_hold_sound,
    mouth_local_hold_binding,
    picture_hold_binding,
    picture_sequence,
    scene_return_binding,
    video_viseme_binding,
    whole_face_binding,
)
from app.evidence.av_binding import TIMING_WINDOWS
from app.evidence.face_boundary_binding import validate_packet as validate_face_boundary
from app.evidence.face_return_binding import (
    DENSE_ENCODING,
    FINE_SCENE_METHOD,
    GUARDED_720P30_METHOD,
    GUARDED_METHODS,
)
from app.evidence.face_return_binding import validate_packet as validate_face_return
from app.evidence.source_geometry import pixel_bounds
from app.evidence.video_packet_versions import (
    BOUNDARY_AVC_PROPOSAL_VERSION,
    DENSE_FACE_VERSIONS,
    GUARDED_720P30_FACE_VERSION,
    GUARDED_CODEC_FACE_VERSION,
    MOVING_AVC_PROPOSAL_VERSION,
    OPPOSITE_PANE_VERSIONS,
    PAIRED_EDGE_VERSION,
    PANE_TRACK_AVC_PROPOSAL_VERSION,
    POSITION_TRACK_AVC_PROPOSAL_VERSION,
    PRIMARY_TRACK_AVC_PROPOSAL_VERSION,
    SECOND_AVC_PROPOSAL_VERSION,
    SECOND_HD_FACE_VERSION,
    SECOND_PANE_TRACK_AVC_PROPOSAL_VERSION,
    SECOND_SHORT_FACE_VERSION,
    SHORT_AVC_PROPOSAL_VERSION,
    SHORT_CARRIER_VERSION,
    SHORT_PROPOSAL_VERSION,
    SHORT_WEBM_PROPOSAL_VERSION,
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
    if proposal:
        anchors = {i: boxes[i] for i in record["shortlist_anchor_indices"]}
        if record["method"] == face_short_proposal_binding.BOUNDARY_AVC_METHOD:
            shortlist = face_short_proposal.boundary_shortlist(reduced, list(anchors.values()), lambda _: None)
        elif record["method"] in face_short_proposal_binding.MOVING_METHODS:
            shortlist = face_short_proposal.moving_shortlist(reduced, anchors, lambda _: None)
        else:
            shortlist = face_short_proposal.shortlist(reduced, list(anchors.values()), lambda _: None)
        if shortlist != record["shortlist_edges"]:
            raise ValueError("The original video does not reproduce its bounded face shortlist")
        if record["method"] in face_short_proposal_binding.PANE_METHODS:
            edges = (record["start_frame_index"], record["end_frame_index"])
            counts = [face_short_proposal.pane_motion(i, reduced, boxes[i-1], record["pane"]) for i in edges]
            if counts != record["off_face_motion_tiles"]:
                raise ValueError("The original video does not reproduce the pane motion counts")
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


def _decode_lip_soundtrack(path: Path, digest: str, record: dict, first_source: dict):
    from app.detectors.audio.decode import VIDEO_AUDIO_MAX_SECONDS, decode_audio, decode_default_second_aac441

    if record["schema"] == video_viseme_binding.SCHEMA_WITH_DEFAULT_AAC441_MOTION:
        decoded = decode_default_second_aac441(path, source_sha256=digest)
        selected = record.get(video_viseme_binding.SELECTED_AUDIO)
        if decoded is None or selected is None or video_viseme_binding.selected_audio(decoded.mapping) != selected:
            raise ValueError("The original video does not reproduce the timing review's selected default soundtrack")
        return decoded, selected

    stereo = record.get("stereo_scope")
    activity = record["inputs"].get(video_viseme_binding.ACTIVITY_KEY)
    decoded = decode_audio(path, source_sha256=digest, max_seconds=VIDEO_AUDIO_MAX_SECONDS,
                           inspect_stereo=stereo is not None, retain_stereo=bool(stereo or activity))
    return decoded, first_source


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
    opposite_pane = evidence["version"] in OPPOSITE_PANE_VERSIONS
    second_avc = opposite_pane or evidence["version"] in {SECOND_AVC_PROPOSAL_VERSION, SECOND_PANE_TRACK_AVC_PROPOSAL_VERSION}
    first_return = next((packet["face_return"] for packet in evidence["packets"] if packet.get("face_return")), {})
    avc_with_guarded_first = second_avc and first_return.get("method") == GUARDED_720P30_METHOD
    guarded_codec = evidence["version"] in {
        GUARDED_CODEC_FACE_VERSION, GUARDED_720P30_FACE_VERSION, SECOND_SHORT_FACE_VERSION, SECOND_HD_FACE_VERSION,
    } or avc_with_guarded_first
    expected_proposals = (1 if avc_with_guarded_first else 2) if second_avc else int(
        evidence["version"] in {SHORT_PROPOSAL_VERSION, SHORT_WEBM_PROPOSAL_VERSION, SHORT_AVC_PROPOSAL_VERSION,
                                MOVING_AVC_PROPOSAL_VERSION, PRIMARY_TRACK_AVC_PROPOSAL_VERSION,
                                POSITION_TRACK_AVC_PROPOSAL_VERSION, PANE_TRACK_AVC_PROPOSAL_VERSION,
                                BOUNDARY_AVC_PROPOSAL_VERSION})
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
    timing = (evidence.get("av", {}).get("coverage") or {}).get(TIMING_WINDOWS)
    if timing:
        from app.detectors.video import lipsync, sampling

        audio = evidence["av"]["source"]
        grid = lipsync.tick_grid(table.times(), sampling.duration_s(table), audio["samples"],
                                 audio_start_s=audio["timeline"][0]["source_pts"] / audio["sample_rate"])
        for row in timing["windows"]:
            if row["status"] == "insufficient_speech":
                continue
            start = row["interval"]["start_sample"] // lipsync.HOP
            end = row["interval"]["end_sample"] // lipsync.HOP
            missing = end > grid.ticks or any(frame < 0 for frame in grid.frames[start:end])
            if (row["status"] == "missing_picture_timestamps") != missing:
                raise ValueError("Timing window coverage differs from the original picture timestamps")
    if paired:
        delivery = probe_container(original, include_codec_tag=True)
        if not face_paired_edge_binding.supported(table, (delivery.get("codec"),delivery.get("pix_fmt"),delivery.get("codec_tag")),
                                                  face_return.source_carrier(original)):
            raise ValueError("The original video differs from its brief face-review source profile")
    if evidence["version"] == SHORT_WEBM_PROPOSAL_VERSION:
        delivery = probe_container(original, include_codec_tag=True)
        if not face_short_proposal_binding.webm_source(table, (delivery.get("codec"), delivery.get("pix_fmt"), delivery.get("codec_tag")),
                                                       face_return.source_carrier(original)):
            raise ValueError("The original video differs from its brief WebM face-review source profile")
    if evidence["version"] in {SHORT_AVC_PROPOSAL_VERSION, SECOND_AVC_PROPOSAL_VERSION, MOVING_AVC_PROPOSAL_VERSION,
                               PRIMARY_TRACK_AVC_PROPOSAL_VERSION, POSITION_TRACK_AVC_PROPOSAL_VERSION,
                               PANE_TRACK_AVC_PROPOSAL_VERSION, SECOND_PANE_TRACK_AVC_PROPOSAL_VERSION,
                               BOUNDARY_AVC_PROPOSAL_VERSION} | OPPOSITE_PANE_VERSIONS:
        delivery = probe_container(original, include_codec_tag=True)
        encoding = delivery.get("codec"), delivery.get("pix_fmt")
        if not face_short_proposal_binding.avc_source(table, (delivery.get("codec"), delivery.get("pix_fmt"), delivery.get("codec_tag")),
                                                      face_return.source_carrier(original)):
            raise ValueError("The original video differs from its brief H.264 face-review source profile")
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
    scene_return_binding.validate(evidence, pts=table.pts, canonical_hash=canonical_hash)
    scene_return_binding.validate_coverage(evidence, canonical_hash=canonical_hash, pts=table.pts)
    video_viseme_binding.validate(evidence, timestamps=table.pts)
    lip_timing = evidence.get(video_viseme_binding.KEY) or {}
    default_second = lip_timing.get("schema") == video_viseme_binding.SCHEMA_WITH_DEFAULT_AAC441_MOTION
    timing_frames = {row["index"]: row for row in lip_timing.get("inputs", {}).get("frames", []) if row["gap_units"] is not None}
    scene_coverage = (evidence.get("coverage") or {}).get(scene_return_binding.COVERAGE) or {}
    scene_committed = source.get(scene_return_binding.COMMITMENT)
    if scene_committed or scene_coverage.get("status") == "no_match":
        delivery = probe_container(original, include_codec_tag=True)
        scene_encoding = (delivery.get("codec"), delivery.get("pix_fmt"), delivery.get("codec_tag"))
        scene_schema = scene_coverage.get("schema", scene_return_binding.SCHEMA)
        if scene_return_carrier.profile(table, scene_encoding, face_return.source_carrier(original)) != scene_schema:
            raise ValueError("Picture-change evidence uses a different video delivery profile")
        verified = 0

        def native_pictures(frames):
            nonlocal verified
            for frame in frames:
                if frame.index != verified or verified >= len(table) or not frame.pts_verified or frame.pts != table.pts[verified]:
                    raise ValueError("Picture-change source frames do not match the verified clock")
                verified += 1
                yield frame

        with closing(decode_stream(original, table, timeout_s=20)) as frames:
            found = scene_return_carrier.detect(native_pictures(frames), table, scene_schema)
        if verified != len(table):
            raise ValueError("The original video did not reproduce the complete picture-change scan")
        if scene_committed:
            [packet] = [item for item in evidence["packets"] if item["id"] == scene_return_binding.PACKET_ID]
            record = packet[scene_return_binding.KEY]
            if found is None or canonical_hash(scene_return_binding.record(
                found, table, [item["decoded_rgb_sha256"] for item in record["anchors"]], schema=scene_schema,
            )) != scene_committed:
                raise ValueError("The standalone picture-change interval or tile support differs from the original")
        elif found is not None:
            raise ValueError("The original video did not reproduce the recorded no-match result")
    hold = evidence.get("picture_hold") or {}
    sequence = hold.get("sequence_review")
    reverse_sequence = hold.get(picture_sequence.REVERSE_KEY)
    motion_records = [(key, record) for key, _, _ in picture_sequence.MOTION_REVIEWS if (record := hold.get(key)) is not None]
    local_mouth = hold.get("mouth_local_hold")
    mouth_limit = 0
    if local_mouth is not None:
        delivery = delivery if delivery is not None else probe_container(original)
        if (not mouth_local_hold_binding.eligible(table, (delivery.get("codec"), delivery.get("pix_fmt")), schema=local_mouth["schema"])
                or (local_mouth["schema"] == mouth_local_hold_binding.HD_SCHEMA and face_return.source_carrier(original) != "mp4")):
            raise ValueError("The original video differs from the mouth-region review profile")
        mouth_limit = local_mouth["frames_scanned"]
    local_eye = hold.get(eye_local_hold_binding.KEY) or {}
    eye_limit = local_eye.get("frames_scanned", 0)
    if eye_limit:
        delivery = delivery if delivery is not None else probe_container(original)
        if not eye_local_hold_binding.eligible(
            table, (delivery.get("codec"), delivery.get("pix_fmt")), face_return.source_carrier(original),
        ):
            raise ValueError("The original video differs from the eye-region review profile")
    eye_match = local_eye.get("match")
    motion_encoding = None
    motion_limit = 0
    crop_limit = 0
    motion_reads = []
    motion_descriptor = picture_motion.descriptor
    if motion_records:
        delivery = delivery if delivery is not None else probe_container(original)
        motion_encoding = delivery.get("codec"), delivery.get("pix_fmt")
        if hold.get(picture_sequence.MOTION_RETIMED_KEY) is not None and face_return.source_carrier(original) != "mp4":
            raise ValueError("Retimed sequence review requires a verified MP4 carrier")
        if hold.get(picture_sequence.MOTION_CROP_KEY) is not None and face_return.source_carrier(original) != "mp4":
            raise ValueError("Cropped sequence review requires a verified MP4 carrier")
        for key, record in motion_records:
            exact_review = reverse_sequence if key == picture_sequence.MOTION_REVERSE_KEY else sequence
            profile = picture_motion.profile(table, motion_encoding, schema=record["schema"])
            if profile is None or record["encoding"] != list(motion_encoding):
                raise ValueError("The original video differs from its motion comparison source profile")
            motion_reads.append((record, exact_review, profile))
            if record["schema"] in picture_sequence.MOTION_CROP_SCHEMAS:
                from app.detectors.video.picture_crop import luminance as crop_luminance

                crop_limit = profile[1]
            else:
                motion_limit = max(motion_limit, profile[1])
            if record["schema"] == picture_sequence.MOTION_REFLECTED_HD_SCHEMA:
                motion_descriptor = picture_motion.hd_descriptor
    extended = "continuity_breaks" in hold
    hold_frames = 0
    if extended or sequence is not None or local_mouth is not None or eye_limit or motion_records:
        picture_hold_binding.validate_source_clock(evidence, table.pts, table.interpolated)
        hold_frames = hold["frames_scanned"]
    pictures, mouths, faces, scenes = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    fine_records, guarded_records, local_records, paired_records, background_indices, background_frames = [], [], [], [], set(), {}
    proposal_records, paired_indices, paired_pixels = [], set(), {}
    window_records = []
    cut_records, cut_indices, cut_pixels = [], set(), {}
    interpolated = set(table.interpolated)
    face_single_boundary_binding.validate(evidence, canonical_hash, table_pts=table.pts,
        interpolated=interpolated, rotation=table.rotation, truncated=table.truncated)
    transition = next((p[face_single_boundary_binding.KEY] for p in evidence["packets"]
                       if face_single_boundary_binding.KEY in p), None)
    transition_frames, transition_scenes, sparse_pixels = {}, {}, {}
    sparse_indices = set()
    transition_shots = None
    coverage = (evidence.get("coverage") or {}).get(face_single_boundary_binding.COVERAGE, {})
    native_method = coverage.get("method")
    native30 = native_method in face_single_boundary_binding.NATIVE_CLOCK_TICKS
    if native30 and coverage["frames_scanned"] > 0:
        if delivery is None or "codec_tag" not in delivery:
            delivery = probe_container(original, include_codec_tag=True)
        encoding = (delivery.get("codec"), delivery.get("pix_fmt"), delivery.get("codec_tag"))
        if not face_single_boundary_binding.native30_source(table, encoding, face_return.source_carrier(original), native_method):
            raise ValueError("The original video differs from its native-clock face transition profile")
    if transition is not None:
        if not native30:
            from app.detectors.video.face_local_detail import supported

            delivery = delivery if delivery is not None else probe_container(original)
            if not supported(table, (delivery.get("codec"), delivery.get("pix_fmt"))) or face_return.source_carrier(original) != "mp4":
                raise ValueError("The original video differs from its single face transition profile")
        transition_frames = {row["index"]: row for row in transition["source_frames"].values()}
        if "scene_context" in transition:
            sparse_indices = face_single_context.required_frames(transition)
        if native30 or sparse_indices:
            transition_shots = ShotDetector(table, min_shot_s=0.)
    packets = [*evidence["packets"], *(evidence.get("av") or {}).get("packets", [])]
    if any(packet.get("semantics") in {whole_face_binding.SAME_FRAME_METHOD, whole_face_binding.DETAIL_METHOD} for packet in packets):
        delivery = probe_container(original, include_codec_tag=True)
        encoding = delivery.get("codec"), delivery.get("pix_fmt"), delivery.get("codec_tag")
        if not whole_face_binding.same_frame_source(table, encoding, face_return.source_carrier(original)):
            raise ValueError("The original video differs from the same-frame face review profile")
    for packet in packets:
        _verify_clip_clock(packet, table, interpolated)
        whole_face_binding.validate_packet(packet, source, pts=table.pts)
        if packet.get("semantics") == whole_face_binding.DETAIL_METHOD:
            for row in packet["whole_face_review"]["samples"]:
                faces[row["frame_index"]].append(row)
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
        if record.get("method") in face_short_proposal_binding.METHODS:
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
        elif record.get("method") in face_portrait_return_binding.METHODS | {face_local_return_binding.METHOD}:
            from app.detectors.video.face_local_detail import supported

            delivery = probe_container(original, include_codec_tag=True)
            encoding = delivery.get("codec"), delivery.get("pix_fmt")
            if record["method"] == face_portrait_return_binding.HEVC_METHOD:
                encoding = (*encoding, delivery.get("codec_tag"), delivery.get("profile"))
                accepts = face_portrait_return_binding.hevc_supported
            elif record["method"] == face_portrait_return_binding.SQUARE_METHOD:
                encoding = (*encoding, delivery.get("codec_tag"))
                accepts = face_portrait_return_binding.square_supported
            elif record["method"] == face_portrait_return_binding.EIGHT_SECOND_METHOD:
                encoding = (*encoding, delivery.get("codec_tag"))
                accepts = face_portrait_return_binding.eight_second_supported
            else:
                accepts = face_portrait_return_binding.supported if record["method"] == face_portrait_return_binding.METHOD else supported
            if not accepts(table, encoding) or face_return.source_carrier(original) != "mp4":
                raise ValueError("The original video differs from the local face review profile")
            local_records.append(record)
            background_indices.update(row["frame_index"] for row in record["scene_frame_inputs"])
    if guarded_codec:
        expected = 2 if evidence["version"] == SECOND_HD_FACE_VERSION else 1
        carrier = face_return.source_carrier(original)
        if (len(guarded_records) != expected or any(encoding != tuple(record["source_encoding"])
                or carrier != record["source_carrier"] for record in guarded_records)):
            raise ValueError("The original video's codec or carrier differs from its guarded face-review scope")
    elif guarded_records:
        raise ValueError("Guarded face evidence requires its dedicated version")
    reading_limit = max(hold_frames, motion_limit, crop_limit)
    indices = set(pictures) | set(mouths) | set(faces) | set(scenes) | set(range(max(reading_limit, eye_limit))) | set(timing_frames)
    if transition_shots is not None:
        indices.update(range(len(table)))
    hold_observations = []
    motion_descriptors = []
    crop_frames = []
    mouth_pictures = []
    eye_pictures = []
    found = set()
    images = patches = alignments = 0
    with closing(decode_frames(original, table, sorted(indices))) as stream:
        for frame in stream:
            index = frame.index
            found.add(index)
            rgb_sha = hashlib.sha256(frame.rgb).hexdigest()
            if transition_shots is not None:
                if not frame.pts_verified or frame.pts != table.pts[index]:
                    raise ValueError("Face transition requires a verified native picture clock")
                transition_shots.update(index, frame.rgb)
                if index in sparse_indices:
                    sparse_pixels[index] = frame.rgb
            if row := timing_frames.get(index):
                _frame_anchor(frame, row["pts"], row["rgb_sha256"], "Speech and mouth timing input", rgb_sha)
                x, y, right, bottom = row["mouth_box"]
                if hashlib.sha256(frame.rgb[y:bottom, x:right].tobytes()).hexdigest() != row["mouth_crop_sha256"]:
                    raise ValueError("The original video does not reproduce the recorded timing crop")
            if index < reading_limit:
                hold_observations.append((frame.pts if frame.pts_verified else None, rgb_sha))
            if index < motion_limit:
                motion_descriptors.append(motion_descriptor(frame.rgb))
            if index < crop_limit:
                crop_frames.append(crop_luminance(frame.rgb))
            if index < mouth_limit:
                _frame_anchor(frame, local_mouth["pts"][index], local_mouth["frame_hashes"][index], "Mouth-region input", rgb_sha)
                mouth_pictures.append(frame.rgb)
            if index < eye_limit:
                _frame_anchor(frame, local_eye["pts"][index], local_eye["frame_hashes"][index], "Eye-region input", rgb_sha)
                if eye_match is not None:
                    eye_pictures.append(frame.rgb)
            if index in pictures:
                with Image.fromarray(frame.rgb) as image:
                    for packet, item in pictures[index]:
                        anchor = item["anchor"]
                        if (anchor["original_sha256"], anchor["stream_selector"], anchor["time_base"]) != (digest, "v:0", table.time_base):
                            raise ValueError(f"{item['id']} is bound to a different video source or stream")
                        _frame_anchor(frame, anchor["pts"], anchor["decoded_rgb_sha256"], item["id"], rgb_sha)
                        images += _verify_images(archive, image, packet, item)
            if index in transition_frames:
                from app.detectors.video.face_detail_inputs import align

                row = transition_frames[index]
                _frame_anchor(frame, row["pts"], row["decoded_rgb_sha256"], "Face appearance transition", rgb_sha)
                crop = align(frame.rgb, face_single_boundary_binding.eyes(row))
                if crop is None or hashlib.sha256(crop.tobytes()).hexdigest() != row["aligned_rgb_sha256"]:
                    raise ValueError("The original video does not reproduce the face transition crop")
                alignments += 1
                if not sparse_indices and index in {transition["boundary"] - 1, transition["boundary"]}:
                    box = row["box"]
                    face = Face(box["x"], box["y"], box["w"], box["h"], 0.)
                    transition_scenes[index] = face_return.Sample(face, None, face_return.scene(frame.rgb, face))
            for packet, row in mouths[index]:
                _frame_anchor(frame, row["frame_pts"], row["decoded_rgb_sha256"], packet["id"], rgb_sha)
                _verify_mouth(archive, frame.rgb, packet, row)
                patches += 1
            for row in faces[index]:
                _frame_anchor(frame, row["pts"], row["decoded_rgb_sha256"], "Aligned face", rgb_sha)
                if "eyes_hex" in row:
                    from app.detectors.video.face_detail_inputs import align
                    from app.evidence.face_detail_binding import eyes

                    crop = align(frame.rgb, eyes(row, table.width, table.height))
                    if crop is None or hashlib.sha256(crop.tobytes()).hexdigest() != row["aligned_rgb_sha256"]:
                        raise ValueError("A recorded facial detail crop differs from the source")
                else:
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
    if transition is not None:
        if transition_shots is not None and (transition_shots.frames_seen != len(table) or transition_shots.boundaries
                                             or set(sparse_pixels) != sparse_indices):
            raise ValueError("The original must reproduce one continuous shot for the face transition")
        if sparse_indices:
            face_single_context.verify(transition, sparse_pixels)
        else:
            edge = transition["boundary"]
            similarity = face_return.scene_lower_quartile(transition_scenes[edge - 1], transition_scenes[edge])
            if similarity is None or not math.isclose(similarity, face_single_boundary_binding.reading(transition, "fine_scene"),
                                                      abs_tol=1e-8, rel_tol=0):
                raise ValueError("The source surroundings differ from the recorded face transition comparison")
    if extended:
        _verify_picture_hold(evidence, table, hold_observations[:hold_frames])
    if sequence is not None and canonical_hash(picture_sequence.scan(hold_observations[:hold_frames], table.tb)) != canonical_hash(sequence):
        raise ValueError("The original video does not reproduce its repeated moving-picture sequences")
    if reverse_sequence is not None and canonical_hash(picture_sequence.scan(hold_observations[:hold_frames], table.tb, reverse=True)) != canonical_hash(reverse_sequence):
        raise ValueError("The original video does not reproduce its reversed moving-picture sequences")
    for record, exact_review, profile in motion_reads:
        if record["schema"] == picture_sequence.MOTION_EXTENDED_SCHEMA:
            opening_profile = picture_motion.profile(table, motion_encoding)
            opening_limit = opening_profile[1]
            opening = picture_motion.scan(hold_observations[:opening_limit], motion_descriptors[:opening_limit],
                                          table.tb, motion_encoding, sequence["pair"], source_profile=opening_profile)
            if opening["status"] != "no_match":
                raise ValueError("The extended motion comparison displaced an opening result")
        limit = profile[1]
        exact_pair = exact_review["pair"] if exact_review is not None else None
        values = crop_frames if record["schema"] in picture_sequence.MOTION_CROP_SCHEMAS else motion_descriptors
        if record["schema"] in {picture_sequence.MOTION_CROP_ALIGNED_SCHEMA, picture_sequence.MOTION_CROP_REVERSE_SCHEMA}:
            for earlier in picture_sequence.MOTION_CROP_FORWARD_PASSES:
                if earlier == record["schema"]:
                    break
                prior = picture_motion.scan(hold_observations[:limit], values[:limit], table.tb, motion_encoding,
                                            exact_pair, source_profile=profile, schema=earlier)
                if prior["status"] != "no_match":
                    raise ValueError("The crop review displaced an earlier crop result")
        if record["schema"] in picture_sequence.MOTION_RETIMED_SCHEMAS:
            for earlier in picture_sequence.MOTION_RETIMED_PASSES:
                if earlier == record["schema"]:
                    break
                prior = picture_motion.scan(hold_observations[:limit], values[:limit], table.tb, motion_encoding,
                                            exact_pair, source_profile=profile, schema=earlier)
                if prior["status"] != "no_match":
                    raise ValueError("The broader retiming comparison displaced an earlier match")
        reproduced = picture_motion.scan(hold_observations[:limit], values[:limit], table.tb, motion_encoding,
                                         exact_pair, source_profile=profile, schema=record["schema"])
        if canonical_hash(reproduced) != canonical_hash(record):
            raise ValueError("The original video does not reproduce its similar moving-picture sequences")
    if local_mouth is not None:
        from app.detectors.video.mouth_local_hold import verify_measurements

        if len(mouth_pictures) != mouth_limit:
            raise ValueError("The source is missing a verified mouth-region input")
        verify_measurements(mouth_pictures, local_mouth["match"], hd=local_mouth["schema"] == mouth_local_hold_binding.HD_SCHEMA)
        if local_mouth.get("sound_context"):
            from app.detectors.audio.decode import VIDEO_AUDIO_MAX_SECONDS, decode_audio
            from app.detectors.video.mouth_hold_sound import source_assets

            decoded_audio = decode_audio(original, source_sha256=digest, max_seconds=VIDEO_AUDIO_MAX_SECONDS)
            packet = next(packet for packet in packets if packet["id"] == mouth_local_hold_binding.PACKET_ID)
            measured = source_assets(decoded_audio.pcm, decoded_audio.mapping, local_mouth, table.time_base,
                time_axis=packet["artifacts"][mouth_hold_sound.WAVEFORM].get("time_axis"))
            for role, content in measured.items():
                if hashlib.sha256(content).hexdigest() != packet["artifacts"][role]["sha256"]:
                    raise ValueError("The sound clip or waveform differs from the supplied soundtrack")
    if eye_match is not None:
        from app.detectors.video.eye_local_hold import verify_measurements as verify_eye_measurements

        verify_eye_measurements(eye_pictures, eye_match)
    if lip_timing.get("inputs"):
        from app.detectors.video.viseme_review import audio_assets
        from app.detectors.video.viseme_stereo import capture
        from app.evidence import video_lip_agreement as agreement
        from app.evidence import video_silent_mouth as silent

        activity = lip_timing["inputs"].get(video_viseme_binding.ACTIVITY_KEY)
        stereo_scope = lip_timing.get("stereo_scope")
        decoded_audio, timing_source = _decode_lip_soundtrack(original, digest, lip_timing, evidence["audio"]["source"])
        if stereo_scope is not None and capture(decoded_audio) != stereo_scope:
            raise ValueError("The original video does not reproduce the timing review's stereo channel measurements")
        limit = video_viseme_binding.sample_limit(lip_timing)
        presented = video_viseme_binding.presented_samples(decoded_audio.mapping, max_samples=limit)
        if activity:
            from app.detectors.video.shots import detect_shots
            from app.detectors.video.viseme_review import activity_eligible
            from app.evidence import video_activity_levels

            with closing(decode_stream(original, table)) as shot_frames:
                shots = detect_shots(shot_frames, table).as_json()
            stereo_pcm = decoded_audio.stereo_pcm
            if (not activity_eligible(table, decoded_audio.mapping, shots["ranges"], shots["frames_scanned"])
                    or stereo_pcm is None or len(stereo_pcm) < presented
                    or hashlib.sha256(stereo_pcm[:presented].astype("<i2", copy=False).tobytes()).hexdigest()
                    != activity["stereo_sha256"]):
                raise ValueError("The source does not reproduce the speech activity review's picture and stereo recording")
            if (video_activity_levels.SQUARES in activity and
                    video_activity_levels.squared_samples(stereo_pcm[:presented]) != activity[video_activity_levels.SQUARES]):
                raise ValueError("The original video does not reproduce the stereo sound levels")
        origin = 0 if activity else video_viseme_binding.audio_origin(decoded_audio.mapping, stereo_scope, max_samples=limit)
        if (decoded_audio.mapping["pcm_sha256"] != lip_timing["pcm_sha256"]
                or presented != lip_timing["inputs"]["samples"] or origin != lip_timing["audio_origin_pts"]):
            raise ValueError("The original video does not reproduce the timing review's soundtrack")
        clock_fields = video_viseme_binding.AAC_PRESENTATION_FIELDS
        if decoded_audio.mapping.get(agreement.AAC_44100_CLOCK):
            clock_fields += video_viseme_binding.AAC_44100_PRESENTATION_FIELDS
        if "source_duration_pts" in decoded_audio.mapping and any(
            decoded_audio.mapping[key] != timing_source.get(key)
            for key in clock_fields
        ):
            raise ValueError("The timing review's AAC presentation clock differs from the original")
        if {agreement.KEY, silent.SHOT_KEY} & lip_timing["inputs"].keys() and (
            decoded_audio.mapping.get("source_codec") != timing_source.get("source_codec")
            or (not default_second and decoded_audio.mapping.get("audio_stream_indices"))
        ):
            raise ValueError("The timing review's mono soundtrack format differs from the original")
        shot_cover = lip_timing.get("shot_coverage")
        motion_aac441 = default_second or lip_timing["schema"] == video_viseme_binding.SCHEMA_WITH_AAC441_MOTION
        if shot_cover or motion_aac441:
            from app.detectors.video.shots import detect_shots

            with closing(decode_stream(original, table)) as shot_frames:
                measured_shots = detect_shots(shot_frames, table).as_json()
            if shot_cover and (measured_shots["frames_scanned"] != len(table)
                               or measured_shots["ranges"] != [row["frames"] for row in shot_cover]):
                raise ValueError("The submitted video does not reproduce the shot review's picture boundaries")
            if motion_aac441:
                from app.detectors.video.viseme_review import supported_aac441_motion

                if not supported_aac441_motion(table, decoded_audio.mapping, measured_shots["ranges"],
                                               measured_shots["frames_scanned"]):
                    raise ValueError("The submitted video does not reproduce the qualified 44.1 kHz mouth and sound scope")
        for packet_id, review in video_viseme_binding.exhibits(lip_timing):
            packet = next(packet for packet in packets if packet["id"] == packet_id)
            for role, content in audio_assets(decoded_audio.pcm, decoded_audio.mapping, review).items():
                if hashlib.sha256(content).hexdigest() != packet["artifacts"][role]["sha256"]:
                    raise ValueError("The timing clip or waveform differs from the supplied soundtrack")
    fine_edges = sum(_verify_scene_support(record, background_frames) for record in fine_records)
    for record in guarded_records:
        _verify_guarded_scene(record, background_frames)
    for record in local_records:
        from app.detectors.video.face_local_detail import shot_changes

        samples = [_scene_sample(row, background_frames) for row in record["scene_frame_inputs"]]
        for pair, pin in zip((samples[:2], samples[2:]), record["scene_correlations_hex"], strict=True):
            observed = face_return.scene_lower_quartile(*pair)
            if observed is None or not math.isclose(observed, float.fromhex(pin), abs_tol=1e-6, rel_tol=0):
                raise ValueError("Local face background support differs from the original video")
        geometry = (face_portrait_return_binding.geometry(record["method"]) if record["method"] in face_portrait_return_binding.METHODS
                    else face_local_return_binding.LANDSCAPE)
        if shot_changes(original, table, record["start_frame_index"], record["end_frame_index"], geometry=geometry):
            raise ValueError("The original video shows a shot change during the local face comparison")
    for record in paired_records:
        _verify_paired_motion(record, {row["frame_index"]:paired_pixels[row["frame_index"]] for row in record["scene_frame_inputs"]})
    if len(proposal_records) != expected_proposals:
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
            + (" Verified the face appearance transition frames, aligned inputs and nearby picture similarity." if transition else "")
            + (f" Recomputed picture-clock coverage for {len(timing['windows'])} timing windows." if timing else "")
            + (f" Recomputed fine background boundaries ({fine_edges})." if fine_edges else "")
            + (" Recomputed 4 guarded background boundaries and verified the source carrier." if len(guarded_records) == 2 else
               " Recomputed both guarded background boundaries and verified the source carrier." if guarded_codec else "")
            + (" Recomputed both paired face boundary motion measurements and verified the source carrier." if paired_records else "")
            + (" Recomputed the bounded face shortlist and both near-head motion boundaries for each reported proposal."
               if second_avc else " Recomputed the bounded face shortlist and both near-head motion boundaries."
               if proposal_records else "")
            + (" Recomputed off-face motion counts on each tracked side at both boundaries of its range." if opposite_pane else
               " Recomputed off-face motion counts on the tracked side at both boundaries of each range."
               if evidence["version"] == SECOND_PANE_TRACK_AVC_PROPOSAL_VERSION else
               " Recomputed off-face motion counts on the tracked side at both boundaries."
               if evidence["version"] == PANE_TRACK_AVC_PROPOSAL_VERSION else "")
            + (" Recomputed each carrier-qualified window shortlist and both native face boundaries." if carrier_windows and window_records else "")
            + (" Recomputed each overlapping window shortlist and both native face boundaries." if windowed and window_records else "")
            + (" Recomputed the surrounding-picture cut context and both native boundaries." if cut_records else "")
            + (f" Replayed every frame in the repeated-picture inspection ({hold_frames})." if hold_frames else "")
            + (" Recomputed the held mouth-region pixel measurements." if local_mouth and local_mouth["match"] else "")
            + (" Recomputed the held eye-region pixel measurements." if eye_match is not None else
               " Verified the eye-region input clock and pictures." if eye_limit else "")
            + (f" Verified the timing review's {'default second ' if default_second else ''}"
               f"soundtrack and {len(timing_frames)} native mouth crops."
               if lip_timing.get("inputs") else "")
            + (" Recomputed the exact moving-sequence comparison." if sequence is not None else "")
            + (" Replayed the motion-aware moving-sequence checks." if motion_records else ""))
