"""Read coordinates for a reviewer from a pinned, source-matched evidence package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from app.evidence import audio_brief_voice_binding, document_native_locations, face_matched_detail_binding
from app.evidence import image_printed_totals_binding as printed_totals
from app.evidence.source_geometry import pixel_bounds
from app.reports import public_verify
from app.reports.verify import verify_package

SCHEMA = "chipmunk.independent-app-positions/1"
EVIDENCE_LIMIT = 64 * 1024 * 1024
TEXT_LOCATOR_FIELDS = (
    "kind", "body_path", "paragraph_index", "part_kind", "part_index", "part_paragraph_index",
    "note_id", "note_paragraph_index", "reference_body_index", "row", "column", "mime_part",
)


def _positive_extent(*sizes: object) -> list[int]:
    if any(type(size) is not int or size <= 0 for size in sizes):
        raise ValueError("The package has an invalid coordinate extent")
    return list(sizes)


def _span(start, stop, extent: int) -> list[int]:
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= extent:
        raise ValueError("A recorded coordinate is outside its source extent")
    return [start, stop]


def _source_commits(path: Path, commitment: dict) -> bool:
    extent = commitment.get("bytes")
    with path.open("rb") as original:
        original.seek(0, 2)
        if type(extent) is not int or original.tell() != extent:
            return False
        original.seek(0)
        return hashlib.file_digest(original, "sha256").hexdigest() == commitment["sha256"]


def _printed_amount_positions(evidence: dict) -> list[dict]:
    comparison = evidence.get(printed_totals.KEY)
    if not comparison:
        return []
    fields = comparison["record"]["fields"]
    return [{"id": printed_totals.PACKET_ID, "affects_overall_confidence": False,
             "amount_boxes": {role: fields[role]["amount_region"] for role in printed_totals.ROLES},
             "row_crop_boxes": {role: fields[role]["source_region"] for role in printed_totals.ROLES}}]


def _brief_speech_positions(evidence: dict, samples: int) -> list[dict]:
    review = evidence.get(audio_brief_voice_binding.KEY) or {}
    return [
        {"id": window["id"], "affects_overall_confidence": False,
         "windows": [_span(window["start_sample"], window["end_sample"], samples)],
         "source_timestamp_ranges": window["source_timestamp_ranges"]}
        for window in review.get("windows", [])
    ]


def _document_packet_positions(packet: dict, native: list[dict], chars: int) -> dict:
    anchors = packet["anchors"]
    row = {"id": packet["id"], "selected_spans": [
        _span(anchor["selected_start"], anchor["selected_end"], chars) for anchor in anchors
    ]}
    if native:
        locations = []
        for location in native:
            anchor = anchors[location["anchor_index"]]
            locator = {field: anchor[field] for field in TEXT_LOCATOR_FIELDS if field in anchor}
            locations.append({**location, "source_locator": locator} if locator else location)
        row["native_text_locations"] = locations
    return row


def _positions(kind: str, evidence: dict) -> tuple[dict, list[dict]]:
    source = evidence["source"]
    packets = evidence.get("packets") or []
    if kind == "image":
        extent = _positive_extent(source["width"], source["height"])
        return ({"basis": "display-oriented bounding boxes", "unit": "pixels", "extent": extent},
                [{"id": packet["id"], "boxes": [list(pixel_bounds(packet["region"], *extent))]}
                 for packet in packets] + _printed_amount_positions(evidence))
    if kind == "audio":
        extent = _positive_extent(source["samples"])
        [rate] = _positive_extent(source["sample_rate"])
        return ({"basis": "analysed mono PCM", "unit": "samples", "extent": extent, "sample_rate": rate,
                 "pcm_sha256": source["pcm_sha256"]},
                [{"id": packet["id"], "windows": [_span(interval["start_sample"], interval["end_sample"], extent[0])]}
                 for packet in packets
                 if (interval := packet.get("interval")) and packet.get("anchor", {}).get("pcm_sha256") == source["pcm_sha256"]]
                + _brief_speech_positions(evidence, extent[0]))
    if kind == "document":
        extent = _positive_extent(source["chars"])
        if source.get("offset_unit") != "unicode_codepoint":
            raise ValueError("The package does not declare Unicode passage coordinates")
        coverage = evidence.get("coverage") or {}
        native = (coverage.get(document_native_locations.KEY) or {}).get("packets", {})
        return ({"basis": "extracted document text", "unit": "unicode_codepoints", "extent": extent,
                 "extracted_text_sha256": source["extracted_text_sha256"]},
                [_document_packet_positions(packet, native.get(packet["id"], []), extent[0])
                 for packet in packets if packet.get("anchors")])
    extent = _positive_extent(source["frame_count"])
    return ({"basis": "native frame indices", "unit": "frames", "extent": extent, "time_base": source["time_base"],
             "frame_table_sha256": source["frame_table_sha256"]},
            [{"id": packet["id"], **_video(packet, extent[0])} for packet in packets])


def _video(packet: dict, frame_count: int) -> dict:
    row = {}
    if frames := packet.get("frames"):
        row["witness_frames"] = sorted({frame["anchor"]["frame_index"] for frame in frames})
        for index in row["witness_frames"]:
            _span(index, index + 1, frame_count)
    if record := packet.get("face_return"):
        row["face_change_span"] = _span(record["start_frame_index"], record["end_frame_index"], frame_count)
    if transition := packet.get("face_single_boundary"):
        boundary = transition["boundary"]
        _span(boundary, boundary + 1, frame_count)
        row["face_transition_index"] = boundary
    if match := packet.get(face_matched_detail_binding.KEY):
        row.update(
            face_change_span=_span(match["start_frame_index"], match["end_frame_index"], frame_count),
            source_pts={"start": match["start_pts"], "end": match["end_pts"], "time_base": match["time_base"]},
            witness_roles={frame["interval_role"]: frame["anchor"]["frame_index"] for frame in packet["frames"]},
            affects_overall_confidence=False,
        )
    return row


def packet_positions(package: Path, original: Path, public_key: Path) -> dict:
    """Verify authenticity and exact original bytes before reading any package positions."""
    verified = verify_package(package, public_key_pem=public_key.read_bytes())
    if not verified.ok:
        raise ValueError("The package did not verify with the supplied public key")
    bundle = verified.bundle
    if bundle["schema"] != public_verify.SCHEMA or not bundle.get("evidence"):
        raise ValueError("Choose a public evidence package with source evidence")
    if not _source_commits(original, bundle["source_commitment"]):
        raise ValueError("The supplied original differs from the signed source commitment")
    reference = bundle["evidence"]
    # Use the signed byte hash from the verified bundle even if the ZIP path
    # changes between verifying its signature and reading its evidence.
    with ZipFile(package) as archive, archive.open("evidence.json") as entry:
        data = entry.read(EVIDENCE_LIMIT + 1)
    if len(data) > EVIDENCE_LIMIT or hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise ValueError("Evidence is oversized or differs from the signed evidence commitment")
    kind = bundle["analysis"]["kind"]
    coordinates, packets = _positions(kind, json.loads(data))
    return {"schema": SCHEMA, "kind": kind, "input_sha256": bundle["source_commitment"]["sha256"],
            "evidence_sha256": reference["sha256"], "coordinates": coordinates,
            "packets": [row for row in packets if len(row) > 1]}
