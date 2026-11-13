"""Public description of the brief face-change passes recorded by the worker."""
from __future__ import annotations

from app.core.canonical import canonical_hash

KEY = "brief_face_review"
COMMITMENT = "brief_face_review_sha256"
_STATES = {"not_applicable", "unassessed", "no_distinct_range", "localized"}
_REASONS = {
    "native_scope": "This clip is outside the supported frame or face-size profile.",
    "time_budget": "The check reached its processing time limit.",
    "native_timestamp_unverified": "A sampled picture's source timestamp could not be verified.",
    "multiple_faces": "More than one face was detected in a sampled picture.",
    "native_measurement_unavailable": "A sampled picture could not be measured.",
    "incomplete_native_decode": "Some scheduled source pictures could not be decoded.",
    "insufficient_same_shot_context": "The sampled pictures did not provide enough comparable single-face context.",
    "surrounding_scene_changed": "The surrounding picture changed too much to support the range.",
    "sparse_range_outside_profile": "The initial scan found a possible change. That scan alone cannot support a brief range for this video profile.",
}
_FOLLOWUPS = ("dense_fallback", "fine_scene_fallback")


def _pass(status, reason, scheduled_frames, measured_face_frames, frame_count):
    if (status not in _STATES or type(scheduled_frames) is not int or type(measured_face_frames) is not int
            or not 0 <= measured_face_frames <= scheduled_frames <= frame_count):
        raise ValueError("Invalid brief face-change coverage")
    if status == "not_applicable":
        if reason != "native_scope" or scheduled_frames:
            raise ValueError("Invalid brief face-change scope")
    elif (not scheduled_frames or (status == "localized" and (reason is not None or measured_face_frames < 9))
          or (status == "no_distinct_range" and reason not in {None, "surrounding_scene_changed"})
          or (status == "unassessed" and reason not in _REASONS.keys()-{"native_scope", "surrounding_scene_changed"})):
        raise ValueError("Invalid brief face-change disposition")
    return {"status": status, "reason": reason, "scheduled_frames": scheduled_frames,
            "measured_face_frames": measured_face_frames}


def _from_raw(raw, frame_count):
    return _pass(raw["status"], raw["reason"], raw["scheduled_frames"], raw["measured_face_frames"], frame_count)


def from_worker(raw, frame_count):
    """Keep separate passes so overlapping samples cannot inflate coverage."""
    subpasses = {name: _from_raw(raw[name], frame_count) for name in _FOLLOWUPS if name in raw}
    initial = raw.get("sparse_context", raw)
    passes = {"initial": _from_raw(initial, frame_count), **subpasses}
    return {"passes": passes, "observations": observations(passes), "affects_overall_confidence": False}


def checked(record, frame_count):
    if (not isinstance(record, dict) or set(record) != {"passes", "observations", "affects_overall_confidence"}
            or record["affects_overall_confidence"] is not False
            or not isinstance(record["passes"], dict) or "initial" not in record["passes"]
            or not record["passes"].keys() <= {"initial", *_FOLLOWUPS}):
        raise ValueError("Invalid brief face-change coverage record")
    for measure in record["passes"].values():
        if not isinstance(measure, dict) or set(measure) != {"status", "reason", "scheduled_frames", "measured_face_frames"}:
            raise ValueError("Invalid brief face-change coverage fields")
        _pass(**measure, frame_count=frame_count)
    if len(record["passes"]) > 1 and record["passes"]["initial"]["status"] in {"not_applicable", "localized"}:
        raise ValueError("Brief face-change follow-up has no eligible initial scan")
    if sum(measure["status"] == "localized" for measure in record["passes"].values()) > 1:
        raise ValueError("Brief face-change coverage reports conflicting passes")
    if record["observations"] != observations(record["passes"]):
        raise ValueError("Brief face-change wording differs from its measured coverage")
    return record


def from_result(result, frame_count=None):
    record = ((result.get("manifest") or {}).get("sampling") or {}).get(KEY)
    if record is None:
        return None
    if frame_count is None:
        # Result projection has no video frame table. Portable evidence checks the real limit.
        frame_count = 240
    return checked(record, frame_count)


def validate(document):
    record = (document.get("coverage") or {}).get(KEY)
    if record is not None:
        checked(record, document["source"]["frame_count"])
        passes = any(item["status"] == "localized" for item in record["passes"].values())
        packets = any(packet.get("face_return") for packet in document["packets"])
        if passes != packets:
            raise ValueError("Brief face-change coverage disagrees with the displayed ranges")
    return record


def validate_commitment(document, expected_hash):
    record = validate(document)
    if (canonical_hash(record) if record is not None else None) != expected_hash:
        raise ValueError("Brief face-change coverage differs from the signed public result")


def observations(record):
    lines = []
    labels = {"initial": "Initial scan", "dense_fallback": "Denser scan", "fine_scene_fallback": "Finer background scan"}
    for name in ("initial", *_FOLLOWUPS):
        if name not in record:
            continue
        item = record[name]
        outcome = _REASONS.get(item["reason"])
        if outcome is None:
            outcome = ("A brief face-change range was recorded." if item["status"] == "localized"
                       else "No distinct brief face-change range was found in this scan.")
        sampled = item["scheduled_frames"]
        counts = f" Comparable face inputs: {item['measured_face_frames']} of {sampled} scheduled pictures." if sampled else ""
        lines.append(f"{labels[name]}. {outcome}{counts}")
    return lines
