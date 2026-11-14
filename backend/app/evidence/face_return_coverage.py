"""Public description of the brief face-change passes recorded by the worker."""
from __future__ import annotations

from app.core.canonical import canonical_hash
from app.evidence import face_short_carrier_binding as carriers
from app.evidence import face_short_proposal_binding as proposal
from app.evidence import face_short_window_binding as windows

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
BOUNDED = "bounded_proposal"
SECOND_AVC = "second_avc"


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


def from_worker(raw, frame_count, bounded=None, second_avc=None):
    """Keep separate passes so overlapping samples cannot inflate coverage."""
    passes = {}
    if raw is not None:
        passes = {"initial": _from_raw(raw.get("sparse_context", raw), frame_count),
                  **{name: _from_raw(raw[name], frame_count) for name in _FOLLOWUPS if name in raw}}
    if bounded is not None:
        passes[BOUNDED] = bounded
    if second_avc is not None:
        passes[SECOND_AVC] = second_avc
    return checked({"passes": passes, "observations": observations(passes), "affects_overall_confidence": False}, frame_count)


def _bounded(item, frame_count):
    if not isinstance(item, dict) or set(item) != {"status", "reason", "pixel_frames", "scheduled_frames", "measured_face_frames"}:
        raise ValueError("Invalid bounded face proposal coverage fields")
    status, reason = item["status"], item["reason"]
    pixels, scheduled, measured = (item[key] for key in ("pixel_frames", "scheduled_frames", "measured_face_frames"))
    if (frame_count not in {120, 240} or any(type(value) is not int for value in (pixels, scheduled, measured))
            or not 0 <= pixels <= 120 or not 0 <= measured <= scheduled <= proposal.MAX_FACE_READS
            or (status in {"localized", "no_distinct_range"} and (reason is not None or pixels != 120 or scheduled < 12 or measured < 4))
            or (status == "localized" and measured < 11)
            or (status == "unassessed" and reason not in {"time_budget", "native_measurement_unavailable",
                "insufficient_same_shot_context", "native_timestamp_unverified", "incomplete_native_decode"})
            or status not in {"localized", "no_distinct_range", "unassessed"}):
        raise ValueError("Invalid bounded face proposal disposition")


def checked(record, frame_count):
    if (not isinstance(record, dict) or set(record) != {"passes", "observations", "affects_overall_confidence"}
            or record["affects_overall_confidence"] is not False
            or not isinstance(record["passes"], dict) or not record["passes"].keys() & {"initial", BOUNDED}
            or not record["passes"].keys() <= {"initial", *_FOLLOWUPS, BOUNDED, SECOND_AVC}):
        raise ValueError("Invalid brief face-change coverage record")
    legacy = {key: value for key, value in record["passes"].items() if key not in {BOUNDED, SECOND_AVC}}
    if BOUNDED in record["passes"]:
        _bounded(record["passes"][BOUNDED], frame_count)
    for measure in legacy.values():
        if not isinstance(measure, dict) or set(measure) != {"status", "reason", "scheduled_frames", "measured_face_frames"}:
            raise ValueError("Invalid brief face-change coverage fields")
        _pass(**measure, frame_count=frame_count)
    if legacy and ("initial" not in legacy or (len(legacy) > 1 and legacy["initial"]["status"] in {"not_applicable", "localized"})):
        raise ValueError("Brief face-change follow-up has no eligible initial scan")
    first_ranges = sum(measure["status"] == "localized" for key, measure in record["passes"].items() if key != SECOND_AVC)
    if first_ranges > 1:
        raise ValueError("Brief face-change coverage reports conflicting passes")
    if SECOND_AVC in record["passes"]:
        _bounded(record["passes"][SECOND_AVC], frame_count)
        if frame_count != 120 or first_ranges != 1:
            raise ValueError("Second AVC face proposal needs a recorded first range")
    if record["observations"] != observations(record["passes"]):
        raise ValueError("Brief face-change wording differs from its measured coverage")
    return record


def from_result(result, frame_count=None):
    record = ((result.get("manifest") or {}).get("sampling") or {}).get(KEY)
    if record is None:
        return None
    if frame_count is None:
        # Result projection has no video frame table. Portable evidence checks the real limit.
        frame_count = 120 if SECOND_AVC in record.get("passes", {}) else 240
    return checked(record, frame_count)


def validate(document):
    record = (document.get("coverage") or {}).get(KEY)
    returns = [packet["face_return"] for packet in document["packets"]
               if packet.get("face_return") and packet["face_return"]["method"] not in {windows.METHOD,carriers.METHOD}]
    proposed = [item for item in returns if item["method"] in proposal.METHODS - {proposal.SECOND_AVC_METHOD}]
    second = [item for item in returns if item["method"] == proposal.SECOND_AVC_METHOD]
    if proposed and (record is None or BOUNDED not in record["passes"]):
        raise ValueError("Brief face proposal is missing its measured coverage")
    if second and (record is None or SECOND_AVC not in record["passes"]):
        raise ValueError("Second AVC face proposal is missing its measured coverage")
    if record is not None:
        checked(record, document["source"]["frame_count"])
        passes = any(item["status"] == "localized" for key, item in record["passes"].items()
                     if key not in {BOUNDED, SECOND_AVC})
        legacy = any(item.get("method") not in proposal.METHODS for item in returns)
        bounded = record["passes"].get(BOUNDED) or {}
        if passes != legacy or (bounded.get("status") == "localized") != bool(proposed):
            raise ValueError("Brief face-change coverage disagrees with the displayed ranges")
        followup = record["passes"].get(SECOND_AVC) or {}
        if (followup.get("status") == "localized") != bool(second):
            raise ValueError("Second AVC face coverage disagrees with the displayed ranges")
        for items, coverage in ((proposed, bounded), (second, followup)):
            if items and (len(items) != 1 or coverage["pixel_frames"] != len(items[0]["scene_frame_inputs"])
                          or coverage["scheduled_frames"] != len(items[0]["scheduled_frame_indices"])
                          or coverage["measured_face_frames"] != len(items[0]["frame_inputs"])):
                raise ValueError("Brief face proposal coverage differs from its measured source inputs")
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
    if item := record.get(BOUNDED):
        outcome = _REASONS.get(item["reason"])
        if outcome is None:
            outcome = ("A brief face-change range was recorded." if item["status"] == "localized"
                       else "No distinct brief face-change range was found in this scan.")
        lines.append(f"Bounded native scan. {outcome} Motion inputs: {item['pixel_frames']} of 120 pictures. "
                     f"Comparable face inputs: {item['measured_face_frames']} of {item['scheduled_frames']} scheduled pictures.")
    if item := record.get(SECOND_AVC):
        outcome = _REASONS.get(item["reason"]) or ("A separate range was recorded." if item["status"] == "localized"
                                                  else "No separate range met the measurement rules.")
        lines.append(f"Second brief face-change scan. {outcome} Motion inputs: {item['pixel_frames']} of 120 pictures. "
                     f"Comparable face inputs: {item['measured_face_frames']} of {item['scheduled_frames']} scheduled pictures.")
    return lines
