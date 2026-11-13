"""Source and display bindings for a short picture change and return."""

from fractions import Fraction

SCHEMA = "picture-change-return/1"
WEBM_SCHEMA = "picture-change-return/2"
KEY = "scene_return"
COMMITMENT = "scene_return_sha256"
COVERAGE = "scene_return_review"
COVERAGE_COMMITMENT = "scene_return_review_sha256"
PACKET_ID = "VSC01"
TITLE = "Short picture change and return"
INTERPRETATION = "An ordinary edit can produce this pattern. The observation carries no weight in overall AI confidence."
ROLES = ("before", "during", "after")
FRAME_COUNT = 120
ENCODINGS = frozenset({("hevc", "yuv420p", "hvc1"), ("h264", "yuv420p", "avc1")})
WEBM_ENCODING = ("vp9", "yuv420p", "[0][0][0][0]")
WEBM_TIME_BASE = Fraction(1, 1000)
LIMITATION = "This check looks for a short change followed by a return to a similar picture. It carries no weight in overall AI confidence."
UNSUPPORTED_WITH_WEBM = "This check requires a supported four-second 720p H.264 or HEVC MP4 or VP9 WebM with verified timing at 30 frames per second."
DESCRIPTIONS = {
    "unsupported_video": "This check requires a supported four-second 720p H.264 or HEVC MP4 with verified timing at 30 frames per second.",
    "profile": "The selected analysis profile did not run this check.",
    "provenance": "The analysis stopped at verified provenance before this observation was issued.",
    "scan_unavailable": "This check could not complete during the analysis.",
    "incomplete_decode": "The decoder could not verify all 120 frames in sequence, so the check reached no result.",
    "missing_anchors": "A possible change was measured, but its before, during and after source frames could not all be verified.",
}


def webm_tick(index):
    return (100 * index + 1) // 3


def valid_webm_clock(time_base, pts, first_pts):
    return (Fraction(time_base) == WEBM_TIME_BASE and type(first_pts) is int and len(pts) == FRAME_COUNT
            and all(tick == first_pts + webm_tick(index) for index, tick in enumerate(pts)))


def coverage(status, *, reason=None, frames_scanned=0, first_pts=None, webm_available=False):
    if status == "matched":
        description = "Checked all 120 source frames and recorded a short picture change with three verified source images."
    elif status == "no_match":
        description = "Checked all 120 source frames. No qualifying short picture change followed by a return was found."
    else:
        description = UNSUPPORTED_WITH_WEBM if reason == "unsupported_video" and webm_available else DESCRIPTIONS[reason]
    value = {"status": status, "reason": reason, "frames_scanned": frames_scanned,
             "packet_id": PACKET_ID if status == "matched" else None, "observation": description,
             "limitation": LIMITATION, "affects_overall_confidence": False}
    if first_pts is not None:
        value.update(schema=WEBM_SCHEMA, source_first_pts=first_pts)
    return value


def validate_coverage(document, *, canonical_hash, pts=None):
    source = document["source"]
    digest = source.get(COVERAGE_COMMITMENT)
    value = (document.get("coverage") or {}).get(COVERAGE)
    if digest is None and value is None:
        return
    if not isinstance(value, dict) or canonical_hash(value) != digest:
        raise ValueError("Picture-change coverage differs from its source commitment")
    status, reason, count = value["status"], value["reason"], value["frames_scanned"]
    supported = (source["width"], source["height"], source["frame_count"]) == (1280, 720, FRAME_COUNT)
    table = document["coverage"]
    complete = (status in {"matched", "no_match"} and reason is None) or (status == "unavailable" and reason == "missing_anchors")
    first = value.get("source_first_pts")
    if value.get("schema") == WEBM_SCHEMA and (
        not supported or not (complete or reason == "incomplete_decode") or type(first) is not int
        or Fraction(source["time_base"]) != WEBM_TIME_BASE or (pts is not None and not valid_webm_clock(source["time_base"], pts, first))
    ):
        raise ValueError("Picture-change coverage differs from its native WebM clock")
    packet = next((p.get(KEY) for p in document.get("packets", ()) if p.get("id") == PACKET_ID), None)
    if packet is not None and (packet.get("source_first_pts") != first or
                               (packet["schema"] == WEBM_SCHEMA) != (value.get("schema") == WEBM_SCHEMA)):
        raise ValueError("Picture-change result and coverage record different source clocks")
    if (type(count) is not int or not 0 <= count <= FRAME_COUNT
            or (status == "matched") != (source.get(COMMITMENT) is not None)
            or (complete and count != FRAME_COUNT)
            or (not complete and (status != "unavailable" or reason not in DESCRIPTIONS))
            or (not complete and reason != "incomplete_decode" and count != 0)
            or ((complete or reason == "incomplete_decode")
                and (not supported or table.get("table_truncated") or table.get("interpolated_frame_indices")))
            or value.get("affects_overall_confidence") is not False
            or value != coverage(status, reason=reason, frames_scanned=count, first_pts=first,
                                 webm_available=value.get("observation") == UNSUPPORTED_WITH_WEBM)):
        raise ValueError("Picture-change coverage misrepresents the recorded inspection")


def frames_for(start, end):
    return start - 1, (start + end) // 2, end


def record(found, table, hashes, *, schema=SCHEMA):
    value = {"schema": schema, "start_frame_index": found.start, "end_frame_index": found.stop,
             "start_pts": table.pts[found.start], "end_pts": table.pts[found.stop], "time_base": table.time_base,
             "start_stream_s": table.seconds(found.start), "end_stream_s": table.seconds(found.stop),
             "tiles": list(found.tiles),
             "anchors": [{"frame_index": i, "pts": table.pts[i], "decoded_rgb_sha256": digest}
                         for i, digest in zip(found.anchors, hashes, strict=True)]}
    if schema == WEBM_SCHEMA:
        value["source_first_pts"] = table.pts[0]
    return value


def observation(value):
    return (f"The picture changed and returned between {value['start_stream_s']:.3f} and "
            f"{value['end_stream_s']:.3f} seconds on the source clock. The end is exclusive. "
            f"Stable texture on both sides supported the two boundaries in {len(value['tiles'])} of 60 picture tiles.")


def validate(document, stored=None, *, pts=None, canonical_hash):
    source = document["source"]
    digest = source.get(COMMITMENT)
    packets = [p for p in document["packets"] if p.get("id") == PACKET_ID or KEY in p or p.get("semantics") in (SCHEMA, WEBM_SCHEMA)]
    if digest is None and not packets and stored is None:
        return
    if len(packets) != 1 or not digest:
        raise ValueError("Picture-change evidence omits its source commitment or packet")
    packet, = packets
    value = packet[KEY]
    schema = value["schema"]
    start, end = value["start_frame_index"], value["end_frame_index"]
    tiles = value["tiles"]
    if (schema not in (SCHEMA, WEBM_SCHEMA) or canonical_hash(value) != digest or (stored is not None and value != stored)
            or any(type(i) is not int for i in (start, end)) or not 4 <= start < end < 117 or not 6 <= end - start <= 24
            or any(type(i) is not int or not 0 <= i < 60 for i in tiles) or tiles != sorted(set(tiles)) or len(tiles) < 24
            or (source["width"], source["height"], source["frame_count"]) != (1280, 720, 120)
            or document.get("coverage", {}).get("interpolated_frame_indices")
            or value["time_base"] != source["time_base"]):
        raise ValueError("Picture-change result differs from its native record")
    quadrants = [0, 0, 0, 0]
    for tile in tiles:
        quadrants[2 * (tile // 10 >= 3) + (tile % 10 >= 5)] += 1
    tb = Fraction(value["time_base"])
    a, b = value["start_pts"], value["end_pts"]
    if schema == WEBM_SCHEMA:
        first = value.get("source_first_pts")
        valid_clock = (type(first) is int and type(a) is int and type(b) is int and tb == WEBM_TIME_BASE
                       and a == first + webm_tick(start) and b == first + webm_tick(end)
                       and (pts is None or valid_webm_clock(tb, pts, first)))
    else:
        valid_clock = "source_first_pts" not in value and tb * (b - a) == Fraction(end - start, 30)
    if (min(quadrants) < 4 or not valid_clock
            or abs(float(a * tb) - value["start_stream_s"]) > 0.000001
            or abs(float(b * tb) - value["end_stream_s"]) > 0.000001
            or (pts is not None and (a, b) != (pts[start], pts[end]))):
        raise ValueError("Picture-change tile support or source timing differs")
    if (packet["id"] != PACKET_ID or packet["semantics"] != schema or packet["title"] != TITLE
            or packet["claim_scope"] != "sampled_window" or packet["evidence_type"] != "anomaly"
            or packet["affects_overall_confidence"] is not False or any(packet[k] for k in ("finding_ids", "signal_refs", "exhibit_refs"))
            or packet["observation"] != observation(value) or packet["interpretation"] != INTERPRETATION
            or packet.get("window") is not None or set(packet["artifacts"]) != set(ROLES)):
        raise ValueError("Picture-change observation misrepresents the measurement")
    for role, index, evidence, frame in zip(ROLES, frames_for(start, end), value["anchors"], packet["frames"], strict=True):
        anchor = frame["anchor"]
        time_delta = (webm_tick(index) - webm_tick(start)) * tb if schema == WEBM_SCHEMA else Fraction(index - start, 30)
        if (frame["interval_role"] != role or frame["score"] is not None or frame["is_peak"] is not False
                or frame["context_region"] is not None or frame["artifact_refs"] != {"original": role, "context": role}
                or (packet["artifacts"][role]["width"], packet["artifacts"][role]["height"]) != (1280, 720)
                or set(evidence) != {"frame_index", "pts", "decoded_rgb_sha256"}
                or evidence["frame_index"] != index or any(anchor[k] != v for k, v in evidence.items())
                or (anchor["width"], anchor["height"], anchor["stream_selector"], anchor["coordinate_space"])
                != (source["width"], source["height"], source["stream_selector"], source["coordinate_space"])
                or tb * (evidence["pts"] - a) != time_delta
                or (schema == WEBM_SCHEMA and type(evidence["pts"]) is not int)
                or (pts is not None and evidence["pts"] != pts[index])):
            raise ValueError("Picture-change frame does not match the recorded source position")
