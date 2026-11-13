"""Source and display bindings for a short picture change and return."""

from fractions import Fraction

SCHEMA = "picture-change-return/1"
KEY = "scene_return"
COMMITMENT = "scene_return_sha256"
PACKET_ID = "VSC01"
TITLE = "Short picture change and return"
INTERPRETATION = "An ordinary edit can produce this pattern. The observation carries no weight in overall AI confidence."
ROLES = ("before", "during", "after")


def frames_for(start, end):
    return start - 1, (start + end) // 2, end


def record(found, table, hashes):
    return {"schema": SCHEMA, "start_frame_index": found.start, "end_frame_index": found.stop,
            "start_pts": table.pts[found.start], "end_pts": table.pts[found.stop], "time_base": table.time_base,
            "start_stream_s": table.seconds(found.start), "end_stream_s": table.seconds(found.stop),
            "tiles": list(found.tiles),
            "anchors": [{"frame_index": i, "pts": table.pts[i], "decoded_rgb_sha256": digest}
                        for i, digest in zip(found.anchors, hashes, strict=True)]}


def observation(value):
    return (f"The picture changed and returned between {value['start_stream_s']:.3f} and "
            f"{value['end_stream_s']:.3f} seconds on the source clock. The end is exclusive. "
            f"Stable texture on both sides supported the two boundaries in {len(value['tiles'])} of 60 picture tiles.")


def validate(document, stored=None, *, pts=None, canonical_hash):
    source = document["source"]
    digest = source.get(COMMITMENT)
    packets = [p for p in document["packets"] if p.get("id") == PACKET_ID or KEY in p or p.get("semantics") == SCHEMA]
    if digest is None and not packets and stored is None:
        return
    if len(packets) != 1 or not digest:
        raise ValueError("Picture-change evidence omits its source commitment or packet")
    packet, = packets
    value = packet[KEY]
    start, end = value["start_frame_index"], value["end_frame_index"]
    tiles = value["tiles"]
    if (value["schema"] != SCHEMA or canonical_hash(value) != digest or (stored is not None and value != stored)
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
    if (min(quadrants) < 4 or tb * (b - a) != Fraction(end - start, 30)
            or abs(float(a * tb) - value["start_stream_s"]) > 0.000001
            or abs(float(b * tb) - value["end_stream_s"]) > 0.000001
            or (pts is not None and (a, b) != (pts[start], pts[end]))):
        raise ValueError("Picture-change tile support or source timing differs")
    if (packet["id"] != PACKET_ID or packet["semantics"] != SCHEMA or packet["title"] != TITLE
            or packet["claim_scope"] != "sampled_window" or packet["evidence_type"] != "anomaly"
            or packet["affects_overall_confidence"] is not False or any(packet[k] for k in ("finding_ids", "signal_refs", "exhibit_refs"))
            or packet["observation"] != observation(value) or packet["interpretation"] != INTERPRETATION
            or packet.get("window") is not None or set(packet["artifacts"]) != set(ROLES)):
        raise ValueError("Picture-change observation misrepresents the measurement")
    for role, index, evidence, frame in zip(ROLES, frames_for(start, end), value["anchors"], packet["frames"], strict=True):
        anchor = frame["anchor"]
        if (frame["interval_role"] != role or frame["score"] is not None or frame["is_peak"] is not False
                or frame["context_region"] is not None or frame["artifact_refs"] != {"original": role, "context": role}
                or (packet["artifacts"][role]["width"], packet["artifacts"][role]["height"]) != (1280, 720)
                or set(evidence) != {"frame_index", "pts", "decoded_rgb_sha256"}
                or evidence["frame_index"] != index or any(anchor[k] != v for k, v in evidence.items())
                or (anchor["width"], anchor["height"], anchor["stream_selector"], anchor["coordinate_space"])
                != (source["width"], source["height"], source["stream_selector"], source["coordinate_space"])
                or tb * (evidence["pts"] - a) != Fraction(index - start, 30)
                or (pts is not None and evidence["pts"] != pts[index])):
            raise ValueError("Picture-change frame does not match the recorded source position")
