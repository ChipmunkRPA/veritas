"""Standard-library validation for frozen mixed-track sound categories."""
import math
import re

def validate(context, source):
    try:
        _validate(context, source)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("Sound categories contain missing fields or invalid types") from error


def _validate(context, source):
    if (context.get("schema") != "chipmunk.sound-events/1" or context.get("status") != "ready"
            or context.get("source") != {k: source[k] for k in ("sha256", "pcm_sha256", "sample_rate", "samples")}
            or source["sample_rate"] != 16000 or not 0 < source["samples"] <= 600 * 16000):
        raise ValueError("Sound context does not match its parent source")
    if not all(re.fullmatch(r"[a-f0-9]{64}", str(source[k])) for k in ("sha256", "pcm_sha256")):
        raise ValueError("Invalid sound-source commitment")
    if context["method"]["window_samples"] != 160000 or context["method"]["hop_samples"] != 160000:
        raise ValueError("Unsupported sound-window sampling")
    previous = -1
    if not isinstance(context.get("windows"), list) or len(context["windows"]) > 60:
        raise ValueError("Invalid sound-window count")
    for i, window in enumerate(context["windows"], 1):
        a, b = window["start_sample"], window["end_sample"]
        if (not isinstance(a, int) or isinstance(a, bool) or not isinstance(b, int) or isinstance(b, bool)
                or not 0 <= a < b <= source["samples"] or b-a > 160000 or a < previous
                or window["id"] != f"C{i:02d}"
                or window["start_s"] not in (a/16000, round(a/16000, 6))
                or window["end_s"] not in (b/16000, round(b/16000, 6))):
            raise ValueError("Invalid sound-window source positions")
        previous = b
        expected_ranges = [{"start_sample": max(a, row["start_sample"]), "end_sample": min(b, row["end_sample"]),
            "source_start_pts": row["source_pts"] + max(a, row["start_sample"]) - row["start_sample"],
            "source_end_pts": row["source_pts"] + min(b, row["end_sample"]) - row["start_sample"], "time_base": "1/16000"}
            for row in source["timeline"] if max(a, row["start_sample"]) < min(b, row["end_sample"])]
        if source["timeline_time_base"] != "1/16000" or window["source_timestamp_ranges"] != expected_ranges:
            raise ValueError("Sound-window recording timestamps differ")
        if not re.fullmatch(r"[a-f0-9]{64}", str(window["raw_output_sha256"])):
            raise ValueError("Sound window has no valid raw-check commitment")
        if not 1 <= len(window["labels"]) <= 3:
            raise ValueError("Invalid sound label count")
        for label in window["labels"]:
            if (not isinstance(label["label"], str) or not label["label"] or len(label["label"]) > 160
                    or not isinstance(label["score"], (int, float)) or isinstance(label["score"], bool)
                    or not math.isfinite(label["score"]) or not 0 <= label["score"] <= 1):
                raise ValueError("Invalid sound label")
        if set(window["artifacts"]) != {"selected_clip", "waveform"}:
            raise ValueError("Missing source-bound sound assets")
        clip = window["artifacts"]["selected_clip"]
        if clip["media_type"] != "audio/wav" or clip["samples"] != b-a or clip["sample_rate"] != 16000 or clip["channels"] != 1:
            raise ValueError("Sound clip differs from its source window")
        picture = window["artifacts"]["waveform"]
        if picture["media_type"] != "image/png" or picture["width"] != 960 or picture["height"] != 280:
            raise ValueError("Invalid sound waveform dimensions")
        if not all(re.fullmatch(r"[a-f0-9]{64}", str(asset["sha256"])) for asset in (picture, clip)):
            raise ValueError("Invalid sound-asset commitment")

