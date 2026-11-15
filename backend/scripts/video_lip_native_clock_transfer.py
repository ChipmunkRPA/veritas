# SPDX-License-Identifier: MIT
"""Test the frozen lip-timing check on original CREMA-D picture clocks."""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import median

from app.evidence import video_viseme_binding as binding
from app.evidence import video_viseme_timing as timing
from scripts import video_lip_viseme_transfer as study

SCHEMA = "chipmunk.video-lip-native-clock-transfer/1"
EDIT_WIDTH = 11_200
SOURCES = 24


def roster(audit):
    sources = []
    for path in sorted(audit.glob("batch-??.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["source_family"] != "recorded_audiovisual_speech" or row["status"] != "decoded":
                continue
            table, audio = row["frame_table"], row["audio_clocks"]
            samples = row["canonical_audio"]["samples"]
            clock = table["pts"]
            if not (audio["contiguous"] and audio["channels"] == 1 and table["time_base"] == "1/1000"
                    and not table["rotation"] and not table["interpolated"] and not table["truncated"]
                    and 24 <= len(clock) <= 120 and 24_000 <= samples <= 64_000
                    and all(b - a in (33, 34) for a, b in zip(clock, clock[1:]))
                    and max(abs(p - clock[0] - Fraction(i * 1001, 30)) for i, p in enumerate(clock)) <= Fraction(3, 2)):
                continue
            sources.append(row)
    ordered = sorted(sources, key=lambda row: hashlib.sha256(f"{SCHEMA}/{row['group']}".encode()).digest())
    study.phone.require(len({row["group"] for row in ordered}) == len(ordered) >= SOURCES,
                        "Need distinct qualified source actors")
    return ordered[:SOURCES]


def run(args):
    sys.path.insert(0, str(args.runtime.resolve()))

    import numpy as np

    from app.detectors.video.decode import FrameTable, decode_stream, probe_frames
    from app.evidence.video_reference_timing import _landmarker, _mesh, available
    from scripts.video_lip_frame_clock_transfer import UNRESOLVED

    rows = roster(args.audit)
    study.phone.require(available(args.geometry), "Expected the pinned facial geometry assets")
    acoustic, device = study.decoder(args.checkpoint)
    counts, details = defaultdict(Counter), []
    for row in rows:
        index = row["index"]
        source = args.media / f"asset-{index:03d}.flv"
        assert study.phone.sha(source) == row["input_sha256"], "Publisher file changed"
        table = probe_frames(source, cap=121, timeout_s=20)
        assert table == FrameTable.from_json(row["frame_table"]), "Publisher picture clock changed"
        decoded = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1",
                                  "-protocol_whitelist", "file", "-i", f"file:{source.resolve()}", "-map", "0:a:0",
                                  "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(timing.RATE), "-f", "s16le", "pipe:1"],
                                 capture_output=True, check=True, timeout=20)
        assert not decoded.stderr.strip(), "Publisher audio did not decode cleanly"
        original = np.frombuffer(decoded.stdout, dtype="<i2")
        samples = len(original)
        assert {"samples": samples, "pcm_sha256": study.phone.digest(original.tobytes())} == {
            key: row["canonical_audio"][key] for key in ("samples", "pcm_sha256")}, "Publisher PCM changed"
        # The native audio audit checks continuity at the FLV clock's millisecond precision.
        # Preserve its first timestamp. Decoded sample indexing supplies the continuous analysis clock.
        origin = round(Fraction(row["audio_clocks"]["frames"][0]["start"]) * timing.RATE)
        frames = []
        with _landmarker(args.geometry) as (mp, task):
            for frame in decode_stream(source, table, max_side=None, timeout_s=20):
                assert frame.index == len(frames) and frame.pts_verified and frame.pts == table.pts[frame.index]
                record = {"index": frame.index, "pts": frame.pts, "gap_units": None}
                if observed := _mesh(task, mp, frame.rgb):
                    features, box = observed
                    x, y, right, bottom = box
                    record.update(gap_units=round(float(median(features[:3])) * timing.SCORE_UNITS), mouth_box=box,
                                  rgb_sha256=study.phone.digest(frame.rgb.tobytes()),
                                  mouth_crop_sha256=study.phone.digest(frame.rgb[y:bottom, x:right].tobytes()))
                frames.append(record)
        assert len(frames) == len(table), "Incomplete source picture scan"
        start = (samples - EDIT_WIDTH) // 2
        edit = [start, start + EDIT_WIDTH]
        results = {}
        for role, pcm in study.parent.treatments(original, edit).items():
            offset = study.parent.EDITS.get(role)
            kind = "local" if offset else "constant" if role.startswith("global_") else "ordinary"
            digest = study.phone.digest(pcm.tobytes())
            mapping = {"samples": samples, "source_sample_rate": timing.RATE, "source_channels": 1,
                       "timeline_time_base": "1/16000", "pcm_sha256": digest,
                       "timeline": [{"start_sample": 0, "end_sample": samples, "source_pts": origin}]}
            measured = binding.measured({"frames": frames, "phones": acoustic(pcm)}, timestamps=table.pts,
                                        time_base=table.time_base, source={"width": table.width, "height": table.height},
                                        mapping=mapping)
            hit = measured["witness"]
            useful = bool(offset and hit and study.parent.useful(
                [value - origin for value in hit["audio_source_pts"]], hit["change_samples"], edit, offset))
            counts[kind].update(files=1, flagged=bool(hit), useful=useful, unresolved=measured["status"] in UNRESOLVED)
            results[role] = {"pcm_sha256": digest, "status": measured["status"], "useful": useful, "witness": hit}
        details.append({"source": index, "sha256": row["input_sha256"], "samples": samples, "audio_origin": origin,
                        "picture_count": len(frames), "edit": edit, "results": results})
        print(f"Checked native-clock recording {len(details)}/{len(rows)}", flush=True)
    args.output.write_text(json.dumps({"schema": SCHEMA, "point": binding.POINT, "reference": binding.REFERENCE,
                                       "source_count": len(details), "device": device, "counts": counts,
                                       "rows": details}, sort_keys=True, indent=2) + "\n")
    print(json.dumps(counts, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "media", "checkpoint", "geometry", "runtime", "output"):
        parser.add_argument(name, type=Path)
    run(parser.parse_args())
