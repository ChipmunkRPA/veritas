"""Replay exact reversed picture matches on generated or retained camera footage."""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from app.detectors.video import picture_hold
from app.detectors.video.decode import probe_frames
from app.evidence import picture_sequence
from scripts.video_exact_sequence_transfer import (
    ENCODINGS,
    FPS,
    FRAMES,
    HEIGHT,
    WIDTH,
    deliver,
    read_source,
    sha,
)
from scripts.video_extended_picture_hold import _frame_hashes
from scripts.video_lossy_sequence_transfer import geometry, locate

CASES = ("original", "ordinary_cut", "static_hold", "exact_copy", "reverse", "reverse_spacing", "reverse_lossy")


def _deliver(raw, path, kind, codec, first, second, count):
    if not kind.startswith("reverse"):
        deliver(raw, path, kind, codec)
        return
    size = WIDTH * HEIGHT * 3
    pixels = memoryview(raw)
    frames = [pixels[i * size:(i + 1) * size] for i in range(FRAMES)]
    frames[second:second + count] = frames[first:first + count][::-1]
    clock = "settb=expr=1/1000,setpts=N*50"
    if kind == "reverse_spacing":
        clock += rf"+15*mod(N\,2)*between(N\,{second}\,{second + count - 1})"
    command = ["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size",
               f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS), "-i", "pipe:0", "-vf", clock, "-fps_mode",
               "passthrough", "-enc_time_base", "1:1000", *ENCODINGS["lossy" if kind == "reverse_lossy" else codec], str(path)]
    subprocess.run(command, input=b"".join(frames), check=True, timeout=30)


def _oracle(pts, hashes, tb):
    """Enumerate matching forward-start and backward-start positions over external hashes."""
    locations = defaultdict(list)
    for index, digest in enumerate(hashes):
        locations[digest].append(index)
    best = None
    for start, digest in enumerate(hashes):
        for last in locations[digest]:
            if last - start < 15:
                continue
            left, right, distinct = start, last, {digest}
            while left + 1 < right - 1 and hashes[left + 1] == hashes[right - 1]:
                step = pts[left + 1] - pts[left]
                if not (0 < step * tb <= Fraction(1, 8) and step == pts[right] - pts[right - 1]):
                    break
                left += 1
                right -= 1
                distinct.add(hashes[left])
            elapsed = pts[left] - pts[start]
            if len(distinct) >= 8 and elapsed * tb >= Fraction(1, 2):
                rank = (elapsed, -start, -right)
                if best is None or rank > best[0]:
                    best = rank, [start, left, right, last]
    return best[1] if best is not None else None


def _bounds(record):
    pair = record.get("pair")
    return ([pair[side][f"{end}_frame_index"] for side in ("first", "second") for end in ("start", "end")]
            if pair is not None else None)


def run(output, archive):
    output.mkdir(mode=0o700)
    results, sources = [], []
    for slot in range(1, 9) if archive else (0,):
        source, digest, _ = locate(archive, "held", slot) if archive else (None, None, None)
        raw = read_source(source)
        first, second, count = geometry("held", slot)
        codec = "h264" if slot % 2 == 0 else "vp9"
        sources.append({"slot": slot, "source_sha256": digest, "raster_sha256": sha(raw), "lossless_encoder": codec})
        authored = [first, first + count - 1, second, second + count - 1]
        for kind in CASES:
            path = output / f"{slot}-{kind}.mkv"
            _deliver(raw, path, kind, codec, first, second, count)
            table = probe_frames(path)
            cover = picture_hold.inspect(path, table, origin=0., sequence_review=True)["coverage"]
            if cover["frames_scanned"] != FRAMES:
                raise ValueError("The native scan did not finish the test clip")
            observed = _bounds(cover[picture_sequence.REVERSE_KEY])
            independent = _oracle(table.pts, _frame_hashes(path, table), table.tb)
            if observed != independent:
                raise ValueError(f"The independent search disagrees for group {slot} {kind}")
            results.append({"slot": slot, "variant": kind, "checked_frames": cover["frames_scanned"],
                            "reverse_authored": authored if kind.startswith("reverse") else None,
                            "reverse_found": observed, "forward_found": _bounds(cover["sequence_review"])})
            path.unlink()
        print(f"Replayed seven deliveries for source {slot}", flush=True)
    summary = {"schema": "chipmunk.video-reverse-sequence-transfer/1", "sources": sources, "outcomes": results,
               "independent_frame_hashes": sum(row["checked_frames"] for row in results),
               "reverse_matches": dict(Counter(row["variant"] for row in results if row["reverse_found"] is not None)),
               "forward_matches": dict(Counter(row["variant"] for row in results if row["forward_found"] is not None))}
    (output / "aggregate.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Reversed matches across {len(results)} deliveries: {summary['reverse_matches']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--archive", type=Path, help="Directory containing the qualified VCD2 v287 archive")
    args = parser.parse_args()
    run(args.output, args.archive)
