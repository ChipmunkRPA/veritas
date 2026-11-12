"""Test native picture and signed-motion agreement across lossy video copies."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence import picture_sequence
from scripts.video_exact_sequence_transfer import FPS, FRAMES, HEIGHT, WIDTH, read_source

SCHEMA = "chipmunk.video-lossy-sequence-transfer/1"
DEVELOPMENT = (1, 2, 4, 7, 8, 11)
HELD = tuple(range(1, 9))
CODECS = {
    "h264": ("-c:v", "libx264", "-crf", "23", "-g", "37", "-preset", "fast"),
    "vp9": ("-c:v", "libvpx-vp9", "-crf", "31", "-b:v", "0", "-g", "49", "-deadline", "realtime", "-cpu-used", "5"),
}
CASES = ("copy", "original", "cut", "hold", "reverse", "spacing")
RULES = tuple(itertools.product((1.0, 1.6, 2.4), (.4, .65, 1.), (.65, .8, .9)))
METHOD = {
    "descriptor": "Full-frame RGB reduced to 40x30 block means with integer luma coefficients 77,150,29. Equal-spacing native paired ranges, frame RMS in gray levels, centered signed adjacent differences, normalized motion cosine, weaker/stronger RMS ratio >= .55.",
    "eligibility": "Nonoverlapping intervals, at least eight matching moving edges, maximum three-edge step between supported edges, span >= .5 s, at least half the transitions must support motion. Unverified clocks, different frame spacing, gaps > 125 ms and dissimilar frame pairs separate runs.",
    "selection": "Prefer zero development control flags. Maximize useful groups, then useful copies, then lower control flags, then mean useful IoU. Break ties toward smaller frame RMS, higher activity and cosine requirements. Never fit on held. Prefer longest matching span, then earliest positions.",
    "gate": "At least four additional useful held copies on different source groups in each codec, no lost exact result, at most one additional ordinary flag in each codec and zero on hold or different spacing. A useful pair overlaps each authored range at least .50 IoU.",
}


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def write(path, data):
    with path.open("x") as file:
        file.write(json.dumps(data, sort_keys=True, indent=2) + "\n")


def locate(archive, role, slot):
    if role == "development":
        folder, filename = "video-vcd2-delivery-v206", "unchanged.mp4"
    else:
        folder, filename = "video-vcd2-unseen-controls-v287", "source.mkv"
    base = archive / folder / f"slot-{slot:02d}-attempt-01"
    path = base / filename
    qualified = load(base / "qualification.json")
    if role == "development":
        media, source = qualified["cases"]["unchanged"]["sha256"], qualified["source_sha256"]
    else:
        media = source = qualified["sha256"]
    if hash_file(path) != media:
        raise ValueError("Retained video differs from its qualified commitment")
    return path, source, media


def geometry(role, slot):
    if role == "development":
        return 20, 130, 22
    return 17 + slot, 114 + 2 * slot, (19, 23, 29)[slot % 3]


def deliver(raw, path, first, second, count, kind, codec):
    size = WIDTH * HEIGHT * 3
    if len(raw) != size * FRAMES:
        raise ValueError("Incomplete nine-second raster")
    src = memoryview(raw)
    frames = [src[size * i:size * (i + 1)] for i in range(FRAMES)]
    a, b = first, second
    if kind in {"copy", "spacing", "reverse"}:
        segment = frames[a:a + count]
        frames[b:b + count] = segment[::-1] if kind == "reverse" else segment
    elif kind == "hold":
        frames[b:b + count] = [frames[b]] * count
    elif kind == "cut":
        raster = np.frombuffer(raw, dtype="u1").reshape(FRAMES, HEIGHT, WIDTH, 3)
        frames[b:b + count] = [frame[:, ::-1].tobytes() for frame in raster[b:b + count]]
    clock = "settb=expr=1/1000,setpts=N*50"
    if kind == "spacing":
        clock += rf"+10*mod(N\,2)*between(N\,{b}\,{b + count - 1})"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}",
                    "-framerate", str(FPS), "-i", "pipe:0", "-filter_threads", "1", "-vf", clock, "-fps_mode", "passthrough",
                    "-enc_time_base", "1:1000", "-threads", "1", "-pix_fmt", "yuv420p", *CODECS[codec], str(path)],
                   input=b"".join(frames), check=True, timeout=30)


def descriptor(rgb):
    means = rgb.reshape(30, 6, 40, 8, 3).mean(axis=(1, 3), dtype=np.float64)
    return (means @ np.array([77., 150., 29.]) / 256).reshape(-1)


def native(path):
    table = probe_frames(path)
    values, hashes = [], []
    for frame in decode_frames(path, table, range(FRAMES), timeout_s=10.):
        if not frame.pts_verified or frame.pts != table.pts[frame.index]:
            raise ValueError("Decoded picture has an unverified native PTS")
        hashes.append(hash_bytes(frame.rgb.tobytes()))
        values.append(descriptor(frame.rgb))
    if len(values) != len(table.pts) or len(values) != FRAMES:
        raise ValueError("Incomplete authored frame table")
    return table, np.stack(values), hashes


def pair_matrices(table, values):
    norm = (values * values).mean(axis=1)
    frame_sq = np.maximum(0, norm[:, None] + norm[None, :] - 2 * values @ values.T / values.shape[1])
    steps = np.diff(table.pts)
    valid = (steps > 0) & (steps <= int(picture_sequence.MAX_STEP / table.tb))
    clock = valid[:, None] & valid[None, :] & (steps[:, None] == steps[None, :])
    motion = np.diff(values, axis=0)
    motion -= motion.mean(axis=1, keepdims=True)
    rms = np.sqrt((motion * motion).mean(axis=1))
    power = rms[:, None] * rms[None, :]
    cos = (motion @ motion.T / values.shape[1]) / np.maximum(power, 1e-12)
    weaker = np.minimum(rms[:, None], rms[None, :])
    stronger = np.maximum(rms[:, None], rms[None, :])
    return frame_sq, clock, cos, weaker, weaker >= .55 * stronger


def segments(indices, max_gap):
    for span in np.split(indices, np.flatnonzero(np.diff(indices) > max_gap) + 1):
        if len(span):
            yield span


def search(pts, tb, matrices, rule, *, motion=True):
    frame_sq, clock, cos, weaker, stable = matrices
    distance, activity, direction = rule
    close = frame_sq <= distance ** 2
    linked = close[1:, 1:] & close[:-1, :-1] & clock
    moving = (weaker >= activity) & (cos >= direction) & stable if motion else np.ones_like(linked)
    best, result = None, None
    for lag in range(9, len(pts) - 8):
        edges = np.flatnonzero(linked.diagonal(lag))
        supports = moving.diagonal(lag)
        for segment in segments(edges, 1):
            for chain in segments(segment[supports[segment]], 3):
                if len(chain) < 8:
                    continue
                for a, start in enumerate(chain[:-7]):
                    b = int(np.searchsorted(chain, start + lag - 1, side="left")) - 1
                    end = int(chain[b]) + 1
                    if b - a + 1 < 8 or 2 * (b - a + 1) < end - start:
                        continue
                    elapsed = int(pts[end] - pts[start])
                    rank = (elapsed, -int(start), -int(start + lag))
                    if elapsed * tb >= picture_sequence.MIN_SPAN and (best is None or rank > best):
                        best, result = rank, [int(start), end, int(start + lag), end + lag]
    return result


def overlap(pair, truth):
    if pair is None or truth is None:
        return 0.
    result = 1.
    for i in (0, 2):
        start, end, known_start, known_end = *pair[i:i + 2], *truth[i:i + 2]
        intersection = max(0, min(end, known_end) - max(start, known_start) + 1)
        union = end - start + known_end - known_start + 2 - intersection
        result = min(result, intersection / union)
    return result


def report(rows):
    output = {}
    for codec in CODECS:
        cases = [r for r in rows if r["codec"] == codec]
        positives = [r for r in cases if r["truth"] is not None]
        controls = [r for r in cases if r["truth"] is None]
        by_arm = {}
        for arm in ("exact", "picture", "motion"):
            good = [r for r in positives if overlap(r[arm], r["truth"]) >= .50]
            flags = [r for r in controls if r[arm] is not None]
            by_arm[arm] = {"useful": len(good), "groups": len({r["group"] for r in good}), "controls_flagged": len(flags),
                           "flags_by_kind": {kind: sum(r["kind"] == kind for r in flags) for kind in CASES if kind != "copy"}}
        baseline = [r for r in positives if overlap(r["exact"], r["truth"]) >= .50]
        added = [r for r in positives if overlap(r["exact"], r["truth"]) < .50 <= overlap(r["motion"], r["truth"])]
        additional = [r for r in controls if r["motion"] is not None and r["exact"] is None]
        output[codec] = {"positives": len(positives), "controls": len(controls), "arms": by_arm,
                         "added_useful_groups": len({r["group"] for r in added}), "lost_exact": sum(overlap(r["motion"], r["truth"]) < .5 for r in baseline),
                         "added_controls": len(additional), "added_clock_or_hold_flags": sum(r["kind"] in {"hold", "spacing"} for r in additional)}
    return output


def choose(rows):
    def rank(rule):
        good = [r for r in rows if r["truth"] and overlap(r["search"][str(rule)], r["truth"]) >= .5]
        flags = sum(r["truth"] is None and r["search"][str(rule)] is not None for r in rows)
        return (flags == 0, len({r["group"] for r in good}), len(good), -flags,
                sum(overlap(r["search"][str(rule)], r["truth"]) for r in good), -rule[0], rule[1], rule[2])
    return max(RULES, key=rank)


def execute(role, run, archive):
    slots = DEVELOPMENT if role == "development" else HELD
    if role == "development":
        run.mkdir(mode=0o700)
        selected = None
    else:
        dev = load(run / "development.json")
        selected = tuple(dev["point"])
        if hash_file(Path(__file__)) != dev["runner_sha256"]:
            raise ValueError("Runner changed after the development point was sealed")
    rows, inputs = [], []
    plans = []
    for slot in slots:
        path, source, media = locate(archive, role, slot)
        if role == "held" and source in {item["original_source_sha256"] for item in dev["sources"]}:
            raise ValueError("Held recording shares an original source with development")
        plans.append({"group": slot, "original_source_sha256": source, "retained_media_sha256": media, "geometry": geometry(role, slot)})
    write(run / f"{role}-plan.json", {"sources": plans, "point": selected, "grid": RULES, "method": METHOD})
    media_dir = run / "media"
    media_dir.mkdir(exist_ok=True)
    for slot, plan in zip(slots, plans, strict=True):
        path, _, _ = locate(archive, role, slot)
        raw = read_source(path)
        first, second, length = plan["geometry"]
        inputs.append(plan | {"input_rgb_sha256": hash_bytes(raw)})
        for codec in CODECS:
            for kind in CASES:
                artifact = media_dir / f"{role}-{slot:02d}-{codec}-{kind}.mkv"
                deliver(raw, artifact, first, second, length, kind, codec)
                table, values, hashes = native(artifact)
                pts = np.asarray(table.pts)
                matrices = pair_matrices(table, values)
                start = time.monotonic()
                truth = [first, first + length - 1, second, second + length - 1] if kind == "copy" else None
                exact = picture_sequence.scan(list(zip(table.pts, hashes, strict=True)), table.tb)["pair"]
                row = {"group": slot, "codec": codec, "kind": kind, "truth": truth,
                       "encoded_sha256": hash_file(artifact), "pts": list(table.pts), "time_base": table.time_base,
                       "rgb_sha256": hashes, "exact": [exact[k][f"{s}_frame_index"] for k in ("first", "second") for s in ("start", "end")] if exact else None}
                if selected is None:
                    row["search"] = {str(rule): search(pts, table.tb, matrices, rule) for rule in RULES}
                    row["picture"] = {str(distance): search(pts, table.tb, matrices, (distance, 0., 0.), motion=False)
                                      for distance in sorted({rule[0] for rule in RULES})}
                else:
                    row["motion"] = search(pts, table.tb, matrices, selected)
                    row["picture"] = search(pts, table.tb, matrices, selected, motion=False)
                row["seconds"] = round(time.monotonic() - start, 4)
                rows.append(row)
        print(role, slot, len(rows), flush=True)
    if selected is None:
        selected = choose(rows)
        for row in rows:
            row["motion"] = row["search"][str(selected)]
            row["picture"] = row["picture"][str(selected[0])]
    summary = report(rows)
    passes = all(s["added_useful_groups"] >= 4 and s["lost_exact"] == 0 and s["added_controls"] <= 1
                 and s["added_clock_or_hold_flags"] == 0 for s in summary.values())
    write(run / f"{role}.json", {"schema": SCHEMA, "role": role, "runner_sha256": hash_file(Path(__file__)), "sources": inputs,
                                "point": list(selected), "rows": rows, "summary": summary, "passes": passes})
    print(role, selected, summary, passes, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("development", "held"))
    parser.add_argument("run", type=Path)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    execute(args.role, args.run, args.archive)
