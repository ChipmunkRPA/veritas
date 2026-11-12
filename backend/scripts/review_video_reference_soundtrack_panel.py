"""Independently verify frozen paired-video timing results and their source evidence."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import wave
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

ROLES = {"local_late", "local_early", "original", "gain", "noise_mix", "unrelated_insert", "global_shift", "short_offset", "picture_change"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def commitment(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def portable(record):
    rows = record["rows"]
    require(record["schema"] == "chipmunk.video-paired-soundtrack-panel/1" and
            [row["ordinal"] for row in rows] == list(range(5, 16)), "Held roster differs")
    totals = {}
    for codec in ("pcm", "aac"):
        correct, alerts, assessed, groups = Counter(), Counter(), Counter(), set()
        for row in rows:
            cases = [case for case in row["cases"] if case["delivery"] == codec]
            require(len(cases) == len(ROLES) and {case["role"] for case in cases} == ROLES, "Held delivery roles differ")
            for case in cases:
                role, state = case["role"], case["state"]
                require(state in {"local_shift", "no_local_shift", "unassessed"}
                        and (state == "local_shift") == bool(case["found"]), "Timing state disagrees with its result")
                assessed[role] += state != "unassessed"
                if not case["found"]:
                    continue
                if role not in {"local_late", "local_early"}:
                    alerts[role] += 1
                    continue
                (a, b), offset = case["found"]
                left, right = case["authored"]
                require(left < right and a < b and (case["offset"] > 0) == (role == "local_late"), "Authored timing differs")
                inter = max(0, min(b, right) - max(a, left))
                if 5 * inter >= 3 * (max(b, right) - min(a, left)) and abs(offset - case["offset"]) <= 16:
                    correct[role] += 1
                    groups.add(row["ordinal"])
        admitted = (sum(correct.values()) >= 16 and correct["local_late"] >= 7 and correct["local_early"] >= 7
                    and len(groups) >= 8 and sum(alerts.values()) <= 1 and not (alerts["global_shift"] or alerts["picture_change"]))
        totals[codec] = {"useful": dict(correct), "false_flags": dict(alerts), "assessed": dict(assessed),
                         "groups": len(groups), "admitted": admitted}
    require(totals == record["results"], "Published totals differ from the frozen held decisions")
    return totals


def command(*args):
    return subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=25).stdout


def inspect(path):
    import numpy as np

    container = json.loads(command("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "frame=best_effort_timestamp:stream=time_base,width,height", "-of", "json", str(path)))
    stream, = container["streams"]
    pts = [int(row["best_effort_timestamp"]) for row in container["frames"]]
    clock = [(tick - pts[0]) * Fraction(stream["time_base"]) for tick in pts]
    checksums = command("ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(path), "-an", "-map", "0:v:0",
                        "-pix_fmt", "rgb24", "-fps_mode", "passthrough", "-threads", "1", "-f", "framehash",
                        "-hash", "sha256", "-").decode().splitlines()
    pictures = [line.rsplit(",", 1)[1].strip() for line in checksums if not line.startswith("#")]
    samples = command("ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-map", "0:a:0", "-ar", "16000", "-ac", "1",
                      "-c:a", "pcm_s16le", "-f", "s16le", "-")
    sound = command("ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=start_pts,time_base",
                    "-of", "json", str(path))
    audio, = json.loads(sound)["streams"]
    origin = int(audio["start_pts"]) * Fraction(audio["time_base"]) - pts[0] * Fraction(stream["time_base"])
    require(len(clock) == len(pictures), "Exact picture count differs from native timestamps")
    return clock, pictures, np.frombuffer(samples, dtype="<i2").astype(np.float32) / 32768, origin, stream


def localize(earlier, submitted):
    import cv2
    import numpy as np

    matches = []
    for start in range(0, len(submitted) - 2559, 640):
        sample = submitted[start:start + 2560]
        left, right = max(0, start - 5760), min(len(earlier) - 2560, start + 5760)
        if float(np.std(sample)) < .003:
            continue
        region = earlier[left:right + 2560]
        result = cv2.matchTemplate(region[None, :], sample[None, :], cv2.TM_CCOEFF_NORMED)[0]
        prefix, squares = np.r_[0, np.cumsum(region, dtype=np.float64)], np.r_[0, np.cumsum(region.astype(np.float64) ** 2)]
        energy = (squares[2560:] - squares[:-2560]) / 2560 - ((prefix[2560:] - prefix[:-2560]) / 2560) ** 2
        result[energy < .003 ** 2] = 0
        best = int(np.argmax(result))
        rivals = result[np.abs(np.arange(len(result)) - best) >= 96]
        if result[best] >= .86 and result[best] - rivals.max(initial=0) >= .06:
            matches.append((start, start - left - best))
    zeros = [start for start, lag in matches if abs(lag) <= 16]
    cuts = []
    for i, (start, lag) in enumerate(matches):
        if abs(lag) < 1920:
            continue
        run = cuts[-1] if cuts else []
        if (not run or matches[i - 1] != run[-1] or abs(run[0][1] - lag) > 16 or start - run[-1][0] > 3840):
            run = []
            cuts.append(run)
        run.append((start, lag))
    outcomes = []
    for run in cuts:
        first, stop = run[0][0], run[-1][0] + 2560
        if len(run) < 4 or 3 * len(run) < 2 * ((run[-1][0] - first) // 640 + 1):
            continue
        before = [start for start in zeros if first - 5760 <= start <= first - 1280]
        after = [start for start in zeros if stop - 1280 <= start <= stop + 5760]
        if len(before) < 2 or len(after) < 2:
            continue
        start, end = max(first, before[-1] + 2560), min(stop, after[0])
        if end - start >= 4480:
            outcomes.append((-len(run), start, end, int(sorted(lag for _, lag in run)[len(run) // 2])))
    best = min(outcomes, default=None)
    return ([[best[1], best[2]], best[3]] if best else None), len(matches)


def _audio_asset(pcm):
    with io.BytesIO() as output:
        with wave.open(output, "wb") as result:
            result.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            result.writeframes((pcm * 32768).astype("<i2").tobytes())
        return hashlib.sha256(output.getvalue()).hexdigest()


def native_group(run, row):
    import numpy as np
    from PIL import Image

    folder = run / f"source-{row['ordinal']:02d}"
    verified = Counter()
    for codec, suffix in (("pcm", "mkv"), ("aac", "mp4")):
        earlier = folder / f"earlier-{codec}.{suffix}"
        a_clock, a_hashes, a_pcm, a_origin, _ = inspect(earlier)
        for case in (case for case in row["cases"] if case["delivery"] == codec):
            role = case["role"]
            path = folder / f"{role}-{codec}.{suffix}"
            require(digest(earlier) == case["earlier_sha256"] and digest(path) == case["submitted_sha256"], "File hash differs")
            clock, hashes, pcm, origin, geometry = inspect(path)
            require(clock == a_clock and len(clock) == case["pictures"] and origin == a_origin and len(pcm) == len(a_pcm),
                    "Picture or audio clock differs")
            require(commitment(hashes) == case["native_picture_rgb_sha256"] and
                    commitment(list(map(str, clock))) == case["relative_picture_pts_sha256"] and
                    hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest() == case["pcm_sha256"], "Native picture or PCM digest differs")
            require((hashes == a_hashes) == (role != "picture_change"), "Picture content relationship differs")
            observed, support = localize(a_pcm, pcm) if hashes == a_hashes else (None, 0)
            if observed:
                (start, end), offset = observed
                if (origin + Fraction(min(start, start - offset), 16000) < 0 or
                        origin + Fraction(max(end, end - offset), 16000) > clock[-1]):
                    observed = None
            require(observed == case["found"] and support == case["windows"], f"Independent timing decision differs for {path.name}")
            if observed:
                index = bisect_right(clock, origin + Fraction(start + end, 32000)) - 1
                picture = command("ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(path), "-map", "0:v:0", "-an",
                                  "-vf", f"select=eq(n\\,{index}),format=rgb24", "-frames:v", "1", "-threads", "1", "-f", "rawvideo", "-")
                require(hashlib.sha256(picture).hexdigest() == hashes[index], "Selected exact picture differs")
                with io.BytesIO() as output, Image.fromarray(np.frombuffer(picture, dtype=np.uint8).reshape(
                        geometry["height"], geometry["width"], 3)) as image:
                    image.save(output, "PNG")
                    image_sha = hashlib.sha256(output.getvalue()).hexdigest()
                assets = {f"soundtrack-shift-picture-{index:03d}.png": image_sha,
                          "soundtrack-shift-earlier.wav": _audio_asset(a_pcm[start - offset:end - offset]),
                          "soundtrack-shift-submitted.wav": _audio_asset(pcm[start:end])}
                require(assets == case["assets"], "Native screenshot or playable sample digest differs")
                verified.update({"images": 1, "excerpts": 2, "excerpt_samples": 2 * (end - start)})
            else:
                require(not case["assets"], "An unselected result has evidence assets")
            verified.update({"deliveries": 1, "pictures": len(hashes), "decoded_samples": len(pcm)})
    print(f"Replayed {row['ordinal']:02d}: {verified['deliveries']} deliveries", flush=True)
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--workers", type=int, default=1, choices=(1, 2))
    args = parser.parse_args()
    record = read(args.receipt)
    totals = portable(record)
    print(json.dumps(totals, sort_keys=True), flush=True)
    if args.run:
        import cv2

        cv2.setNumThreads(1)
        require(digest(args.run / "freeze.json") == record["freeze_sha256"], "Frozen plan hash differs")
        counters = Counter()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(lambda row: native_group(args.run, row), record["rows"]):
                counters.update(result)
        print(json.dumps(dict(counters), sort_keys=True))


if __name__ == "__main__":
    main()
