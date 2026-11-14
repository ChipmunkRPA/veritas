# SPDX-License-Identifier: Apache-2.0
"""Evaluate brief local timing shifts over a continuous shot in one video upload."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.video_short_soundtrack_transfer import sha, sources

ROOT = Path(__file__).resolve().parents[2]
ROSTER = ROOT / "docs/evaluation-panels/video-mouth-hd-hold-2026-11-12.json"
SCHEMA = "chipmunk.video-lip-offset-path/1"
FIT = frozenset({"diego", "ioanna", "jan", "mckensie", "piotr", "regina", "simon"})
ROLES = ("original", "gain", "lowpass", "noise", "global_early", "local_early", "global_late", "local_late",
         "matched_picture_early", "matched_picture_late", "dub", "camera_cut")
POSITIVE = {"local_early", "local_late"}
DISPLACEMENT = (5, 7)
LENGTHS = (8, 12, 16, 22)
TICKS, HOP = 160, 640
PACKED_AUDIO = f"aresample=16000:first_pts=0,apad=whole_len={TICKS * HOP},atrim=end_sample={TICKS * HOP}"
DESIGN = {
    "method": "Eight regularized canonical correlation directions from same-clock speech. Compare one constant offset "
              "with one temporary offset using identical observed ticks and both neighboring flanks within one shot.",
    "input": "6.4 s, 480 x 360 H.264 at 25 fps, 16 kHz mono PCM with fixed resampling endpoints. Two frozen offsets per person.",
    "controls": "Originals, local gain, filter and noise, constant offsets, synchronized picture/sound observations, "
                "whole-track dubbing from another retained speaker and a rendered cut between synchronized scenes.",
    "point": "Strict excess above the largest ordinary response on four separate calibration people plus 0.02, "
             "with a 0.10 floor. Freeze before held scoring.",
    "pass": "Add at least four correct locations spanning three held people. Allow at most two extra control flags "
            "and two extra misplaced edits, with zero extra whole-track, synchronized, dubbed or camera-cut flags.",
    "utility": "At least 0.40 temporal intersection over union on the soundtrack clock and a signed offset within 80 ms of the known edit.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_plan():
    prior = json.loads(ROSTER.read_bytes())["source_plan"]["rows"]
    rows = [{"file": item["name"] + ".webm", "sha256": item["source_sha256"], "page": item["commons"],
             "license": item["license"], "attribution": item["credit"], "offsets": [item["offset_s"] + 1, item["offset_s"] + 15],
             "split": "fit" if item["name"] in FIT else "point"} for item in prior]
    rows.extend(item | {"offsets": [52, 79], "split": "held"} for item in sources())
    return rows


def _sha_pcm(wave):
    return hashlib.sha256(wave.tobytes()).hexdigest()


def _clip(source, offset, path):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", str(offset), "-i", str(source), "-t", "6.4",
                    "-map", "0:v:0", "-map", "0:a:0", "-vf", "fps=25,scale=480:360:force_original_aspect_ratio=decrease,"
                    "pad=480:360:(ow-iw)/2:(oh-ih)/2,setsar=1", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", "-threads", "1", "-af", PACKED_AUDIO,
                    "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", str(path)],
                   check=True, timeout=40, capture_output=True)


def _camera_cut(before, after, path):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(before), "-i", str(after), "-t", "6.4",
                    "-filter_complex", "[0:v]trim=end=3.2,setpts=PTS-STARTPTS[v1];[0:a]atrim=end=3.2,asetpts=PTS-STARTPTS[a1];"
                    "[1:v]trim=end=3.2,setpts=PTS-STARTPTS[v2];[1:a]atrim=end=3.2,asetpts=PTS-STARTPTS[a2];"
                    f"[v1][a1][v2][a2]concat=n=2:v=1:a=1[v][out];[out]{PACKED_AUDIO}[a]", "-map", "[v]", "-map", "[a]", "-c:v", "libx264",
                    "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-threads", "1", "-c:a", "pcm_s16le",
                    "-ar", "16000", "-ac", "1", "-y", str(path)], check=True, timeout=40, capture_output=True)


def _observe(path, detector):
    import numpy as np

    from scripts.video_lipsync_local_delay_transfer import inputs

    grid, table, origin, wave, removed, faces, patches, anchors = inputs(path.parent, {"file": path.name, "sha256": sha(path)}, detector)
    require(grid.ticks == TICKS and len(wave) == TICKS * HOP and len(table.pts) == TICKS and origin == 0 and removed == 0,
            f"Unqualified source grid: {grid.ticks} ticks, {len(table.pts)} pictures, {len(wave)} samples, origin {origin}, {removed} removed")
    flags = subprocess.check_output(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vf", "scale=32:24",
                                     "-an", "-c:v", "rawvideo", "-threads", "1", "-pix_fmt", "gray", "-f", "rawvideo", "-"], timeout=30)
    y = np.frombuffer(flags, dtype=np.uint8).reshape(TICKS, 768).astype(np.int16)
    jump = np.abs(np.diff(y, axis=0))
    cuts = (jump.mean(axis=1) >= 18) & ((jump >= 16).mean(axis=1) >= .45)
    return {"grid": grid, "table": table, "wave": wave, "faces": faces, "patches": patches, "anchors": anchors,
            "shot": np.cumsum(np.r_[False, cuts]), "sha256": sha(path)}


def interventions(observed, donor, cut, group, excerpt):
    import numpy as np

    from scripts.video_lipsync_local_delay_transfer import shift

    wave = observed["wave"]
    width = LENGTHS[(group + excerpt) % len(LENGTHS)]
    start = 61 + 7 * excerpt + group % 7
    end = start + width
    a, b = start * HOP, end * HOP
    delta = DISPLACEMENT[(group + excerpt) % len(DISPLACEMENT)]
    identity = tuple(range(TICKS))
    unchanged = wave[a:b]
    blend = np.ones(len(unchanged))
    blend[:160] = np.arange(160) / 160
    blend[-160:] = blend[:160][::-1]

    def change(samples):
        out = wave.copy()
        out[a:b] = np.rint(unchanged * (1 - blend) + samples * blend).clip(-32768, 32767).astype("<i2")
        return out

    cases = {"original": (wave, None, None, identity), "gain": (change(.55 * unchanged), None, None, identity),
             "lowpass": (change(np.convolve(unchanged, np.ones(7) / 7, mode="same")), None, None, identity)}
    variance = float(unchanged.std())
    noise = np.random.default_rng(group * 12 + excerpt).normal(0, .70 * variance, b - a)
    cases["noise"] = (change(unchanged + noise), None, None, identity)
    for lag, suffix in ((-delta, "early"), (delta, "late")):
        shifted = shift(wave, lag * HOP)
        staged = change(shifted[a:b])
        require(np.count_nonzero(staged[a:b] != unchanged) >= HOP, "The edit contains too few changed source samples")
        cases[f"global_{suffix}"] = shifted, None, None, identity
        cases[f"local_{suffix}"] = staged, [start, end], lag, identity
        moved = tuple(i - lag if start <= i < end else i for i in identity)
        cases[f"matched_picture_{suffix}"] = staged, None, None, moved
    cases["dub"] = donor["wave"], None, None, identity
    cases["camera_cut"] = cut["wave"], None, None, identity
    return cases


def _run_case(model, observed, wave, indices):
    import numpy as np

    from app.detectors.video import lipsync, xcorr
    from scripts import video_lip_offset_path as candidate
    from scripts.video_lipsync_signed_transitions import current

    grid = observed["grid"]
    frames = tuple(grid.frames[i] for i in indices)
    pictures = replace(grid, frames=frames)
    patches = {tick: observed["patches"][source] for tick, source in enumerate(indices) if source in observed["patches"]}
    values, speech = candidate.grid_scores(model, patches, frames, wave)
    proposal = candidate.compare(values, speech, observed["shot"][np.array(indices)])
    old = current(observed["faces"], patches, {tick: xcorr.aperture(patch) for tick, patch in patches.items()},
                  lipsync.vad(wave.astype(np.float32) / 32768.), pictures)
    if best := proposal["best"]:
        start, end = best["span"]
        best["witness"] = [dict(observed["anchors"][frames[t]], tick=t, source_tick=indices[t])
                           for t in (start - 1, (start + end) // 2, end) if frames[t] >= 0]
        a, b = ((tick + best["changed"]) * HOP for tick in (start, end))
        best["audio_samples"] = [a, b]
        best["audio_excerpt_sha256"] = _sha_pcm(wave[a:b])
    return {"existing": old, "candidate": proposal, "pcm_sha256": _sha_pcm(wave)}


def _useful(span, signed, truth, lag):
    a, b = span
    x, y = truth
    return signed is not None and abs(signed - lag) <= 2 and 5 * (min(b, y) - max(a, x)) >= 2 * (max(b, y) - min(a, x))


def point(rows):
    return round(max(.10, max((c["candidate"]["best"]["score"] for row in rows for c in row["cases"]
                              if c["role"] not in POSITIVE and c["candidate"]["best"]), default=0.) + .02), 8)


def review(document):
    design = document["freeze"]
    require(design["schema"] == SCHEMA and design["design"] == DESIGN and design["sources"] == source_plan(), "Wrong frozen source roles")
    expected = [(group, excerpt, source["split"]) for group, source in enumerate(design["sources"], 1)
                if source["split"] != "fit" for excerpt in range(2)]
    rows = document["rows"]
    require([(r["group"], r["excerpt"], r["split"]) for r in rows] == expected, "Incomplete roster or overlap across roles")
    cutoff = point([r for r in rows if r["split"] == "point"])
    require(cutoff == document["point"], "The held operating point differs from calibration")
    totals = defaultdict(Counter)
    gains = set()
    for row in rows:
        held = row["split"] == "held"
        group, excerpt = row["group"], row["excerpt"]
        require([c["role"] for c in row["cases"]] == list(ROLES), "A paired delivery is missing")
        for c in row["cases"]:
            role, truth, lag = c["role"], c["truth"], c["lag"]
            if role in POSITIVE:
                a = 61 + 7 * excerpt + group % 7
                require(truth == [a, a + LENGTHS[(group + excerpt) % len(LENGTHS)]]
                        and lag == DISPLACEMENT[(group + excerpt) % 2] * (-1 if role.endswith("early") else 1), "Incorrect positive timing")
            else:
                require(truth is None and lag is None, "An ordinary clip has a manipulated reference label")
            earlier, proposal = c["existing"], c["candidate"]["best"]
            if proposal:
                a, b = proposal["span"]
                q, p = proposal["reference"], proposal["changed"]
                lo, middle, hi, fit = proposal["gains"]
                require(b - a in LENGTHS and 20 <= a < b <= TICKS - 20 and abs(q) <= 6 and abs(p) <= 9 and 3 <= abs(q-p) <= 8
                        and abs(proposal["score"] - min(2*lo, middle, 2*hi, fit)) <= 2e-8
                        and proposal["audio_samples"] == [(a + p) * HOP, (b + p) * HOP], "Invalid candidate score, signed lag or native interval")
                require(all(ref["tick"] in {a - 1, (a + b) // 2, b} and 0 <= ref["frame_index"] < TICKS
                            for ref in proposal["witness"]), "Wrong native picture witness")
            if not held:
                continue
            counts = totals[role]
            counts.update(files=1, old_measured=bool(earlier["starts"]), measured=bool(c["candidate"]["tested"]))
            selected = bool(proposal and proposal["score"] > cutoff)
            if truth:
                old = any(shift is not None and _useful([a / 2 + shift, b / 2 + shift], shift, truth, lag)
                          for _, a, b, shift in earlier["ranges"])
                good = bool(proposal and _useful([value // HOP for value in proposal["audio_samples"]],
                                                proposal["changed"] - proposal["reference"], truth, lag))
                counts.update(old_correct=old, useful_before_point=good, correct=selected and good,
                              gained=selected and good and not old, added_misplaced=selected and not good and not earlier["ranges"])
                if selected and good and not old:
                    gains.add(group)
            else:
                counts.update(ordinary_flag=selected, old_ordinary_flag=bool(earlier["ranges"]),
                              added_ordinary_flag=selected and not earlier["ranges"])
    supported = sum(t["gained"] for t in totals.values())
    ordinary = sum(t["added_ordinary_flag"] for t in totals.values())
    critical = sum(t["added_ordinary_flag"] for k, t in totals.items() if k.startswith(("global_", "matched_picture_", "dub", "camera_cut")))
    wrong = sum(t["added_misplaced"] for t in totals.values())
    return {"results": {k: dict(t) for k, t in totals.items()}, "added_locations": supported, "sources_with_gain": len(gains),
            "added_ordinary_flags": ordinary, "critical_added_flags": critical, "added_misplaced": wrong,
            "passes": supported >= 4 and len(gains) >= 3 and ordinary <= 2 and wrong <= 2 and not critical}


def run(cache, held, output):
    import cv2

    from app.detectors.image.faces import load_yunet
    from scripts import video_lip_offset_path as model
    from scripts.video_lipsync_window_schedule import FACE_FILE, FACE_SHA

    roster = source_plan()
    require(sha(FACE_FILE) == FACE_SHA, "The face asset differs from the published license-reviewed weights")
    inputs = [((held if item["split"] == "held" else cache) / item["file"], item) for item in roster]
    require(all(sha(path) == item["sha256"] for path, item in inputs), "A published source differs from its commitment")
    frozen = {"schema": SCHEMA, "design": DESIGN, "sources": roster, "face_sha256": FACE_SHA,
              "producers": {Path(file).name: sha(Path(file)) for file in (__file__, model.__file__)},
              "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0], "opencv": cv2.__version__}
    output.mkdir(mode=0o700)
    (output / "freeze.json").write_text(json.dumps(frozen, indent=2) + "\n")
    cv2.setNumThreads(1)
    detector = load_yunet(FACE_FILE)
    observed, excerpts, rows, fit_inputs = {}, {}, [], []
    with TemporaryDirectory(prefix="lip-offset-native-") as temporary:
        work = Path(temporary)
        for split in ("fit", "point", "held"):
            bank = [g for g, (_, item) in enumerate(inputs, 1) if item["split"] == split]
            for group in bank:
                source, item = inputs[group - 1]
                for j, offset in enumerate(item["offsets"]):
                    path = work / f"clip-{group}-{j}.mov"
                    _clip(source, offset, path)
                    excerpts[group, j] = path
                    clip = _observe(path, detector)
                    observed[group, j] = clip
                    if split == "fit":
                        fit_inputs.append((clip["patches"], clip["grid"].frames, clip["wave"]))
                    print(split, "observed", group, j, len(clip["patches"]), flush=True)
            if split == "fit":
                fitted, fit_support = model.fit(fit_inputs)
                continue
            for i, group in enumerate(bank):
                donor_group = bank[(i + 1) % len(bank)]
                for j in range(2):
                    original, donor = observed[group, j], observed[donor_group, j]
                    path = work / "cut.mov"
                    _camera_cut(excerpts[group, j], excerpts[donor_group, j], path)
                    cut = _observe(path, detector)
                    rows.append({"group": group, "excerpt": j, "split": split, "input_sha256": original["sha256"],
                                 "cut_sha256": cut["sha256"], "detected_cut_ticks": [i for i in range(1, TICKS) if cut["shot"][i] != cut["shot"][i-1]],
                                 "crops": len(original["patches"]), "cases": []})
                    cases = interventions(original, donor, cut, group, j)
                    for role in ROLES:
                        wave, truth, lag, indices = cases[role]
                        item = cut if role == "camera_cut" else original
                        rows[-1]["cases"].append({"role": role, "truth": truth, "lag": lag, **_run_case(fitted, item, wave, indices)})
                    print(split, "measured", group, j, flush=True)
            if split == "point":
                cutoff = point(rows)
                (output / "point.json").write_text(json.dumps({"point": cutoff, "rows": rows}) + "\n")
    document = {"freeze": frozen, "fit_support": fit_support, "point": cutoff, "rows": rows}
    document["summary"] = review(document)
    (output / "receipt.json").write_text(json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps(document["summary"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--held", type=Path)
    args = parser.parse_args()
    if args.action == "run":
        if not args.cache or not args.held:
            parser.error("Both pinned media folders are required")
        run(args.cache, args.held, args.path)
    else:
        document = json.loads(args.path.read_bytes())
        calculated = review(document)
        require(calculated == document["summary"], "The measured totals differ from their recorded values")
        print(json.dumps(calculated, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
