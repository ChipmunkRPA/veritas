# SPDX-License-Identifier: Apache-2.0
"""Check whether a wider picture comparison excludes short whole-frame loops."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image.faces import load_yunet
from app.detectors.video import mouth_local_hold
from scripts import video_mouth_sequence_reuse_transfer as base

CONTRACT = {
    "point": [0.28, 0.38, 0.48, 0.58],
    "control_fit": "At most one control flag and zero whole-picture or matched audio/video flags. Maximize correct pairs then prefer the stricter point.",
    "paired_arms": "Frozen earlier .28 comparator, and the same candidate with a wider-picture difference guard.",
    "support": "Compare stable-coordinate texture in two wider picture sectors away from the mouth. The middle of both repeated intervals must have at least two active wider tiles in both sectors on three fifths of paired pictures.",
    "source_scope": "Wikitongues speaker roles from the earlier HD study, rematerialized on a two-second 25fps H.264 picture and 16kHz PCM clock. Audio is explicitly trimmed to the first 32,000 resampled samples. Speakers are disjoint from the GRID/CREMA-D banks and from each other.",
    "gate": {
        "correct_minimum": 10,
        "groups_minimum": 4,
        "max_controls": 1,
        "max_loops_or_av_controls": 0,
        "minimum_assessed": 0.75,
        "iou": 0.6,
    },
}
OLD_POINT = 0.28
POSITIVES = base.POSITIVES
CONTROLS = base.CONTROLS


def ffmpeg(*args):
    return subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-filter_threads", "1", *map(str, args)],
        check=True,
        timeout=35,
        capture_output=True,
    ).stdout


def freeze(archive, parent):
    retained = base.read(archive / "run/freeze.json")
    receipt = base.read(archive / "run/receipt.json")
    base.require(
        receipt["freeze_sha256"] == base.digest(archive / "run/freeze.json"), "Archived source ledger changed"
    )
    rows = retained["rows"]
    base.require(
        len(rows) == 11
        and len({r["source_sha256"] for r in rows}) == 11
        and len({r["name"] for r in rows}) == len({r["commons"] for r in rows}) == 11,
        "Distinct retained speakers required",
    )
    for row in rows:
        base.require(
            row["license"].startswith(
                ("https://creativecommons.org/licenses/by/", "https://creativecommons.org/licenses/by-sa/")
            ),
            "Unexpected source rights",
        )
    point = base.read(parent / "point.json")
    base.require(point["point"] == OLD_POINT, "Earlier independently selected point differs")
    return {
        "contract": CONTRACT,
        "previous_development_point_sha256": base.digest(parent / "point.json"),
        "previous_freeze_sha256": base.digest(parent / "freeze.json"),
        "source_roles_sha256": receipt["freeze_sha256"],
        "candidate_sha256": base.digest(Path(__file__)),
        "base_code_sha256": base.digest(Path(base.__file__)),
        "rows": rows,
        "opencv": cv2.__version__,
    }


def original(archive, row):
    source = archive / "sources" / (row["name"] + ".webm")
    base.require(base.digest(source) == row["source_sha256"], "Source differs from the publisher archive")
    picture = ffmpeg(
        "-ss",
        row["offset_s"],
        "-i",
        source,
        "-map",
        "0:v:0",
        "-an",
        "-frames:v",
        base.N,
        "-vf",
        "fps=25,scale=640:360:force_original_aspect_ratio=increase:flags=area,crop=480:360,setsar=1",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    )
    sound = ffmpeg(
        "-ss",
        row["offset_s"],
        "-i",
        source,
        "-map",
        "0:a:0",
        "-vn",
        "-af",
        "aresample=16000,atrim=end_sample=32000",
        "-ar",
        base.SR,
        "-ac",
        1,
        "-f",
        "s16le",
        "-",
    )
    base.require(
        len(picture) == base.N * 480 * 360 * 3 and len(sound) == base.N * base.HOP * 2,
        "Incomplete two-second materialization",
    )
    return list(np.frombuffer(picture, dtype=np.uint8).reshape(base.N, 360, 480, 3)), np.frombuffer(
        sound, dtype="<i2"
    )


def wider_tiles(gray, mouth):
    size = max(8, round(0.48 * mouth.width))
    x, y = round(mouth.x), round(mouth.y)
    h, w = gray.shape[1:]
    upper, lower = [], []
    for dy in (-5, -4, -3, 2, 3, 4):
        for dx in (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4):
            left, top = x + dx * size, y + dy * size
            if not (0 <= left <= w - size and 0 <= top <= h - size):
                continue
            values = base.centered(base.crop(gray, (left, top, size, size)))
            if float(np.median(values.std(axis=1))) >= 4.0:
                (upper if dy < 0 else lower).append(base.differences(values))
    return (
        [np.sort(group, axis=0)[-2] for group in (upper, lower)] if min(len(upper), len(lower)) >= 2 else []
    )


def qualify(choices, sectors):
    if not sectors:
        return []
    accepted = []
    context = np.minimum(*sectors)
    for choice in choices:
        (a, end), (b, _) = choice["ranges"]
        # Drop both edges so similar whole-picture repeats cannot qualify from flanking native movement.
        start = np.arange(a + 2, end - 2)
        comparison = context[start, start + b - a]
        if np.count_nonzero(comparison >= 0.9) >= 0.60 * len(comparison) and np.median(comparison) >= 1.1:
            accepted.append(choice | {"wider_difference": float(np.median(comparison))})
    return accepted


def run(folder, archive, row, group, detector):
    frames, pcm = original(archive, row)
    geometry = mouth_local_hold.stable_geometry(frames, detector)
    local = folder / row["name"]
    local.mkdir(mode=0o700)
    box = None
    if geometry:
        width, height = round(geometry.width * 1.52), round(geometry.width * 1.10)
        x, y = round(geometry.x - width / 2), round(geometry.y - height / 2)
        if 0 <= x <= 480 - width and 0 <= y <= 360 - height:
            box = [x, y, width, height]
    if box is None or row["control_only"]:
        cases = [("untouched", frames, pcm, None, None), ("tandem", frames, pcm, None, None)]
    else:
        cases = base.author(frames, pcm, box, group)
    records = []
    for role, pictures, audio, donor, target in cases:
        path = local / (role + ".mov")
        table, decoded, observed = base.deliver(local, path, pictures, audio, role)
        stable = mouth_local_hold.stable_geometry(decoded, detector)
        gray = np.stack([cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) for rgb in decoded]).astype(np.float32)
        choices = base.pairs(gray, observed, stable) if stable else []
        sectors = wider_tiles(gray, stable) if stable else []
        guarded = qualify(choices, sectors)
        selected = [base.best(guarded, point) for point in base.POINTS]
        unguarded = base.best(choices, OLD_POINT)
        records.append(
            {
                "role": role,
                "source": donor,
                "target": target,
                "assessed": bool(sectors),
                "baseline_assessed": stable is not None,
                "matches": selected,
                "baseline": unguarded,
                "sha256": base.digest(path),
                "pcm_sha256": base.pcm_sha(observed),
                "native": [base.witness(x, decoded, table.pts) for x in (unguarded, *selected)],
            }
        )
        if role not in {"untouched", "mouth_later", "mouth_earlier"}:
            path.unlink()
    base.save(
        local / "result.json",
        {
            "bank": "wikitongues",
            "group": group,
            "rows": records,
            "geometry_qualified": box is not None,
            "control_only": row["control_only"],
            "source_sha256": row["source_sha256"],
            "pcm_source_sha256": base.pcm_sha(pcm),
            "normalized_rgb_sha256": hashlib.sha256(b"".join(f.tobytes() for f in frames)).hexdigest(),
        },
    )
    print(row["split"], group, len(records), "complete", flush=True)


def count(groups, index):
    totals = {
        role: dict(assessed=0, correct=0, false=0, wrong=0, exact=0, n=0) for role in POSITIVES + CONTROLS
    }
    useful = set()
    for group in groups:
        for case in group["rows"]:
            role = case["role"]
            entry = totals[role]
            entry["n"] += 1
            entry["assessed"] += int(case["baseline_assessed" if index is None else "assessed"])
            match = case["baseline"] if index is None else case["matches"][index]
            if not match:
                continue
            if not case["target"]:
                entry["false"] += 1
                continue
            expected = sorted((case["source"], case["target"]))
            accurate = match["lag"] == expected[1][0] - expected[0][0] and all(
                base.iou(found, truth) >= base.Fraction(3, 5)
                for found, truth in zip(match["ranges"], expected, strict=True)
            )
            entry["correct" if accurate else "wrong"] += 1
            if accurate:
                useful.add(group["group"])
                entry["exact"] += int(match["ranges"] == expected)
    return {
        "roles": totals,
        "useful_groups": len(useful),
        "total": {key: sum(r[key] for r in totals.values()) for key in next(iter(totals.values()))},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "development", "point", "held", "report"))
    p.add_argument("folder", type=Path)
    p.add_argument("archive", type=Path)
    p.add_argument("earlier", type=Path)
    args = p.parse_args()
    cv2.setNumThreads(1)
    plan = freeze(args.archive, args.earlier)
    folder = args.folder
    if args.phase == "freeze":
        folder.mkdir(mode=0o700)
        base.save(folder / "freeze.json", plan)
        return
    base.require(base.read(folder / "freeze.json") == plan, "Frozen source or method changed")
    if args.phase in {"development", "held"}:
        if args.phase == "held":
            base.require(
                base.read(folder / "point.json")["point"] in base.POINTS, "No frozen operating point"
            )
        detector = load_yunet(base.sources.WEIGHTS)
        for i, row in enumerate(plan["rows"]):
            if row["split"] == args.phase:
                run(folder, args.archive, row, i, detector)
        return
    split = "development" if args.phase == "point" else "held"
    rows = [base.read(folder / row["name"] / "result.json") for row in plan["rows"] if row["split"] == split]
    if args.phase == "point":
        options = [count(rows, i) for i in range(len(base.POINTS))]
        allowed = [
            i
            for i, r in enumerate(options)
            if r["total"]["false"] <= 1
            and not sum(r["roles"][role]["false"] for role in ("whole_sequence", "matched_picture_audio"))
        ]
        chosen = max(allowed, key=lambda i: (options[i]["total"]["correct"], -i), default=0)
        base.save(
            folder / "point.json", {"point": base.POINTS[chosen], "options": options, "eligible": allowed}
        )
        return
    point = base.read(folder / "point.json")["point"]
    result = {
        "point": point,
        "baseline_point": OLD_POINT,
        "baseline": count(rows, None),
        "guarded": count(rows, base.POINTS.index(point)),
        "rows": rows,
    }
    base.save(folder / "result.json", result)
    print({key: value for key, value in result.items() if key != "rows"}, flush=True)


if __name__ == "__main__":
    main()
