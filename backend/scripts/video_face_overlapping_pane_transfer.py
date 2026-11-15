# SPDX-License-Identifier: Apache-2.0
"""Evaluate coincident brief changes on two independently measured face tracks."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.review_video_brief_face_120_carrier import SHA, require
from scripts.video_face_opposite_pane_transfer import GROUPS, PARENT, SIDES, checked_proposal, run, sha
from scripts.video_face_primary_track_transfer import replay_pixels

SCHEMA = "chipmunk.video-face-overlapping-pane/1"
MIN_OVERLAP = 3
CASES = ("aligned", "staggered", "left_only", "right_only", "unchanged", "speaker_cut", "light_pulse")
PROTOCOL = (
    "Keep the existing first and same-side or separated opposite-side ranges. Admit an independently "
    "guarded proposal on the opposite side only when it overlaps at least three native frames and "
    "the existing follow-up is empty. Use fixed face and surrounding-motion points. Require at least "
    "eight correct additional intervals across four source groups, no additional control ranges and "
    "no misplaced added interval on edited files."
)


def composition(parent_rows, group, condition):
    index = GROUPS.index(group)
    peer = GROUPS[(index + 3) % len(GROUPS)]
    a, b = parent_rows[group, "swaps_left"]["truth"][0]
    c, d = parent_rows[peer, "swaps_left"]["truth"][0]
    shift = a - c + (3 if condition == "staggered" and index % 2 else -3 if condition == "staggered" else 0)
    changed = {"aligned", "staggered"}
    intervals = []
    if condition in changed | {"left_only"}:
        intervals.append({"pane": "left", "frames": [a, b]})
    if condition in changed | {"right_only"}:
        intervals.append({"pane": "right", "frames": [c + shift, d + shift]})
    return [group, peer], shift, intervals


def compose_frames(maps, actors, shift, edits, condition, light_range):
    import numpy as np

    for i in range(120):
        panes = []
        for side, actor, offset in zip(SIDES, actors, (0, shift), strict=True):
            start, stop = edits.get(side, (-1, -1))
            kind = "swaps_left" if start <= i < stop else "original"
            if condition == "speaker_cut":
                kind = "ordinary_speaker_cut"
            panes.append(maps[actor][kind][max(0, min(119, i - offset))])
        frame = np.concatenate(panes, axis=1)
        if condition == "light_pulse" and light_range[0] <= i < light_range[1]:
            frame = np.minimum(frame.astype(np.uint16) + 24, 255).astype(np.uint8)
        yield frame


def prepare(parent, bank, output):
    from app.detectors.video.decode import FrameTable, decode_frames
    from scripts.video_face_720p30_transfer import authored, describe

    manifest = bank / "media.json"
    require(sha(manifest) == parent["media_sha256"], "Earlier pane bank differs")
    inputs = {(r["group"], r["condition"]): r for r in json.loads(manifest.read_bytes())["rows"]}
    output.mkdir(mode=0o700)
    media = output / "media"
    media.mkdir(mode=0o700)

    def pixels(actor, kind):
        row = inputs[actor, kind]
        path = bank / "media" / row["file"]
        table = FrameTable.from_json(row["clock"])
        require(sha(path) == row["sha256"], "Parent clip differs")
        frames = []
        for frame in decode_frames(path, table, range(120), timeout_s=40):
            require(frame.pts_verified and frame.pts == table.pts[frame.index]
                    and hashlib.sha256(frame.rgb).hexdigest() == row["rgb_sha256"][frame.index],
                    "Parent picture differs")
            frames.append(frame.rgb[:, :640].copy())
        require(len(frames) == 120, "Missing source pictures")
        return frames

    rows = []
    for group in GROUPS:
        actors, _, _ = composition(inputs, group, "aligned")
        maps = {actor: {kind: pixels(actor, kind) for kind in ("original", "swaps_left", "ordinary_speaker_cut")}
                for actor in actors}
        for condition in CASES:
            actors, shift, truth = composition(inputs, group, condition)
            edits = {row["pane"]: row["frames"] for row in truth}
            path = media / f"g{group}-{condition}.mp4"
            authored(path, pictures=compose_frames(maps, actors, shift, edits, condition, inputs[group, "swaps_left"]["truth"][0]))
            rows.append(describe(path, group, condition, truth) | {"participants": actors, "right_shift": shift})
            print("Prepared", path.name, flush=True)
    (output / "media.json").write_text(json.dumps({"parent": sha(manifest), "rows": rows}, indent=2) + "\n")


def candidate(row):
    first = row["first"]
    if not first or row["same"] or row["opposite"]:
        return None
    side = "right" if first["pane"] == "left" else "left"
    other = row["methods"][f"{side}_guarded"]["proposal"]
    if other is not None:
        start, end = first["frames"]
        a, b = other["frames"]
        if min(end, b) - max(start, a) >= MIN_OVERLAP:
            return {**other, "pane": side}
    return None


def readout(rows, parent):
    require([(row["group"], row["condition"]) for row in rows] == [(g, case) for g in GROUPS for case in CASES],
            "The frozen clip roster differs")
    stored = {(row["group"], row["condition"]): row for row in parent["rows"]}
    counts = {case: Counter(files=0, known=0, baseline_exact=0, expanded_exact=0,
                            baseline_extras=0, added_exact=0, added_wrong=0) for case in CASES}
    sources = set()
    for row in rows:
        actors, shift, truth = composition(stored, row["group"], row["condition"])
        require(row["truth"] == truth and row["participants"] == actors and row["right_shift"] == shift
                and SHA.fullmatch(row["sha256"]), "Edit timing differs")
        reported = [row[key] for key in ("first", "same", "opposite") if row[key]]
        added = candidate(row)
        require(row["overlapping_opposite"] == added, "The selected overlapping range differs")
        for record in (*reported, *([added] if added else [])):
            checked_proposal(record, row)
            require(len(record["pane_tiles"]) == 2 and all(0 <= n < 8 for n in record["pane_tiles"]),
                    "A proposed range lacks its own pane motion support")
        hits = sum({"pane": item["pane"], "frames": item["frames"]} in truth for item in reported)
        exact = added is not None and {"pane": added["pane"], "frames": added["frames"]} in truth
        counts[row["condition"]].update(files=1, known=len(truth), baseline_exact=hits, expanded_exact=hits + exact,
                                        baseline_extras=len(reported) - hits, added_exact=exact,
                                        added_wrong=added is not None and not exact)
        if exact:
            sources.add(row["group"])
    gains = sum(c["added_exact"] for c in counts.values())
    extras = sum(c["added_wrong"] for c in counts.values())
    return {"cases": counts, "additional_source_groups": len(sources),
            "qualifies": gains >= 8 and len(sources) >= 4 and extras == 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "review", "pixels"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--bank", type=Path)
    args = parser.parse_args()
    parent = json.loads(PARENT.read_bytes())
    if args.action == "prepare":
        prepare(parent, args.bank, args.path)
        return
    freeze = {"schema": SCHEMA, "protocol": PROTOCOL, "parent_sha256": sha(PARENT), "weights": parent["weights"]}
    if args.action == "run":
        manifest = args.bank / "media.json"
        prepared = json.loads(manifest.read_bytes())
        rows = run(args.bank, parent)
        for row, item in zip(rows, prepared["rows"], strict=True):
            require(row["sha256"] == item["sha256"], "Measured input differs from freeze")
            row.update(right_shift=item["right_shift"], overlapping_opposite=candidate(row))
        receipt = {**freeze, "media_sha256": sha(manifest), "rows": rows, "result": readout(rows, parent)}
        with args.path.open("x") as output:
            print(json.dumps(receipt, sort_keys=True, indent=2), file=output)
    else:
        receipt = json.loads(args.path.read_bytes())
        require(all(value == receipt[key] for key, value in freeze.items())
                and SHA.fullmatch(receipt["media_sha256"])
                and receipt["result"] == readout(receipt["rows"], parent), "Receipt does not replay")
        if args.action == "pixels":
            manifest = args.bank / "media.json"
            require(sha(manifest) == receipt["media_sha256"], "Composed deliveries differ")
            paths = {row["sha256"]: args.bank / "media" / row["file"] for row in json.loads(manifest.read_bytes())["rows"]}
            keys = ("first", "same", "opposite", "overlapping_opposite")
            rows = [row | {"methods": {key: {"proposal": row[key]} for key in keys}} for row in receipt["rows"]]
            print(json.dumps(replay_pixels(rows, paths), sort_keys=True))
    print(json.dumps(receipt["result"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
