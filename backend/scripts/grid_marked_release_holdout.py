# SPDX-License-Identifier: Apache-2.0
"""Test marked waveform fallback on different GRID actors and newly authored shifts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from contextlib import closing
from fractions import Fraction
from pathlib import Path, PurePosixPath
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from app.evidence import video_reference as route
from app.evidence import video_reference_timing as timing
from scripts import grid_fresh_release_labels as labels
from scripts import grid_marked_release_transfer as transfer
from scripts import grid_paired_closure_panel as video
from scripts import grid_phone_alignment_v161 as words
from scripts import grid_reserve_neural_mouth as mesh
from scripts import grid_reserve_word_clocks as grid
from scripts.review_grid_single_clip_release_timing import RGB_SIZE, digest, native, read, sha
from scripts.video_common_sources_v89 import write

SCHEMA = "chipmunk.grid-marked-release-holdout/1"
GROUPS = tuple(range(1, 7))
WIDTHS = ((4480, 5120), (5120, 7040))
ROLES = (("original", 0), ("gain", 0), ("audio_early", -3840), ("audio_late", 3840))
POLICY = {
    "source": "Distinct pilot actor roles outside the prior marked timing panel. First two unused exact /p/ utterances in hash order. These speaker identities have project exposure.",
    "mark": "First of the two selected originals with an eligible muted closure bracket and enough room for both authored shifts. No acoustic event supplied.",
    "spans": "Even and odd actor groups alternate between 600 ms and 760 ms local edit ranges on the native clock.",
    "fallback": "Keep every assessed production disposition. Invoke the fixed short template only after an unassessed support outcome.",
    "useful": "Signed result within 80 ms and the entire displayed audio template within the exact shifted sample range.",
    "advance": "Preserve all useful production results. Add at least two different actor groups with no new control alerts.",
}


def code():
    return {
        path.name: digest(path)
        for path in map(Path, (__file__, transfer.__file__, timing.__file__, route.__file__))
    }


def freeze(panel, run):
    _, plan = grid.plan()
    frozen = read(panel / "freeze.json")
    assert frozen["prior_plan_sha256"] == plan
    archive_plan = grid.planning.TRIAL / "bind-00/output/private-plan.json"
    assert digest(archive_plan) == plan
    all_rows = read(archive_plan)["sources"]
    used = {PurePosixPath(row["input"]["planned"]["member"]["name"]).stem for row in all_rows}
    used.update(PurePosixPath(row["member"]).stem for row in frozen["roster"])
    assert not set(GROUPS) & {row["group"] for row in frozen["roster"]}
    assert (
        grid.publisher_record()["alignments.zip"]["checksum"]
        == "md5:" + hashlib.md5(words.ALIGNMENTS.read_bytes(), usedforsecurity=False).hexdigest()
    )
    run.mkdir(mode=0o700)
    sources = run / "sources"
    sources.mkdir(mode=0o700)
    roster = []
    with ZipFile(words.ALIGNMENTS) as alignment:
        directories = defaultdict(dict)
        for item in alignment.infolist():
            path = PurePosixPath(item.filename)
            if (
                len(path.parts) == 3
                and path.parts[0] == "alignments"
                and path.suffix == ".align"
                and not path.name.startswith("._")
            ):
                directories[path.parent.name][path.stem] = item
        for group in GROUPS:
            actor = next(
                row for row in all_rows if row["component_index"] == group and row["role"] == "pilot"
            )
            _, matched, members = grid.inventory(actor, directories)
            archive = (
                grid.planning.ARCHIVE
                / "grid-acquisition-continuation-v135"
                / f"source-{actor['source_slot']:02}"
                / "archive.zip"
            )
            chosen = 0
            with ZipFile(archive) as movie:
                for stem in sorted(
                    members.keys() - used, key=lambda s: sha(f"grid-marked-held-v1/{group}/{s}".encode())
                ):
                    aligned = alignment.read(directories[matched][stem])
                    intervals, total = words.publisher_words(aligned)
                    at = next(
                        (a for word, a, _ in intervals if word == "please" and 38500 <= a <= 57750), None
                    )
                    if at is None:
                        continue
                    assert total == 74500
                    member = members[stem]
                    raw = movie.read(member)
                    chosen += 1
                    path = sources / f"g{group:02}-{chosen}.mpg"
                    path.write_bytes(raw)
                    roster.append(
                        {
                            "group": group,
                            "order": chosen,
                            "file": str(path),
                            "file_sha256": sha(raw),
                            "archive": str(archive),
                            "member": member.filename,
                            "member_crc32": member.CRC,
                            "word_start": at,
                            "word_sha256": sha(aligned),
                        }
                    )
                    used.add(stem)
                    if chosen == 2:
                        break
            assert chosen == 2
    write(
        run / "freeze.json",
        {
            "schema": SCHEMA,
            "policy": POLICY,
            "code": code(),
            "panel_sha256": digest(panel / "freeze.json"),
            "archive_plan_sha256": plan,
            "widths": WIDTHS,
            "roster": roster,
        },
    )


def locked(run):
    data = read(run / "freeze.json")
    assert data["code"] == code() and data["policy"] == POLICY
    return data["roster"]


def author(run):
    module, task = mesh.mp_task()
    originals, exclusions = {}, []
    with closing(task):
        for row in locked(run):
            group = row["group"]
            if group in originals:
                continue
            path = Path(row["file"])
            assert digest(path) == row["file_sha256"]
            _, frames, pcm = video.video(path)
            geometry = [timing._mesh(task, module, frame) for frame in frames]
            proof = labels.muted_video_only(
                [None if item is None else item[0] for item in geometry],
                round(Fraction(row["word_start"] * 16000, 25000)),
            )
            if proof["status"] != "observed_reopening":
                exclusions.append({"group": group, "order": row["order"], "status": proof["status"]})
                continue
            mark = proof["bracket_frames"][0]
            before, after = WIDTHS[group % len(WIDTHS)]
            start, end = mark * 640 - before, mark * 640 + after
            if start < 3840 or end + 3840 > len(pcm):
                exclusions.append({"group": group, "order": row["order"], "status": "outside_sample_bounds"})
                continue
            originals[group] = row | {"mark": mark, "span": [start, end], "pcm_sha256": sha(pcm.tobytes())}
    write(
        run / "marks.json",
        {"freeze_sha256": digest(run / "freeze.json"), "originals": originals, "exclusions": exclusions},
    )
    media = run / "media"
    media.mkdir(mode=0o700)
    cases = []
    for group, row in originals.items():
        source = Path(row["file"])
        _, pictures, pcm = video.video(source)
        assert sha(pcm.tobytes()) == row["pcm_sha256"]
        a, b = row["span"]
        for role, shift in ROLES:
            output, edited = source, pcm.copy()
            if role != "original":
                edited[a:b] = pcm[a - shift : b - shift] if shift else np.rint(pcm[a:b] * 0.80)
                output = media / f"g{group:02}-{role}.mkv"
                wav = output.with_suffix(".wav")
                video.save_pcm(wav, edited)
                subprocess.run(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-v",
                        "error",
                        "-i",
                        str(source),
                        "-i",
                        str(wav),
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-c:v",
                        "copy",
                        "-c:a",
                        "pcm_s16le",
                        str(output),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=15,
                )
                wav.unlink()
                _, stored_frames, stored_pcm = video.video(output)
                assert np.array_equal(edited, stored_pcm)
                assert all(np.array_equal(x, y) for x, y in zip(pictures, stored_frames, strict=True))
            cases.append(
                {
                    "group": group,
                    "role": role,
                    "shift": shift,
                    "span": row["span"],
                    "mark": row["mark"],
                    "file": str(output),
                    "file_sha256": digest(output),
                    "pcm_sha256": sha(edited.tobytes()),
                }
            )
    write(run / "media.json", {"marks_sha256": digest(run / "marks.json"), "cases": cases})
    print("Marked actors and authored deliveries", len(originals), len(cases), flush=True)


def score(run):
    locked(run)
    marks = read(run / "marks.json")
    media = read(run / "media.json")
    assert marks["freeze_sha256"] == digest(run / "freeze.json") and media["marks_sha256"] == digest(
        run / "marks.json"
    )
    sys.path.insert(0, str(mesh.runtime.SITE))
    model = mesh.ASSETS / "permissions/face_landmarker.task"
    rows = []
    for row in media["cases"]:
        source = marks["originals"][str(row["group"])]
        paths = Path(source["file"]), Path(row["file"])
        assert [digest(path) for path in paths] == [source["file_sha256"], row["file_sha256"]]
        opts = {"include_local": False, "timing_event_s": Fraction(row["mark"], 25), "timing_model": model}
        first, assets = route.compare(*paths, **opts)
        transfer.check_assets(first["timing"], assets)
        with patch.object(timing, "audio_shift", transfer.short_audio):
            second, assets = route.compare(*paths, **opts)
        transfer.check_assets(second["timing"], assets)
        assert first["picture"]["matching_frames"] == second["picture"]["matching_frames"] == 75
        rows.append(
            {
                "group": row["group"],
                "role": row["role"],
                "current": first["timing"],
                "short": second["timing"],
            }
        )
    write(run / "readout.json", {"media_sha256": digest(run / "media.json"), "rows": rows})


def verify_result(record, decoded, case, arm):
    if record["status"] == "unassessed":
        return False, False, False
    starts, frames = [], []
    for side, (pts, tb, rgb, pcm) in zip(("earlier", "submitted"), decoded, strict=True):
        proof = record[side]
        k, (a, b) = proof["frame_index"], proof["pcm_samples"]
        raw = rgb[k * RGB_SIZE : (k + 1) * RGB_SIZE]
        x0, y0, x1, y1 = proof["crop_xyxy"]
        crop = np.frombuffer(raw, np.uint8).reshape(288, 360, 3)[y0:y1, x0:x1]
        assert proof["picture_pts"] == pts[k] and Fraction(proof["picture_time_base"]) == tb
        assert pts[k] * tb == Fraction(k, 25)
        assert (
            sha(raw) == proof["native_picture_rgb_sha256"]
            and sha(crop.tobytes()) == proof["native_crop_rgb_sha256"]
        )
        assert sha(pcm[a:b].tobytes()) == proof["pcm_s16_sha256"] and b - a == (
            960 if arm == "short" else 2560
        )
        starts.append(a)
        frames.append(k)
    assert frames[0] == case["mark"] and not record["affects_overall_confidence"]
    lag = 640 * (frames[1] - frames[0]) - (starts[1] - starts[0])
    assert abs(lag - record["offset_ms"] * 16) < 1e-7
    alert = record["status"] == "timing_difference"
    assert alert == (abs(lag) >= 1920)
    exact = bool(case["shift"] and alert and lag == -case["shift"])
    lo, hi = case["span"]
    a, b = record["submitted"]["pcm_samples"]
    useful = bool(case["shift"] and alert and abs(lag + case["shift"]) <= 1280 and lo <= a < b <= hi)
    return alert, useful, exact


def review(run, public):
    roster = locked(run)
    marks, media, outcome = (read(run / (name + ".json")) for name in ("marks", "media", "readout"))
    assert marks["freeze_sha256"] == digest(run / "freeze.json")
    assert media["marks_sha256"] == digest(run / "marks.json") and outcome["media_sha256"] == digest(
        run / "media.json"
    )
    for row in roster:
        with ZipFile(row["archive"]) as archive:
            info = archive.getinfo(row["member"])
            assert (
                row["member_crc32"] == info.CRC
                and sha(archive.read(info)) == digest(Path(row["file"])) == row["file_sha256"]
            )
    counts, additions, totals = defaultdict(Counter), set(), Counter()
    for case, result in zip(media["cases"], outcome["rows"], strict=True):
        assert (case["group"], case["role"]) == (result["group"], result["role"])
        earlier = marks["originals"][str(case["group"])]
        decoded = native(Path(earlier["file"])), native(Path(case["file"]))
        assert decoded[0][2] == decoded[1][2]
        source, submitted = decoded[0][3], decoded[1][3]
        assert (
            sha(source.tobytes()) == earlier["pcm_sha256"] and sha(submitted.tobytes()) == case["pcm_sha256"]
        )
        a, b = case["span"]
        shift = case["shift"]
        expected = source.copy()
        if case["role"] != "original":
            expected[a:b] = source[a - shift : b - shift] if shift else np.rint(source[a:b] * 0.80)
        assert np.array_equal(expected, submitted)
        current, short = (verify_result(result[arm], decoded, case, arm) for arm in ("current", "short"))
        fallback = short if result["current"].get("reason") == "support" else current
        if fallback[1] and not current[1]:
            additions.add(case["group"])
        assert not current[1] or fallback[1]
        entry = counts[case["role"]]
        entry["views"] += 1
        for label, values in (("current", current), ("short", short), ("fallback", fallback)):
            for metric, value in zip(("alerts", "useful", "exact"), values, strict=True):
                entry[f"{label}_{metric}"] += value
        totals["native_readback_pairs"] += 1
        totals["replayed_dispositions"] += 3
        totals["image_crop_and_pcm_records"] += 2 * sum(
            result[arm]["status"] != "unassessed" for arm in ("current", "short")
        )
    result = {
        "schema": SCHEMA,
        "policy": POLICY,
        "source_actors": len(GROUPS),
        "marked_actors": len(marks["originals"]),
        "new_actor_groups_localized": len(additions),
        "control_alerts_added": sum(
            counts[role]["fallback_alerts"] - counts[role]["current_alerts"] for role in ("original", "gain")
        ),
        "by_role": dict(counts),
        "audit": dict(totals),
        "freeze_sha256": digest(run / "freeze.json"),
        "readout_sha256": digest(run / "readout.json"),
    }
    write(public, result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "author", "score", "review"))
    parser.add_argument("panel", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--public", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.panel, args.run)
    elif args.phase == "review":
        assert args.public is not None
        review(args.run, args.public)
    else:
        {"author": author, "score": score}[args.phase](args.run)


if __name__ == "__main__":
    main()
