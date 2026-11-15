# SPDX-License-Identifier: Apache-2.0
"""Verify the bounded face review on 120 exact native pictures at 30000/1001 fps."""

import argparse
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

from scripts import video_face_clip_boundary_transfer as boundary
from scripts.review_video_brief_face_120_carrier import require

RATE = Fraction(30_000, 1001)
SCHEMA = "chipmunk.video-face-fractional-avc/1"
PANELS = Path(__file__).resolve().parents[2] / "docs/evaluation-panels"
REFERENCE = PANELS / "video-face-clip-boundary-held-2026-11-15.json"
GROUPS = {"retimed": 8, "native": 4}
FILTER = "setts=pts=round(PTS*1001/1000):dts=round(DTS*1001/1000):duration=1001"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def retime(previous, output):
    from scripts.video_face_720p30_transfer import describe

    old = previous / "held"
    new = output / "held"
    new.mkdir(parents=True, mode=0o700)
    plan = read(previous / "freeze.json")
    write(output / "freeze.json", plan)
    expected = {(row["group"], row["condition"]): row for row in read(REFERENCE)["rows"]}
    media = []
    for row in read(old / "media.json"):
        source, delivery = old / row["file"], new / row["file"]
        reference = expected.pop((row["group"], row["condition"]))
        require(digest(source) == row["sha256"] == reference["sha256"], "Reference video differs")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-map", "0:v:0", "-c:v", "copy",
                        "-bsf:v", FILTER, "-video_track_timescale", "30000", str(delivery)], check=True, timeout=30)
        measured = describe(delivery, row["group"], row["condition"], row["truth"], rate=RATE)
        require(measured["rgb_sha256"] == row["rgb_sha256"], "A clock-only conversion changed the decoded pixels")
        media.append(measured)
    require(not expected, "Reference deliveries are missing")
    write(new / "media.json", media)


def prepare_native(previous, output):
    output.mkdir(mode=0o700)
    sources = boundary.source_plan(previous / "freeze.json", rate=RATE, groups=GROUPS["native"], development=0)
    write(output / "freeze.json", {"sources": sources, "previous_sha256": digest(previous / "freeze.json")})
    boundary.prepare(output, "held", rate=RATE)


def same_readings(now, prior):
    return all(all(current[key] == earlier[key] for key in ("selected", "measured", "scheduled"))
               and [(x["index"], x["rgb_sha256"]) for x in current.get("anchors", ())]
               == [(x["index"], x["rgb_sha256"]) for x in earlier.get("anchors", ())]
               for current, earlier in zip((now[key] for key in boundary.READINGS),
                                           (prior[key] for key in boundary.READINGS), strict=True))


def review(record):
    require(record["schema"] == SCHEMA and record["frame_rate"] == str(RATE) and record["reference_sha256"] == digest(REFERENCE),
            "The fractional clock or reference differs")
    expected = read(REFERENCE)
    panels = record["panels"]
    require(set(panels) == set(GROUPS), "A transfer panel is missing")
    totals = {}
    for label, group_count in GROUPS.items():
        panel = panels[label]["measurements"]
        totals[label] = boundary.review(panel, rate=RATE, group_count=group_count)
        anchors = sum(len({item["index"] for reading in row["readings"].values() for item in reading.get("anchors", ())})
                      for row in panel["rows"])
        files = len(panel["rows"])
        require(panels[label]["independent_decode"] == {"native_times_verified": 120 * files, "files": files,
                                                        "verified_anchors": anchors}, "Independent decode count differs")
    current = panels["retimed"]["measurements"]
    require(current["groups"] == expected["groups"] and current["weights"] == expected["weights"], "Reference roster or weights differ")
    require(all((now["group"], now["condition"], now["truth"]) == (prior["group"], prior["condition"], prior["truth"])
                and same_readings(now["readings"], prior["readings"])
                for now, prior in zip(current["rows"], expected["rows"], strict=True)),
            "Changed pictures or proposal selections on the fractional clock")
    ids = {group["index"] for group in current["groups"]}
    require(ids.isdisjoint(group["index"] for group in panels["native"]["measurements"]["groups"]), "A native source repeated")
    require(record["identical_decoded_pictures"] == 120 * len(current["rows"]), "Incomplete pixel comparison")
    return totals


def publish(root, output):
    record = {"schema": SCHEMA, "frame_rate": str(RATE), "reference_sha256": digest(REFERENCE),
              "panels": {label: {"measurements": read(root / label / "held.json"),
                        "independent_decode": boundary.pixels(root / label, "held", rate=RATE, group_count=count)}
                         for label, count in GROUPS.items()}, "identical_decoded_pictures": 120 * GROUPS["retimed"] * len(boundary.CASES)}
    review(record)
    write(output, record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase, names in (("retime", ("previous", "output")), ("native", ("previous", "output")),
                         ("publish", ("root", "output")), ("review", ("receipt",))):
        command = phases.add_parser(phase)
        for name in names:
            command.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.phase == "review":
        print(json.dumps(review(read(args.receipt)), indent=2, sort_keys=True))
    elif args.phase == "publish":
        publish(args.root, args.output)
    else:
        label = "retimed" if args.phase == "retime" else "native"
        action = retime if label == "retimed" else prepare_native
        action(args.previous, args.output)
        boundary.score(args.output, "held", rate=RATE, group_count=GROUPS[label])
