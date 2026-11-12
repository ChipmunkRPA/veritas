# SPDX-License-Identifier: Apache-2.0
"""Independently reopen 720p30 source clocks, edit recipes and selected face crops."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import video_face_two_returns_hd as study
from scripts.video_common_sources_v89 import read, require, sha, write


def includes_half(observed, truth):
    a, b = observed
    x, y = truth
    return 2*(min(b, y)-max(a, x)) >= max(b, y)-min(a, x)


def digest(rgb):
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def view_anchors(path, receipt, record, output=None):
    from PIL import Image

    from app.detectors.image.faces import Face
    from app.detectors.video import face_return
    from app.detectors.video.decode import decode_frames, probe_frames

    table = probe_frames(path, timeout_s=20)
    encoding = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt",
         "-of", "csv=p=0", str(path)], timeout=10, text=True).strip()
    require(sha(path) == receipt["sha256"] and table.to_json() == receipt["clock"] and encoding == "h264,yuv420p"
            and (table.width, table.height, table.coded_width, table.coded_height) == (1280, 720, 1280, 720)
            and face_return.source_carrier(path) == "mp4", "Delivered source size, codec, carrier or file commitment differs")
    wanted, edge = {}, {}
    if record:
        left, middle, right = record["support"]
        a, b = record["frames"]
        require(left[-1] < a <= middle[0] <= middle[-1] < b <= right[0]
                and all(i % 2 == 0 for i in [a,b,*left,*middle,*right]) and 0 < Fraction(b-a,30) < Fraction(1,2)
                and list(map(Fraction, record["clock_s"])) == [table.pts[i]*table.tb for i in (a,b)],
                "Selected interval disagrees with native measured anchors")
        for label, i in zip(("before", "during", "after"), (left[-1], middle[len(middle)//2], right[0]), strict=True):
            item = record["anchors"][label]
            require(item["index"] == i, "Original frame evidence has the wrong role")
            wanted[i] = label, item
        edge = {item["frame_index"]: item for item in record["scene_inputs"]}
        require(list(edge) == record["scene_frames"] == [a-2,a,b-2,b], "Guarded scene pictures differ")
    count, crops, scenes = 0, {}, {}
    for frame in decode_frames(path, table, range(120), timeout_s=45):
        index, pts, rgb = frame.index, frame.pts, frame.rgb
        h = digest(rgb)
        require(index == count and frame.pts_verified and pts == table.pts[index]
                and (pts-table.pts[0])*table.tb == Fraction(index,30) and h == receipt["rgb_sha256"][index],
                "A delivered picture differs from its original-clock receipt")
        if index in wanted:
            label, item = wanted[index]
            x,y,w,d = item["box"]
            require(item["pts"] == pts and item["rgb_sha256"] == h and 0 <= x < x+w <= table.width
                    and 0 <= y < y+d <= table.height, "Face crop does not reference its exact native frame")
            crop = rgb[y:y+d,x:x+w]
            crops[label] = {"pts": pts, "frame": index, "native_rgb_sha256": h, "crop_rgb_sha256": digest(crop)}
            if output:
                Image.fromarray(rgb).save(output/f"{label}-picture.png")
                Image.fromarray(crop).save(output/f"{label}-crop.png")
        if index in edge:
            item = edge[index]
            require(item["pts"] == pts and item["decoded_rgb_sha256"] == h, "Background input differs from the native picture")
            face = Face(*item["face_box"], 1.0) if item["face_box"] is not None else None
            scenes[index] = face_return.Sample(face, None, face_return.scene(rgb, face))
        count += 1
    require(count == 120 and len(crops) == len(wanted) and set(scenes) == set(edge), "Native source readback is incomplete")
    if record:
        a,b,c,d = record["scene_frames"]
        for pos, (x,y) in enumerate(((a,b),(c,d))):
            median = face_return.scene_correlation(scenes[x], scenes[y])
            lower = face_return.scene_lower_quartile(scenes[x], scenes[y])
            require(median is not None and lower is not None and median >= .55 and lower >= .60
                    and abs(median-record["scene_correlations"][pos]) <= 1e-6
                    and abs(lower-record["scene_lower_quartiles"][pos]) <= 1e-6, "Guarded background measurements differ")
    return count, crops


def rebuild(folder, frozen, media):
    import cv2

    cv2.setNumThreads(1)
    items = {(row["group"],row["condition"]):row for row in media["rows"]}
    pins = frozen["sources"]
    matched = [[-j+1, corr] for corr,j in study.closest_background(pins)]
    require(matched == media["cut_matches"], "Source-only picture-cut ranking differs")
    count = 0
    with TemporaryDirectory(prefix="verify-hd-delivery-", dir=folder) as name:
        for group, pin in enumerate(pins, 1):
            first, second = (range(*span) for span in pin["placements"])
            host = study.native(pin,"original",range(120))
            originals = {**study.native(pin,"swap_a",first), **study.native(pin,"swap_b",second)}
            guest = study.native(pins[matched[group-1][0]-1],"original",tuple(first)+tuple(second))
            for role, replaced in (("two_swaps", originals), ("cross_recording_cuts", guest)):
                receipt = items[group,role]
                require({str(i):digest(rgb) for i,rgb in replaced.items()} == receipt["authored_hashes"]
                        and all(digest(rgb) != digest(host[i]) for i,rgb in replaced.items()),
                        "Known edited source positions differ from their authored native pixels")
                out = Path(name)/receipt["file"]
                study.hd.authored(out, pictures=(replaced.get(i,host[i]) for i in range(120)))
                require(sha(out) == receipt["sha256"], "Native authoring reconstruction differs")
                if role == "two_swaps":
                    tandem = items[group,"tandem_two_swaps"]
                    recoded = Path(name)/tandem["file"]
                    study.hd.authored(recoded, source=out, crf=35)
                    require(sha(recoded) == tandem["sha256"], "Tandem reconstruction changed the delivered bytes")
                    recoded.unlink()
                    count += 1
                out.unlink()
                count += 1
            print("independently reconstructed", group, flush=True)
    return count


def audit(folder):
    import cv2

    cv2.setNumThreads(1)
    frozen, media, aggregate = (read(folder/name) for name in ("freeze.json", "media.json", "aggregate.json"))
    require(frozen == study.plan() and media["freeze_sha256"] == aggregate["freeze_sha256"] == sha(folder/"freeze.json")
            and aggregate["runtime"]["held_sha256"] == sha(folder/"held.jsonl"), "Inputs differ from the frozen comparison")
    counts, examples, pictures, crops = defaultdict(Counter), {}, 0, 0
    with (folder/"held.jsonl").open() as stream:
        for row, source in zip(map(json.loads,stream),media["rows"],strict=True):
            require(all(row[k] == source[k] for k in ("file","group","condition","truth","sha256")), "Measured media binding differs")
            role, current, added = row["condition"], row["baseline"], row["second"]
            pts = source["clock"]["pts"]
            before = [tuple(pts.index(p[k]) for k in ("start_pts","end_pts")) for key in ("coarse","boundary_refinement","short_range")
                      if (p:=current[key])]
            candidate = added["range"]
            if candidate:
                require(current["short_range"] and added["status"] == "measured" and len(added["previous"]) == len(before)
                        and all(end+3 <= candidate["frames"][0] or candidate["frames"][1]+3 <= start for start,end in before)
                        and float.fromhex(candidate["response_hex"]) > .10, "Additional range violates source separation or review point")
            old = [span for span in row["truth"] if any(includes_half(a,span) for a in before)]
            new = [span for span in row["truth"] if span not in old and candidate and includes_half(candidate["frames"],span)]
            counts[role].update(files=1,true_ranges=len(row["truth"]),useful_before=len(old),useful_after=len(old)+len(new),
                                full_dual_before=len(old)==2,full_dual_after=len(old)+len(new)==2,
                                files_flagged_before=bool(before),files_flagged_after=bool(before or candidate),
                                second_assessed=added["status"]=="measured",wrong_additions=bool(candidate and not new))
            target = None
            if new and role not in examples:
                target = folder/f"private-example-{role}"
                target.mkdir(mode=0o700)
            n, anchors = view_anchors(folder/"media"/row["file"],source,candidate,target)
            pictures += n
            crops += len(anchors)
            if target:
                write(target/"exact_source_anchors.json", anchors)
                examples[role] = {"observed_s":candidate["clock_s"],"authored_s":[str(Fraction(i,30)) for i in new[0]],
                                  "anchor_s":[str(Fraction(item["pts"])*Fraction(source["clock"]["time_base"]))
                                              for item in anchors.values()]}
    require(counts == aggregate["strata"] and pictures == 7680, "Independent case totals differ")
    authored = rebuild(folder,frozen,media)
    require(authored == 24, "Independent edit and compression reconstructions incomplete")
    receipt = {"status":"verified","reopened_exact_native_pictures":pictures,"replayed_source_crops":crops,
               "byte_exact_reconstructions":authored,"independently_counted_decisions":64,"examples":examples,
               "aggregate_sha256":sha(folder/"aggregate.json")}
    write(folder/"independent-audit.json",receipt)
    print(receipt,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    args = parser.parse_args()
    audit(args.folder)


if __name__ == "__main__":
    main()
