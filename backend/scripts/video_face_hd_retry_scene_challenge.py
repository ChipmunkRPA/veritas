# SPDX-License-Identifier: Apache-2.0
"""Challenge the unpublished 720p sparse route with ordinary same-layout cuts."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import acquire_grid_transfer_v130 as grid
from scripts import review_video_face_hd_sparse_retry as prior_review
from scripts import review_video_face_two_returns_hd as review
from scripts import video_face_hd_sparse_retry as retry
from scripts import video_face_return_eligibility as spans
from scripts.video_common_sources_v89 import read, require, sha, write

hd, earlier, worker = retry.hd, retry.earlier, retry.worker
PREVIOUS = retry.PREVIOUS.parent.parent / "video-face-hd-sparse-retry-2026-11-12-v2/freeze.json"
GRID_RECEIPTS = grid.BASE / "grid-native-inputs-v137"
GRID_ROLES = ("ordinary", "person_switch", "tandem_person_switch", "same_person_jump", "same_person_reframe")
SWAP_ROLES = ("original", "publisher_swap", "tandem_publisher_swap")
PLACEMENTS = ((63, 74), (85, 96))
SPEC = {
    "schema": "chipmunk.video-face-hd-retry-scene-challenge/1",
    "delivery": [1280, 720, 30, 120, "h264", "yuv420p", "mp4"],
    "placements": list(map(list, PLACEMENTS)),
    "placement_rule": "Both half-open intervals contain three 8 Hz samples and no fourth. Alternate phases by cohort ordinal.",
    "genuine_controls": "Pair sixteen separately archived genuine GRID speakers in archive order. Compose a whole centered "
                        "webcam picture on a fixed geometric surround. Test ordinary viewing, a different speaker cutaway, "
                        "tandem compression of that cutaway, earlier motion of the same speaker, and a same-person punch-in.",
    "genuine_clock": "From native 25 Hz picture i use floor(n*5/6)-12 for delivery frame n with reflection at the "
                     "source edges. The genuine guest uses the same rule. No generated intermediate pictures.",
    "webcam_tile": {"xywh": [360, 136, 560, 448], "reframe_source_crop": [24, 20, 336, 270], "jump_delivery_frames": 18},
    "surround": "Deterministic fixed low-contrast geometric slide texture supplies unchanged border tiles.",
    "swaps": "All seven remaining exact-clock archival RTFS groups separated by known host URL and donor from recent "
             "HD brief-range and retry panels. Use one publisher swap, alternating family, and its tandem encode.",
    "crf": [23, 35],
    "gate": retry.SPEC["gate"],
    "maximum_media_bytes": 100 * 1024**2,
}


def plan():
    import cv2

    from app.detectors.video import face_return
    from app.detectors.video.adapters import face_appearance
    from scripts import video_face_six_picture_transfer as panel

    record = grid.METADATA / "grid-record.json"
    require(sha(record) == grid.RECORD_SHA, "GRID publisher metadata differs")
    require(read(record)["metadata"]["license"]["id"] == "cc-by-4.0", "GRID license differs")
    ordinary = []
    for number in range(16):
        receipt = GRID_RECEIPTS / f"source-{number:02}/bind-00/output/native-inputs.json"
        entry = read(receipt)[0]
        require(entry["status"] == "native_video_and_audio_available" and len(entry["pictures"]) == 75
                and entry["source_license"] == "CC-BY-4.0", "GRID source qualification differs")
        ordinary.append({"source": number, "receipt_sha256": sha(receipt), "source_sha256": entry["source_sha256"]})
    require(len({r["source_sha256"] for r in ordinary}) == 16, "Genuine recordings must be distinct")

    archive, common, known, _, history = hd.plan_sources()
    previous = (earlier.EARLIER, retry.PREVIOUS, PREVIOUS)
    excluded = set(known) | {r["index"] for path in previous for r in read(path)["sources"]}
    documents = [read(p) for p in sorted(common.DATA.glob("group-*-complete.json"))]
    metadata = panel.trial.prior.comparison.metadata()
    exposed = [metadata[r["group"]] for r in documents if r["index"] in excluded]
    urls = {r["url"] for r in exposed}
    donors = set().union(*(r["donors"] for r in exposed))
    swaps = []
    for row in documents:
        meta = metadata[row["group"]]
        if meta["url"] in urls or meta["donors"] & donors or "child" in meta["tags"]:
            continue
        cases = {r["condition"]: r for r in row["cases"]}
        table = panel.FrameTable.from_json(cases["original"]["frame_table"])
        span = PLACEMENTS[len(swaps) % len(PLACEMENTS)]
        role = ("swap_a", "swap_b")[len(swaps) % 2]
        if ((table.width, table.height) not in ((1280, 720), (1920, 1080)) or len(table) < 120
                or hd.relative(table) != [Fraction(i, 30) for i in range(len(table))]
                or not set(range(*span)) <= set(cases[role]["actual_changed_frames"])):
            continue
        pin = panel.encoded.prior.archive_io.source_pin(row["index"], archive)
        swaps.append({"pin": pin, "span": list(span), "role": role})
        urls.add(meta["url"])
        donors.update(meta["donors"])
    require(len(swaps) == 7, "The seven source-separated RTFS groups are required")
    bindings = [Path(module.__file__) for module in (face_return, face_appearance, worker, earlier, hd, retry, review)]
    return {"protocol": SPEC, "ordinary": ordinary, "swaps": swaps, "history": [*history, *map(sha, previous)],
            "publisher": sha(record), "models": common.bindings(), "opencv": cv2.__version__,
            "code": {p.name: sha(p) for p in [Path(__file__), *bindings]}}


def genuine_frames(pin):
    import numpy as np

    from app.detectors.video.decode import decode_frames, probe_frames

    receipt = GRID_RECEIPTS / f"source-{pin['source']:02}/bind-00/output/native-inputs.json"
    require(sha(receipt) == pin["receipt_sha256"], "GRID receipt changed")
    row = read(receipt)[0]
    source = Path(row["source_file"])
    require(sha(source) == pin["source_sha256"], "GRID source changed")
    table = probe_frames(source, timeout_s=15)
    require(table.to_json() == row["native_video_table"] and (table.width, table.height) == (360, 288)
            and hd.relative(table) == [Fraction(i, 25) for i in range(75)]
            and not (table.rotation or table.interpolated or table.truncated), "GRID source clock changed")
    rgb = []
    for frame in decode_frames(source, table, range(75), timeout_s=25):
        proof = row["pictures"][frame.index]
        require(frame.pts_verified and frame.index == len(rgb) == proof["frame_index"] and frame.pts == proof["pts"]
                and review.digest(frame.rgb) == proof["rgb_sha256"], "GRID source picture changed")
        rgb.append(frame.rgb)
    require(len(rgb) == 75, "Incomplete GRID source pictures")
    return np.stack(rgb)


def poster():
    import numpy as np

    y, x = np.ogrid[:720, :1280]
    tone = 62 + ((x//15+y//21) % 2)*24 + ((x//41+2*(y//53)) % 3)*12 + (x//320)*7
    return np.stack([tone, tone+9, tone+17], axis=-1).astype("uint8")


def composed(host, guest, span, role):
    import cv2

    still = poster()
    x, y, w, h = SPEC["webcam_tile"]["xywh"]
    left, top, right, bottom = SPEC["webcam_tile"]["reframe_source_crop"]
    for n in range(120):
        in_span = span[0] <= n < span[1]
        k = n - SPEC["webcam_tile"]["jump_delivery_frames"] if in_span and role == "same_person_jump" else n
        k = (k*5)//6 - 12
        k = abs(k) if k < 0 else min(k, 148-k)
        picture = guest[k] if in_span and role == "person_switch" else host[k]
        if in_span and role == "same_person_reframe":
            picture = picture[top:bottom, left:right]
        canvas = still.copy()
        canvas[y:y+h, x:x+w] = cv2.resize(picture, (w, h), interpolation=cv2.INTER_LINEAR)
        yield canvas


def describe(path, group, role, truth, authored_span):
    return {**hd.describe(path, group, role, truth), "authored_span": authored_span}


def prepare(folder, frozen):
    import cv2
    import numpy as np

    cv2.setNumThreads(1)
    media = folder/"media"
    media.mkdir(mode=0o700)
    rows = []
    nbytes = 0
    ordinary = frozen["ordinary"]
    for group, (first, second) in enumerate(zip(ordinary[::2], ordinary[1::2], strict=True), 1):
        host, guest = genuine_frames(first), genuine_frames(second)
        span = PLACEMENTS[(group-1) % len(PLACEMENTS)]
        for role in GRID_ROLES:
            path = media/f"o{group}-{role}.mp4"
            if role == "tandem_person_switch":
                hd.authored(path, source=media/f"o{group}-person_switch.mp4", crf=35)
            else:
                hd.authored(path, pictures=composed(host, guest, span, role))
            rows.append(describe(path, group, role, [], None if role == "ordinary" else list(span)))
            nbytes += path.stat().st_size
        print("prepared ordinary group", group, flush=True)
        require(nbytes <= SPEC["maximum_media_bytes"], "Encoded delivery limit exceeded")
    for group, row in enumerate(frozen["swaps"], 1):
        pin, span = row["pin"], row["span"]
        host = earlier.native(pin, "original", range(120))
        changed = earlier.native(pin, row["role"], range(*span))
        require(all(not np.array_equal(host[i], rgb) for i, rgb in changed.items()), "A known changed picture did not change")
        for role in SWAP_ROLES:
            path = media/f"p{group}-{role}.mp4"
            if role == "tandem_publisher_swap":
                hd.authored(path, source=media/f"p{group}-publisher_swap.mp4", crf=35)
            else:
                hd.authored(path, pictures=(changed[i] if role == "publisher_swap" and i in changed else host[i] for i in range(120)))
            edited = role != "original"
            rows.append(describe(path, group, role, [span] if edited else [], span if edited else None))
            nbytes += path.stat().st_size
        print("prepared swap group", group, flush=True)
        require(nbytes <= SPEC["maximum_media_bytes"], "Encoded delivery limit exceeded")
    require(len(rows) == len({r["sha256"] for r in rows}) == 61, "Missing or repeated deliveries")
    write(folder/"media.json", {"freeze_sha256": sha(folder/"freeze.json"), "media_bytes": nbytes, "rows": rows})


def exact_media(folder):
    lock, ledger = read(folder/"freeze.json"), read(folder/"media.json")
    require(lock == plan() and ledger["freeze_sha256"] == sha(folder/"freeze.json"), "Frozen inputs differ")
    return ledger


def screen(folder):
    import cv2

    from app.detectors.image.faces import load_yunet
    from app.detectors.video import face_return
    from app.detectors.video.adapters import face_appearance as adapter
    from app.detectors.video.adapters.face_sequence import SequenceInputs
    from app.detectors.video.decode import FrameTable
    from app.detectors.video.frame_store import FrameStore

    cv2.setNumThreads(2)
    rows = exact_media(folder)["rows"]
    weights = grid.ROOT/"models/weights"
    detector = load_yunet(weights/"face_detection_yunet_2023mar.onnx")
    recognizer = cv2.FaceRecognizerSF.create(str(weights/"face_recognition_sface_2021dec.onnx"), "")
    began = time.monotonic()
    screened = []
    for row in rows:
        path = folder/"media"/row["file"]
        require(sha(path) == row["sha256"], "Sparse input differs from the source receipt")
        table = FrameTable.from_json(row["clock"])
        inputs = SequenceInputs(table, FrameStore(path, table, max_bytes=32*1024**2), (), (), ("h264", "yuv420p"))
        schedule = face_return.guarded_720p30_schedule(table, inputs.encoding, path)
        require(schedule is not None, "Sparse profile differs")
        details, state = {}, {"measured_face_frames": 0}
        adapter._return_pass(details, inputs, detector, recognizer, None, None, schedule, state,
                             face_return.METHOD, time.monotonic()+adapter.RETURN_BUDGET_S)
        record = details.get("face_return")
        screened.append({"file": row["file"], "sha256": row["sha256"], "state": state,
                         "sparse_frames": None if record is None else list(spans.interval(record, table.pts))})
        print("sparse", path.name, state["status"], state["reason"], flush=True)
    write(folder/"sparse.json", {"media_sha256": sha(folder/"media.json"), "rows": screened, "elapsed_s": time.monotonic()-began})


def score(folder):
    state = worker.configure(folder)
    os.environ["VERITAS_VIDEO_FACE_SECOND_RETURN_REVIEW"] = "true"
    rows = exact_media(folder)["rows"]
    sparse = read(folder/"sparse.json")
    require(sparse["media_sha256"] == sha(folder/"media.json")
            and all((r["file"], r["sha256"]) == (s["file"], s["sha256"])
                    for r, s in zip(rows, sparse["rows"], strict=True)), "Sparse qualification differs")
    run = worker.scorer(state)
    began = time.monotonic()
    with (folder/"held.jsonl").open("x") as output:
        for row in rows:
            answer = retry.test_file(run, folder/"media"/row["file"], row)
            output.write(json.dumps({k: row[k] for k in ("file", "group", "condition", "truth", "sha256")} | answer,
                                    allow_nan=False)+"\n")
            output.flush()
            print("Full", row["file"], (answer["before"]["short_coverage"] or {}).get("reason"),
                  "retry" if answer["retry"] else "", flush=True)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write(folder/"runtime.json", {"held_sha256": sha(folder/"held.jsonl"), "sparse_sha256": sha(folder/"sparse.json"),
                                 "elapsed_s": time.monotonic()-began, "rss_bytes": rss if sys.platform == "darwin" else rss*1024})


def rebuild(folder, cases):
    import cv2

    cv2.setNumThreads(1)
    frozen = read(folder/"freeze.json")
    count = 0
    with TemporaryDirectory(prefix="rebuild-", dir=folder) as location:
        destination = Path(location)
        for media, _, row in cases:
            if row["retry"] is None:
                continue
            group, role, span = media["group"], media["condition"], media["authored_span"]
            output = destination/media["file"]
            is_tandem = role.startswith("tandem_")
            base_role = role.removeprefix("tandem_")
            original = output if not is_tandem else destination/("first-"+output.name)
            if role in GRID_ROLES:
                source = frozen["ordinary"][(group-1)*2:group*2]
                host, guest = map(genuine_frames, source)
                span = span or PLACEMENTS[(group-1) % len(PLACEMENTS)]
                hd.authored(original, pictures=composed(host, guest, span, base_role))
            else:
                source = frozen["swaps"][group-1]
                host = earlier.native(source["pin"], "original", range(120))
                changed = {} if base_role == "original" else earlier.native(source["pin"], source["role"], range(*span))
                hd.authored(original, pictures=(changed.get(i, host[i]) for i in range(120)))
            if is_tandem:
                hd.authored(output, source=original, crf=35)
                original.unlink()
            require(sha(output) == media["sha256"], "Rebuilt branch delivery differs")
            output.unlink()
            count += 1
    return count


def report(folder):
    ledger = exact_media(folder)
    screening, runtime = read(folder/"sparse.json"), read(folder/"runtime.json")
    require(runtime["held_sha256"] == sha(folder/"held.jsonl") and runtime["sparse_sha256"] == sha(folder/"sparse.json")
            and screening["media_sha256"] == sha(folder/"media.json"), "A frozen observation changed")
    with (folder/"held.jsonl").open() as stream:
        cases = list(zip(ledger["rows"], screening["rows"], map(json.loads, stream), strict=True))
    strata = {role: Counter() for role in (*GRID_ROLES, *SWAP_ROLES)}
    gains, contributing = Counter(), set()
    exact, crops, examples = 0, 0, {}
    from app.detectors.video.decode import FrameTable

    for media, sparse, row in cases:
        require(all(row[k] == media[k] for k in ("file", "group", "condition", "truth", "sha256"))
                and sparse["file"] == row["file"] and sparse["sha256"] == row["sha256"], "Observed delivery differs")
        base, candidate = row["before"], row["retry"]
        route = (base["short_coverage"] or {}).get("reason") == "sparse_range_outside_profile"
        require(route == (candidate is not None), "Measured retry differs from the incumbent route")
        path = folder/"media"/row["file"]
        inspected, _ = review.view_anchors(path, media, None)
        exact += inspected
        table = FrameTable.from_json(media["clock"])
        original = [base[key] for key in ("coarse", "boundary_refinement", "short_range") if base[key]] + row["incumbent_second"]
        before = [list(spans.interval(record, table.pts)) for record in original]
        after = [list(spans.interval(record, table.pts)) for record in candidate["ranges"]] if candidate else []
        truth = row["truth"]
        found = {i for i, span in enumerate(truth) if any(review.includes_half(r, span) for r in before)}
        new = {i for i, span in enumerate(truth) if any(review.includes_half(r, span) for r in after)} - found
        role = row["condition"]
        wrong = sum(not any(review.includes_half(r, span) for span in truth) for r in after)
        count = strata[role]
        count.update(files=1, truth=len(truth), sparse_screen=sparse["state"]["status"] == "localized", full_retry=route,
                     dense_measured=bool(candidate and candidate["coverage"]["status"] in ("localized", "no_distinct_range")),
                     before_useful=len(found), combined_useful=len(found|new), before_flags=bool(before),
                     combined_flags=bool(before or after), retry_ranges=len(after), wrong_retry_ranges=wrong,
                     retry_at_ordinary_intervention=sum(bool(media["authored_span"] and not truth
                                                            and review.includes_half(r, media["authored_span"])) for r in after))
        if new:
            contributing.add(row["group"])
            gains["added_correct"] += len(new)
        if after and not truth:
            cut = role not in ("ordinary", "original")
            gains["added_cut_flags" if cut else "added_ordinary_flags"] += 1
        if truth:
            gains["added_misplaced_ranges"] += wrong
        for record in candidate["ranges"] if candidate else []:
            compact = prior_review.compact(record, table)
            _, matched = review.view_anchors(path, media, compact)
            crops += len(matched)
            if role not in examples:
                examples[role] = {"frames": compact["frames"], "clock_s": compact["clock_s"],
                                  "anchors": [v["frame"] for v in matched.values()]}
    gains["sources_with_gain"] = len(contributing)
    gate = {key: gains[key] <= limit if key in ("added_ordinary_flags", "added_cut_flags", "added_misplaced_ranges")
            else gains[key] >= limit for key, limit in SPEC["gate"].items() if key != "minimum_overlap"}
    write(folder/"aggregate.json", {"protocol": SPEC, "freeze_sha256": sha(folder/"freeze.json"),
                                    "media_sha256": sha(folder/"media.json"), "runtime": runtime,
                                    "native_frames": exact, "crops": crops, "guarded_boundaries": 2*(crops//3),
                                    "exact_branch_rebuilds": rebuild(folder, cases),
                                    "strata": strata, "gains": {key: gains[key] for key in gate},
                                    "gate": gate, "advances": all(gate.values()), "examples": examples})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "screen", "score", "report"))
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        args.folder.mkdir(mode=0o700, parents=True)
        write(args.folder/"freeze.json", plan())
    elif args.phase == "prepare":
        lock = read(args.folder/"freeze.json")
        require(lock == plan(), "Frozen source roster or method changed")
        prepare(args.folder, lock)
    elif args.phase == "screen":
        screen(args.folder)
    elif args.phase == "score":
        score(args.folder)
    else:
        report(args.folder)


if __name__ == "__main__":
    main()
