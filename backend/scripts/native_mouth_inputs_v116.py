"""Bind native mouth-edit pictures to qualified speech states; decoder-only."""

from __future__ import annotations

import argparse
import json
import subprocess
import types
from collections import Counter
from pathlib import Path

from scripts import av_source_encoder_v114 as speech
from scripts import video_appearance_runtime_v108 as runtime
from scripts import video_native_sources_v92 as native
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = speech.ROOT, speech.BASE, speech.PANEL
TRIAL = BASE / "native-mouth-inputs-v116"
CACHE = native.ARCHIVE / "native-mouth-inputs-v116"
PUBLIC = PANEL / "native-mouth-input-freeze-v116.json"
CODE = (
    "backend/scripts/native_mouth_inputs_v116.py",
    "backend/tests/test_native_mouth_inputs_v116.py",
    "docs/native-mouth-inputs-v116-method.md",
)
read, write, sha, require = speech.read, speech.write, speech.sha, speech.require


def edit_support(plan):
    edit = plan["native20_edit"]
    if edit is None:
        return []
    start, end = edit["frames_half_open"]
    require(
        type(start) is int and type(end) is int and end - start == 20, "Exact frozen 20-picture edit required"
    )
    indices = list(range(start - 2, end + 2))
    frames = {f["video_frame_index"]: f for f in plan["frames"]}
    require(
        min(indices) >= 0
        and all(i in frames and frames[i]["source_sample_status"] == "supported" for i in indices),
        "Complete before/during/after source/audio support required",
    )
    return indices


def planned():
    parent = read(BASE / "av-sentence-support-v104/preparation/plan.json")["records"]
    native_rows = {r["index"]: r for r in read(native.TRIAL / "qualified-plan.json")["assets"]}
    assets = {r["index"]: r for r in native.assets()}
    states = [
        row
        for batch in range(8)
        for row in speech.lines(speech.TRIAL / f"extract-{batch:02d}/output/source-states.jsonl")
    ]
    plans = speech.source_records()
    require(len(parent) == len(states) == len(plans) == 48, "All original source inputs required")
    rows = []
    for p, state, (clock, _) in zip(parent, states, plans, strict=True):
        index = p["asset_index"]
        n, a = native_rows[index], assets[index]
        require(
            index == state["asset_index"] == clock["asset_index"]
            and p["id"] == state["source_group"] == clock["source_group"] == n["group"] == a["group"]
            and p["study_role"] == state["study_role"] == clock["study_role"]
            and p["input_sha256"] == n["input_sha256"] == a["sha256"]
            and state["encoder_routing_qualified"] is True
            and state["generator_conditioning_qualified"] is False,
            "Exact native source/speech/role correspondence required",
        )
        indices = edit_support(clock)
        rows.append(
            {
                "asset_index": index,
                "source_group": p["id"],
                "study_role": p["study_role"],
                "source_path": str(native.MEDIA / a["file"]),
                "input_sha256": a["sha256"],
                "table": n["frame_table"],
                "frame_indices": indices,
                "expected_rgb_sha256": {str(i): n["rgb_sha256"][i] for i in indices},
                "clock_frames": [f for f in clock["frames"] if f["video_frame_index"] in indices],
                "edit_frames_half_open": None
                if clock["native20_edit"] is None
                else clock["native20_edit"]["frames_half_open"],
                "hidden_file": state["hidden_file"],
                "hidden_file_sha256": state["hidden_file_sha256"],
                "status": "planned" if indices else "unsupported_short_source",
            }
        )
    return rows


def identity():
    speech.verify()
    native.checked()
    require(
        read(speech.TRIAL / "review-v115/result.json") == read(PANEL / "av-source-encoder-result-v115.json"),
        "Published complete source-state review required",
    )
    require(read(TRIAL / "plan.json") == planned(), "Unchanged source-only planning required")
    return {
        "schema": "chipmunk.native-mouth-input-freeze/116",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "plan_sha256": sha(TRIAL / "plan.json"),
        "source_state_freeze_sha256": sha(speech.PUBLIC),
        "source_state_result_sha256": sha(PANEL / "av-source-encoder-result-v115.json"),
        "native_source_freeze_sha256": sha(native.TRIAL / "freeze.json"),
        "native_qualified_plan_sha256": sha(native.TRIAL / "qualified-plan.json"),
        "source_role_plan_sha256": sha(BASE / "av-sentence-support-v104/preparation/plan.json"),
        "sources": 48,
        "eligible_sources": 47,
        "retained_short_sources": 1,
        "planned_pictures": 1128,
        "new_neural_calls": 0,
        "scope": "Native frame bytes, clocks and speech-state bindings only. No face eligibility, mouth render, achieved synchronization, authenticity prediction or serving change.",
    }


def freeze():
    rows = planned()
    require(
        len(rows) == 48
        and sum(bool(r["frame_indices"]) for r in rows) == 47
        and sum(len(r["frame_indices"]) for r in rows) == 1128,
        "Complete unchanged native edit panel required",
    )
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "plan.json", rows)
    body = identity()
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(
        json.dumps({"status": "frozen", "sources": 48, "planned_pictures": 1128, "new_neural_calls": 0}),
        flush=True,
    )


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Published unchanged input contract required",
    )


def routed_popen(argv, **kwargs):
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind"],
        "Only fixed decoder binding can be routed",
    )
    routed = list(argv)
    routed[3] = "scripts.native_mouth_inputs_v116"
    return subprocess.Popen(routed, **kwargs)


def run(batch):
    require(type(batch) is int and 0 <= batch < 8, "One fixed decoder batch required")
    proxy = types.SimpleNamespace(
        Popen=routed_popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "prerequisites": lambda command, batch: True,
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("bind", batch)


def binding(batch, output):
    import cv2
    import numpy as np
    from PIL import Image

    from app.detectors.video.decode import FrameTable, decode_frames

    rows = read(TRIAL / "plan.json")[batch * 6 : (batch + 1) * 6]
    require(len(rows) == 6, "Complete prepared batch required")
    cache = CACHE / f"batch-{batch:02d}"
    cache.mkdir(parents=True, exist_ok=False)
    result = []
    for offset, row in enumerate(rows):
        path = Path(row["source_path"])
        require(
            sha(path) == row["input_sha256"] and sha(Path(row["hidden_file"])) == row["hidden_file_sha256"],
            "Exact native source and private states required",
        )
        files = []
        if row["frame_indices"]:
            folder = cache / f"source-{offset:02d}"
            folder.mkdir(exist_ok=False)
            table = FrameTable.from_json(row["table"])
            for frame in decode_frames(path, table, row["frame_indices"], timeout_s=120):
                expected = row["expected_rgb_sha256"][str(frame.index)]
                require(
                    speech.speech.fingerprint(frame.rgb) == expected
                    and frame.pts == row["table"]["pts"][frame.index],
                    "Actual native picture bytes/clock differ",
                )
                png = folder / f"frame-{frame.index:04d}.png"
                require(
                    cv2.imwrite(str(png), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)),
                    "Could not retain source picture",
                )
                with Image.open(png) as image:
                    actual = np.asarray(image.convert("RGB"))
                require(actual.tobytes() == frame.rgb.tobytes(), "Separate saved PNG decode differs")
                files.append(
                    {
                        "frame_index": frame.index,
                        "pts": frame.pts,
                        "file": str(png),
                        "file_sha256": sha(png),
                        "rgb_sha256": expected,
                    }
                )
            require(
                [r["frame_index"] for r in files] == row["frame_indices"],
                "Complete exact picture sequence required",
            )
        require(sha(path) == row["input_sha256"], "Native source changed during decode")
        result.append(
            {
                "planned": row,
                "pictures": files,
                "face_geometry_qualified": False,
                "generator_conditioning_qualified": False,
            }
        )
    write(output / "bindings.json", result)
    receipt = {
        "schema": "chipmunk.native-mouth-input-binding/116",
        "status": "complete",
        "batch": batch,
        "sources": 6,
        "eligible_sources": sum(bool(r["pictures"]) for r in result),
        "retained_short_sources": sum(not r["pictures"] for r in result),
        "pictures": sum(len(r["pictures"]) for r in result),
        "bindings_sha256": sha(output / "bindings.json"),
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "new_neural_calls": 0,
        "defaults_changed": False,
    }
    write(output / "result.json", receipt)
    print(json.dumps(receipt), flush=True)


def summarise():
    rows, batches = [], []
    for batch in range(8):
        folder = TRIAL / f"bind-{batch:02d}"
        p, r = read(folder / "profile.json"), read(folder / "output/result.json")
        require(
            p["child_exit_code"] == 0
            and p["guard"] is None
            and p["sample_errors"] == 0
            and p["worker_receipt_sha256"] == sha(folder / "output/result.json")
            and r["status"] == "complete"
            and r["batch"] == batch
            and r["freeze_sha256"] == sha(TRIAL / "freeze.json")
            and r["bindings_sha256"] == sha(folder / "output/bindings.json"),
            "Complete successful exact input stage required",
        )
        bound = read(folder / "output/bindings.json")
        require(
            [r["planned"] for r in bound] == read(TRIAL / "plan.json")[batch * 6 : (batch + 1) * 6],
            "No substituted or omitted source",
        )
        rows.extend(bound)
        batches.append(
            {
                "batch": batch,
                "result_sha256": sha(folder / "output/result.json"),
                "profile_sha256": sha(folder / "profile.json"),
                "pictures": r["pictures"],
                "duration_s": p["duration_s"],
                "sampled_tree_rss_lower_bound_bytes": p["sampled_tree_rss_lower_bound_bytes"],
                "minimum_available_ram_bytes": p["minimum_available_ram_bytes"],
            }
        )
    require(
        len(rows) == 48 and sum(len(r["pictures"]) for r in rows) == 1128,
        "Complete original input panel required",
    )
    result = {
        "schema": "chipmunk.native-mouth-input-result/116",
        "status": "complete",
        "sources": 48,
        "eligible_sources": 47,
        "retained_short_sources": 1,
        "pictures": 1128,
        "role_counts": dict(Counter(r["planned"]["study_role"] for r in rows)),
        "batches": batches,
        "new_neural_calls": 0,
        "face_geometry_qualified_sources": 0,
        "generator_conditioning_qualified_sources": 0,
        "defaults_changed": False,
        "scope": "Exact native pictures, source clocks and private speech state bindings only; no mouth rendering, achieved alignment or detector accuracy.",
    }
    write(TRIAL / "result.json", result)
    write(PANEL / "native-mouth-input-result-v116.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "batches"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "bind", "summarise"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.command == "freeze":
        require(not args.arguments, "Freeze has no arguments")
        freeze()
    elif args.command == "run":
        require(len(args.arguments) == 1, "One batch required")
        run(int(args.arguments[0]))
    else:
        verify()
        if args.command == "summarise":
            require(not args.arguments, "Summary has no arguments")
            summarise()
        else:
            require(len(args.arguments) == 2, "Exact guarded output/batch required")
            output = Path(args.arguments[0])
            batch = int(args.arguments[1])
            require(
                type(batch) is int and 0 <= batch < 8 and output.resolve().is_relative_to(TRIAL.resolve()),
                "Private fixed stage required",
            )
            output.mkdir(exist_ok=False)
            binding(batch, output)
