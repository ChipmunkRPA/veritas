"""Separate source-to-output pixel/mask/clock replay; no model inference."""

from __future__ import annotations

import argparse
import json
import subprocess
import types
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np

from scripts import native_mouth_clips_v120 as clips
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = clips.ROOT, clips.BASE, clips.PANEL
TRIAL = BASE / "native-mouth-clip-review-v121"
PUBLIC = PANEL / "native-mouth-clip-review-freeze-v121.json"
CODE = (
    "backend/scripts/review_native_mouth_clips_v121.py",
    "backend/tests/test_review_native_mouth_clips_v121.py",
    "docs/native-mouth-clip-review-v121-method.md",
)
read, write, sha, require = clips.read, clips.write, clips.sha, clips.require


def changed_pixels(original, output, mask):
    require(
        original.shape == output.shape
        and original.dtype == output.dtype == np.uint8
        and original.ndim == 3
        and original.shape[2] == 3,
        "Complete corresponding native RGB pictures required",
    )
    require(
        mask.shape == original.shape[:2] and mask.dtype == np.uint8 and np.isin(mask, [0, 255]).all(),
        "Exact binary native pixel mask required",
    )
    actual = np.any(original != output, axis=2)
    require(
        np.array_equal(actual, mask != 0), "Recorded mask differs from actual source/output pixel changes"
    )
    return int(actual.sum())


def crop_transform(proof):
    x0, y0, x1, y1 = proof["geometry"]["bbox"]
    require(
        all(type(v) is int for v in (x0, y0, x1, y1)) and x1 > x0 and y1 > y0,
        "Native source crop bounds required",
    )
    sx, sy = 256 / (x1 - x0), 256 / (y1 - y0)
    expected = [[sx, 0.0, (0.5 - x0) * sx - 0.5], [0.0, sy, (0.5 - y0) * sy - 0.5]]
    require(
        proof["source_to_crop_matrix_hex"] == [[float(x).hex() for x in row] for row in expected],
        "Source crop pixel-edge transform differs",
    )


def identity():
    clips.verify()
    return {
        "schema": "chipmunk.native-mouth-clip-review-freeze/121",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "generation_freeze_sha256": sha(clips.PUBLIC),
        "generation_plan_sha256": sha(clips.TRIAL / "plan.json"),
        "scope": "Separate data-only source-to-output pixel masks, source clocks, crop transforms and unchanged soundtrack replay. No independent face-alpha reconstruction, generator forward reproduction, achieved alignment or detector accuracy.",
    }


def freeze():
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(json.dumps({"status": "frozen", "new_neural_calls": 0}), flush=True)


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Published unchanged source-mask audit required",
    )


def routed_popen(argv, **kwargs):
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind"],
        "Only fixed data review can be routed",
    )
    routed = list(argv)
    routed[3] = "scripts.review_native_mouth_clips_v121"
    return subprocess.Popen(routed, **kwargs)


def run(batch):
    require(type(batch) is int and 0 <= batch < 8, "One complete source batch required")
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


def review(batch, output):
    from PIL import Image

    from app.detectors.audio.decode import decode_audio, source_ranges
    from app.detectors.video.decode import FrameTable, decode_frames, probe_frames

    folder = clips.TRIAL / f"extract-{batch:02d}"
    receipt, profile = read(folder / "output/result.json"), read(folder / "profile.json")
    require(
        receipt["status"] == "complete"
        and receipt["batch"] == batch
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["worker_receipt_sha256"] == sha(folder / "output/result.json")
        and receipt["cases_sha256"] == sha(folder / "output/cases.jsonl")
        and receipt["freeze_sha256"] == sha(clips.TRIAL / "freeze.json"),
        "Successful exact complete source generation required",
    )
    cases = [json.loads(l) for l in (folder / "output/cases.jsonl").read_text().splitlines()]
    plans = read(clips.TRIAL / "plan.json")[batch * 6 : (batch + 1) * 6]
    require(
        len(cases) == 36 and [r["condition"] for r in cases] == list(clips.CONDITIONS) * 6,
        "Complete ordered condition records required",
    )
    results = []
    with (output / "reviewed-cases.jsonl").open("x") as journal:
        for offset, plan in enumerate(plans):
            source = plan["source"]
            path = Path(source["source_path"])
            table = FrameTable.from_json(source["table"])
            require(sha(path) == source["input_sha256"], "Original source changed")
            original = [f.rgb for f in decode_frames(path, table, range(len(table)), timeout_s=120)]
            require(
                [clips.inputs.speech.speech.fingerprint(x) for x in original] == plan["native_rgb_sha256"],
                "Original full source RGB replay differs",
            )
            wav = plan["waveform"]
            wavepath = clips.inputs.speech.clock.CACHE / wav["file"]
            require(sha(wavepath) == wav["sha256"], "Original qualified waveform changed")
            with wave.open(str(wavepath), "rb") as stream:
                expected_pcm = stream.readframes(stream.getnframes())
            expected_origin = Fraction(
                clips.inputs.speech.source_records()[batch * 6 + offset][0]["frames"][0][
                    "source_audio_origin_seconds"
                ]
            )
            for case in cases[offset * 6 : (offset + 1) * 6]:
                require(
                    case["asset_index"] == source["asset_index"]
                    and case["source_group"] == source["source_group"]
                    and case["study_role"] == source["study_role"]
                    and case["known_ai_provenance"] == case["condition"].startswith("ai_")
                    and case["achieved_synchronization_qualified"] is False,
                    "Exact source/role/provenance and unqualified alignment required",
                )
                counts = {"pictures": 0, "edit_picture_masks": 0, "changed_pixels": 0, "audio_samples": 0}
                if case["status"] == "ready":
                    file = Path(case["file"])
                    require(sha(file) == case["file_sha256"], "Constructed media file changed")
                    t = probe_frames(file)
                    require(
                        t.pts == table.pts
                        and t.time_base == table.time_base
                        and (t.width, t.height) == (table.width, table.height)
                        and not t.interpolated
                        and not t.truncated,
                        "Native source/output picture clocks or dimensions differ",
                    )
                    bounds = None if case["condition"] == "original" else source["edit_frames_half_open"]
                    require(case["edit_frames_half_open"] == bounds, "Actual condition edit bounds differ")
                    proofs = {r["frame_index"]: r for r in case["edit_proofs"]}
                    require(
                        list(proofs) == ([] if bounds is None else list(range(*bounds))),
                        "Exact complete native mask positions required",
                    )
                    for frame in decode_frames(file, t, range(len(t)), timeout_s=120):
                        counts["pictures"] += 1
                        if frame.index not in proofs:
                            require(
                                frame.rgb.tobytes() == original[frame.index].tobytes(),
                                "Unedited or context source picture changed",
                            )
                        else:
                            proof = proofs[frame.index]
                            require(proof["pts"] == table.pts[frame.index], "Mask timestamp differs")
                            mask_path = Path(proof["mask_file"])
                            require(sha(mask_path) == proof["mask_sha256"], "Mask bytes changed")
                            with Image.open(mask_path) as image:
                                mask = np.asarray(image)
                            n = changed_pixels(original[frame.index], frame.rgb, mask)
                            require(n == proof["changed_pixels"], "Claimed changed-pixel count differs")
                            crop_transform(proof)
                            counts["edit_picture_masks"] += 1
                            counts["changed_pixels"] += n
                    require(counts["pictures"] == len(table), "Full output decode required")
                    audio = decode_audio(file, source_sha256=case["file_sha256"], max_seconds=30)
                    pcm = np.rint(audio.pcm * 32768).astype("<i2").tobytes()
                    require(pcm == expected_pcm, "Actual soundtrack differs from original qualified samples")
                    ranges = source_ranges(audio.mapping, 0, len(audio.pcm))
                    require(
                        len(ranges) == 1
                        and Fraction(ranges[0]["source_start_pts"], 16000) == expected_origin,
                        "Actual soundtrack origin or continuity differs",
                    )
                    counts["audio_samples"] = len(audio.pcm)
                    if source["edit_frames_half_open"] is not None:
                        a, b = source["edit_frames_half_open"]
                        require(
                            case["before_during_after"]
                            == {
                                "before": [a - 2, a - 1],
                                "during": [a, (a + b) // 2, b - 1],
                                "after": [b, b + 1],
                            },
                            "Representative context positions differ",
                        )
                else:
                    require(
                        case["status"] == "unavailable" and bool(case["reason"]),
                        "Explicit construction unavailability required",
                    )
                result = {
                    "asset_index": source["asset_index"],
                    "study_role": source["study_role"],
                    "condition": case["condition"],
                    "status": case["status"],
                    **counts,
                }
                results.append(result)
                journal.write(json.dumps(result) + "\n")
                journal.flush()
            del original
    result = {
        "schema": "chipmunk.native-mouth-clip-review-result/121",
        "status": "complete",
        "batch": batch,
        "cases": 36,
        "ready_cases": sum(r["status"] == "ready" for r in results),
        "unavailable_cases": sum(r["status"] == "unavailable" for r in results),
        "pictures": sum(r["pictures"] for r in results),
        "edit_picture_masks": sum(r["edit_picture_masks"] for r in results),
        "changed_pixels": sum(r["changed_pixels"] for r in results),
        "audio_samples": sum(r["audio_samples"] for r in results),
        "private_review_journal_sha256": sha(output / "reviewed-cases.jsonl"),
        "generation_receipt_sha256": sha(folder / "output/result.json"),
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "new_neural_calls": 0,
        "defaults_changed": False,
        "scope": "Source/output difference masks, native timing, representatives, crop transforms and soundtrack equality. No independently assessed geometry unavailability, reconstructed face alpha, neural forward reproduction, achieved synchronization or detector accuracy.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "bind"))
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
        require(len(args.arguments) == 2, "Exact output and batch required")
        output = Path(args.arguments[0])
        batch = int(args.arguments[1])
        require(
            0 <= batch < 8 and output.resolve().is_relative_to(TRIAL.resolve()),
            "Exclusive fixed private stage required",
        )
        output.mkdir(exist_ok=False)
        review(batch, output)
