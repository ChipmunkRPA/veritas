"""Test whether remote crop responses identify shared processing artifacts."""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import image_bg_real_insertion_score as evidence
from scripts import image_bg_real_insertion_transfer as intake
from scripts import image_crop_context_transfer as crop
from scripts import image_relative_patch_transfer as shared

RELATIVE_POINT = 8.313690900802614
SUPPORT = ("original", "processed", "edited")


def historical_context(previous):
    keys, known, commits = intake.exposure()
    _, plan, rows = evidence.checked_inputs(previous)
    for case, row in zip(plan["cases"], rows, strict=True):
        keys.update(intake.roster.source_keys(case))
        for fingerprint in row.get("fingerprints", {}).values():
            known[0].append(fingerprint)
            known[1].add(fingerprint["encoded_sha256"])
            known[2].add(fingerprint["rgb_sha256"])
    commits["prior_insertion_roster"] = shared.file_digest(previous / "plan.json")
    commits["prior_insertion_qualification"] = shared.file_digest(previous / "result.json")
    return keys, known, commits


def selection(stage, previous, metadata, output):
    keys, known, commitments = historical_context(previous)
    plan = intake.candidate_rows(metadata, keys) | {
        "exposure_sha256": commitments,
        "script_sha256": shared.file_digest(Path(__file__)),
        "dataset_revision": intake.roster.REVISION,
        "generator_revision": intake.GENERATOR_REVISION,
    }
    if stage == "freeze-inputs":
        output.mkdir(mode=0o700)
        shared.write(output / "plan.json", plan)
        print(f"{len(plan['cases'])} planned independent source groups", flush=True)
    else:
        assert plan == json.loads((output / "plan.json").read_bytes())
        shared.write(output / "result.json", intake.acquire(plan, output, keys, known))


def disjoint(first, second):
    left, top, right, bottom = first
    x0, y0, x1, y1 = second
    return right <= x0 or x1 <= left or bottom <= y0 or y1 <= top


def readout(boxes, logits):
    peak, responses = crop.scores(logits)
    remote = [score for box, score in zip(boxes, logits, strict=True) if disjoint(boxes[peak], box)]
    assert len(remote) >= 8
    return boxes[peak], responses["relative_crop"], float(np.median(remote))


def inference(checkpoint):
    import torch

    from app.detectors.image import commfor

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = commfor.load_model(checkpoint, "cpu")
    mean = torch.tensor(commfor.MEAN)[None, :, None, None]
    std = torch.tensor(commfor.STD)[None, :, None, None]

    def score(image):
        boxes, pixels = zip(*crop.crops(image), strict=True)
        logits = []
        for start in range(0, len(pixels), 5):
            tensor = torch.from_numpy(np.stack(pixels[start:start + 5])).permute(0, 3, 1, 2).float() / 255
            logits.extend(commfor.logits(model, (tensor - mean) / std, "cpu"))
        return readout(boxes, logits)

    return score


def evaluate(previous, held_out, weights, output, stage):
    prior, old_plan, old_rows = evidence.checked_inputs(previous)
    new, new_plan, new_rows = evidence.checked_inputs(held_out)
    assert old_plan["cases"] and new_plan["cases"] and new["qualified"] > 0
    assert not {row["flickr_owner"] for row in old_plan["cases"]} & {row["flickr_owner"] for row in new_plan["cases"]}
    identity = {"schema": "chipmunk.image-crop-remote-context/1",
                "study_sha256": shared.file_digest(Path(__file__)),
                "previous_inputs_sha256": shared.file_digest(previous / "result.json"),
                "held_out_inputs_sha256": shared.file_digest(held_out / "result.json"),
                "weight_sha256": shared.file_digest(weights),
                "relative_point": RELATIVE_POINT,
                "remote_point_rule": "strictly below the second lowest remote median among prior flagged processing controls"}
    assert identity["weight_sha256"] == shared.WEIGHT_SHA
    if stage == "freeze-test":
        output.mkdir(mode=0o700)
        shared.write(output / "freeze.json", identity)
        return
    assert identity == json.loads((output / "freeze.json").read_bytes())
    start = time.monotonic()
    predict = inference(weights)
    fit = []
    for _, _, pixels, _ in evidence.panels(previous, old_plan, old_rows):
        _, score, remote = predict(Image.fromarray(pixels["processed"]))
        fit.append({"relative": score, "remote": remote})
    reference = sorted(row["remote"] for row in fit if row["relative"] >= RELATIVE_POINT)
    assert len(reference) >= 2, "At least two prior false highlights are required to fit a control point"
    point = reference[1]
    shared.write(output / "points.json", {"relative": RELATIVE_POINT, "remote_upper_strict": point,
                                         "eligible_fit_controls": len(reference), "fit_group_count": prior["qualified"]})
    rows = []
    with (output / "rows.jsonl").open("x") as journal:
        for slot, _, pixels, truth in evidence.panels(held_out, new_plan, new_rows):
            for role in SUPPORT:
                (left, top, right, bottom), score, remote = predict(Image.fromarray(pixels[role]))
                reference_flag = score >= RELATIVE_POINT
                mask = np.zeros_like(truth)
                mask[top:bottom, left:right] = True
                empty = np.zeros_like(truth)
                expected = truth if role == "edited" else empty
                row = {"slot": slot, "role": role, "delivery": "png", "relative": score, "remote": remote,
                       "measurements": {"relative_crop": evidence.measure(mask if reference_flag else empty, expected),
                                        "remote_control": evidence.measure(mask if reference_flag and remote < point else empty,
                                                                           expected)}}
                rows.append(row)
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                journal.flush()
            print(f"qualified group {slot} complete", flush=True)
    paired = defaultdict(lambda: defaultdict(int))
    for row in rows:
        before = row["measurements"]["relative_crop"]["reported"] > 0
        after = row["measurements"]["remote_control"]["reported"] > 0
        paired[row["role"]][f"{int(before)}_to_{int(after)}"] += 1
    assert len(rows) == len(SUPPORT) * new["qualified"]
    shared.write(output / "result.json", {
        "schema": identity["schema"], "serving_admitted": False,
        "freeze_sha256": shared.file_digest(output / "freeze.json"),
        "points_sha256": shared.file_digest(output / "points.json"),
        "rows_sha256": shared.file_digest(output / "rows.jsonl"),
        "assessed_groups": new["qualified"], "planned_groups": new["planned"],
        "input_crops": (len(rows) + len(fit)) * len(crop.STARTS)**2,
        "results": evidence.summarize(rows), "paired_region_flags": paired,
        "elapsed_s": round(time.monotonic() - start, 3),
        "peak_rss_bytes_mac": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name, fields in (("freeze-inputs", ("previous", "metadata", "output")),
                         ("acquire", ("previous", "metadata", "output")),
                         ("freeze-test", ("previous", "holdout", "weights", "output")),
                         ("run", ("previous", "holdout", "weights", "output"))):
        command = commands.add_parser(name)
        for field in fields:
            command.add_argument(field, type=Path)
    args = parser.parse_args()
    if args.action in {"freeze-inputs", "acquire"}:
        selection(args.action, args.previous, args.metadata, args.output)
    else:
        evaluate(args.previous, args.holdout, args.weights, args.output, args.action)


if __name__ == "__main__":
    main()
