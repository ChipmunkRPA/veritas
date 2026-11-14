# SPDX-License-Identifier: Apache-2.0
"""Compare local JPEG-history evidence with the existing luma check on pinned photographs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from scripts.image_small_region_repeat import read, require, save, sha, useful

SCHEMA = "chipmunk.image-jpeg-history-gap/1"
ROOT = Path(__file__).resolve().parents[2]
ROSTER = ROOT / "docs/evaluation-panels/image-edge-context-new-sources-2026-11-14.json"
ROLES = ("sharp", "feathered", "original", "local_tone", "local_blur", "local_copy", "global_resize", "prior_jpeg")
FAMILIES = {"lama": 3, "sd15": 3, "kandinsky": 2}
PROFILES = {"jpeg75": 75, "jpeg90": 90}
HISTORIES = ("direct", "edited_after_earlier_save")
PLAN = {
    "profiles": PROFILES,
    "histories": list(HISTORIES),
    "roles": list(ROLES),
    "calibration": FAMILIES,
    "points": "Per final profile, strict exceedance of the second-largest source maximum on all ordinary calibration "
              "images, with a minimum of 0.24. Positives are evaluated only on other sources.",
    "earlier_qualities": [50, 55, 60],
    "scope": "Exact native pre-delivery change against the corresponding base; experimental rectangle is visible "
             "compression-history evidence with zero AI authorship weight.",
    "comparison": "Keep aligned luma finding if selected. Try local history gap on files with no baseline region. "
                  "Record ungated geometry, candidate alone and combined outcomes on the same final bytes.",
    "advance": "For each final profile and history separately, add useful locations on at least three source photos "
               "across two families, add at most two ordinary source groups and at most four misplaced edited files. "
               "Direct delivery must be assessed separately from edits staged over an earlier JPEG.",
}


def _split(sources):
    calibration = set()
    for family, count in FAMILIES.items():
        group = [case for case in sources if case["generator"] == family]
        group.sort(key=lambda case: sha(f"{SCHEMA}/split/{case['recipient']}".encode()))
        calibration.update(case["slot"] for case in group[:count])
    require(len(sources) == 26 and len(calibration) == 8, "Incomplete source split")
    return calibration


def _inputs(directory):
    import numpy as np

    from scripts.image_edge_context_new_sources import read_source
    from scripts.image_residual_encoder_transfer import authored, decode, encode

    sources = read(ROSTER)["sources"]
    calibration = _split(sources)
    for case in sources:
        slot, family, record = case["slot"], case["generator"], case["proof"]
        blobs = {role: (directory / str(slot) / f"{role}.png").read_bytes() for role in record["assets"]}
        require({role: sha(body) for role, body in blobs.items()} == record["assets"], "Changed publisher assets")
        views, truth, box, _ = read_source(case, blobs)
        require(sha(truth.tobytes()) == record["truth_sha256"] and box == record["box"], "Changed publisher truth")
        source = case["original"]["sha256"]
        split = "calibration" if slot in calibration else "held"
        original = views["original"]
        earlier_quality = PLAN["earlier_qualities"][int(source[:6], 16) % 3]
        for history in HISTORIES:
            base = original if history == "direct" else decode(encode(original, earlier_quality))
            edited = np.where(truth[..., None], views["edited"], base)
            ordinary, positive = authored({"original": base, "edited": edited}, truth, source)
            images = ordinary | positive | {"prior_jpeg": ordinary["second_jpeg"]}
            for role in ROLES:
                if split == "calibration" and role in ROLES[:2]:
                    continue
                pixels = images[role]
                target = np.any(pixels != base, axis=2) if role in ROLES[:2] else np.zeros_like(truth)
                for profile, quality in PROFILES.items():
                    body = encode(pixels, quality)
                    case_id = {"source": source, "family": family, "split": split, "history": history, "profile": profile,
                               "earlier_quality": earlier_quality if history != "direct" else None, "role": role,
                               "jpeg_sha256": sha(body), "truth_sha256": sha(target.tobytes()),
                               "changed": int(target.sum()), "width": target.shape[1], "height": target.shape[0]}
                    yield case_id, body, target


def _count_mask(mask, truth):
    return [int(mask.sum()), int((mask & truth).sum())]


def _measure(body, truth):
    import cv2
    import numpy as np

    from app.detectors.image import jpeg_dq
    from scripts import image_jpeg_history_gap

    old = jpeg_dq.analyse(body)
    coefficients = jpeg_dq.read_coefficients(body)
    require(old is not None and coefficients is not None, "A frozen standard JPEG could not be read")
    bias = old.localised
    baseline = [0, 0]
    if bias is not None:
        pixels = old.reliable & (old.posterior() >= jpeg_dq.REGION_POSTERIOR)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(pixels.astype(np.uint8), connectivity=8)
        rows = [label for label, row in enumerate(stats[1:], 1)
                if tuple(row.tolist()) == (bias.x, bias.y, bias.w, bias.h, bias.blocks)]
        require(len(rows) == 1, "Could not bind the aligned result to its actual JPEG blocks")
        mask = np.repeat(np.repeat(labels == rows[0], 8, axis=0), 8, axis=1)[:old.height, :old.width]
        baseline = _count_mask(mask, truth)
    gap = image_jpeg_history_gap.measure(coefficients)
    proposal = [0, 0]
    if gap and gap.bounds:
        x, y, w, h = gap.bounds
        proposal = [w * h, int(truth[y:y+h, x:x+w].sum())]
    return {"existing": baseline, "existing_has_alignment": old.double_compression,
            "gap": asdict(gap) if gap else None, "proposal": proposal}


def review(frozen, rows):
    require(frozen["schema"] == SCHEMA and frozen["design"] == PLAN, "Invalid comparison plan")
    require(len(rows) == len(frozen["cases"]) == 768, "Incomplete JPEG delivery comparisons")
    cal = defaultdict(dict)
    assignments = defaultdict(set)
    identities = set()
    for case, row in zip(frozen["cases"], rows, strict=True):
        source, split, profile, history, role = (case[key] for key in ("source", "split", "profile", "history", "role"))
        identity = source, profile, history, role
        require(identity not in identities and case["jpeg_sha256"] == row["jpeg_sha256"], "Duplicate or unbound row")
        identities.add(identity)
        require(case["family"] in FAMILIES and history in HISTORIES and profile in PROFILES and role in ROLES,
                "Unexpected source role")
        require(bool(case["changed"]) == (role in ROLES[:2]) and split in {"calibration", "held"}, "Invalid reference label")
        assignments[source].add((split, case["family"]))
        for size, overlap in (row["existing"], row["proposal"]):
            require(0 <= size <= case["width"] * case["height"] and 0 <= overlap <= min(size, case["changed"]),
                    "Out-of-bounds mask measurement")
        gap = row["gap"]
        if gap and gap["bounds"]:
            x, y, w, h = gap["bounds"]
            require(0 <= x < x + w <= case["width"] and 0 <= y < y + h <= case["height"]
                    and w * h == row["proposal"][0] and gap["score"] > 0., "Out-of-bounds proposal")
        else:
            require(row["proposal"] == [0, 0] and (gap is None or gap["score"] == 0.), "Missing proposal geometry")
        if split == "calibration":
            require(role not in ROLES[:2], "A calibration view used a positive")
            score = gap["score"] if gap else 0.
            cal[profile][source] = max(score, cal[profile].get(source, 0.))
    require(len(assignments) == 26 and all(len(values) == 1 for values in assignments.values()), "Source crossover")
    require(Counter(next(iter(values))[1] for values in assignments.values() if next(iter(values))[0] == "calibration")
            == FAMILIES, "Wrong calibration split")
    points = {profile: max(.24, sorted(scores.values())[-2]) for profile, scores in cal.items()}
    totals = defaultdict(lambda: defaultdict(Counter))
    gains, false_groups, incorrect = defaultdict(set), defaultdict(set), Counter()
    alignments = defaultdict(Counter)
    for case, row in zip(frozen["cases"], rows, strict=True):
        if case["split"] == "calibration":
            continue
        history, profile, role, truth = (case[key] for key in ("history", "profile", "role", "changed"))
        panel = f"{profile}/{history}"
        gap = row["gap"]
        existing, proposal = row["existing"], row["proposal"]
        selected = proposal if gap and gap["score"] > points[profile] else [0, 0]
        combined = existing if existing[0] else selected
        alignments[panel].update(files=1, existing=row["existing_has_alignment"], measured=gap is not None,
                                 correct_earlier=bool(gap and gap["quality"] == case["earlier_quality"]))
        for arm, measurement in (("existing", existing), ("top_before_point", proposal), ("candidate", selected),
                                 ("combined", combined)):
            hit = useful(measurement, truth)
            totals[panel][arm].update(edited=bool(truth), ordinary=not truth, useful=hit,
                                      misplaced=bool(truth and measurement[0] and not hit),
                                      ordinary_flags=bool(not truth and measurement[0]))
        if not existing[0] and selected[0]:
            if not truth:
                false_groups[panel].add(case["source"])
            elif useful(selected, truth):
                gains[panel].add((case["family"], case["source"]))
            else:
                incorrect[panel] += 1
    decisions = {}
    for panel in sorted(totals):
        gained = gains[panel]
        decisions[panel] = {"gained_sources": len(gained), "gained_families": sorted({family for family, _ in gained}),
                            "added_ordinary_groups": len(false_groups[panel]), "added_misplaced": incorrect[panel],
                            "passes": len(gained) >= 3 and len({family for family, _ in gained}) >= 2
                            and len(false_groups[panel]) <= 2 and incorrect[panel] <= 4}
    return {"points": points, "counts": totals, "decisions": decisions, "alignment": alignments}


def _producer():
    paths = [Path(__file__), Path(__file__).with_name("image_jpeg_history_gap.py"),
             ROOT / "backend/app/detectors/image/jpeg_dq.py"]
    return {path.relative_to(ROOT).as_posix(): sha(path.read_bytes()) for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "measure", "review", "replay"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--sources", type=Path)
    args = parser.parse_args()
    output = args.output
    if args.action == "review":
        receipt = read(output)
        calculated = review(receipt["freeze"], receipt["rows"])
        require(calculated == receipt["summary"], "The published counts do not replay")
        print(json.dumps(calculated, indent=2, sort_keys=True))
        return
    if args.sources is None:
        parser.error("Native steps require --sources with the pinned publisher files")
    inputs = _inputs(args.sources)
    if args.action == "freeze":
        output.mkdir(mode=0o700)
        frozen = {"schema": SCHEMA, "design": PLAN, "roster_sha256": sha(ROSTER.read_bytes()),
                  "producer": _producer(), "cases": [case for case, _, _ in inputs]}
        save(output / "freeze.json", frozen)
        print(f"Froze {len(frozen['cases'])} deliveries", flush=True)
        return
    frozen = read(output / "freeze.json")
    require(frozen["roster_sha256"] == sha(ROSTER.read_bytes()) and frozen["producer"] == _producer(), "Frozen code or roster changed")
    rows = []
    for index, (entry, case) in enumerate(zip(inputs, frozen["cases"], strict=True), 1):
        identity, body, truth = entry
        require(identity == case, "A JPEG or its truth differs from the freeze")
        rows.append({"jpeg_sha256": case["jpeg_sha256"], **_measure(body, truth)})
        if index % 96 == 0:
            print(f"Measured {index} JPEGs", flush=True)
    receipt = {"freeze": frozen, "rows": rows, "summary": review(frozen, rows)}
    if args.action == "replay":
        require(json.loads(json.dumps(receipt)) == read(output / "receipt.json"), "Native reprobe differs from recorded rows")
    else:
        save(output / "receipt.json", receipt)
    print(json.dumps(receipt["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
