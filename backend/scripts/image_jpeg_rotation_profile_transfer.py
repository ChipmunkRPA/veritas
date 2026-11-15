# SPDX-License-Identifier: Apache-2.0
"""Freeze and verify rotation-copy evidence on additional JPEG profiles."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts import image_jpeg_turned_copy_transfer as prior
from scripts.image_exact_turned_copy_transfer import digest, useful

SCHEMA = "chipmunk.image-jpeg-rotation-profile-transfer/1"
QUALITIES = (85, 95)
GROUPS = 24
GATE = {"minimum_gained_groups": 4, "maximum_added_ordinary": 2, "maximum_added_repetition": 0, "maximum_misplaced": 2}
PANELS = Path("docs/evaluation-panels")


def read(path):
    return json.loads(path.read_bytes())


def write(path, data):
    with path.open("x", encoding="utf-8") as output:
        json.dump(data, output, sort_keys=True, separators=(",", ":"), allow_nan=False)
        output.write("\n")


def freeze(graph_path, commons, destination):
    import numpy as np

    from scripts import image_jpeg_copy_blocks_transfer as sources
    from scripts import image_offgrid_patch_transfer as qualification
    from scripts.image_affine_native_scale import decode

    graph = qualification.committed(graph_path, "image-fresh-source-cache-v39-results.json", "private_groups_sha256")
    excluded = set()
    for panel in sorted(PANELS.glob("image-*copy*.json")):
        rows = [read(panel)]
        while rows:
            row = rows.pop()
            if isinstance(row, dict):
                excluded.update(item for key, item in row.items()
                                if key in {"source_sha256", "source_jpeg_sha256"} and isinstance(item, str))
                rows.extend(row.values())
            elif isinstance(row, list):
                rows.extend(row)
    by_key = {row["key"]: row for row in graph["sources"]}

    def rank(row):
        return digest(f"{SCHEMA}/{row['source_jpeg_sha256']}".encode())

    hard_controls = qualification.hard_controls(commons)
    authors = {row["author"].casefold() for row in hard_controls}
    authors.update(row["author"].casefold() for row in graph["sources"] if row.get("source_jpeg_sha256") in excluded)
    eligible = []
    for component in graph["candidate_only_unlinked_components"]:
        members = [by_key[key] for key in component]
        if any(row["author"].casefold() in authors or row.get("source_jpeg_sha256") in excluded for row in members):
            continue
        matches = [row for row in members if row["license_url"] in sources.LICENSES and min(row["size"]) >= 480
                   and max(row["size"]) >= 640]
        if matches:
            eligible.append(min(matches, key=rank))
    roster = sorted(eligible, key=rank)[:GROUPS]
    assert len(roster) == len({row["author"].casefold() for row in roster}) == GROUPS
    destination.mkdir(mode=0o700)
    cases = []

    def encode(group, source_sha256, treatments):
        for role, image, truth in treatments:
            for quality in QUALITIES:
                encoded = prior.encode_jpeg(image, quality)
                name = f"{group:02d}-{role}-{quality}.jpg"
                (destination / name).write_bytes(encoded)
                cases.append({"file": name, "group": group, "source_sha256": source_sha256, "role": role,
                              "sha256": digest(encoded), "quality": quality, "truth": truth, "shape": list(image.shape[1::-1])})

    local_sources = []
    for group, source in enumerate(roster):
        _, image = sources.photo(source)
        encode(group, source["source_jpeg_sha256"], prior.deliveries(np.asarray(image), group + 219))
        local_sources.append({key: source[key] for key in ("author", "photo_page", "license_url", "source_jpeg_sha256")})
    for group, source in enumerate(hard_controls, GROUPS):
        image = decode(source["path"].read_bytes())
        x, y = (image.width - 704) // 2, (image.height - 480) // 2
        pixels = image.crop((x, y, x + 704, y + 480))
        encode(group, source["jpeg_sha256"], [("repetition", np.asarray(pixels), None)])
    write(destination / "inputs.json", {"schema": SCHEMA, "gate": GATE, "cases": cases, "source_graph_sha256": digest(graph_path.read_bytes()),
        "credited_edit_sources": local_sources,
        "credited_repeat_sources": [{key: row[key] for key in ("slot", "author", "page", "licence_url", "jpeg_sha256")}
                                    for row in hard_controls]})


def measure(directory, case):
    from io import BytesIO

    import numpy as np
    from PIL import Image

    from app.detectors.image import jpeg_copy_blocks, jpeg_turned_copy

    data = (directory / case["file"]).read_bytes()
    assert digest(data) == case["sha256"]
    with Image.open(BytesIO(data), formats=["JPEG"]) as opened:
        assert list(opened.size) == case["shape"]
        pixels = np.array(opened)
    baseline = prior._existing(pixels, data)
    trial = baseline["status"] == "no_pair" or (case["quality"] == 95 and baseline["status"] == "unsupported_profile"
                                                and jpeg_copy_blocks.analyse(data)[0] == "no_pair")
    status, matches = jpeg_turned_copy.analyse(pixels) if trial else ("earlier_check", [])
    return {**case, "prior": baseline, "rotation": {"status": status, "matches": matches}}


def summarize(rows):
    results = {}
    for quality in QUALITIES:
        groups = set()
        totals = Counter(positive=0, ordinary=0, repetitions=0, prior_useful=0, added_useful=0, misplaced=0,
                         added_ordinary=0, added_repetition=0, earlier_unassessed=0, trial_unassessed=0, trial_ran=0)
        for row in (row for row in rows if row["quality"] == quality):
            truth, earlier, rotation = row["truth"], row["prior"], row["rotation"]
            pairs = [match["pair"] for match in rotation["matches"]]
            totals["earlier_unassessed"] += earlier["status"] not in {"no_pair", "supported_pair"}
            totals["trial_ran"] += rotation["status"] != "earlier_check"
            totals["trial_unassessed"] += rotation["status"] in {"ambiguous", "unassessed_match_budget"}
            if not truth:
                key = "repetitions" if row["role"] == "repetition" else "ordinary"
                totals[key] += 1
                totals["added_repetition" if key == "repetitions" else "added_ordinary"] += bool(pairs)
                continue
            totals["positive"] += 1
            old = useful(earlier["pairs"] + earlier["affine_pairs"], truth)
            hit = useful(pairs, truth)
            totals["prior_useful"] += old
            totals["added_useful"] += hit and not old
            totals["misplaced"] += bool(pairs) and not hit
            if hit and not old:
                groups.add(row["group"])
        results[str(quality)] = {**totals, "gained_groups": len(groups),
            "advances": len(groups) >= GATE["minimum_gained_groups"] and totals["misplaced"] <= GATE["maximum_misplaced"]
            and totals["added_ordinary"] <= GATE["maximum_added_ordinary"] and totals["added_repetition"] <= GATE["maximum_added_repetition"]}
    return results


def review(receipt):
    rows = receipt["rows"]
    identities = {(row["group"], row["role"], row["quality"]) for row in rows}
    expected = {(group, role, quality) for group in range(GROUPS + 10) for quality in QUALITIES
                for role in ((*prior.EDITS, *prior.CONTROLS) if group < GROUPS else ("repetition",))}
    assert receipt["schema"] == SCHEMA and receipt["gate"] == GATE and identities == expected and len(rows) == len(expected)
    for row in rows:
        rotation = row["rotation"]
        assert bool(row["truth"]) == (row["role"] in prior.EDITS)
        assert len(rotation["matches"]) == (rotation["status"] == "supported_pair")
        assert rotation["status"] in {"supported_pair", "no_pair", "ambiguous", "unassessed_match_budget", "earlier_check"}
        for match in rotation["matches"]:
            a, b = match["pair"]
            assert match["turns"] in (1, 2) and b[2:] == (a[2:][::-1] if match["turns"] == 1 else a[2:])
            assert not prior.overlap(a, b)
            width, height = row["shape"]
            assert all(0 <= x <= width - w and 0 <= y <= height - h and 32 <= min(w, h) <= max(w, h) <= 118
                       for x, y, w, h in (a, b))
            witness = match["measurement"]
            mean, p90 = witness["interior_mean"], witness["interior_p90"]
            assert 0 <= mean <= 3.2 and 0 <= p90 <= 6
            assert sum(v is not None and v + .00005 >= max(7, 2.5 * mean) for v in witness["perimeter_means"]) >= 3
    assert receipt["summary"] == summarize(rows)
    return receipt["summary"]


def capture(directory, output, workers):
    from concurrent.futures import ProcessPoolExecutor
    from itertools import repeat

    plan_file = directory / "inputs.json"
    plan = read(plan_file)
    assert plan["schema"] == SCHEMA and plan["gate"] == GATE
    with ProcessPoolExecutor(max_workers=workers, initializer=prior._setup) as pool:
        rows = list(pool.map(measure, repeat(directory), plan["cases"]))
    data = {"schema": SCHEMA, "gate": GATE, "input_plan_sha256": digest(plan_file.read_bytes()), "rows": rows,
            "summary": summarize(rows)}
    print(json.dumps(review(data), indent=2, sort_keys=True))
    write(output, data)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("freeze", "capture", "review"))
    cli.add_argument("path", type=Path)
    cli.add_argument("output", nargs="?", type=Path)
    cli.add_argument("--commons", type=Path)
    cli.add_argument("--workers", type=int, choices=range(1, 5), default=3)
    args = cli.parse_args()
    if args.mode == "review":
        print(json.dumps(review(read(args.path)), indent=2, sort_keys=True))
    elif args.mode == "freeze":
        freeze(args.path, args.commons, args.output)
    else:
        capture(args.path, args.output, args.workers)
