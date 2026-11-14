# SPDX-License-Identifier: Apache-2.0
"""Replay exact earlier-copy WebP evidence on the pinned PNG benchmark bank."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.image_edge_context_new_sources import deliveries, public_roster
from scripts.image_small_region_repeat import read, require, save, sha, useful

SCHEMA = "chipmunk.image-reference-webp-transfer/1"
ROLES = ("sharp", "feathered", "unchanged", "ordinary_tone", "wrong_framing")


def recount(rows):
    stats, groups = defaultdict(Counter), defaultdict(set)
    for row in rows:
        key = row["source"], row["role"]
        family, role = row["family"], row["role"]
        require(role not in groups[row["source"]] and role in ROLES, "Duplicate or unknown comparison")
        groups[row["source"]].add(role)
        changed, selected, overlap = (row[key] for key in ("changed_pixels", "selected_pixels", "overlap"))
        require(0 <= overlap <= min(selected, changed) and (row["compared"] or selected == 0)
                and all(len(bytes.fromhex(row[field])) == 32 for field in ("earlier_sha256", "webp_sha256")),
                f"Invalid source measurement {key}")
        for name in ("all", family):
            stats[f"{name}/{role}"].update(files=1, compared=row["compared"], flags=bool(selected),
                                           useful=useful((selected, overlap), changed),
                                           exact=bool(changed and changed == selected == overlap))
    require(len(groups) == 26 and all(group == set(ROLES) for group in groups.values()), "Incomplete paired bank")
    return dict(stats)


def measure(directory, sources):
    import numpy as np
    from PIL import Image

    from app.evidence.image_reference import inspect

    by_group = defaultdict(dict)
    for item in deliveries(directory, sources):
        _, group, _, role, body, rgb, _, _ = item
        by_group[group][role] = body, rgb
    rows = []
    with TemporaryDirectory(prefix="chipmunk-webp-benchmark-") as temporary:
        work = Path(temporary)
        earlier, delivered = work / "earlier.png", work / "submitted.webp"
        for case in sources:
            source = case["original"]["sha256"]
            views = by_group[source]
            original, base = views["original"]
            earlier.write_bytes(original)
            for index, role in enumerate(ROLES, 1):
                rgb = views[role][1] if role in ROLES[:2] else (
                    views["local_tone"][1] if role == "ordinary_tone" else
                    np.roll(base, 2, axis=1) if role == "wrong_framing" else base)
                Image.fromarray(rgb).save(delivered, lossless=True, method=0)
                with Image.open(delivered) as decoded:
                    require(np.array_equal(np.asarray(decoded), rgb), "Lossless delivery changed source samples")
                exhibits = work / f"{case['slot']}-{index}"
                result = inspect(earlier, delivered, evidence_dir=exhibits)
                truth = np.any(base != rgb, axis=-1)
                compared = result["status"] == "comparable"
                if compared:
                    proof = result["exact_webp"]
                    require(proof["changed_pixels"] == int(truth.sum()), "Exact decoded difference count changed")
                selected = overlap = 0
                if result["regions"]:
                    with Image.open(exhibits / result["mask"]["file"]) as image:
                        measured = np.asarray(image) == 255
                    require(not (measured & ~truth).any(), "A mask pixel matched the earlier source")
                    selected, overlap = int(measured.sum()), int((measured & truth).sum())
                rows.append({"source": source, "family": case["generator"], "role": role, "compared": compared,
                             "earlier_sha256": sha(original), "webp_sha256": sha(delivered.read_bytes()),
                             "changed_pixels": int(truth.sum()), "selected_pixels": selected, "overlap": overlap})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("measure", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "review":
        receipt = read(args.path)
        require(receipt["schema"] == SCHEMA and recount(receipt["rows"]) == receipt["summary"], "Count mismatch")
    else:
        sources = read(args.receipt)["sources"] if args.receipt else public_roster(args.sources)
        rows = measure(args.sources, sources)
        receipt = {"schema": SCHEMA, "rows": rows, "summary": recount(rows)}
        save(args.path, receipt)
    print(json.dumps(receipt["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
