"""Verify source runs and export a model-independent face-swap comparison."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.bench import independent
from scripts import video_face_frequency_fit as fit
from scripts import video_face_residual_fit as residual
from scripts.video_face_temporal_fit import read, save, sha


def native(study, output):
    from scripts.audit_video_face_quality import read_checks
    from scripts.audit_video_face_residuals import asset_clock, delivery_receipt
    from scripts.audit_video_face_temporal import cadence

    roster = study / "held-roster.json"
    plans = read(roster)["groups"]
    manifest = read(study / "held/manifest.json")
    assert manifest["roster_sha256"] == sha(roster)
    deliveries = {(row["group"], row["family"], row["delivery"]): row for row in manifest["rows"]}
    assert len(deliveries) == len(manifest["rows"]) == len(plans) * 6
    acquired = []
    for plan in plans:
        for family, asset in plan["files"].items():
            direct, tandem = (deliveries[plan["group"], family, mode] for mode in ("direct", "tandem"))
            assert direct["native_map"] == tandem["native_map"] and direct["source_sha256"] == tandem["source_sha256"] == asset["sha256"]
            acquired.append((asset, study / "source" / f"{plan['index']:02d}-{family}.mp4", direct["native_map"]))
    pairs = Counter("comparable" if item["comparable"] else "different" for item in cadence(manifest).values())
    with ThreadPoolExecutor(max_workers=3) as pool:
        mapped = sum(pool.map(lambda case: asset_clock(*case), acquired))
        reports = {}
        for name in ("held", "camera"):
            folder, journal = study / name, study / f"{name}.jsonl"
            _, rows = residual.journal(journal)
            traces = sum(pool.map(lambda row, directory=folder: read_checks(directory / row["file"], row),
                                  (row for row in rows if row["quantization"])))
            reports[name] = delivery_receipt(folder, journal, pool=pool) | {"qp_clocks": traces}
    result = {"roster_sha256": sha(roster), "publisher_assets": len(acquired), "source_picture_mappings": mapped,
              "paired_source_cadence": dict(pairs), "panels": reports}
    save(output, result)
    print(json.dumps(result, sort_keys=True))


def _case(row, split):
    camera = row["origin"] == "camera"
    condition = f"{'webcam' if camera else 'matched swap set'}/{row['delivery']}"
    return {"id": row["sha256"], "modality": "video", "split": split, "group": row["group"],
            "input_sha256": row["sha256"], "source_sha256": row["source_sha256"], "lineage": [row["group"]],
            "condition": condition, "family": row["family"], "label": int(row["family"] != "ordinary"),
            "license": {"id": "CDLA-Permissive-2.0" if camera else "CC-BY-SA-4.0 dataset declaration",
                        "evaluation_permitted": True, "redistribution_permitted": False}}


def _response(row, method, point):
    score = row["scores"][method] if method is not None else None
    if row["current"] is None and score is None:
        return {"id": row["sha256"], "input_sha256": row["sha256"], "status": "inconclusive", "response": None,
                "reason": "Insufficient sampled face evidence"}
    flag = residual.baseline(row) or residual.above(score, point)
    return {"id": row["sha256"], "input_sha256": row["sha256"], "status": "ok", "response": int(flag)}


def bundle(result):
    fit.verify(result)
    locked = result["freeze"]
    panels = result["panels"]
    development = locked["selection"]
    confirmation = [row for panel in panels.values() for row in panel["rows"]]
    split_rows = {"development": development, "confirmation": confirmation}
    manifest = {"schema": independent.SCHEMA, "purpose": "Paired whole-clip face swap comparison at two delivery qualities",
                "cases": [_case(row, split) for split, rows in split_rows.items() for row in rows]}
    assets = {digest for items in panels["held"]["reference"]["weights"].values() for digest in items}
    chosen = locked["selection_order"][0]
    documents, pair = {"manifest": manifest}, []
    for name, method in (("baseline", None), ("candidate", chosen)):
        point = locked["points"][method]["wider"] if method else None
        weights = assets | ({digest for v in locked["headers"][0]["weights"].values() for digest in v}
                            | {independent.digest(locked["heads"][method])} if method else set())
        recipe = {"binding": locked["code_sha256"], "method": method or "paired plus same-frame review", "point": point}
        identity = {"manifest_sha256": independent.digest(manifest), "system": name,
                    "implementation_sha256": independent.digest(recipe), "weight_sha256": sorted(weights),
                    "score_kind": "raw", "method_kind": "weighted", "response_units": "frozen binary alert"}
        receipts = {split: identity | {"split": split, "predictions": [_response(row, method, point) for row in rows]}
                    for split, rows in split_rows.items()}
        gate = independent.freeze(manifest, receipts["development"], thresholds={"video": .5})
        receipts["confirmation"]["freeze_sha256"] = independent.digest(gate)
        pair.extend((receipts["confirmation"], gate))
        documents.update({f"{name}-{split}": document for split, document in (receipts | {"freeze": gate}).items()})
    documents["comparison"] = independent.compare(manifest, *pair, require_freeze_commitment=True)
    return documents


def export(result, output):
    documents = bundle(read(result))
    output.mkdir(mode=0o700)
    for filename, document in documents.items():
        save(output / f"{filename}.json", document)
    print(json.dumps(documents["comparison"]["paired_differences"], sort_keys=True))


def release(result, native_receipt, output):
    data, audit = read(result), read(native_receipt)
    assert audit["roster_sha256"] == data["freeze"]["roster_sha256"]
    for name, panel in data["panels"].items():
        check = audit["panels"][name]
        assert check["manifest_sha256"] == panel["header"]["manifest_sha256"]
        assert check["deliveries"] == len(panel["rows"]) and check["picture_clocks"] == 100 * len(panel["rows"])
        assert check["rgb_anchors"] == sum(len(row["frames"]) for row in panel["rows"])
        assert check["qp_clocks"] == 100 * sum(row["qp"] is not None for row in panel["rows"])
    fit.verify(data)
    save(output, data | {"native_audit": audit})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_subparsers(dest="task", required=True)
    for name, implementation, fields in (("native", native, ("study", "output")), ("export", export, ("result", "output")),
                                        ("release", release, ("result", "native_receipt", "output"))):
        command = tasks.add_parser(name)
        for field in fields:
            command.add_argument(field, type=Path)
        command.set_defaults(implementation=implementation, fields=fields)
    args = parser.parse_args()
    args.implementation(*(getattr(args, name) for name in args.fields))
