"""Convert the image panels to the shared local evaluation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from app.bench import independent
from app.bench.independent_locations import TRUTH_SCHEMA, count_locations
from scripts.image_region_rank_compare import require, verify
from scripts.image_region_rank_inputs import POSITIVE, QUALITY
from scripts.video_face_temporal_fit import read, save, sha


def pack_truth(mask_path, expected, output, extent):
    import numpy as np
    from PIL import Image

    require(sha(mask_path) == expected, "Private truth bytes differ from the public ledger")
    with Image.open(mask_path, formats=["PNG"]) as image:
        require(list(image.size) == extent, "Mask dimensions differ from the original raster")
        mask = np.array(image.convert("L")) > 0
    raw = f"P4\n{extent[0]} {extent[1]}\n".encode("ascii") + np.packbits(mask, axis=1).tobytes()
    output.write_bytes(raw)
    return {"file": output.name, "sha256": hashlib.sha256(raw).hexdigest()}


def export(results, media, output):
    verify(results)
    locked = results["freeze"]
    require(locked["manifest_sha256"] == sha(media / "manifest.json"), "The private renders differ from the study commitment")
    output.mkdir(mode=0o700)
    mask_dir = output / "masks"
    mask_dir.mkdir(mode=0o700)
    groups = {row["tag"]: row for row in results["source_groups"]}
    checked = {}
    for quality in QUALITY:
        directory = output / f"jpeg{quality}"
        directory.mkdir(mode=0o700)
        rows = [row for row in locked["calibration"] + results["held"] if row["quality"] == quality]
        manifest = {"schema": independent.SCHEMA, "cases": []}
        selection = locked["chosen"][str(quality)]
        cutoff = locked["levels"][f"{quality}/{selection}"]["wider"]
        truth_rows = {"development": [], "confirmation": []}
        predictions = {"development": [], "confirmation": []}
        expected = {}
        files = {}
        for row in rows:
            name = f"{row['group']}-{row['role']}"
            split = "development" if row["split"] == "calibration" else "confirmation"
            h, w = row["shape"]
            source = groups[row["group"]]["source"]
            terms = "CC BY 2.0 photos and CC BY 4.0 dataset contributions" if row["split"] == "transfer" else (
                "CC BY 4.0 originals and CC BY-SA 4.0 edit release")
            manifest["cases"].append({"id": name, "modality": "image", "split": split, "group": row["group"],
                                      "label": int(row["role"] in POSITIVE), "family": row["family"],
                                      "condition": f"{row['split']}-{row['role']}", "source_sha256": source,
                                      "input_sha256": row["sha256"], "lineage": [source],
                                      "license": {"id": terms, "evaluation_permitted": True, "redistribution_permitted": True},
                                      "localization_truth": {"unit": independent.LOCALIZATION_UNITS["image"], "source_units": w*h,
                                                             "target_units": row["changed_pixels"]}})
            require(sha(media / row["file"]) == row["sha256"], "A tested JPEG changed")
            extent = [w, h]
            truth = {"id": name, "input_sha256": row["sha256"], "extent": extent, "regions": []}
            if row["changed_pixels"]:
                truth.pop("regions")
                mask = mask_dir / f"{name}.pbm"
                if name not in checked:
                    checked[name] = pack_truth(media / row["mask"], row["mask_sha256"], mask, extent)
                truth["mask"] = checked[name]
            truth_rows[split].append(truth)
            method = row["checks"][selection]
            listed = method["score"] > cutoff
            predictions[split].append({"id": name, "status": "ok", "input_sha256": row["sha256"], "response": method["score"],
                                       "localization": {"status": "ok", "regions": [method["box"]] if listed else []}})
            expected[name] = (method["selected"], method["intersection"]) if listed else (0, 0)
            files[name] = row["file"]
        scope = independent.audit_manifest(manifest)
        manifest_sha = scope["manifest_sha256"]
        centre = locked["heads"][str(quality)][selection]
        weights = [independent.digest(centre)]
        if selection != "residual":
            weights.extend(locked["features"]["weight_sha256"])
        save(directory / "manifest.json", manifest)
        save(directory / "relative-deliveries.json", files)
        lock = None
        for split in ("development", "confirmation"):
            truth = {"schema": TRUTH_SCHEMA, "manifest_sha256": manifest_sha, "split": split, "cases": truth_rows[split]}
            receipt = {"manifest_sha256": manifest_sha, "split": split, "system": f"image-region-ranking-{selection}",
                       "implementation_sha256": locked["fitter_sha256"], "weight_sha256": weights,
                       "score_kind": "raw", "response_units": "paired rank response", "predictions": predictions[split]}
            if lock is not None:
                receipt["freeze_sha256"] = independent.digest(lock)
            measured = count_locations(manifest, truth, receipt, mask_directory=mask_dir)
            require(all((r["localization"]["reported_units"], r["localization"]["overlap_units"]) == expected[r["id"]]
                        for r in measured["predictions"]), "Shared overlap counter differs from the native audit")
            if split == "development":
                # The shared counter uses >=. The experiment's frozen comparison uses >.
                lock = independent.freeze(manifest, measured, thresholds={"image": math.nextafter(cutoff, math.inf)})
                save(directory / "freeze.json", lock)
            save(directory / f"{split}-locations.json", receipt)
            save(directory / f"{split}-truth.json", truth)
            if split == "confirmation":
                summary = independent.summarize(manifest, measured, lock, require_freeze_commitment=True)
                save(directory / "aggregate.json", summary)
        print(json.dumps({"quality": quality, "cases": scope["cases"], "groups": scope["groups"],
                          "manifest_sha256": manifest_sha, "aggregate_sha256": sha(directory / "aggregate.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("result", "deliveries", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    export(read(args.result), args.deliveries, args.output)
