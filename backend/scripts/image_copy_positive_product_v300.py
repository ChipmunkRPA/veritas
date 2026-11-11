"""Guarded current-product comparison on v300 generated and ordinary variants."""

from __future__ import annotations

import io
import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scripts import image_copy_corroboration_v295 as cue
from scripts import image_copy_positive_transfer_v300 as transfer

require, sha, write, flux = transfer.require, transfer.sha, transfer.write, transfer.flux
FORMATS, VARIANTS = transfer.FORMATS, transfer.VARIANTS
ROUTE_KEY = {"native_png": 0, "jpeg90": 75}


def product(fmt, batch, expected):
    require(fmt in FORMATS, "Exact frozen product format required")
    transfer.guarded(
        f"product-{fmt}",
        batch,
        expected,
        sys.executable,
        f"_product-{fmt}",
        lambda: transfer.generated(batch, expected),
    )


def encoded_view(rgb, fmt):
    buffer = io.BytesIO()
    image = Image.fromarray(rgb)
    if fmt == "jpeg90":
        image.save(buffer, format="JPEG", quality=cue.JPEG_QUALITY)
    else:
        image.save(buffer, format="PNG")
    payload = buffer.getvalue()
    with Image.open(io.BytesIO(payload)) as decoded:
        view = decoded.convert("RGB")
    require(view.size == image.size, "Product encoding changed native coordinates")
    if fmt == "native_png":
        require(np.array_equal(rgb, np.asarray(view)), "Lossless product view changed source pixels")
    return payload, view


def measure_selected(pixels, selected, generated, ordinary, row, kind):
    from scripts.localization_benchmark import mask_metrics

    # The product selection stays intact; measure the cue on each retained native connected component.
    before = flux.digest(selected)
    total, labels = cv2.connectedComponents(selected.astype(np.uint8), connectivity=8)
    regions = []
    for label in range(1, total):
        component = labels == label
        evidence = cue.corroborate(pixels, component)
        require(evidence["proposal_mask_sha256"] == flux.digest(component), "Product region changed")
        supported = evidence["status"] == "copy_consistent"
        donor = evidence["donor"]
        expected_transform = {
            "translated_copy": "translation",
            "reflected_copy": "horizontal_reflection",
            "remote_reflected_copy": "horizontal_reflection",
            "mixed": "horizontal_reflection",
        }.get(kind)
        ordinary_overlap = int((component & ordinary).sum())
        generated_metrics = mask_metrics(component, generated)
        regions.append(
            {
                **evidence,
                "generated_intersection": generated_metrics["intersection"],
                "generated_iou": generated_metrics["iou"],
                "generated_recall": generated_metrics["recall"],
                "ordinary_intersection": ordinary_overlap,
                "copy_supported_generated_intersection": generated_metrics["intersection"]
                if supported
                else 0,
                "exact_ordinary_donor": bool(
                    ordinary_overlap
                    and donor
                    and donor["transform"] == expected_transform
                    and donor["box_xywh"] == row["donor_box_xywh"]
                ),
            }
        )
    require(flux.digest(selected) == before, "Product selection changed")
    return {
        "source_mask_sha256": before,
        "generated": mask_metrics(selected, generated),
        "ordinary": mask_metrics(selected, ordinary),
        "cue": regions,
    }


def worker(fmt, batch, expected):
    from app.detectors.image.regional_support import readout
    from scripts import image_flux_product_comparison_v288 as original

    _, stage, destination = transfer.stage_paths(f"product-{fmt}", batch)
    require(
        (stage / "resources.jsonl").is_file()
        and (stage / "worker.log").is_file()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Exact offline product worker guard required",
    )
    network = []

    def offline(event, _args):
        if event in {
            "socket.connect",
            "socket.sendto",
            "socket.getaddrinfo",
            "socket.gethostbyname",
            "socket.gethostbyaddr",
        }:
            network.append(event)
            raise RuntimeError("Offline product worker refused network activity")

    sys.addaudithook(offline)
    selected, output, generated_receipts = transfer.generated(batch, expected)
    contract = transfer.product_contract()
    route_key = ROUTE_KEY[fmt]
    require(
        contract["routes"][fmt] == original.ROUTES[route_key]
        and contract["configured_product_route"] == original.installed_product_routes()
        and contract["versions"] == original.original.fixed_versions(),
        "Exact v288 runtime and current product path required",
    )
    assets = {
        path: value
        for path, value in contract["asset_sha256"].items()
        if ("/splice/" in path) == (fmt == "native_png")
    }
    require(
        len(assets) == (1 if fmt == "native_png" else 2)
        and all(sha(transfer.ROOT / path) == digest for path, digest in assets.items()),
        "Exact v288 route weights required",
    )
    cv2.setNumThreads(1)
    predict = original.original.v28.load_predictor(
        "baseline", original.ROUTES[route_key]["predictor_format"], None
    )
    destination.mkdir(mode=0o700)
    journal = destination / "rows.jsonl"
    rows, repeated = [], False
    with journal.open("x") as stream:
        for before, after, _ in selected:
            variants, _ = transfer.source_material(before, after, output)
            for kind in VARIANTS:
                row = {"source_index": before["source_index"], "kind": kind, "format": fmt}
                if kind not in variants:
                    row["status"] = "unassessed_generator_geometry"
                else:
                    rgb, generated, ordinary = variants[kind]
                    payload, image = encoded_view(rgb, fmt)
                    raw, details = predict(image)
                    require(
                        raw.shape == original.response_shape("baseline", route_key, image),
                        "Unexpected product raw geometry",
                    )
                    if batch == 1 and not repeated:
                        repeat, repeat_details = predict(image)
                        require(
                            np.array_equal(raw, repeat) and details == repeat_details,
                            "First product input did not replay",
                        )
                        repeated = True
                    gate, threshold = original.current_gate(raw, details, route_key)
                    pixels = np.asarray(image)
                    selection = readout(raw, gate, threshold, payload, pixels).source_mask(
                        image.width, image.height
                    )
                    row.update(
                        status="scored",
                        image_gate=gate,
                        raw_map_sha256=flux.digest(raw),
                        submitted_rgb_sha256=flux.digest(pixels),
                        generated_truth_sha256=flux.digest(generated),
                        ordinary_truth_sha256=flux.digest(ordinary),
                        result=measure_selected(pixels, selection, generated, ordinary, before, kind),
                    )
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
    require(
        not network and len(rows) == len(selected) * len(VARIANTS),
        "Every product view must be retained offline",
    )
    unique = sum(row["status"] == "scored" for row in rows)
    write(
        destination / "result.json",
        {
            "schema": "chipmunk.image-copy-positive-product-stage/300",
            "status": "complete",
            "freeze_sha256": expected,
            "format": fmt,
            "batch": batch,
            "planned_views": len(rows),
            "unique_views_scored": unique,
            "repeat_forwards": int(repeated),
            "new_neural_calls": unique + int(repeated),
            "network_attempts": 0,
            "generated_batch_receipts": generated_receipts,
            "rows_sha256": sha(journal),
        },
    )


def verified_rows(fmt, batch, expected):
    _, stage, output = transfer.stage_paths(f"product-{fmt}", batch)
    receipt, profile_path = output / "result.json", stage / "profile.json"
    result, profile = (json.loads(path.read_bytes()) for path in (receipt, profile_path))
    journal = output / "rows.jsonl"
    require(
        result["status"] == "complete"
        and result["freeze_sha256"] == expected
        and (result["format"], result["batch"]) == (fmt, batch)
        and result["network_attempts"] == 0
        and profile["child_started"]
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and not profile["resource_limit_observations"]
        and profile["runtime_sha256"] == sha(Path(transfer.__file__))
        and profile["worker_receipt_sha256"] == sha(receipt)
        and profile["resource_journal_sha256"] == sha(stage / "resources.jsonl")
        and result["rows_sha256"] == sha(journal)
        and result["new_neural_calls"] == profile["new_neural_calls"]
        and result["repeat_forwards"] == (batch == 1),
        "Complete bounded product stage required",
    )
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    require(
        [(row["source_index"], row["kind"], row["format"]) for row in rows]
        == [(index, kind, fmt) for index in range(2 * (batch - 1), 2 * batch) for kind in VARIANTS]
        and result["planned_views"] == len(rows)
        and sum(row["status"] == "scored" for row in rows) == result["unique_views_scored"],
        "Exact expected stage rows required",
    )
    return (
        rows,
        {"result_sha256": sha(receipt), "profile_sha256": sha(profile_path)},
        result["new_neural_calls"],
    )


def summary(rows, contract):
    measured = [row["result"] for row in rows if row["status"] == "scored"]
    return {
        "planned": len(rows),
        "statuses": dict(sorted(Counter(row["status"] for row in rows).items())),
        "groups_with_highlights": sum(bool(row["cue"]) for row in measured),
        "groups_with_useful_generated_localization": sum(
            row["generated"]["reference_positive_pixels"] > 0
            and row["generated"]["iou"] >= contract["useful_iou"]
            and row["generated"]["recall"] >= contract["useful_recall"]
            for row in measured
        ),
        "groups_with_useful_generated_component": sum(
            any(
                region["generated_recall"] is not None
                and region["generated_iou"] >= contract["useful_iou"]
                and region["generated_recall"] >= contract["useful_recall"]
                for region in row["cue"]
            )
            for row in measured
        ),
        "groups_with_ordinary_overlap": sum(row["ordinary"]["intersection"] > 0 for row in measured),
        "groups_with_copy_supported_generated_overlap": sum(
            any(region["copy_supported_generated_intersection"] > 0 for region in row["cue"])
            for row in measured
        ),
        "groups_with_exact_ordinary_donor": sum(
            any(region["exact_ordinary_donor"] for region in row["cue"]) for row in measured
        ),
        "component_cue_outcomes": dict(
            sorted(Counter(region["status"] for row in measured for region in row["cue"]).items())
        ),
    }


def seal(expected):
    transfer.frozen_rows(expected)
    contract = transfer.product_contract()
    result_path = transfer.TRIAL / "private-product-result.json"
    require(not result_path.exists(), "Preserve existing product result")
    stages, formats, neural_calls = {}, {}, 0
    for fmt in FORMATS:
        collected = []
        for batch in range(1, transfer.BATCHES + 1):
            rows, stages[f"{fmt}-{batch:02d}"], calls = verified_rows(fmt, batch, expected)
            collected.extend(rows)
            neural_calls += calls
        formats[fmt] = {
            kind: summary([row for row in collected if row["kind"] == kind], contract) for kind in VARIANTS
        }
    result = {
        "schema": "chipmunk.image-copy-positive-product-aggregate/300",
        "freeze_sha256": expected,
        "source_groups": transfer.GROUPS,
        "planned_views": transfer.GROUPS * len(FORMATS) * len(VARIANTS),
        "new_neural_calls": neural_calls,
        "stage_receipts": stages,
        "by_format": formats,
        "serving_admitted": False,
    }
    write(result_path, result)
    return {**result, "private_result_sha256": sha(result_path)}
