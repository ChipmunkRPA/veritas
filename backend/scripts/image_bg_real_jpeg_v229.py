"""Evaluate frozen JPEG localization policies on the qualified BG-REAL roster."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from scripts import image_bg_real_acquisition_v221 as acquisition
from scripts import image_bg_real_transport_v225 as transport
from scripts import image_jpeg_diagnostic_v189 as original
from scripts.video_appearance_corrected_v112 import scoped

ROOT, PANELS = original.ROOT, original.PANELS
SOURCE = original.DATA / "image-bg-real-acquisition-v225"
QUALIFICATION = PANELS / "image-bg-real-pixel-qualification-complete-v225.json"
TRANSPORT_FREEZE = PANELS / "image-bg-real-transport-freeze-v225.json"
POLICY_FREEZE = PANELS / "image-jpeg-diagnostic-freeze-v189.json"
PUBLISHED_SHA = {
    QUALIFICATION: "3a46ad593be6a095f25e5998f7c51d9d592de6d413915348b57271111ec50306",
    TRANSPORT_FREEZE: "d3c239bfad6a332ef027dd991ffc2fa1ee016d1c788f245853e29c88735863c4",
    POLICY_FREEZE: "cd0333f18623441dba7177e4f79db69a31174713bc823822d5112a07dd4bba5e",
}
PHOTOS = 24
PHASES = (PHOTOS * len(original.KINDS) + original.CHUNK - 1) // original.CHUNK
CODE = (
    "backend/scripts/image_bg_real_jpeg_v229.py",
    "backend/scripts/image_bg_real_jpeg_runtime_v229.py",
    "backend/tests/test_image_bg_real_jpeg_v229.py",
    "docs/image-bg-real-jpeg-v229-method.md",
)
KINDS, QUALITIES, ARMS = original.KINDS, original.QUALITIES, original.ARMS
read, write, sha, require = original.read, original.write, original.sha, original.require


def public_contract():
    require(
        all(sha(path) == digest for path, digest in PUBLISHED_SHA.items()),
        "Published v225/v189 bytes changed",
    )
    summary, transport_freeze, policy = map(read, (QUALIFICATION, TRANSPORT_FREEZE, POLICY_FREEZE))
    require(
        summary["schema"] == "chipmunk.image-bg-real-pixel-qualification/225"
        and summary["qualified"] == summary["attempted"] == PHOTOS
        and summary["backup_attempted"] == summary["neural_calls"] == summary["target_shortfall"] == 0
        and summary["status_counts"] == {"qualified": PHOTOS}
        and summary["accepted_mask_bindings"] == {"publisher_asset_commitment": PHOTOS}
        and summary["saved_assets_by_host"] == {transport.AWS: PHOTOS * len(acquisition.ASSETS)}
        and summary["plan_sha256"] == transport_freeze["private_plan_sha256"]
        and transport_freeze["schema"] == "chipmunk.image-bg-real-acquisition-plan/225"
        and policy["sensitive_policy"] == original.POLICY_NAME
        and policy["qualities"] == list(QUALITIES)
        and policy["kinds"] == list(KINDS)
        and policy["arms"] == list(ARMS)
        and policy["methods"] == list(original.METHODS)
        and policy["asset_sha256"] == original.PREDICTOR_ASSETS
        and policy["v42_freeze_sha256"] == sha(original.V42)
        and policy["v46_freeze_sha256"] == sha(original.V46)
        and policy["training_chain_sha256"] == sha(original.CHAIN),
        "Changed source admission, model lineage or numerical policy",
    )
    inherited = (
        policy["code_sha256"]
        | transport_freeze["code_commitments"]
        | transport_freeze["successor_code_commitments"]
    )
    require(
        all(sha(ROOT / name) == value for name, value in inherited.items()), "A frozen dependency changed"
    )
    return summary, policy, inherited


def source_cases(summary):
    private = SOURCE / "qualified-private.json"
    require(
        sha(SOURCE / "summary-v225.json") == PUBLISHED_SHA[QUALIFICATION]
        and sha(SOURCE / "summary.json") == summary["transport_base_summary_sha256"]
        and sha(SOURCE / "plan.json") == summary["plan_sha256"]
        and sha(SOURCE / "attempts.jsonl") == summary["acquisition_journal_sha256"]
        and sha(private) == summary["qualified_private_sha256"],
        "The exact sealed v225 private qualification is required",
    )
    sealed = read(private)
    require(
        sealed["schema"] == "chipmunk.image-bg-real-pixel-qualified-private/221"
        and sealed["plan_sha256"] == summary["plan_sha256"]
        and sealed["acquisition_journal_sha256"] == summary["acquisition_journal_sha256"]
        and sealed["attempt_statuses"] == [{"ordinal": n, "status": "qualified"} for n in range(PHOTOS)],
        "Complete v225 pixel qualification required",
    )
    cache = Path(sealed["cache"])
    require(
        str(cache) == read(SOURCE / "plan.json")["private_cache"]
        and cache.is_absolute()
        and cache == cache.resolve(),
        "Qualified source cache path changed",
    )
    cases, assets = [], {}
    for n, row in enumerate(sealed["qualified"]):
        source, measurement = row["source"], row["measurement"]
        source_id = source["recipient"]
        require(
            source_id not in assets
            and source["ordinal"] == measurement["ordinal"] == n
            and source["tier"] == measurement["tier"] == "primary"
            and source["publisher_split"] == measurement["publisher_split"]
            and measurement["pair_failure"] is None
            and measurement["prior_matches"] == []
            and measurement["measured"]["semantic_mask_binding"] == "publisher_asset_commitment"
            and row["truth_file"] == f"{n:02d}/exact_truth.png",
            "Qualified source, grouping or order changed",
        )
        cases.append(
            {
                "source_id": source_id,
                "component": f"bg-real-{n:02d}",
                "publisher_split": source["publisher_split"],
            }
        )
        assets[source_id] = row
    require(
        len(cases) == PHOTOS
        and {
            name: len({r["source"][name] for r in assets.values()})
            for name in ("flickr_photo_id", "flickr_owner")
        }
        == {"flickr_photo_id": PHOTOS, "flickr_owner": PHOTOS}
        and dict(Counter(case["publisher_split"] for case in cases))
        == summary["qualified_by_publisher_split"],
        "Each of the 24 frozen sources needs its own deduplicated component",
    )
    return cases, {"cache": cache, "assets": assets}


def compose_views(source, edited, semantic, exact):
    require(
        source.shape == edited.shape == semantic.shape
        and exact.shape == source.shape[:2]
        and source.ndim == 3
        and source.shape[2] == 3
        and source.dtype == edited.dtype == semantic.dtype == np.uint8
        and np.array_equal(semantic, np.repeat(semantic[:, :, :1], 3, axis=2))
        and np.isin(semantic[:, :, 0], (0, 255)).all()
        and exact.dtype == bool
        and bool(exact.any())
        and np.array_equal(exact, np.any(source != edited, axis=2))
        and not np.any(exact & (semantic[:, :, 0] == 0)),
        "Inconsistent exact edit or support",
    )
    before = Image.fromarray(source)
    alpha = Image.fromarray(semantic[:, :, 0])
    tone = np.clip(
        np.rint((source.astype(np.float64) - 127.5) * 1.12 + 127.5 + np.asarray([3, -2, 1])),
        0,
        255,
    ).astype(np.uint8)
    images = {
        "ai": Image.fromarray(edited),
        "original": before,
        "tone": Image.composite(Image.fromarray(tone), before, alpha),
        "blur": Image.composite(before.filter(ImageFilter.GaussianBlur(1.25)), before, alpha),
        "copy_move": original.ordinary_copy(before, alpha),
    }
    footprints = {"ai": exact, **{kind: original.changed_pixels(before, images[kind]) for kind in KINDS[1:]}}
    require(
        all(not np.any(mask & (semantic[:, :, 0] == 0)) for mask in footprints.values()),
        "A conventional control changed pixels outside the published semantic region",
    )
    return images, footprints


def material(cases, index, position):
    cached = index["assets"][cases[position]["source_id"]]
    cache = index["cache"]
    case, observation = cached["source"], cached["measurement"]
    pixels = []
    for role in acquisition.ASSETS:
        body = transport.verify_file(cache, case, role, observation["files"][role])
        image = acquisition.png_pixels(body, case["size"])
        if role != "semantic_mask":
            require(
                acquisition.fingerprint(body, image) == observation["fingerprints"][role],
                "Qualified source RGB and sketch changed",
            )
        pixels.append(image)
    old, edited, semantic = pixels
    exact, support = acquisition.edit_truth(old, edited, semantic, case["semantic_edit_area"])
    measured = observation["measured"]
    require(
        int(exact.sum()) == measured["changed_pixels"]
        and support == measured["support_pixels"]
        and original.digest(exact.astype(np.uint8).tobytes()) == measured["exact_mask_pixel_sha256"]
        and observation["files"]["semantic_mask"]["sha256"] == measured["semantic_mask_sha256"],
        "Exact source-relative pixel truth differs from v225",
    )
    path = cache / cached["truth_file"]
    require(
        path.resolve() == path and sha(path) == cached["truth_png_sha256"], "Sealed exact truth PNG changed"
    )
    saved = acquisition.png_pixels(path.read_bytes(), case["size"])
    require(
        np.array_equal(saved, np.repeat((exact.astype(np.uint8) * 255)[:, :, None], 3, axis=2)),
        "Saved exact truth pixels differ",
    )
    return compose_views(old, edited, semantic, exact)


tasks = scoped(original.tasks, {"PHOTOS": PHOTOS, "PHASES": PHASES})
batch_views = scoped(original.batch_views, {"tasks": tasks, "material": material})
prep_profile = original.prep_profile


def frozen_identity(summary, policy, inherited):
    require(
        original.fixed_versions() == policy["versions"]
        and original.rotary_files() == policy["rotary_source_sha256"],
        "Unchanged v189 numerical runtime required",
    )
    return {
        "schema": "chipmunk.image-bg-real-jpeg-comparison-freeze/229",
        "source_family": "BG-REAL v2 LaMa object removal",
        "photos": PHOTOS,
        "source_groups": PHOTOS,
        "qualification_sha256": PUBLISHED_SHA[QUALIFICATION],
        "qualified_private_sha256": summary["qualified_private_sha256"],
        "qualification_transport_freeze_sha256": PUBLISHED_SHA[TRANSPORT_FREEZE],
        "v189_policy_freeze_sha256": PUBLISHED_SHA[POLICY_FREEZE],
        "publisher_split_counts": summary["qualified_by_publisher_split"],
        "inherited_code_sha256": inherited,
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "batches": PHASES,
        "views_per_batch": original.CHUNK,
        "views_per_arm_quality": PHOTOS * len(KINDS),
        "unique_model_calls": PHOTOS * len(KINDS) * len(QUALITIES) * len(ARMS),
        "exact_first_input_repeats": len(QUALITIES) * len(ARMS),
        "neural_calls_during_preparation": 0,
        "serving_admitted": False,
        **{
            name: policy[name]
            for name in (
                "qualities",
                "kinds",
                "arms",
                "methods",
                "sensitive_policy",
                "useful_iou",
                "useful_recall",
                "bootstrap_draws",
                "bootstrap_seed",
                "device",
                "dtype",
                "cpu_threads",
                "asset_sha256",
                "candidate_checkpoint_sha256",
                "v42_freeze_sha256",
                "v46_freeze_sha256",
                "training_chain_sha256",
                "versions",
                "rotary_source_sha256",
            )
        },
    }


def prepare(trial, raw_root):
    trial, raw_root = trial.resolve(), raw_root.resolve()
    summary, policy, inherited = public_contract()
    cases, index = source_cases(summary)
    cache = index["cache"]
    for destination in (trial, raw_root):
        require(
            all(
                not destination.is_relative_to(path) and not path.is_relative_to(destination)
                for path in (SOURCE, cache)
            ),
            "New comparison destinations must preserve v225 sources",
        )
    require(
        not trial.exists()
        and raw_root.is_dir()
        and not any(raw_root.iterdir())
        and not trial.is_relative_to(raw_root)
        and not raw_root.is_relative_to(trial),
        "New trial and separate empty private map directory required",
    )
    freeze = frozen_identity(summary, policy, inherited)
    chain = original.prior_chain()
    require(
        all(sha(ROOT / path) == expected for path, expected in original.PREDICTOR_ASSETS.items())
        and sha(Path(chain["final_checkpoint"])) == freeze["candidate_checkpoint_sha256"],
        "Pinned v189 model bytes changed",
    )
    trial.mkdir(mode=0o700)
    (trial / "masks").mkdir()
    entries = []
    for n, case in enumerate(cases):
        images, masks = material(cases, index, n)
        target = trial / "masks" / f"{n:03d}.npz"
        with target.open("xb") as stream:
            np.savez_compressed(stream, **masks)
        entries.append(
            {
                **case,
                "masks_sha256": sha(target),
                "changed_pixels": {kind: int(mask.sum()) for kind, mask in masks.items()},
                "inputs": {
                    str(q): {
                        kind: original.view_binding(*original.jpeg_view(images[kind], q)) for kind in KINDS
                    }
                    for q in QUALITIES
                },
            }
        )
    write(trial / "inputs.json", {"raw_root": str(raw_root), "sources": entries})
    freeze["private_input_bindings_sha256"] = sha(trial / "inputs.json")
    write(trial / "freeze.json", freeze)
    print(sha(trial / "freeze.json"), flush=True)


def lease(trial, expected, profile=None):
    require(sha(trial / "freeze.json") == expected, "Exact v229 freeze SHA-256 required")
    summary, policy, inherited = public_contract()
    freeze = frozen_identity(summary, policy, inherited)
    freeze["private_input_bindings_sha256"] = sha(trial / "inputs.json")
    require(read(trial / "freeze.json") == freeze, "Source, JPEG bindings or frozen policy changed")
    inputs = read(trial / "inputs.json")
    cases, index = source_cases(summary)
    require(
        cases
        == [
            {key: row[key] for key in ("source_id", "component", "publisher_split")}
            for row in inputs["sources"]
        ],
        "Prepared v225 source groups or order changed",
    )
    if profile is not None:
        from scripts.image_bg_real_jpeg_runtime_v229 import assert_profile

        assert_profile(profile, "prepare", ["prepare", str(trial), inputs["raw_root"]])
    return inputs, cases, index


def phase_rows(trial, name, command, args, expected):
    from scripts.image_bg_real_jpeg_runtime_v229 import assert_profile

    part = trial / name
    assert_profile(trial / f"{name}-profile", command, args)
    finished = read(part / "completed.json")
    journal = part / "rows.jsonl"
    require(
        finished["freeze_sha256"] == expected and sha(journal) == finished["rows_sha256"],
        "Completed phase or row journal changed",
    )
    return finished, [json.loads(line) for line in journal.read_text().splitlines()]


producer_rows = scoped(original.producer_rows, {"phase_rows": phase_rows, "tasks": tasks})
score = scoped(original.score, {"lease": lease, "batch_views": batch_views, "tasks": tasks})
readout = scoped(
    original.readout,
    {
        "lease": lease,
        "producer_rows": producer_rows,
        "batch_views": batch_views,
        "tasks": tasks,
    },
)
coverage = scoped(
    original.coverage,
    {
        "lease": lease,
        "producer_rows": producer_rows,
        "PHASES": PHASES,
        "PHOTOS": PHOTOS,
    },
)


def result_write(path, data):
    require(path.name == "summary.json", "The seal writes one aggregate only")
    freeze = read(path.parent / "freeze.json")
    receipts = {}
    for name in data["phase_profile_sha256"].keys() - {"prepare"}:
        receipt = path.parent / name / "completed.json"
        receipts[name] = {"completed_sha256": sha(receipt), "rows_sha256": read(receipt)["rows_sha256"]}
    write(
        path,
        {
            **data,
            "schema": "chipmunk.image-bg-real-jpeg-comparison-result/229",
            "qualification_sha256": freeze["qualification_sha256"],
            "v189_policy_freeze_sha256": freeze["v189_policy_freeze_sha256"],
            "publisher_split_counts": freeze["publisher_split_counts"],
            "phase_output_sha256": receipts,
            "scope": (
                "BG-REAL v2 LaMa object removal. Both qualities reuse the same 24 deduplicated source groups. "
                "Paired group bootstrap intervals are conditional and pointwise. "
                "Cue localization is separate from whole-file authorship confidence."
            ),
        },
    )


seal = scoped(
    original.seal,
    {
        "lease": lease,
        "producer_rows": producer_rows,
        "phase_rows": phase_rows,
        "PHASES": PHASES,
        "tasks": tasks,
        "write": result_write,
    },
)


def parse_args(*, monitor=False):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("trial", type=Path)
    build.add_argument("raw_root", type=Path)
    names = ("score", "readout", "seal") if monitor else ("score", "readout", "seal", "coverage")
    for name in names:
        command = commands.add_parser(name)
        command.add_argument("trial", type=Path)
        command.add_argument("freeze_sha256")
        if name == "score":
            command.add_argument("arm", choices=ARMS)
        if name in {"score", "readout"}:
            command.add_argument("quality", type=int, choices=QUALITIES)
            command.add_argument("phase", type=int, choices=range(PHASES))
    args = parser.parse_args()
    return args.command, list(vars(args).values())[1:]


def main():
    command, args = parse_args()
    args[0] = args[0].resolve()
    if command == "prepare":
        args[1] = args[1].resolve()
    {"prepare": prepare, "score": score, "readout": readout, "seal": seal, "coverage": coverage}[command](
        *args
    )


if __name__ == "__main__":
    main()
