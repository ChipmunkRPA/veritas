"""Verify earlier-copy comparison across lossless progressive JPEG rewrites."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

SCHEMA = "chipmunk.image-reference-progressive-transfer/1"
PLAN = {
    "source": "First six supported CC BY 2.0 LaMa sources by original SHA-256 from each previously measured BG-REAL roster.",
    "roles": ["unchanged", "sharp", "feathered", "local_blur", "local_tone", "local_copy", "global_resize",
              "second_jpeg", "unrelated", "shifted", "reflection"],
    "pairs": ["baseline", "earlier_progressive", "submitted_progressive", "both_progressive"],
    "parity": "Require identical RGB, exact-block evidence and masks across lossless coefficient transcodes.",
    "scope": "Compare baseline and progressive standard quality-90 RGB JPEG pairs with matching dimensions.",
}
ORDINARY = {"unchanged", "global_resize", "second_jpeg"}
INCOMPATIBLE = {"unrelated", "shifted", "reflection"}
EDITED = {"sharp", "feathered"}
EXACT_SCHEMA = "chipmunk.image-reference-comparison/14"


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def summarize(rows):
    totals = {arm: Counter() for arm in ("previous", "extended")}
    for row in rows:
        role, target = row["role"], row["changed"]
        for arm in totals:
            status, _, selected, overlap = row[arm]
            counts = totals[arm]
            counts["measured"] += status == "comparable"
            counts["known_edits_located"] += role in EDITED and 2 * overlap >= target and 10 * overlap >= 3 * (selected + target - overlap)
            counts["ordinary_highlights"] += role in ORDINARY and selected > 0
            counts["local_ordinary_highlights"] += role.startswith("local_") and selected > 0
            counts["incompatible_highlights"] += role in INCOMPATIBLE and selected > 0
    return {key: dict(values) for key, values in totals.items()}


def review(receipt):
    rows = receipt["rows"]
    check(receipt["schema"] == SCHEMA and receipt["plan"] == PLAN and len(rows) == 132, "Unexpected source panel")
    roster = receipt["sources"]
    check(len(roster) == len(set(roster)) == 12 and len({(row["group"], row["role"]) for row in rows}) == 132,
          "Source groups or delivery roles repeat")
    for row in rows:
        check(1 <= row["group"] <= len(roster) and row["source_sha256"] == roster[row["group"] - 1]
              and row["role"] in PLAN["roles"] and row["progressive_pairs"] == 3, "Invalid source or transport commitment")
        check(row["baseline_decode"] == row["progressive_decode"] and row["extended"] == row["baseline"],
              "Progressive encoding changed the RGB pixels or complete comparison")
        for arm in ("previous", "extended", "baseline"):
            status, schema, selected, overlap = row[arm]
            check(0 <= overlap <= min(row["changed"], selected) and 0 <= selected <= row["pixels"], "Invalid overlap")
            check(status == "comparable" or selected == 0, "Unassessed comparison has a highlight")
            if arm == "extended" and row["previous"] != row["extended"]:
                check(schema == EXACT_SCHEMA and row["previous"][0] != "comparable", "A previous comparison was replaced")
        check(row["role"] != "unchanged" or row["extended"][2] == 0, "Unchanged pixels were highlighted")
    totals = summarize(rows)
    check(receipt["summary"] == totals, "Frozen results do not recount")
    return totals


def run(args):
    import cv2
    import numpy as np
    from PIL import Image

    from app.detectors.image import jpeg_bitstream, reference_same_jpeg
    from app.evidence.image_reference import inspect
    from scripts.image_reference_delivery_envelope_lama import source_pixels
    from scripts.image_residual_encoder_transfer import authored, encode

    binary = shutil.which("jpegtran")
    check(binary is not None, "Install jpegtran to regenerate the comparison")
    cv2.setNumThreads(1)
    cases, seen = [], set()
    for directory in args.rosters:
        raw = (directory / "qualified-private.json").read_bytes()
        recorded = json.loads((directory / "summary.json").read_bytes())
        check(digest(raw) == recorded["qualified_private_sha256"], "The publisher intake record differs")
        roster = json.loads(raw)
        candidates = [row for row in roster["qualified"] if
                      512 <= min(row["source"]["size"]) <= max(row["source"]["size"]) <= 768]
        check(len(candidates) >= 6, "Insufficient qualified source photos")
        for row in sorted(candidates, key=lambda item: item["source"]["original"]["sha256"])[:6]:
            source = row["source"]
            original_sha = source["original"]["sha256"]
            identity = {str(source["flickr_photo_id"]), source["flickr_owner"].casefold(),
                        source["photo_attribution"]["Author"].casefold(), original_sha}
            check(not seen & identity and source["photo_attribution"]["License"] == "https://creativecommons.org/licenses/by/2.0/",
                  "Photo identity or terms differ")
            seen.update(identity)
            original, edited, _ = source_pixels(Path(roster["cache"]), row)
            cases.append((original_sha, original, edited))

    def transcode(body):
        out = subprocess.run([binary, "-copy", "all", "-progressive"], input=body, capture_output=True, check=True, timeout=10).stdout
        initial = jpeg_bitstream.read(body)
        progressive = jpeg_bitstream.progressive_frame(out)
        check(progressive is not None and initial.fingerprint == progressive[0].fingerprint,
              "JPEG transport changed quantization tables or sampling")
        with Image.open(io.BytesIO(body)) as before, Image.open(io.BytesIO(out)) as after:
            check(before.mode == after.mode == "RGB" and before.size == after.size and before.tobytes() == after.tobytes(),
                  "JPEG transport changed decoded RGB")
            return out, digest(before.tobytes()), digest(after.tobytes())

    def measurements(result, truth, output):
        mask = result.get("mask")
        if mask:
            with Image.open(output / mask["file"]) as image:
                selected = np.asarray(image.convert("L")) > 0
            counts = int(selected.sum()), int((selected & truth).sum())
        else:
            counts = 0, 0
        shutil.rmtree(output)
        return [result["status"], result["schema"], *counts]

    rows = []
    with TemporaryDirectory() as folder:
        root = Path(folder)
        paths = [root / f"{name}.jpg" for name in ("earlier", "earlier-progressive", "submitted", "submitted-progressive")]
        output = root / "evidence"
        common = {"allow_proportional_resize": True, "allow_exact_jpeg_identity": True, "allow_shifted_jpeg": True,
                  "allow_structure_jpeg": False, "allow_jpeg_detail_loss": False}
        # Isolate the exact-block fallback. The separate baseline-only tone
        # and structure route can win before it on sequential JPEGs.
        for group, (pin, original, edited) in enumerate(cases, 1):
            ordinary, generated = authored({"original": original, "edited": edited}, np.any(original != edited, axis=2), pin)
            ordinary["unchanged"] = ordinary.pop("original")
            donor = cv2.resize(cases[group % len(cases)][1], original.shape[1::-1], interpolation=cv2.INTER_AREA)
            variants = ordinary | generated | {"unrelated": donor, "shifted": np.roll(original, 16, axis=1),
                                               "reflection": original[:, ::-1]}
            paths[0].write_bytes(encode(original, 90))
            progressive, earlier_rgb, transcode_earlier_rgb = transcode(paths[0].read_bytes())
            paths[1].write_bytes(progressive)
            for role in PLAN["roles"]:
                pixels = variants[role]
                truth = np.any(pixels != original, axis=2) if role in EDITED or role.startswith("local_") else np.zeros(original.shape[:2], bool)
                paths[2].write_bytes(encode(pixels, 90))
                progressive, submitted_rgb, transcode_submitted_rgb = transcode(paths[2].read_bytes())
                paths[3].write_bytes(progressive)
                baseline = inspect(paths[0], paths[2], evidence_dir=output, allow_same_jpeg=True, **common)
                # The full-route result may be accepted by an earlier general
                # check. The existing JPEG-only proof must also replay exactly.
                with Image.open(paths[0]) as a, Image.open(paths[2]) as b:
                    exact, _ = reference_same_jpeg.compare(a, b, paths[0].read_bytes(), paths[2].read_bytes())
                actual = measurements(baseline, truth, output)
                previous = inspect(paths[1], paths[3], evidence_dir=output, allow_same_jpeg=False, **common)
                old = measurements(previous, truth, output)
                for i, j in ((1, 2), (0, 3), (1, 3)):
                    result = inspect(paths[i], paths[j], evidence_dir=output, allow_same_jpeg=True, **common)
                    normalized = {k: v for k, v in result.items() if k not in {"reference", "submitted"}}
                    expected = {k: v for k, v in baseline.items() if k not in {"reference", "submitted"}}
                    metrics = measurements(result, truth, output)
                    check(normalized == expected and metrics == actual,
                          f"Complete evidence differs for progressive pair {group}/{role}/{i}/{j}")
                    with Image.open(paths[i]) as a, Image.open(paths[j]) as b:
                        candidate, _ = reference_same_jpeg.compare(a, b, paths[i].read_bytes(), paths[j].read_bytes())
                    check(candidate == exact, "Exact-block proof changed across JPEG transports")
                rows.append({"group": group, "source_sha256": pin, "role": role, "pixels": int(truth.size),
                             "changed": int(truth.sum()), "baseline_decode": [earlier_rgb, submitted_rgb],
                             "progressive_decode": [transcode_earlier_rgb, transcode_submitted_rgb], "progressive_pairs": 3,
                             "baseline": actual, "previous": old, "extended": actual})
            print(f"Compared {group} of 12 source photos", flush=True)
    receipt = {"schema": SCHEMA, "plan": PLAN, "sources": [pin for pin, *_ in cases], "rows": rows, "summary": summarize(rows)}
    review(receipt)
    with args.receipt.open("x") as output:
        json.dump(receipt, output, indent=2, sort_keys=True)
        output.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "review"))
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--rosters", type=Path, nargs=2)
    args = parser.parse_args()
    if args.phase == "run":
        check(args.rosters, "Supply the two earlier qualified public-source rosters")
        run(args)
    print(json.dumps(review(json.loads(args.receipt.read_bytes())), sort_keys=True))


if __name__ == "__main__":
    main()
