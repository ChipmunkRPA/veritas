# SPDX-License-Identifier: Apache-2.0
"""Replay exact picture locations and ordinary incremental saves in PDFs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
from pathlib import Path

from scripts.audio_native_alignment_panel import read, save
from scripts.pdf_saved_wording_panel import _canonical_save

WIDTH, HEIGHT = 420, 320
LEFT, TOP = 56, 142
BOX = (LEFT, TOP, LEFT + WIDTH, TOP + HEIGHT)
ENCODINGS = ("png", "jpeg")
CASES = ("generated_patch", "later_note", "local_tone", "metadata", "note", "reencoded", "global_tone", "covered", "full_save")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def encode(values, mode, quality=88):
    from PIL import Image

    with io.BytesIO() as body:
        options = {"quality": quality, "subsampling": 2} if mode == "jpeg" else {}
        Image.fromarray(values).save(body, mode.upper(), **options)
        return body.getvalue()


def source_pixels(path, digest):
    import numpy as np
    from PIL import Image

    from app.detectors.image.reference_heif import display_pixels

    data = path.read_bytes()
    if sha(data) != digest:
        raise ValueError(f"Photo does not match its pinned source: {path.name}")
    with Image.open(io.BytesIO(data)) as image:
        rgb = Image.fromarray(display_pixels(image))
        return np.array(rgb.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS), dtype=np.uint8)


def sample_pair(source, earlier, edited):
    import numpy as np

    hashes = source["fingerprints"]
    before, donor = (source_pixels(path, hashes[key]["encoded_sha256"])
                     for path, key in ((earlier, "original"), (edited, "edited")))
    delta = np.max(np.abs(before.astype(np.int16) - donor.astype(np.int16)), axis=2)
    # The stage uses the densest known changed area in a fixed grid, before any detection.
    width, height = 76, 68
    _, x, y = max((int(np.count_nonzero(delta[y:y+height, x:x+width] > 8)), x, y)
                  for y in range(25, HEIGHT - height - 20, 25) for x in range(25, WIDTH - width - 20, 25))
    selected = np.s_[y:y+height, x:x+width]
    changed = before.copy()
    changed[selected] = donor[selected]
    tone = before.copy()
    tone[selected] = np.clip(tone[selected] * .64 + 8, 0, 255).astype(np.uint8)
    target = [2 * value for value in (LEFT + x, TOP + y, LEFT + x + width, TOP + y + height)]
    return before, changed, tone, target


def build(output, before, changed, tone, encoding, rotation):
    import numpy as np
    import pymupdf

    base = output / "earlier.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((56, 65), "Field report with a retained picture")
        page.insert_text((56, 91), "The illustration below accompanies the findings from the site survey.")
        page.insert_text((56, 110), "The field notes and photograph were assembled in a separate review before publication.")
        xref = page.insert_image(pymupdf.Rect(BOX), stream=encode(before, encoding))
        page.set_rotation(rotation)
        _canonical_save(pdf, base)
    for case in CASES:
        path = output / f"{case}.pdf"
        shutil.copyfile(base, path)
        with pymupdf.open(path) as pdf:
            page = pdf[0]
            if case in ("generated_patch", "later_note", "full_save"):
                page.replace_image(xref, stream=encode(changed, encoding))
            elif case == "local_tone":
                page.replace_image(xref, stream=encode(tone, encoding))
            elif case == "metadata":
                pdf.set_metadata(pdf.metadata | {"subject": "An update to the report index"})
            elif case == "note":
                page.add_text_annot((53, 500), "A note outside the picture")
            elif case == "covered":
                page.replace_image(xref, stream=encode(changed, encoding))
                page.draw_rect(pymupdf.Rect(83, 166, 191, 275), color=None, fill=(0, 0, 0), overlay=True)
            elif case == "reencoded":
                page.replace_image(xref, stream=encode(before, "jpeg", quality=74))
            elif case == "global_tone":
                global_tone = np.round(before * .77 + 14).astype(np.uint8)
                page.replace_image(xref, stream=encode(global_tone, encoding))
            if case == "full_save":
                with pymupdf.open(stream=pdf.tobytes(garbage=4, no_new_id=True), filetype="pdf") as rewrite:
                    path.unlink()
                    _canonical_save(rewrite, path)
            else:
                _canonical_save(pdf, path, incremental=True)
        if case == "later_note":
            with pymupdf.open(path) as pdf:
                pdf[0].add_text_annot((56, 504), "A later note outside the picture")
                _canonical_save(pdf, path, incremental=True)
        yield case, path


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def located(regions, target):
    for x0, y0, x1, y1 in regions:
        support = area([max(target[0], x0), max(target[1], y0), min(target[2], x1), min(target[3], y1)])
        union = area(target) + area([x0, y0, x1, y1]) - support
        if 2 * support >= area(target) and support >= .30 * union:
            return True
    return False


def summarize(rows):
    results = {}
    for encoding in ENCODINGS:
        results[encoding] = {}
        for kind in CASES:
            subset = [r for r in rows if r["case"] == kind and r["encoding"] == encoding]
            results[encoding][kind] = {"total": len(subset), "compared": sum(row["compared"] > 0 for row in subset),
                                       "located": sum(located(row["regions"], row["target"]) for row in subset if row["target"]),
                                       "flagged": sum(bool(row["regions"]) for row in subset),
                                       "wording_flags": sum(row["wording_status"] == "changed" for row in subset)}
    return results


def source_pairs(origin):
    indexed = origin / "sources.json"
    if indexed.is_file():
        for row in read(indexed):
            yield row, origin / "photo-fixtures" / f"{row['slot']}-photo.png", origin / "photo-fixtures" / f"{row['slot']}-edit.png"
        return
    for slot, row in enumerate(read(origin / "qualification.json")["accepted"]):
        source = row["source"]
        yield ({"slot": slot, "photo": source["flickr_photo_id"], "author": source["flickr_owner"],
                "attribution": source["photo_attribution"], "generator": source["generator"],
                "fingerprints": {"original": {"encoded_sha256": source["original"]["sha256"]},
                                 "edited": {"encoded_sha256": source["edited"]["sha256"]}}},
               origin / row["folder"] / "original.png", origin / row["folder"] / "edited.png")


def run(output, inputs, prior):
    from app.evidence import document_pdf_history, document_pdf_picture_history

    earlier_sources = read(prior)["sources"] if prior else []
    seen_photos = {row["photo"] for row in earlier_sources}
    seen_authors = {row["author"] for row in earlier_sources}
    rows, sources = [], []
    for origin in inputs:
        for source, earlier, edited in source_pairs(origin):
            slot = source["slot"]
            if source["photo"] in seen_photos or source["author"] in seen_authors:
                raise ValueError("A photo or contributor occurs twice")
            seen_photos.add(source["photo"])
            seen_authors.add(source["author"])
            sources.append(source)
            before, changed, tone, target = sample_pair(source, earlier, edited)
            for encoding in ENCODINGS:
                group = output / source["photo"] / encoding
                group.mkdir(parents=True)
                for case, path in build(group, before, changed, tone, encoding, slot % 4 * 90):
                    data = path.read_bytes()
                    reading, _ = document_pdf_picture_history.inspect(data)
                    comparisons = reading["comparisons"]
                    changes = [row for _, row in document_pdf_picture_history.binding.locations(reading)]
                    rows.append({"source": source["photo"], "case": case, "encoding": encoding,
                                 "sha256": sha(data), "wording_status": document_pdf_history.inspect(data)["status"],
                                 "status": reading["status"], "reason": reading["reason"],
                                 "saves": reading["saves"], "compared": max(
                                     (sum(row["compared"] for row in comparison["pages"]) for comparison in comparisons), default=0),
                                 "regions": [row["region_pixels"] for row in changes],
                                 "target": target if case in {"generated_patch", "later_note", "local_tone"} else None})
    save(output / "results.json.gz", {"sources": sources, "rows": rows, "summary": summarize(rows),
                                     "prior_receipt_sha256": sha(prior.read_bytes()) if prior else None})
    print(json.dumps(summarize(rows), indent=2))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    command = cli.add_subparsers(dest="command", required=True)
    measure = command.add_parser("run")
    measure.add_argument("output", type=Path)
    measure.add_argument("sources", type=Path, nargs="+")
    measure.add_argument("--prior", type=Path)
    verify = command.add_parser("verify")
    verify.add_argument("receipt", type=Path)
    verify.add_argument("--prior", type=Path)
    args = cli.parse_args()
    if args.command == "run":
        run(args.output, args.sources, args.prior)
    else:
        recorded = read(args.receipt)
        sources = recorded["sources"]
        if args.prior:
            previous = read(args.prior)["sources"]
            if sha(args.prior.read_bytes()) != recorded["prior_receipt_sha256"]:
                raise ValueError("The confirmation receipt commits to a different earlier panel")
            for key in ("photo", "author"):
                if {row[key] for row in sources} & {row[key] for row in previous}:
                    raise ValueError("A confirmation photo or contributor already occurs in the earlier panel")
        cases = {(row["source"], row["encoding"], row["case"]) for row in recorded["rows"]}
        expected = {(row["photo"], enc, kind) for row in sources for enc in ENCODINGS for kind in CASES}
        if cases != expected or len(cases) != len(recorded["rows"]):
            raise ValueError("At least one frozen delivery is missing, duplicated or unplanned")
        summary = summarize(recorded["rows"])
        if summary != recorded["summary"]:
            raise ValueError("Picture location counts differ from the recorded inputs")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
