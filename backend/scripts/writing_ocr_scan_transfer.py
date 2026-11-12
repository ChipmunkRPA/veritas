# SPDX-License-Identifier: Apache-2.0
"""Measure which exact writing inputs survive a controlled scan."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.writing-ocr-scan-transfer/1"
VIEWS = {"clear": {"dpi": 200, "format": "PNG", "contrast": 1.0},
         "soft_jpeg": {"dpi": 110, "format": "JPEG", "contrast": .64, "quality": 42}}


def read(path):
    return json.loads(path.read_bytes())


def save(path, record):
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rasterize(original, destination, view):
    import pymupdf
    from PIL import Image, ImageEnhance

    scale = view["dpi"] / 72
    with pymupdf.open(original) as source, pymupdf.open() as scanned:
        for page in source:
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csGRAY, alpha=False)
            image = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
            if view["contrast"] != 1.0:
                image = ImageEnhance.Contrast(image).enhance(view["contrast"])
            with io.BytesIO() as stream:
                options = {"quality": view["quality"]} if "quality" in view else {}
                image.save(stream, view["format"], **options)
                scan = scanned.new_page(width=page.rect.width, height=page.rect.height)
                scan.insert_image(scan.rect, stream=stream.getvalue())
        scanned.save(destination, garbage=4, deflate=True, no_new_id=True)


def freeze(formats, run, source_file):
    parent = formats / "freeze.json"
    groups = defaultdict(list)
    for case in read(parent)["cases"]:
        groups[(case["panel"], case["group"])].append(case)
    ranked = sorted(groups, key=lambda pair: hashlib.sha256(f"ocr-scan/1/{pair}".encode()).digest())
    selected = [key for panel, count in (("writing-grammar-seams", 8), ("writing-option-order-transfer", 6))
                for key in [key for key in ranked if key[0] == panel][:count]]
    assert len(selected) == 14
    run.mkdir(mode=0o700)
    records = []
    for group_number, group in enumerate(selected):
        cases = groups[group]
        original, = [case for case in cases if case["kind"] == "original"]
        human, = [case for case in cases if case["kind"] == "human_splice"]
        generated = [case for case in cases if case["kind"] not in {"original", "human_splice"}]
        chosen = sorted(generated, key=lambda case: hashlib.sha256(case["text_sha256"].encode()).digest())[0]
        for role, case in zip(("original", "human_splice", "generated"), (original, human, chosen), strict=True):
            folder = formats / "inputs" / f"{case['index']:03d}"
            source = folder / "excerpt.pdf"
            assert digest(folder / "excerpt.txt") == case["text_sha256"]
            for name, view in VIEWS.items():
                output = run / f"{case['index']:03d}-{name}.pdf"
                rasterize(source, output, view)
                records.append({"index": case["index"], "group": group_number, "role": role, "view": name,
                                "scan": output.name, "scan_sha256": digest(output), "native_pdf_sha256": digest(source),
                                "text_sha256": case["text_sha256"], "native_contexts": case["contexts"]})
    return {"schema": SCHEMA, "views": VIEWS, "parent_sha256": digest(parent), "runner_sha256": digest(Path(__file__)),
            "extraction_sha256": digest(source_file), "records": records}


def word_distance(reference, recognized):
    previous = list(range(len(recognized) + 1))
    for i, word in enumerate(reference, 1):
        current = [i]
        for j, candidate in enumerate(recognized, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (word != candidate)))
        previous = current
    return previous[-1]


def measure(formats, run, lock, source_file):
    import pymupdf

    from app.detectors.text.extract import _ocr_runtime, extract, text_hash
    from app.detectors.text.segment import segment
    from scripts.writing_checkpoint_agreement import native_contexts

    assert lock["parent_sha256"] == digest(formats / "freeze.json")
    assert lock["runner_sha256"] == digest(Path(__file__)) and lock["extraction_sha256"] == digest(source_file)
    assert _ocr_runtime()["available"], "Pinned local OCR data are required"
    output = []
    for i, case in enumerate(lock["records"], 1):
        source = formats / "inputs" / f"{case['index']:03d}"
        native = source / "excerpt.pdf"
        scan = run / case["scan"]
        truth = (source / "excerpt.txt").read_bytes().decode()
        assert digest(scan) == case["scan_sha256"] and digest(native) == case["native_pdf_sha256"]
        assert text_hash(truth) == case["text_sha256"]
        with pymupdf.open(scan) as document:
            assert all(not page.get_text().strip() and len(page.get_images()) == 1 for page in document)
        document = extract(scan, ".pdf")
        pages = document.meta["ocr"]["pages"]
        assert pages and all(row["status"] in {"extracted", "unchecked"} for row in pages)
        contexts = [text_hash(" ".join(sentence.text for sentence in context))
                    for context in native_contexts(segment(document))]
        matching = Counter(contexts) & Counter(case["native_contexts"])
        output.append({"group": case["group"], "role": case["role"], "view": case["view"],
                       "fully_extracted": all(row["status"] == "extracted" for row in pages),
                       "exact_text": document.text == truth, "native_words": len(truth.split()),
                       "word_edits": word_distance(truth.split(), document.text.split()),
                       "exact_contexts": matching.total(), "native_contexts": len(case["native_contexts"]),
                       "ordered_contexts_equal": contexts == case["native_contexts"],
                       "pages": len(pages), "extracted_pages": sum(row["status"] == "extracted" for row in pages),
                       "mapped_words": sum(len(row.get("fragments") or ()) for row in document.source_map)})
        if i % 14 == 0:
            print(f"Scanned {i}/{len(lock['records'])} deliveries", flush=True)
    return output


def summarize(rows):
    totals = {}
    for row in rows:
        group = totals.setdefault(row["view"], Counter())
        group["deliveries"] += 1
        group.update({key: int(value) for key, value in row.items() if key not in {"group", "role", "view"}})
    return totals


def review(receipt):
    rows = receipt["rows"]
    identities = {(row["view"], row["group"], row["role"]) for row in rows}
    expected = {(view, group, role) for view in VIEWS for group in range(14)
                for role in ("original", "human_splice", "generated")}
    assert len(rows) == len(expected) and identities == expected
    for row in rows:
        assert 0 <= row["exact_contexts"] <= row["native_contexts"]
        assert row["fully_extracted"] == (row["pages"] == row["extracted_pages"])
        assert not row["ordered_contexts_equal"] or row["exact_contexts"] == row["native_contexts"]
        assert not row["exact_text"] or (row["word_edits"] == 0 and row["ordered_contexts_equal"])
    assert receipt["totals"] == summarize(rows)
    print(json.dumps(receipt["totals"], sort_keys=True), flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "measure", "review"))
    cli.add_argument("path", type=Path)
    cli.add_argument("formats", type=Path, nargs="?")
    args = cli.parse_args()
    if args.phase == "review":
        review(read(args.path))
        return
    assert args.formats is not None, "Supply the retained format panel directory"
    source_file = Path(__file__).resolve().parents[1] / "app/detectors/text/extract.py"
    if args.phase == "freeze":
        save(args.path / "freeze.json", freeze(args.formats, args.path, source_file))
        print(digest(args.path / "freeze.json"), flush=True)
        return
    lock_path = args.path / "freeze.json"
    lock = read(lock_path)
    assert lock["views"] == VIEWS and lock["schema"] == SCHEMA
    rows = measure(args.formats, args.path, lock, source_file)
    receipt = {"schema": SCHEMA, "views": VIEWS, "freeze_sha256": digest(lock_path),
               "rows": rows, "totals": summarize(rows)}
    review(receipt)
    save(args.path / "result.json", receipt)


if __name__ == "__main__":
    main()
