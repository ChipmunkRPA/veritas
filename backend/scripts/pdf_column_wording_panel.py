"""Compare earlier wording on controlled reflows and intact published PDF pages."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.pdf-column-wording-panel/1"
ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "docs/evaluation-panels"
GAO_ID = re.compile(rb"\bGAO-\d{2}-\d{1,6}(?:SP|R|T)?\b", re.IGNORECASE)
PILOT = {"GAO-15-149", "GAO-20-530", "GAO-21-104648"}
UNAVAILABLE = {"mapped": False, "sources": [], "ranges": [], "native_locations": [], "sections": []}
SHARDS = (
    ("test", "gpt-5.4", "9e28d157d336a343106487c8acab242ee41829f49ceb748dcb5f9fd6a9f9fb33"),
    ("train", "gemini-2.5-flash", "c4af2313124b86f5003f6635bf451960f6cbd015a849fb33cd35935c89cf778e"),
)


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())


def save(path, data):
    raw = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw + b"\n")


def ranked(items, seed):
    return sorted(items, key=lambda name: hash_bytes(f"{SCHEMA}/{seed}/{name}".encode()))


def exclusions(directory):
    exposed = set(PILOT)
    files = {}
    for path in sorted(directory.iterdir()):
        if not path.name.startswith(("writing-", "text-", "document-gao-")) or path.suffix not in {".gz", ".json"}:
            continue
        raw = path.read_bytes()
        exposed.update(match.decode().upper() for match in GAO_ID.findall(gzip.decompress(raw) if path.suffix == ".gz" else raw))
        files[path.name] = hash_bytes(raw)
    return exposed, files


def official_rows(library, used):
    output = []
    for first, last in (("2009", "2012"), ("2013", "2016"), ("2017", "2020")):
        names = [key for key, row in library.items() if first <= row["published_date"][:4] <= last and key not in used
                 and re.fullmatch(r"GAO-\d{2}-\d{1,6}", key) and 2_400 <= len(row["text"]) <= 6_000]
        for key in ranked(names, f"official/{first}")[:8]:
            output.append({"id": key, "url": f"https://files.gao.gov/assets/{key.lower()}.pdf",
                           "reference_in_pack": True, "report_sha256": library[key]["source_record_sha256"]})
    return output


def source_roster(plan):
    return {**plan, "works": [{**work, "variants": [{k: v for k, v in row.items() if k != "text"}
                                                   for row in work["variants"]]} for work in plan["works"]]}


def acquire(plan, output):
    from concurrent.futures import ThreadPoolExecutor
    from subprocess import run

    output.mkdir(mode=0o700)

    def download(row):
        name = row["id"].lower()
        url = f"https://files.gao.gov/assets/{name}.pdf"
        if not re.fullmatch(r"gao-\d{2}-\d{1,6}", name) or row["url"] != url:
            raise ValueError("An official source must use its expected publisher URL")
        pending = output / f"{name}.part"
        outcome = run(["curl", "--fail", "--location", "--silent", "--show-error", "--proto", "=https",
                       "--proto-redir", "=https", "--max-time", "35", "--max-filesize", "18000000",
                       "--output", str(pending), url], capture_output=True, check=False)
        if outcome.returncode == 0:
            with pending.open("rb") as received:
                pdf = received.read(5) == b"%PDF-"
            if pdf:
                pending.rename(pending.with_suffix(".pdf"))
                return name, "downloaded"
        pending.unlink(missing_ok=True)
        return name, "unavailable"

    with ThreadPoolExecutor(max_workers=4) as pool:
        statuses = dict(pool.map(download, read(plan)["official"]))
    save(output / "downloads.json", statuses)
    print(f"Acquired {sum(s == 'downloaded' for s in statuses.values())} of {len(statuses)} official reports", flush=True)


def prepare(cache, pack, output, *, prior=None, roster=None):
    from scripts import writing_report_sources as source

    if hash_bytes(pack.read_bytes()) != source.GAO_SHA256:
        raise ValueError("The source archive does not match the pinned edition")
    library = read(pack)["reports"]
    frozen = read(roster) if roster else None
    if frozen:
        excluded, receipts = set(), frozen["exclusions"]
        official = frozen["official"]
    else:
        excluded, receipts = exclusions(REPORTS)
        if prior:
            previous = read(prior)
            excluded.update(row["id"] for row in [*previous["works"], *previous["official"]])
        official = official_rows(library, excluded)
    seen = excluded | {row["id"] for row in official}
    works = []
    for split, family, pin in SHARDS:
        pairs = source.test_pairs(cache / f"default/{split}/reports_{family}.csv", family, split, pin)
        accepted = 0
        picks = ([row["dataset_id"] for row in frozen["works"] if row["family"] == family]
                 if frozen else ranked(pairs, family))
        for identifier in picks:
            key = identifier.removeprefix("GAO_").upper()
            if key in seen or set(pairs[identifier]) != {"v0", "v1"}:
                continue
            work, _ = source.verified_gao_work(identifier, pairs[identifier], library.get(key), family, split,
                                                max_edits=4, changed_only=True)
            if work is None:
                continue
            variants = [row for row in work["variants"] if row["layout"] == "published"]
            if any(not 2_400 <= len(row["text"]) <= 4_900 for row in variants):
                continue
            works.append({"id": key, "dataset_id": identifier, "source_record_sha256": work["source_record_sha256"],
                          "family": family, "publisher_split": split, "publisher_ids": work["publisher_ids"],
                          "variants": [{k: row[k] for k in ("version", "text", "native_text_sha256", "truth")} for row in variants]})
            seen.add(key)
            accepted += 1
            if accepted == 18:
                break
        if accepted < 18:
            raise ValueError(f"Only {accepted} fresh report pairs for {family}")
    plan = {"schema": SCHEMA, "archive_sha256": source.GAO_SHA256, "publisher_revision": source.OPAI_REVISION,
            "shards": list(map(list, SHARDS)), "exclusions": receipts, "works": works, "official": official}
    if frozen and source_roster(plan) != frozen:
        raise ValueError("The source roster differs from its pinned files")
    save(output, plan)
    print(f"Frozen source plan with {len(works)} reflow pairs and {len(official)} official downloads", flush=True)


def _wrap(tokens, font, size, width):
    from reportlab.pdfbase.pdfmetrics import stringWidth

    lines, line = [], ""
    for token in tokens:
        candidate = f"{line} {token}".strip()
        if line and stringWidth(candidate, font, size) > width:
            lines.append(line)
            line = token
        else:
            line = candidate
    return [*lines, line] if line else lines


def authored_pdf(text, variant, font, *, covered=False):
    from io import BytesIO

    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    if "PanelSans" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("PanelSans", str(font)))
    raw = BytesIO()
    page = canvas.Canvas(raw, pagesize=(612, 792), invariant=1)
    words = text.split()
    count = len(words)
    if variant == "balanced":
        bands, origins, widths = [0, count // 2, count], [38, 322], [250, 250]
    elif variant == "sidebar":
        bands, origins, widths = [0, count // 3, count], [38, 222], [152, 350]
    elif variant == "three_columns":
        bands, origins, widths = [0, count // 3, 2 * count // 3, count], [32, 225, 418], [156, 156, 156]
    else:
        bands, origins, widths = [0, count], [40], [510]
    for column, (start, end) in enumerate(zip(bands, bands[1:])):
        y = 747 - 5 * column
        for offset in range(start, end, 44):
            draft = page.beginText(origins[column], y)
            draft.setFont("PanelSans", 8.5, leading=11)
            for line in _wrap(words[offset:min(offset + 44, end)], "PanelSans", 8.5, widths[column]):
                draft.textLine(line)
            y = draft.getY() - 8
            page.drawText(draft)
        if y < 30:
            raise ValueError("A staged column would leave the page")
    if covered:
        page.setFillColorRGB(1, 1, 1)
        page.rect(34, 672, 240, 86, stroke=0, fill=1)
    page.showPage()
    page.save()
    return raw.getvalue(), origins


def _highlights(path, output, report_id):
    import pymupdf

    if not path.exists():
        return None
    with pymupdf.open(path) as source:
        for page in source[:5]:
            text = " ".join(page.get_text().split()).casefold()
            if (report_id.casefold() in text and "what gao found" in text and "why gao did this study" in text):
                with pymupdf.open() as extract:
                    extract.insert_pdf(source, from_page=page.number, to_page=page.number)
                    extract.save(output, no_new_id=True, garbage=4, deflate=True)
                return page.number + 1
    return None


def compact_offsets(text):
    offsets = [0]
    for char in text:
        offsets.append(offsets[-1] + (not char.isspace()))
    return offsets


def score_ranges(doc, plain, origins):
    """Verify the delivered PDF against the authored read order independently of the check."""
    from bisect import bisect_right

    def key(block):
        x, y = block["bbox"][:2]
        return bisect_right(origins, x + 1), y

    ordered = sorted(doc.source_map, key=key)
    offsets, cursor = [], 0
    for block in ordered:
        a, b = block["start"], block["end"]
        prefix = compact_offsets(doc.text[a:b])
        offsets.append((a, b, cursor, prefix))
        cursor += prefix[-1]
    if "".join("".join(doc.text[row["start"]:row["end"]].split()) for row in ordered) != "".join(plain.split()):
        raise ValueError("The presented PDF text does not match the authored document")
    return offsets, cursor


def measures(rows):
    totals = defaultdict(Counter)
    for row in rows:
        for arm in ("current", "columns"):
            reading = row[arm]
            totals[f'{row["layout"]}/{row["version"]}/{arm}'].update(
                files=1, unavailable="unavailable" in row, unmapped_truth=row.get("unmapped_truth", False), mapped=reading["mapped"],
                matched=bool(reading["sources"]), wrong_source=bool(set(reading["sources"]) - {row["id"]}),
                with_differences=bool(reading["native_locations"]), rewrite_sentences=len(row["truth"]),
                located_sentences=0 if row.get("unmapped_truth") else len({i for i, (a, b) in enumerate(row["truth"])
                                       if any(max(a, lo) < min(b, hi) for lo, hi in reading["ranges"])}))
    return dict(sorted(totals.items()))


def record_reading(doc, meta, library, index, archive_hash, *, enabled, offsets=None):
    from app.core.canonical import canonical_hash
    from app.evidence import document_published_wording as wording

    reading = {"mapped": wording.eligible(".pdf", doc.text, doc.source_map, meta, None, pdf_columns=enabled),
               "sources": [], "ranges": [], "native_locations": [], "sections": []}
    if not reading["mapped"]:
        return reading
    result = wording.capture(doc.text, doc.paragraphs, doc.source_map, meta, None, archive_hash, library, index,
                             pdf_columns=enabled)
    if not result:
        return reading
    statement = {"coverage": {wording.KEY: result}, "source": {"method": "pymupdf", "chars": len(doc.text),
                                                            wording.COMMITMENT: canonical_hash(result)}}
    wording.verify_source(statement, doc.text, doc.source_map, meta, None)
    for section in result.get("sections", [result]):
        reading["sources"].append(section["reference"]["id"])
        reading["sections"].append([section.get("page"), section.get("column"), section["submitted_range"]])
        for change in section["changes"]:
            for place in change["locations"]:
                a, b = place["start"], place["end"]
                reading["native_locations"].append([place["page"], a, b, place["boxes"]])
                if offsets is None:
                    reading["ranges"].append([a, b])
                elif change["start"] < change["end"]:
                    lo, hi, base, local = next(item for item in offsets if item[0] <= a < b <= item[1])
                    reading["ranges"].append([base + local[a - lo], base + local[b - lo]])
    return reading


def execute(plan_path, pack, downloads, font, output):
    from app.detectors.text.extract import extract
    from app.evidence.document_gao_archive import library

    output.mkdir(mode=0o700)
    fixtures = output / "files"
    fixtures.mkdir()
    plan = read(plan_path)
    reports, index = library(pack, plan["archive_sha256"])
    rows = []

    def observe(row, path, text=None, origins=None):
        digest = hash_bytes(path.read_bytes())
        row.update(file=path.name, input_sha256=digest)
        try:
            doc = extract(path, ".pdf", allow_ocr=False)
        except ValueError:
            rows.append(row | {"unavailable": "missing native text", "current": UNAVAILABLE, "columns": UNAVAILABLE})
            return
        offsets, size = None, len(doc.text)
        if text is not None:
            try:
                offsets, size = score_ranges(doc, text, origins)
            except ValueError:
                row["unmapped_truth"] = True
        meta = doc.meta | {"original_sha256": digest}
        row.update(source_size=size, extracted_sha256=hash_bytes(doc.text.encode()))
        for arm in ("current", "columns"):
            row[arm] = record_reading(doc, meta, reports, index, plan["archive_sha256"],
                                      enabled=arm == "columns", offsets=offsets)
        rows.append(row)

    for file in plan["works"]:
        key = file["id"]
        for variant in file["variants"]:
            version, text = variant["version"], variant["text"]
            truth = [span["compact"] for span in variant["truth"]]
            layouts = ["balanced", "sidebar"]
            if version == "ordinary":
                layouts += ["three_columns"]
            for layout in layouts:
                row = {"id": key, "version": version, "layout": layout, "truth": truth}
                try:
                    pdf, origins = authored_pdf(text, layout, font)
                except ValueError:
                    rows.append(row | {"unavailable": "render exceeded a page", "current": UNAVAILABLE, "columns": UNAVAILABLE})
                    continue
                path = fixtures / f"{key}-{version}-{layout}.pdf"
                path.write_bytes(pdf)
                observe(row, path, text, origins)
        print(f"Measured {key}", flush=True)

    for item in plan["official"]:
        key = item["id"]
        src = downloads / f"{key.lower()}.pdf"
        path = fixtures / f"{key}-published.pdf"
        try:
            page = _highlights(src, path, key)
        except (OSError, RuntimeError, ValueError):
            page = None
        row = {"id": key, "version": "ordinary", "layout": "published_pdf", "url": item["url"], "truth": [],
               "publisher_sha256": hash_bytes(src.read_bytes()) if src.is_file() else None, "original_page": page}
        if page is None:
            rows.append(row | {"unavailable": "no highlights page found", "current": UNAVAILABLE, "columns": UNAVAILABLE})
            continue
        observe(row, path)
    result = {"schema": SCHEMA, "plan_sha256": hash_bytes(plan_path.read_bytes()), "font_sha256": hash_bytes(font.read_bytes()),
              "rows": rows, "counts": measures(rows)}
    save(output / "result.json.gz", result)
    print(json.dumps(result["counts"], sort_keys=True), flush=True)


def bindings(receipt, downloads):
    """Export an input manifest other evaluators can use on the same deliveries."""
    from shutil import copyfile

    from app.bench import independent
    from app.bench.independent_inputs import INDEX_SCHEMA, verify_inputs

    rows = read(receipt)["rows"]
    root = receipt.parent
    by_case = {(row["id"], row["layout"], row["version"]): row for row in rows}
    (root / "publisher").mkdir()
    manifest, index = [], []
    for row in rows:
        if "file" not in row:
            continue
        key, layout, role = row["id"], row["layout"], row["version"]
        if layout == "published_pdf":
            source_name = f"{key.lower()}.pdf"
            parent = f"publisher/{source_name}"
            copyfile(downloads / source_name, root / parent)
            parent_sha = row["publisher_sha256"]
            license_id = "U.S. GAO terms of use"
        else:
            earlier = by_case[key, layout, "ordinary"]
            if "file" not in earlier:
                continue
            parent = f'files/{earlier["file"]}'
            parent_sha = earlier["input_sha256"]
            license_id = "GovReport CC BY 4.0; OpAI-Bench Apache-2.0"
        case = Path(row["file"]).stem
        manifest.append({"id": case, "modality": "text", "split": "confirmation", "label": int(role == "mixed"),
                         "group": key, "lineage": [f"GAO report {key}"], "family": "short wording rewrite" if role == "mixed" else "ordinary",
                         "condition": layout, "input_sha256": row["input_sha256"], "source_sha256": parent_sha,
                         "license": {"id": license_id, "evaluation_permitted": True, "redistribution_permitted": False}})
        index.append({"id": case, "input": f'files/{row["file"]}', "source": parent})
    document = {"schema": independent.SCHEMA, "cases": manifest}
    file_index = {"schema": INDEX_SCHEMA, "manifest_sha256": independent.audit_manifest(document)["manifest_sha256"], "cases": index}
    verified = verify_inputs(document, file_index, root)
    for name, value in (("manifest", document), ("file-index", file_index), ("verified", verified)):
        save(root / f"{name}.json", value)
    save(root / "bindings.json.gz", {"manifest": document, "file_index": file_index, "verified": verified})
    print(json.dumps(verified, sort_keys=True))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="mode", required=True)
    for name, params in (("prepare", ("cache", "pack", "output")), ("acquire", ("plan", "output")),
                         ("run", ("plan_path", "pack", "downloads", "font", "output")),
                         ("review", ("receipt",)), ("strip-sources", ("plan", "output")), ("bindings", ("receipt", "downloads"))):
        parser = modes.add_parser(name)
        for param in params:
            parser.add_argument(param, type=Path)
        if name == "prepare":
            replay = parser.add_mutually_exclusive_group()
            replay.add_argument("--prior", type=Path)
            replay.add_argument("--roster", type=Path)
    args = vars(cli.parse_args())
    mode = args.pop("mode")
    if mode == "review":
        result = read(args["receipt"])
        computed = measures(result["rows"])
        if result["schema"] != SCHEMA or computed != result["counts"]:
            raise ValueError("The recorded comparison totals differ")
        print(json.dumps(computed, sort_keys=True))
    elif mode == "strip-sources":
        save(args["output"], source_roster(read(args["plan"])))
    else:
        {"prepare": prepare, "acquire": acquire, "run": execute, "bindings": bindings}[mode](**args)
