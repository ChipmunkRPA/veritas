"""Replay retained public-archive changes in native Word positions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from docx import Document as WordDocument

from app.detectors.text.extract import from_string
from app.evidence import document_archive, document_archive_word
from app.evidence import document_revision as revision

ROOT = Path(__file__).resolve().parents[2]
RECEIPTS = ROOT / "docs/evaluation-panels"
ARCHIVE = RECEIPTS / "writing-archive-source-evidence-2026-11-12.json"
ARTICLES = RECEIPTS / "writing-multiparagraph-localization-transfer-2026-11-12.json"
NOTICES = RECEIPTS / "writing-practical-notice-transfer-2026-11-11.json"
OUTPUT = RECEIPTS / "writing-archive-word-transfer-2026-11-12.json"
SCHEMA = "chipmunk.writing-archive-word-transfer/1"
ARCHIVE_KINDS = {"original", "human_swap", "qwen", "smol2", "unknown"}
PLAN = {
    "inputs": "62 known-state and out-of-index diplomatic texts, 10 genuine Kubernetes articles, and 18 genuine Federal Register controls.",
    "views": "Render every text as ordinary top-level Word paragraphs. Add one blank paragraph and duplicate horizontal spaces. Split each raw paragraph into 23-codepoint native text runs.",
    "extra_controls": "Render the first six diplomatic originals in table cells only. Require zero source claims.",
    "baseline": "Current plain-text comparison and its committed prior diplomatic attribution and change coordinates. Word was outside the live route.",
    "release": "Require correct citations and exact positions for every retained plain-text change, zero new source claims on all out-of-index and table controls, and exact raw Word text and run slices from independent XML replay.",
}
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def cases(archive, articles, notices):
    a = read(ARCHIVE)["audit"]["commitments"]
    b = read(ARTICLES)["audit"]["commitments"]
    c = read(NOTICES)["source"]["private_controls_sha256"]
    require(sha((archive / "outcomes.json").read_bytes()) == a["outcomes.json"]
            and sha((articles / "originals.json").read_bytes()) == b["originals.json"]
            and sha((notices / "private-controls.jsonl").read_bytes()) == c, "Retained source commitments differ")
    result = []
    for row in read(archive / "outcomes.json"):
        if row["kind"] in ARCHIVE_KINDS:
            raw = (archive / "fixtures" / row["file"]).read_bytes()
            require(sha(raw) == row["file_sha256"], "Diplomatic text differs")
            result.append((row["kind"], raw.decode(), row["source_comparison"], False))
    result.extend(("kubernetes", "\n\n".join(part["text"] for part in row["paragraphs"]), None, False)
                  for row in read(articles / "originals.json"))
    for raw in (notices / "private-controls.jsonl").read_bytes().splitlines():
        row = json.loads(raw)
        require(sha(row["text"].encode()) == row["text_sha256"], "Notice text differs")
        result.append(("notice", row["text"], None, False))
    table_texts = [text for kind, text, _, _ in result if kind == "original"][:6]
    result.extend(("table_only", text, None, True) for text in table_texts)
    require(Counter(row[0] for row in result) == {"original": 16, "human_swap": 16, "qwen": 13, "smol2": 8,
                                                "unknown": 9, "kubernetes": 10, "notice": 18, "table_only": 6},
            "Held Word roster differs")
    return result


def freeze(roster):
    files = (Path(__file__), Path(document_archive.__file__), Path(document_archive_word.__file__), Path(revision.__file__))
    return {"schema": SCHEMA, "method": PLAN, "implementation": {path.name: sha(path.read_bytes()) for path in files},
            "source_receipts": {path.name: sha(path.read_bytes()) for path in (ARCHIVE, ARTICLES, NOTICES)},
            "cases": [[role, sha(text.encode())] for role, text, _, _ in roster]}


def render(text, table_only):
    word = WordDocument()
    if table_only:
        cell = word.add_table(rows=1, cols=1).cell(0, 0)
        for value in text.split("\n\n"):
            cell.add_paragraph(value)
        word.add_paragraph("The submitted material is in the table.")
    else:
        for index, value in enumerate(text.split("\n\n")):
            if index == 1:
                word.add_paragraph()
            raw = "  " + value.replace(" ", "  ") + "  "
            paragraph = word.add_paragraph()
            for cursor in range(0, len(raw), 23):
                paragraph.add_run(raw[cursor:cursor + 23]).bold = cursor % 2 == 1
    stream = BytesIO()
    word.save(stream)
    return stream.getvalue()


def verify_xml(raw, change):
    with ZipFile(BytesIO(raw)) as archive:
        body = ET.fromstring(archive.read("word/document.xml")).find(W + "body")
    children = list(body)
    submitted = change["submitted"]
    location = submitted["docx_location"]
    index = location["body_index"]
    paragraphs = [i for i, item in enumerate(children) if item.tag == W + "p"]
    require(paragraphs[location["paragraph"] - 1] == index and location["runs_complete"], "Word paragraph witness differs")
    runs = ["".join(node.text or "" for node in run.iter(W + "t")) for run in children[index].findall(W + "r")]
    text = "".join(runs)
    start, end = location["paragraph_range"]
    require(" ".join(text[start:end].split()) == submitted["text"], "Word change excerpt differs")
    require([item["extracted_range"] for item in location["changed_tokens"]] == submitted["changed_tokens"],
            "Word changed-token roster differs")
    for token in location["changed_tokens"]:
        a, b = token["extracted_range"]
        expected = submitted["text"][a - submitted["start"]:b - submitted["start"]]
        lo, hi = token["paragraph_range"]
        actual = "".join(runs[item["run"] - 1][slice(*item["range"])] for item in token["runs"])
        require(text[lo:hi] == actual == expected, "Word token or run positions differ")
    return len(location["changed_tokens"]), sum(len(item["runs"]) > 1 for item in location["changed_tokens"])


def measure(roster):
    cases = []
    for kind, text, prior, table_only in roster:
        original = document_archive.compare(from_string(text), sha(text.encode())) if not table_only else None
        if prior is not None:
            require(original["status"] == prior["status"] and (original["reference"] or {}).get("id") == prior["source"],
                    "Retained diplomatic attribution differs")
            retained = [change["later_changed_tokens"] for edit in prior["edits"] for change in edit["changes"]]
            require([change["submitted"]["changed_tokens"] for change in original["changes"]] == retained,
                    "Retained diplomatic changed coordinates differ")
        raw = render(text, table_only)
        word = document_archive_word.compare(raw, sha(raw))
        if original is not None:
            for key in ("status", "reference", "coverage", "extracted_text_sha256", "authorship_confidence_contribution"):
                require(word[key] == original[key], "Word and plain source comparison differ")
        else:
            require(word["reference"] is None and word["status"] == "out_of_scope", "Table-only source was attributed")
        tokens, split = 0, 0
        for index, change in enumerate(word["changes"]):
            stripped = change | {"submitted": {k: v for k, v in change["submitted"].items() if k != "docx_location"}}
            require(stripped == original["changes"][index], "Word changed ranges differ from plain text")
            count, cross_run = verify_xml(raw, change)
            tokens += count
            split += cross_run
        require(len(word["changes"]) == (len(original["changes"]) if original else 0), "A Word edit is missing")
        require(kind not in {"unknown", "kubernetes", "notice", "table_only"} or word["reference"] is None,
                "Unexpected source claim on out-of-index control")
        cases.append({"role": kind, "status": word["status"], "reference": (word["reference"] or {}).get("id"),
                      "changes": len(word["changes"]), "changed_tokens": tokens, "split_run_tokens": split})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "publish", "verify"))
    parser.add_argument("lock", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("articles", type=Path)
    parser.add_argument("notices", type=Path)
    args = parser.parse_args()
    roster = cases(args.archive, args.articles, args.notices)
    seal = freeze(roster)
    if args.phase == "freeze":
        write(args.lock, seal)
        return
    require(read(args.lock) == seal, "Word comparison freeze differs")
    observed = measure(roster)
    result = {"schema": SCHEMA, "method": PLAN, "freeze_sha256": sha(args.lock.read_bytes()),
              "cases": observed, "case_count": len(observed), "matches": sum(bool(row["reference"]) for row in observed),
              "changed_tokens": sum(row["changed_tokens"] for row in observed),
              "split_run_tokens": sum(row["split_run_tokens"] for row in observed)}
    if args.phase == "publish":
        write(OUTPUT, result)
    else:
        require(read(OUTPUT) == result, "The independent Word position replay differs")
    print({key: value for key, value in result.items() if key not in {"cases", "method"}})


if __name__ == "__main__":
    main()
