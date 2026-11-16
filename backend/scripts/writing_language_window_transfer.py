# SPDX-License-Identifier: Apache-2.0
"""Measure how much English writing can be recovered in bilingual submissions."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.writing-language-window-transfer/1"
BOOKS = {
    "es": ("15027", "52e1b89a8391f254dc71923f8da9a4a69859fcbf7f8e9a676da2c224f041a38f"),
    "fr": ("4650", "1cc10ab3268a76aa01bd3f9722ae2ae0e5e36b869a18b0bb0832b75bc0327806"),
    "de": ("22367", "359d3f5983c812393f4bd7ee49a350ffffc7d476015e58d6b84b6c848dd2b0e9"),
    "it": ("45334", "4cdd962f1910a49091b945eff4233798b8c13f74b011cee75a8cccc7467d3612"),
}
TITLES = {"es": "La vida de Lazarillo", "fr": "Candide ou l’optimisme",
          "de": "Die Verwandlung", "it": "I promessi sposi"}
INPUT_SHA256 = "ac5491aa566418b9d41fa361e71c06dafa5148f425d118ce6252fe2bcce60e63"
CASE_TYPES = ("clean", "title", "paragraph", "inline", "clause", "other_only")
ARMS = ("sentence", "halves")
WORD = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
MAX_SENTENCES = 48
MAX_SENTENCE_CHARS = 1200
WINDOW_WIDTH = 3


def sha(data):
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def write(path, document):
    raw = (json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
           + "\n").encode("utf-8")
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)


def _long_enough(text):
    from app.detectors.text.language_scope import MIN_PASSAGE_LETTERS

    return sum(ch.isalpha() for ch in text) >= MIN_PASSAGE_LETTERS


def _pieces(text):
    words = text.split()
    middle = len(words) // 2
    halves = (" ".join(words[:middle]), " ".join(words[middle:]))
    return [part for part in halves if _long_enough(part)]


def candidates(sentences):
    """Return positions for three-sentence contexts within the same paragraph."""
    from app.detectors.text.language_scope import _recognised

    if len(sentences) > MAX_SENTENCES:
        return {arm: [] for arm in ARMS}
    sentence_admission, fragment_admission = [], []
    for sentence in sentences:
        text = sentence.text
        english = len(text) <= MAX_SENTENCE_CHARS and _long_enough(text) and _recognised(text) == "en"
        sentence_admission.append(english)
        fragment_admission.append(english and all(_recognised(part) == "en" for part in _pieces(text)))
    positions = [start for start in range(len(sentences) - WINDOW_WIDTH + 1)
                 if sentences[start].paragraph == sentences[start + WINDOW_WIDTH - 1].paragraph]
    return {
        name: [start for start in positions if all(admitted[start:start + WINDOW_WIDTH])]
        for name, admitted in (("sentence", sentence_admission), ("halves", fragment_admission))
    }


def foreign_bank(books, roster=BOOKS, *, count=28):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    banks = {}
    for language, (book, commitment) in roster.items():
        raw = (books / f"{book}.txt").read_bytes()
        assert sha(raw) == commitment
        interior = re.split(r"(?m)^\*\*\* START [^\n]+\n", raw.decode("utf-8-sig"), maxsplit=1)[1]
        interior = re.split(r"(?m)^\*\*\* END ", interior, maxsplit=1)[0]
        paragraphs = re.split(r"\n\s*\n", interior)
        sample = []
        for text in paragraphs[len(paragraphs) // 3:2 * len(paragraphs) // 3]:
            doc = from_string(text)
            for sentence in segment(doc):
                # Avoid headers, marginalia, dialogue and unusual stops using only source shape.
                candidate = sentence.text
                if (19 <= len(WORD.findall(candidate)) <= 31 and candidate.endswith(".")
                        and not any(char in candidate for char in ':?![]_*“”„«»"—–')
                        and candidate.count(".") == 1 and candidate[0].isupper()):
                    sample.append(candidate)
        candidates = sorted(set(sample), key=sha)
        assert len(candidates) >= count, (language, len(candidates))
        banks[language] = candidates[:count]
    return banks


def delivery(text, spans, case, foreign, title):
    offset = spans[1][0]
    if case == "title":
        return f"{title}\n\n{text}", None
    if case == "paragraph":
        start = len(text) + 2
        return f"{text}\n\n{foreign}", (start, start + len(foreign))
    if case == "inline":
        return text[:offset] + foreign + " " + text[offset:], (offset, offset + len(foreign))
    if case == "clause":
        words = list(WORD.finditer(text[spans[1][0]:spans[1][1]]))
        offset += words[len(words) // 2].start()
        fragment = foreign.rstrip(".")
        return text[:offset] + fragment + " " + text[offset:], (offset, offset + len(fragment))
    if case == "other_only":
        return " ".join([foreign] * 4), (0, len(foreign))
    return text, None


def freeze(plan_path, books, output):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    assert sha(plan_path.read_bytes()) == INPUT_SHA256
    plan = read(plan_path)
    rows = sorted(plan["held"], key=lambda row: (row["group"], row["delivery"]))
    groups = sorted({row["group"] for row in rows})
    assert len(groups) == 112
    banks = foreign_bank(books)
    assignments = {group: (lang, banks[lang][i // len(BOOKS)]) for i, group in enumerate(groups)
                   for lang in (tuple(BOOKS)[i % len(BOOKS)],)}
    inputs = []
    for row in rows:
        language, foreign = assignments[row["group"]]
        text, spans = row["text"], row["spans"]
        assert sha(text) == row["sha256"]
        original = segment(from_string(text))
        all_english = {" ".join(s.text for s in original[i:i + 3])
                       for i in range(len(original) - 2)}
        centers = {text[a:b] for i in row["target"] for a, b in (spans[i],)}
        for case in CASE_TYPES:
            if case == "other_only" and row["delivery"] != "original":
                continue
            source, target = delivery(text, spans, case, foreign, TITLES[language])
            doc = from_string(source)
            # Check the planned range against the extracted text before recording it.
            if case in {"inline", "clause", "paragraph"}:
                assert doc.text[target[0]:target[1]] == foreign.rstrip(".") + ("" if case == "clause" else ".")
            inputs.append({"group": row["group"], "family": row["family"], "genre": row["genre"],
                           "role": row["delivery"], "case": case, "language": language,
                           "source": source, "sha256": sha(source), "foreign_range": target,
                           "original_windows": sorted(sha(value) for value in all_english),
                           "target_centers": sorted(sha(value) for value in centers),
                           "foreign_sha256": sha(foreign)})
    write(output, {"schema": SCHEMA, "input_plan_sha256": INPUT_SHA256, "inputs": inputs})


def measure(row, native_inputs):
    from app.detectors.text import language_scope
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    doc = from_string(row["source"])
    sentences = segment(doc)
    arms = candidates(sentences)
    start, end = row["foreign_range"] or (-1, -1)
    readings = {"scope": language_scope.classify(doc.text, sentences), "windows": []}
    for i, (a, center, b) in enumerate(zip(sentences, sentences[1:], sentences[2:], strict=False)):
        if a.paragraph != b.paragraph:
            continue
        value = " ".join((a.text, center.text, b.text))
        commitment = sha(value)
        overlaps = (a.start < end and b.end > start) or row["case"] == "other_only"
        is_original = commitment in row["original_windows"]
        if is_original and row["case"] == "clean":
            native_inputs[commitment] = value
        readings["windows"].append({
            "range": [a.start, b.end], "center": [center.start, center.end], "source_sha256": commitment,
            "foreign": overlaps, "preserved_context": is_original, "target_center": sha(center.text) in row["target_centers"],
            "arms": [arm for arm in ARMS if i in arms[arm]],
        })
    return readings


def tally(records, *, by_delivery=False):
    result = {}
    for case in CASE_TYPES:
        selected = [row for row in records if row["case"] == case]
        if not selected:
            continue
        counts = Counter()
        for row in selected:
            counts["n"] += 1
            counts["baseline_english"] += row["scope"] == "english"
            counts["baseline_other"] += row["scope"] == "unsupported"
            counts["baseline_mixed"] += row["scope"] == "mixed"
            for arm in ARMS:
                windows = [w for w in row["windows"] if arm in w["arms"]]
                counts[f"{arm}_any"] += bool(windows)
                counts[f"{arm}_foreign"] += any(w["foreign"] for w in windows)
                counts[f"{arm}_preserved"] += any(w["preserved_context"] for w in windows)
                counts[f"{arm}_recovered"] += row["scope"] != "english" and any(w["preserved_context"] for w in windows)
                counts[f"{arm}_preserved_target"] += any(w["preserved_context"] and w["target_center"] for w in windows)
                counts[f"{arm}_native_flag"] += any(w["preserved_context"] and w["reading"] >= 0.5 for w in windows)
                counts[f"{arm}_native_target_flag"] += any(w["preserved_context"] and w["target_center"]
                                                           and w["reading"] >= 0.5 for w in windows)
        result[case] = dict(counts)
    if by_delivery:
        result["by_delivery"] = {
            role: tally([row for row in records if row["role"] == role])
            for role in ("original", "edited")
        }
    return result


def capture(frozen, checkpoint, output):
    from scripts.writing_two_sided_context import open_classifier

    plan = read(frozen)
    assert plan["schema"] == SCHEMA and plan["input_plan_sha256"] == INPUT_SHA256
    inputs, records = {}, []
    for row in plan["inputs"]:
        assert sha(row["source"]) == row["sha256"]
        result = measure(row, inputs)
        records.append({name: value for name, value in row.items() if name not in
                        {"source", "original_windows", "target_centers"}} | result)
    classifier = open_classifier(checkpoint)
    keys = sorted(inputs)
    scores = {}
    for start in range(0, len(keys), 32):
        keys_batch = keys[start:start + 32]
        readings = classifier.predict([inputs[key] for key in keys_batch], max_len=512, batch=4, check=lambda _: None)
        for key, value in zip(keys_batch, readings, strict=True):
            scores[key] = round(float(value), 8)
        if start % 128 == 0:
            print(f"Scored {min(start + 32, len(keys))}/{len(keys)} exact English contexts", flush=True)
    for row in records:
        for window in row["windows"]:
            window["reading"] = scores.get(window["source_sha256"]) if window["preserved_context"] else None
            assert window["reading"] is not None if window["preserved_context"] else window["reading"] is None
    receipt = {"schema": SCHEMA, "freeze_sha256": sha(frozen.read_bytes()), "records": records,
               "results": tally(records, by_delivery=True)}
    write(output, receipt)
    print(json.dumps(receipt["results"], sort_keys=True), flush=True)


def review(receipt):
    assert receipt["schema"] == SCHEMA
    rows = receipt["records"]
    groups = {row["group"] for row in rows}
    assert len(groups) == 112
    expected = {(group, role, case) for group in groups for case in CASE_TYPES
                for role in (("original",) if case == "other_only" else ("original", "edited"))}
    assert len(rows) == len(expected) == 1232
    assert {(row["group"], row["role"], row["case"]) for row in rows} == expected
    inputs = {}
    for row in rows:
        key = row["group"], row["role"]
        assert "source" not in row and row["language"] in BOOKS
        if row["case"] == "clean":
            assert key not in inputs
            inputs[key] = {window["source_sha256"]: window["reading"] for window in row["windows"]}
    for row in rows:
        for window in row["windows"]:
            assert set(window["arms"]) <= set(ARMS)
            if "halves" in window["arms"]:
                assert "sentence" in window["arms"]
            a, b = window["range"]
            s, e = window["center"]
            assert 0 <= a <= s < e <= b
            if window["preserved_context"]:
                assert not window["foreign"] and window["reading"] == inputs[row["group"], row["role"]][window["source_sha256"]]
            else:
                assert window["reading"] is None
    counts = tally(rows, by_delivery=True)
    assert counts == receipt["results"]
    paired = {}
    for case in ("paragraph", "inline", "clause"):
        findings = {"original": set(), "edited": set()}
        for row in rows:
            if row["case"] == case and any(
                "halves" in window["arms"] and window["preserved_context"] and window["target_center"]
                and window["reading"] >= 0.5 for window in row["windows"]
            ):
                findings[row["role"]].add(row["group"])
        paired[case] = {
            "only_edited": len(findings["edited"] - findings["original"]),
            "only_original": len(findings["original"] - findings["edited"]),
            "both": len(findings["edited"] & findings["original"]),
        }
    return counts | {"paired_target_flags": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="command", required=True)
    prepare = modes.add_parser("freeze")
    for name in ("plan", "books", "output"):
        prepare.add_argument(name, type=Path)
    run = modes.add_parser("capture")
    for name in ("freeze", "checkpoint", "output"):
        run.add_argument(name, type=Path)
    replay = modes.add_parser("review")
    replay.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.plan, args.books, args.output)
    elif args.command == "capture":
        capture(args.freeze, args.checkpoint, args.output)
    else:
        print(json.dumps(review(read(args.receipt)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
