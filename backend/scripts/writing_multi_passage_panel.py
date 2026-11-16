# SPDX-License-Identifier: MIT
"""Build a source-bound test of two separated synthetic passages in one document."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

REVISION = "c8a59b129c34cfd9c67f739ae049abfff26fc2d3"
DATASET = "open-text-detector/heterogeneous-ai-spans"
SCHEMA = "chipmunk.writing-multi-passage/1"
FORMATS = ("mixed", "mixed_layout_neutral")
GENRES = {"gutenberg_selected": "historical_prose", "standardebooks": "historical_prose", "beigebook": "economic_report"}
EBOOKS = {
    "upton-sinclair_the-jungle": "32695122feed71de0991195f4cef7ab8983f0325",
    "virginia-woolf_mrs-dalloway": "0de73d9a62b20d33803b86ef5b4c548c5bf9d5a9",
    "willa-cather_death-comes-for-the-archbishop": "dd92c6b3fd2174031015d937cbc536bc51784309",
}
PINS = {
    "mixed-test.parquet": "31979a49fe8cfc331a90313e00be4759423dc9e127e20bcf2ec98959add03189",
    "mixed-validation.parquet": "7e1d1fb0612e18300fba5fc4d5cae32e112dadc864a0eee6bd508e5173c5a2f3",
    "mixed_layout_neutral-test.parquet": "c08dc08400a1f9ec0c18f563051a4e1032d51c6f423f2136dae2aeabdb23d8be",
    "mixed_layout_neutral-validation.parquet": "bfd99f7e28717051e666b134128ff74df326467470c9864145dcca444c688f8a",
}
CAP = {"test": {"historical_prose": 36, "economic_report": 20},
       "validation": {"historical_prose": 32, "economic_report": 24}}


def digest(body):
    return hashlib.sha256(body).hexdigest()


def read(file):
    body = file.read_bytes()
    return json.loads(gzip.decompress(body) if file.suffix == ".gz" else body)


def save(path, value):
    data = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data + b"\n")


def compact(text):
    return "".join(text.split())


def _prefix(text):
    positions = [0]
    for character in text:
        positions.append(positions[-1] + (not character.isspace()))
    return positions


def _validate(row):
    raw, original = row["text"], row["source_text"]
    assert digest(raw.encode()) == row["text_sha256"]
    assert row["offset_unit"] == "unicode_codepoint" and row["end_exclusive"] and row["kind"] == "mixed"
    previous, original_previous = 0, 0
    for span in row["spans"]:
        a, b = span["start"], span["end"]
        x, y = span["source_start"], span["source_end"]
        assert a == previous and a < b <= len(raw) and x == original_previous and x < y <= len(original)
        assert span["label"] in {"human", "ai"}
        if span["label"] == "human":
            assert raw[a:b] == original[x:y]
        else:
            assert raw[a:b] != original[x:y] and span["replacement_id"]
        previous, original_previous = b, y
    assert previous == len(raw) and original_previous == len(original)


def _paired(raw, neutral):
    assert all(raw[key] == neutral[key] for key in ("id", "source_dataset", "generator_backend", "group_id", "split"))
    assert compact(raw["source_text"]) == compact(neutral["source_text"])
    for a, b in zip(raw["spans"], neutral["spans"], strict=True):
        assert a["label"] == b["label"] and a["replacement_id"] == b["replacement_id"]
        assert compact(raw["text"][a["start"]:a["end"]]) == compact(neutral["text"][b["start"]:b["end"]])


def native(raw, truth):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    doc = from_string(raw)
    if compact(raw) != compact(doc.text):
        return None
    sentences = segment(doc)
    if not 12 <= len(sentences) <= 128:
        return None
    starts = _prefix(doc.text)
    prefix = _prefix(raw)
    return {"text": doc.text, "native_text_sha256": digest(doc.text.encode()), "original_text_sha256": digest(raw.encode()),
            "nonspace_characters": starts[-1],
            "sentences": [[s.start, s.end, starts[s.start], starts[s.end], s.paragraph, s.n_words] for s in sentences],
            "truth": [{"source": [a, b], "compact": [prefix[a], prefix[b]]} for a, b in truth]}


def load_files(folder, split):
    import pyarrow.parquet as pq

    rows = []
    for fmt in FORMATS:
        filename = f"{fmt}-{split}.parquet"
        source = folder / filename
        assert digest(source.read_bytes()) == PINS[filename], filename
        result = pq.read_table(source, filters=[("source_dataset", "in", sorted(GENRES))]).to_pylist()
        assert all(row["split"] == split for row in result)
        rows.append({row["id"]: row for row in result})
    assert set(rows[0]) == set(rows[1])
    return rows


def _source_id(row):
    source = row["source_dataset"]
    if source == "beigebook":
        assert re.match(r"^https://www\.federalreserve\.gov/monetarypolicy/beigebook\d+\.htm$", row["source_reference"])
        assert "Board website public domain" in row["source_license"]
        return row["group_id"]
    if source == "standardebooks":
        assert "dedicated CC0" in row["source_license"]
        assert row["source_reference"] in {f"https://github.com/standardebooks/{book}/tree/{pin}" for book, pin in EBOOKS.items()}
    else:
        assert "Public domain in USA declared by Project Gutenberg" in row["source_license"]
        assert re.match(r"^https://www\.gutenberg\.org/(ebooks|files|cache)/", row["source_reference"])
    return row["source_author"].casefold().strip()


def _variants(a, b, split):
    variants = []
    for layout, row in zip(("published", "neutral"), (a, b), strict=True):
        doc = native(row["source_text"], ())
        if doc is None:
            return None
        if split == "test":
            prefix = _prefix(row["source_text"])
            doc["matched_targets"] = [[prefix[item["source_start"]], prefix[item["source_end"]]]
                                      for item in row["spans"] if item["label"] == "ai"]
        variants.append({"layout": layout, "version": "ordinary"} | doc)
        if split == "test":
            truth = [(item["start"], item["end"]) for item in row["spans"] if item["label"] == "ai"]
            edited = native(row["text"], truth)
            if edited is None:
                return None
            variants.append({"layout": layout, "version": "mixed"} | edited)
    return variants


def prepare(folder, output):
    test, validation = (load_files(folder, split) for split in ("test", "validation"))
    eligible = Counter()
    held_authors = set()
    selected, excluded = [], Counter()
    for split, (base, normalized) in (("test", test), ("validation", validation)):
        use_count, by_group, by_generator = Counter(), Counter(), Counter()
        for key, row in sorted(base.items(), key=lambda item: digest(f"multi-passage/1/{item[0]}".encode())):
            genre = GENRES[row["source_dataset"]]
            group = _source_id(row)
            family = row["generator_backend"]
            if split == "test" and sum(s["label"] == "ai" for s in row["spans"]) != 2:
                continue
            eligible[(split, genre)] += 1
            limit = 12 if genre == "historical_prose" else 8
            if (split == "validation" and group in held_authors or by_group[group] >= limit
                    or (split == "test" and by_generator[genre, family] >= 12) or use_count[genre] == CAP[split][genre]):
                continue
            neutral = normalized[key]
            _validate(row)
            _validate(neutral)
            try:
                _paired(row, neutral)
            except AssertionError:
                excluded["layout_not_whitespace_equivalent"] += 1
                continue
            variants = _variants(row, neutral, split)
            if variants is None:
                excluded["native_geometry"] += 1
                continue
            use_count[genre] += 1
            by_group[group] += 1
            by_generator[genre, family] += 1
            if split == "test":
                held_authors.add(group)
            selected.append({
                "document": key, "split": split, "group_sha256": digest(group.encode()), "genre": genre,
                "source_reference": row["source_reference"], "source_author": row["source_author"],
                "source_title": row["source_title"], "source_license": row["source_license"],
                "generator_backend": family, "requested_model": row["requested_model"],
                "model_identity_status": row["model_identity_status"], "variants": variants,
            })
        print(split, dict(use_count), flush=True)
    assert len({work["document"] for work in selected}) == len(selected)
    assert not ({work["group_sha256"] for work in selected if work["split"] == "test"}
                & {work["group_sha256"] for work in selected if work["split"] == "validation"})
    for split in ("test", "validation"):
        for genre in GENRES.values():
            assert sum(work["split"] == split and work["genre"] == genre for work in selected) >= 8
    save(output, {"schema": SCHEMA, "dataset": DATASET, "revision": REVISION, "publisher_files_sha256": PINS,
                  "exclusions": dict(excluded), "eligible": {"-".join(k): v for k, v in eligible.items()},
                  "works": selected})


def _inputs(variant):
    text, sentences = variant["text"], variant["sentences"]
    isolated = [text[a:b] for a, b, *_ in sentences]
    context = [" ".join(isolated[max(0, i-1):i+2]) for i in range(len(isolated))]
    return isolated, context


def score(plan, snapshot, output):
    import torch

    from app.detectors.text.classifier import FourClassTextClassifier
    from scripts.audio_strong_traffic_view import network_free
    from scripts.writing_focus_v3 import BASELINE_ASSETS, BASELINE_REVISION
    from scripts.writing_word_edit_transfer import model_assets

    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    network_free()
    model_assets(snapshot)
    classifier = FourClassTextClassifier.from_snapshot(snapshot, "cpu")
    gathered = {"validation": set(), "test": set()}
    for work in plan["works"]:
        for variant in work["variants"]:
            for view in _inputs(variant):
                gathered[work["split"]].update(view)
    measured = {}
    for split in ("validation", "test"):
        waiting = sorted(gathered[split] - measured.keys(), key=lambda line: (len(line), digest(line.encode())))
        for start in range(0, len(waiting), 192):
            subset = waiting[start:start + 192]
            values = classifier.predict(subset, batch=12, check=lambda _: None)
            for text, value in zip(subset, values, strict=True):
                score_value = float(value)
                measured[text] = round(score_value, 8) if math.isfinite(score_value) else None
            if start % (5 * 192) == 0:
                print(f"{split} readings {start + len(subset)}/{len(waiting)}", flush=True)
    result = []
    for work in plan["works"]:
        variants = []
        for variant in work["variants"]:
            isolated, context = _inputs(variant)
            variants.append({key: value for key, value in variant.items() if key != "text"} | {
                "isolated": [measured[i] for i in isolated], "context": [measured[i] for i in context],
                "inputs_sha256": [digest(item.encode()) for item in context],
            })
        result.append(work | {"variants": variants})
    save(output, plan | {"works": result, "model_revision": BASELINE_REVISION, "model_assets": BASELINE_ASSETS,
                         "input_count": len(measured), "input_plan_sha256": digest(json.dumps(plan, sort_keys=True).encode())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phase = parser.add_subparsers(dest="action", required=True)
    building = phase.add_parser("prepare")
    building.add_argument("folder", type=Path)
    building.add_argument("output", type=Path)
    scoring = phase.add_parser("score")
    for key in ("plan", "snapshot", "output"):
        scoring.add_argument(key, type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.folder, args.output)
    else:
        score(read(args.plan), args.snapshot, args.output)


if __name__ == "__main__":
    main()
