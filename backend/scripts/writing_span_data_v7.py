"""Private controlled documents copied from the exact licensed TXD-22 source."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

from app.detectors.text.extract import from_string
from scripts.prepare_writing_focus_v3 import project_labels
from scripts.writing_span_geometry_v7 import require, truth_spans
from scripts.writing_specialists_v5 import SOURCE_SHA, source_components

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "backend/data/independent-evaluation/writing-specialists-v5"
SOURCE = ROOT / "backend/data/independent-evaluation/specialists-v3-sources/txd22.csv"
ROLE_MAP = {"specialist_cal": "fit", "fusion_fit": "cal", "fusion_cal": "point", "confirmation": "diagnostic"}
GENUINE = {"human", "human-small-matched", "human-large-matched", "human-many-matched"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def recipes(group):
    p = 31 + int(text_sha(group + ":small")[:8], 16) % 60
    q = 31 + int(text_sha(group + ":large")[:8], 16) % 40
    return {
        "human": [(0, 0, 160)],
        "ai": [(1, 0, 160)],
        "mixed-small": [(0, 0, p), (1, 0, 12), (0, p, 148)],
        "mixed-large": [(0, 0, q), (1, 0, 60), (0, q, 100)],
        "mixed-human-gap": [(1, 0, p), (0, 0, 12), (1, p, 148)],
        "mixed-many": [(1, 0, 20), (0, 0, 30), (1, 20, 60), (0, 30, 60), (1, 60, 100)],
        "human-small-matched": [(0, 12, 12 + p), (0, 0, 12), (0, 12 + p, 160)],
        "human-large-matched": [(0, 60, 60 + q), (0, 0, 60), (0, 60 + q, 160)],
        "human-many-matched": [(0, 0, 20), (0, 90, 120), (0, 120, 160), (0, 20, 50), (0, 50, 90)],
    }


def compose(group, role, name, parts, sources):
    chunks, labels, provenance = [], [], []
    for truth, start_word, end_word in parts:
        source = sources[truth]
        words = list(re.finditer(r"\S+", source["text"]))
        require(
            len(words) >= 160 and 0 <= start_word < end_word <= 160, "Exact160-word source support required"
        )
        start, end = words[start_word].start(), words[end_word - 1].end()
        text = source["text"][start:end]
        if chunks:
            labels.extend(["User", "User"])
        chunks.append(text)
        labels.extend(["Bot" if truth else "User"] * len(text))
        provenance.append(
            {
                "row": source["row"],
                "source_start": start,
                "source_end": end,
                "source_text_sha256": source["text_sha256"],
                "label": truth,
            }
        )
    raw = "\n\n".join(chunks)
    require(
        len(raw.split()) == 160 and len(labels) == len(raw) and raw.isascii(),
        "All160 original ASCII words required",
    )
    doc = from_string(raw)
    projected = project_labels(doc, labels)
    local = [
        1 if projected.get(i) == "Bot" else 0 if projected.get(i) == "User" else -1
        for i in range(len(doc.text))
    ]
    spans = truth_spans(doc.text, local)
    require(bool(spans) == (name not in GENUINE), "Recipe source truth changed")
    return {
        "id": text_sha("writing-span-v7:" + group + ":" + name),
        "group": group,
        "role": role,
        "recipe": name,
        "label": int(bool(spans)),
        "text": doc.text,
        "labels": local,
        "text_sha256": text_sha(doc.text),
        "raw_sha256": text_sha(raw),
        "raw_text": raw,
        "source_map": doc.source_map,
        "source_parts": provenance,
        "truth_spans": spans,
        "offset_unit": "Unicode codepoints, end exclusive",
    }


def prepare():
    require(sha(SOURCE) == SOURCE_SHA, "Exact original licensed CSV required")
    old_lock = json.loads((OLD / "freeze.json").read_text())
    require(sha(OLD / "plan.json") == old_lock["plan_sha256"], "Original connected-group plan changed")
    plan = json.loads((OLD / "plan.json").read_text())
    rows = list(csv.DictReader(io.StringIO(SOURCE.read_bytes().decode("latin1"), newline="")))
    require(len(rows) == 66000, "Publisher row count changed")
    require(
        len(Counter(r["Source"] for r in rows)) == 22
        and set(Counter(r["Source"] for r in rows).values()) == {3000},
        "All22 publisher classes required",
    )
    components = {tuple(sorted(v)) for v in source_components(rows)}
    require(
        len(plan["groups"]) == 128
        and Counter(g["role"] for g in plan["groups"]) == {r: 32 for r in ROLE_MAP},
        "Original128 source-group roles required",
    )
    output, roles, originals = [], {}, {}
    for group in sorted(plan["groups"], key=lambda g: g["id"]):
        require(
            tuple(sorted(group["component_rows"])) in components,
            "Original full prompt/answer component differs",
        )
        role = ROLE_MAP[group["role"]]
        sources = {s["label"]: s for s in group["sources"]}
        require(set(sources) == {0, 1}, "Both original source classes required")
        for label, source in sources.items():
            row = rows[source["row"]]
            require(
                row["Answer"] == source["text"]
                and row["Source"] == source["generator"]
                and text_sha(row["Answer"]) == source["text_sha256"]
                and text_sha(row["Question"]) == source["question_sha256"],
                "Actual publisher row differs",
            )
            require(
                label == int(row["Source"] != "Human written") and source["text"].isascii(),
                "Original source-class/encoding eligibility changed",
            )
            prior = originals.setdefault(source["text_sha256"], group["id"])
            require(prior == group["id"], "An original answer crosses source groups")
        roles[group["id"]] = role
        for name, parts in recipes(group["id"]).items():
            if role == "point" and name not in GENUINE:
                continue
            output.append(compose(group["id"], role, name, parts, sources))
    require(len(output) == len({r["id"] for r in output}) == 992, "Complete992-case design required")
    require(
        Counter(r["role"] for r in output) == {"fit": 288, "cal": 288, "point": 128, "diagnostic": 288},
        "All planned role denominators required",
    )
    by_text = {}
    for row in output:
        prior = by_text.setdefault(row["text_sha256"], row["group"])
        require(prior == row["group"], "Constructed exact text crosses source groups")
    return {
        "schema": "chipmunk.writing-span-inputs/7",
        "source_sha256": SOURCE_SHA,
        "original_plan_sha256": sha(OLD / "plan.json"),
        "source_groups": 128,
        "cases": output,
        "roles": roles,
        "new_model_forwards": 0,
        "training_started": False,
        "limits": [
            "Consumed prompt/duplicate groups, no writer IDs or verified external training exclusion.",
            "Publisher source labels; controlled160-word compositions, not natural article or email provenance.",
            "ASCII-only original slices; no unassisted-human population or new-source accuracy claim.",
        ],
    }
