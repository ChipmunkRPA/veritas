"""Pure, privacy-preserving source audit; no scoring, split assignment or admission.

Original events and final-text representations describe the same documents.
Raw event-ID reuse supplies conservative potential links, never verified sessions.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict

from app.detectors.text.extract import from_string

MAX_CHARS = 120_000
MAX_ROWS = 1_000
MAX_EVENTS = 100_000
CORPUS_FILES = ("human.jsonl", "ai_local.jsonl", "ai_claude_train.jsonl")
NORMALIZATIONS = ("exact", "nfc_whitespace", "production")
_DEVELOPMENT_ID = re.compile(r"(?:corpus:(?:human|ai_local|ai_claude_train)\.jsonl:[0-9]+|hc3:[0-9]+:chatgpt_answers:0)")


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)  # Reasons are fixed strings; never interpolate source content.


def _text(value: object) -> str:
    _require(isinstance(value, str) and bool(value.strip()) and len(value) <= MAX_CHARS,
             "invalid or unbounded text")
    return value


def _sha(value: str) -> str:
    try:
        encoded = value.encode("utf-8")
    except UnicodeError:
        raise ValueError("invalid source Unicode") from None
    return hashlib.sha256(encoded).hexdigest()


def _identity(*parts: object) -> str:
    return _sha(json.dumps(parts, ensure_ascii=False, separators=(",", ":"), allow_nan=False))


def fingerprints(text: str) -> dict[str, str]:
    """Hash exact, NFC/whitespace and production-normalized bounded final text."""
    text = _text(text)
    return {"exact": _sha(text), "nfc_whitespace": _sha(" ".join(unicodedata.normalize("NFC", text).split())),
            "production": _sha(from_string(text).text)}


def _answers(row: dict, field: str) -> list[str]:
    values = row.get(field, [])
    _require(isinstance(values, list) and all(isinstance(v, str) and len(v) <= MAX_CHARS for v in values),
             "invalid HC3 answer list")
    return values


def reconstruct_development(corpus: dict[str, list[dict]], hc3: list[dict], hc3_ai: int = 250) -> list[dict]:
    """Reproduce trainer input ordering/HC3 selection, retaining source row identities.

Returns private text-bearing working records, not a publishable audit report.
The caller must bind source bytes and actual historical training arguments.
"""
    _require(isinstance(corpus, dict) and set(corpus) == set(CORPUS_FILES), "require the three development corpus files")
    _require(type(hc3_ai) is int and hc3_ai > 0, "HC3 development count must be a positive integer")
    _require(isinstance(hc3, list) and all(isinstance(r, dict) for r in hc3), "invalid HC3 rows")
    result = []
    for name in CORPUS_FILES:
        _require(isinstance(corpus[name], list), "invalid development corpus")
        for index, row in enumerate(corpus[name]):
            _require(isinstance(row, dict) and type(row.get("label")) is int and row["label"] in (0, 1)
                     and isinstance(row.get("source"), str) and bool(row["source"].strip()), "invalid development record")
            result.append({"id": f"corpus:{name}:{index}", "text": _text(row.get("text")),
                           "label": row["label"], "source": row["source"]})
    order = list(enumerate(hc3))
    random.Random(33).shuffle(order)
    chosen = 0
    for index, row in order:
        answers = _answers(row, "chatgpt_answers")
        first = answers[0] if answers else ""
        if 100 <= len(first.split()) <= 400:
            result.append({"id": f"hc3:{index}:chatgpt_answers:0", "text": first,
                           "label": 1, "source": "hc3_chatgpt"})
            chosen += 1
        if chosen >= hc3_ai:
            break
    _require(chosen == hc3_ai, "insufficient eligible HC3 development answers")
    return result


def _metadata(row: object) -> dict:
    _require(isinstance(row, dict) and isinstance(row.get("metadata"), dict), "invalid source representation")
    m = row["metadata"]
    _require(set(m) == {"author", "human_only", "model", "temperature", "setting", "task"}, "invalid metadata fields")
    _require(all(isinstance(m[k], str) and bool(m[k].strip()) for k in ("author", "setting", "task"))
             and isinstance(m["model"], str) and type(m["human_only"]) is bool
             and type(m["temperature"]) in (int, float) and math.isfinite(m["temperature"]), "invalid metadata values")
    return m


def _segments(row: dict) -> tuple[str, list[dict]]:
    data = row.get("data")
    _require(isinstance(data, list) and 0 < len(data) <= MAX_EVENTS, "invalid final-text segments")
    for part in data:
        _require(isinstance(part, dict) and isinstance(part.get("text"), str) and len(part["text"]) <= MAX_CHARS
                 and part.get("author") in ("User", "Bot") and isinstance(part.get("queries"), list)
                 and all(isinstance(q, str) for q in part["queries"]), "invalid final-text segment")
    return _text("".join(p["text"] for p in data)), data


def _pairs(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate event JSON key")
        result[key] = value
    return result


def _constant(_: str) -> None:
    raise ValueError("nonfinite event JSON value")


def _events(row: dict) -> list[dict]:
    data = row.get("data")
    _require(isinstance(data, list) and len(data) <= MAX_EVENTS, "invalid original event list")
    result = []
    for value in data:
        _require(isinstance(value, str) and len(value) <= 2_000_000, "invalid original event encoding")
        try:
            event = json.loads(value, object_pairs_hook=_pairs, parse_constant=_constant)
        except (ValueError, TypeError, RecursionError):
            raise ValueError("invalid original event JSON") from None
        _require(isinstance(event, dict), "invalid original event object")
        pending = [event]
        while pending:
            value = pending.pop()
            _require(not isinstance(value, float) or math.isfinite(value), "nonfinite original event value")
            if isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
        result.append(event)
    return result


def _token(value: object) -> tuple[str, str | int] | None:
    if type(value) is int:
        return "int", value
    if isinstance(value, str) and value.strip() and len(value) <= 1024:
        return "str", value
    return None


def _lineage(events: list[dict], add) -> dict:
    """Local references are checked before publishing only hashed potential links."""
    counts = Counter()
    counts["original_events"] = len(events)
    definitions = {name: Counter() for name in ("edit", "request", "response")}
    refs = []
    for event in events:
        kinds = [name for key, name in (("operationType", "edit"), ("documentEditId", "request"),
                                        ("requestId", "response")) if key in event]
        if len(kinds) != 1:
            counts["ambiguous_event_kind" if kinds else "unclassified_events"] += 1
            continue
        kind = kinds[0]
        counts[f"{kind}_events"] += 1
        token = _token(event.get("id"))
        if token is None:
            counts[f"{kind}_missing_or_invalid_id"] += 1
        else:
            definitions[kind][token] += 1
            add(f"potential_raw_{kind}_id", _identity("aitdna", kind, *token))
        reference = {"request": ("documentEditId", "edit"), "response": ("requestId", "request")}.get(kind)
        if reference:
            field, destination = reference
            target = _token(event.get(field))
            if target is None:
                counts[f"{kind}_missing_or_invalid_reference"] += 1
            else:
                refs.append((kind, destination, target))
                add(f"potential_raw_{destination}_id", _identity("aitdna", destination, *target))
        if kind == "request":
            query = event.get("fullQuery")
            if not isinstance(query, str) or not query.strip():
                counts["request_missing_full_query"] += 1
            else:
                _require(len(query) <= MAX_CHARS, "unbounded full query")
                add("publisher_full_query_exact", _sha(query))
                add("publisher_full_query_nfc_whitespace", _sha(" ".join(unicodedata.normalize("NFC", query).split())))
        if kind == "response" and event.get("accepted") is None:
            counts["response_acceptance_null_or_missing"] += 1
    for kind, values in definitions.items():
        counts[f"{kind}_duplicate_local_ids"] = sum(n - 1 for n in values.values())
    for kind, destination, token in refs:
        count = definitions[destination][token]
        if count == 0:
            counts[f"{kind}_dangling_local_reference"] += 1
        elif count > 1:
            counts[f"{kind}_ambiguous_local_reference"] += 1
    return dict(sorted(counts.items()))


def _external_matches(fps: list[dict], records) -> list[dict]:
    lookup = {kind: defaultdict(list) for kind in NORMALIZATIONS}
    for index, fp in enumerate(fps):
        for kind, digest in fp.items():
            lookup[kind][digest].append(index)
    matches = defaultdict(list)
    for record_id, text, label in records:
        for kind, digest in fingerprints(text).items():
            for index in lookup[kind].get(digest, ()):
                matches[(index, kind, digest)].append({"id": record_id, "label": label})
    return [{"row_index": index, "normalization": kind, "sha256": digest,
             "records": sorted(records, key=lambda r: r["id"])}
            for (index, kind, digest), records in sorted(matches.items())]


def audit_relations(originals: list[dict], spans: list[dict], documents: list[dict], targets: set[int],
                    prior: set[int], development: list[dict], hc3: list[dict]) -> dict:
    """Audit every supplied study row; output only hashes, indices and fixed labels.

Errors contain no source text. Technical integrity failures raise; missing event
lineage and contradictory assistance labels remain explicit in row diagnostics.
"""
    _require(all(isinstance(r, list) for r in (originals, spans, documents))
             and 0 < len(originals) == len(spans) == len(documents) <= MAX_ROWS, "unaligned source row counts")
    size = len(spans)
    for members in (targets, prior):
        _require(isinstance(members, set) and all(type(i) is int and 0 <= i < size for i in members), "invalid role indices")
    _require(bool(targets) and not targets & prior, "empty or overlapping audit roles")
    links = defaultdict(set)
    rows, fps = [], []
    for index, (original, span, document) in enumerate(zip(originals, spans, documents, strict=True)):
        metadata = _metadata(span)
        _require(_metadata(original) == metadata == _metadata(document), "representation metadata mismatch")
        text, parts = _segments(span)
        other, _ = _segments(document)
        _require(text == other, "representation final-text mismatch")
        fp = fingerprints(text)
        fps.append(fp)

        def add(kind, digest, row_index=index):
            links[(kind, digest)].add(row_index)

        author_sha = _sha(metadata["author"])
        add("author_pseudonym", _identity("aitdna", "author", author_sha))
        for kind, digest in fp.items():
            add(f"final_text_{kind}", digest)
        bot = [p for p in parts if p["author"] == "Bot" and p["text"].strip()]
        truth_valid = (not bot) if metadata["human_only"] else bool(bot)
        counters = _lineage(_events(original), add)
        rows.append({"row_index": index, "author_sha256": author_sha, "fingerprints": fp,
                     "human_only": metadata["human_only"], "assistance_truth": (not metadata["human_only"]) if truth_valid else None,
                     "truth_consistent": truth_valid, "characters": len(text),
                     "bot_characters": sum(len(p["text"]) for p in parts if p["author"] == "Bot"),
                     "nonempty_bot_spans": len(bot), "bot_spans_without_queries": sum(not p["queries"] for p in bot),
                     "span_query_entries": sum(len(p["queries"]) for p in parts),
                     "span_query_orphans": None, "span_query_semantics": "unverified_namespace", "lineage": counters})
    _require(isinstance(development, list), "invalid development records")
    dev = []
    seen = set()
    for record in development:
        _require(isinstance(record, dict) and isinstance(record.get("id"), str)
                 and _DEVELOPMENT_ID.fullmatch(record["id"]) is not None and record["id"] not in seen
                 and type(record.get("label")) is int and record["label"] in (0, 1), "invalid development identity or label")
        seen.add(record["id"])
        dev.append((record["id"], _text(record.get("text")), record["label"]))
    _require(isinstance(hc3, list) and all(isinstance(r, dict) for r in hc3), "invalid HC3 rows")
    hc = []
    empty_answers = 0
    for index, row in enumerate(hc3):
        for field, label in (("human_answers", 0), ("chatgpt_answers", 1)):
            for answer, text in enumerate(_answers(row, field)):
                if not text.strip():
                    empty_answers += 1
                    continue
                hc.append((f"hc3:{index}:{field}:{answer}", text, label))
    matches = {"development": _external_matches(fps, dev), "hc3": _external_matches(fps, hc)}
    parent = list(range(size))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    relations = []
    for (kind, digest), members in sorted(links.items()):
        if len(members) < 2:
            continue
        indices = sorted(members)
        for index in indices[1:]:
            a, b = find(indices[0]), find(index)
            parent[max(a, b)] = min(a, b)
        relations.append({"kind": kind, "identity_sha256": digest, "row_indices": indices})
    grouped = defaultdict(list)
    for i in range(size):
        grouped[find(i)].append(i)
    touched = {kind: {r["row_index"] for r in match} for kind, match in matches.items()}
    components = []
    for indices in sorted(grouped.values(), key=lambda v: v[0]):
        members = set(indices)
        other = sorted(members - targets - prior)
        local_issues = [i for i in indices if any(
            count and any(word in key for word in ("invalid", "duplicate", "dangling", "ambiguous", "unclassified"))
            for key, count in rows[i]["lineage"].items())]
        reasons = []
        if members & targets:
            for reason, applies in (("prior_exposure_relation", bool(members & prior)),
                                    ("other_exclusion_relation", bool(other)),
                                    ("known_development_match", bool(members & touched["development"])),
                                    ("broader_hc3_match", bool(members & touched["hc3"])),
                                    ("unresolved_local_event_identity", bool(local_issues)),
                                    ("inconsistent_assistance_truth", any(not rows[i]["truth_consistent"] for i in members))):
                if applies:
                    reasons.append(reason)
        components.append({"row_indices": indices, "target_row_indices": sorted(members & targets),
                           "prior_row_indices": sorted(members & prior), "other_excluded_row_indices": other,
                           "touches_prior": bool(members & prior), "touches_other_exclusions": bool(other),
                           "development_match_row_indices": sorted(members & touched["development"]),
                           "hc3_match_row_indices": sorted(members & touched["hc3"]),
                           "local_lineage_issue_row_indices": local_issues,
                           "target_exclusion_reasons": reasons})
    return {"schema": "veritas.text-lineage-audit/1", "model_scores_used": False, "split_assignment": False,
            "independent_validation_eligible": False, "admitted": False,
            "target_row_indices": sorted(targets), "prior_row_indices": sorted(prior),
            "counts": {"source_rows": size, "target_rows": len(targets), "prior_rows": len(prior),
                       "other_excluded_rows": size - len(targets) - len(prior), "components": len(components),
                       "target_components": sum(bool(c["target_row_indices"]) for c in components),
                       "source_authors": len({r["author_sha256"] for r in rows}),
                       "target_authors": len({rows[i]["author_sha256"] for i in targets}),
                       "target_human_controls": sum(rows[i]["human_only"] for i in targets),
                       "target_truth_inconsistent": sum(not rows[i]["truth_consistent"] for i in targets),
                       "development_records": len(dev), "hc3_answer_records": len(hc),
                       "hc3_human_answers": sum(label == 0 for _, _, label in hc),
                       "hc3_ai_answers": sum(label == 1 for _, _, label in hc),
                       "hc3_empty_answers": empty_answers,
                       "matched_source_rows": {k: len(v) for k, v in touched.items()},
                       "matched_target_rows": {k: len(v & targets) for k, v in touched.items()}},
            "rows": rows, "relations": relations, "components": components, "matches": matches,
            "limitations": ["Author pseudonyms do not authenticate participants or exclude aliases across sessions.",
                            "Raw typed event-ID reuse forms conservative potential relations; global ID scope and sessions are not authenticated.",
                            "Relative event times do not establish absolute generation dates or session identity.",
                            "Publisher fullQuery is reconstructed; exact/normalized equality does not resolve shared templates or semantic prompt reuse.",
                            "Span query strings have unverified semantics; no request-ID or orphan interpretation is inferred.",
                            "Original is an event representation of the same document, not an independent human-original sample.",
                            "Bot genesis indicates retained assistance, not a calibrated authorship fraction.",
                            "Development and broader HC3 matches are separate exclusion signals; absence is not upstream training independence.",
                            "This single-source audit does not resolve rights, source independence, calibration or admission."]}
