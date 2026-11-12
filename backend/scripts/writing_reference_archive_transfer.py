"""Measure local retrieval and exact difference evidence against licensed archive texts."""

from __future__ import annotations

import argparse
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from scripts import writing_paired_revision_replication as generation
from scripts.video_appearance_native_v108 import read, require, write
from scripts.writing_blind_rewrite_probe import generator_configuration, text_hash, valid_sentence, words
from scripts.writing_focus_v3 import BASELINE_ASSETS, digest

SCHEMA = "chipmunk.writing-reference-archive-transfer/1"
ANCHOR = "216cdb7649a8e29504377d4f2858351dc46331a8"
VOLUMES = {
    "frus1961-63v09": ("e2f31e6d66b76f4d81c9311661a48ad3679aaf8dbaaddd9a48e2d9f2cae65d31", "d5d210386f9882e9c8beb49be6d6309b6a869f5dc233cbd31c05b665701b07d5"),
    "frus1964-68v08": ("2b11b34daf9ba996254ce33b5dfe68518216e45beeab0199417e519f01e04c86", "014189190aa483b2e5601288c8f2a89bf5074a064e84c488fec1beb858c1e4cb"),
    "frus1964-68v15": ("b0925e5f64418901b75f0a3a749c70cf239719cd59a8a90d503787e6313bea5a", "570f2a0bfdcd9e6099c492f9ae1cca6948e34427354cfa2e93157496c1fe7bd9"),
    "frus1964-68v24": ("c0fd378ef1ef3ffc33b28a67a33730672cbd725ca83bb4f2920f60e7be4820a9", "99eda4695ad73d86b73000636fde4296f4923acafc216b4173e8314f441cfe08"),
}
LICENSE_SHA = "832ba2061e56721ec56f349dac4fa2bc27683c1e937a768088050f302cd73ca4"
TYPO_VOLUME = "frus1964-68v24"
TYPO_PARENT = "c860bd1cdf5d5b206f2b63e92a1bd1ae24b4c66f"
TYPO_FIX = "41eae325d63e2f1513c8d56b56dde6a6a7e8b7bd"
TYPO_SHA = "d2ecaf45c4ec63e1636eae07da27a4de7a774b33de27d302644ec4c431d70ceb"
T = "{http://www.tei-c.org/ns/1.0}"
X = "{http://www.w3.org/XML/1998/namespace}"
WORD = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)*", re.UNICODE)
AUTHORITY = re.compile(r"^\d+[a-z]?\. (?:"
    r"(?:Telegram|Airgram) From (?:the )?(?:Embassy in|Department of State|U\.S\. Mission|Consulate)"
    r"|(?:Memorandum|Letter|Telegram) From (?:the )?(?:Secretary of (?:State|Defense|the Treasury)|Under Secretary of (?:State|the Treasury)|President(?: Kennedy| Johnson|[’']s)|Deputy Under Secretary of State))",
    re.I)
FORBIDDEN = {"quote", "q", "list", "table", "figure", "foreign", "gap", "unclear", "choice"}
SKIP = {"note", "pb", "lb", "milestone", "anchor"}
NGRAM = 8
METHOD = {
    "rights": "Use four pinned U.S. State Department archival TEI volumes under the repository's public-domain and CC0 dedication. Include direct body paragraphs only from U.S. official sender headings and omit embedded quoted, foreign and editorial material.",
    "source_sampling": "Hash order within each volume. Register up to 72 six-paragraph source windows per volume. Split up to four distinct official sender groups per volume into in-index evaluation and three into open-set holdout. Keep publisher corrections in a separate source group stratum.",
    "text_retrieval": "Index case-folded eight-word fingerprints with source positions. Suppress grams common to more than four archive documents. Require at least 20 supported words and 55% lexical support per paragraph, three strictly ordered body paragraphs and 65% whole-body support. Refuse competing source matches within 10 percentage points. Require 60 total supported words.",
    "alignment": "Retain strongly matched paragraphs and a single anchored gap between consecutive known anchors. Equal gaps can bind one changed paragraph. Unequal single gaps can identify an inserted or deleted paragraph. Leave exterior and longer ambiguous gaps unaligned.",
    "differences": "Use the existing exact word-and-punctuation comparison on each aligned paragraph. Report original decoded Unicode ranges and precise changed tokens. Combine each paragraph's changed-token runs into an envelope for overlap measurement, retaining the exact separate runs.",
    "controls": "Unchanged archive excerpts, exact same-source human sentence swaps, pre-correction publisher excerpts, a missing-source condition, quotes from a registered work inside unrelated genuine body text, and exact duplicate-source ambiguity. Generation adds no authorship label to the archive method.",
    "publisher_probe": "Separately include one identified three-paragraph excerpt around an actual U.S. archive typo correction, d585 in frus1964-68v24. Keep the six-paragraph pre-correction intake and its observed dispositions.",
    "current": "Measure first six incumbent native context windows at a raw response at least .5 on registered English original, ordinary sentence swap and generated files only. This is a location comparator.",
    "metric": "Useful query mismatch range requires .30 nonwhitespace character IoU and .50 recall with known replacement. Publisher corrections use independently measured exact query and reference character edit coordinates. Source availability and attribution, ordinary differences and false mismatches are distinct outcomes.",
    "interpretation": "This is a bounded public-source comparison. The result describes actual text matching or difference with the cited archive version. It contributes zero AI-origin or authorship confidence.",
}


def raw_contract(run):
    files = run / "raw"
    license_path = files / "LICENSE.md"
    require(digest(license_path) == LICENSE_SHA and "CC0 1.0 Universal" in license_path.read_text(),
            "Archive rights differ")
    by_volume = {}
    for volume, checks in VOLUMES.items():
        pair = (files / (volume + ".xml"), files / (volume + ".prior.xml"))
        require(tuple(map(digest, pair)) == checks, "Pinned archived XML changed")
        prior = (files / (volume + ".prior-commit")).read_text().strip()
        history = read(files / (volume + ".history.json"))
        change_prefix = "000b082d9484e3a7" if volume in tuple(VOLUMES)[:2] else "17d7e33cca061ca"
        require(re.fullmatch(r"[0-9a-f]{40}", prior) and any(
            row["sha"].startswith(change_prefix) and row["parents"][0]["sha"] == prior for row in history),
            "Publisher revision-parent witness differs")
        by_volume[volume] = {"current": ANCHOR, "earlier": prior, "xml_sha256": list(checks)}
    typo = read(files / f"{TYPO_FIX}.commit.json")
    require(typo["sha"] == TYPO_FIX and typo["parents"][0]["sha"] == TYPO_PARENT
            and digest(files / f"{TYPO_VOLUME}.typo-parent.xml") == TYPO_SHA, "Typo correction source pin differs")
    return {"schema": SCHEMA, "method": METHOD, "volume_pins": by_volume, "license_sha256": LICENSE_SHA,
            "publisher_probe": {"volume": TYPO_VOLUME, "doc_id": "d585", "commit": TYPO_FIX,
                                "prior_commit": TYPO_PARENT, "prior_xml_sha256": TYPO_SHA},
            "code_sha256": digest(Path(__file__))}


def rendered(node):
    from lxml import etree

    if node.text is None and not len(node):
        return ""
    found = [node.text or ""]
    for child in node:
        tag = etree.QName(child).localname
        if tag not in SKIP:
            found.append(rendered(child))
        found.append(child.tail or "")
    return "".join(found)


def normalized(value):
    return " ".join(unicodedata.normalize("NFC", value).split())


def source_rows(path, volume):
    from lxml import etree

    parsed = etree.parse(path, etree.XMLParser(resolve_entities=False, no_network=True, remove_comments=True))
    root = parsed.getroot()
    require(root.get(X + "id") == volume and not parsed.docinfo.doctype, "Wrong XML volume")
    docs = {}
    for node in root.iter(T + "div"):
        if node.get("type") != "document":
            continue
        doc_id = node.get(X + "id", "")
        if not re.fullmatch(r"d[0-9]+[a-z]?", doc_id) or node.find(T + "head") is None:
            continue
        header = node.find(T + "head")
        title = normalized(rendered(header))
        if not AUTHORITY.match(title):
            continue
        person = next((normalized(rendered(p)) for p in header.findall(T + "persName") if p.get("type") == "from"), "")
        signer = next((normalized(rendered(p)) for p in node.findall(f"{T}closer/{T}signed") if normalized(rendered(p))), "")
        identity = person or signer or re.split(r"\s+to\s+", re.split(r"\s+From\s+", title, maxsplit=1)[-1], maxsplit=1)[0]
        identity = normalized(identity).casefold().strip(". ,")
        if not identity or len(identity) > 110:
            continue
        paragraphs = []
        for i, element in enumerate(node.findall(T + "p")):
            unsupported = any(etree.QName(child).localname in FORBIDDEN for child in element.iterdescendants())
            text = normalized(rendered(element))
            allowed = not unsupported and 22 <= len(words(text)) <= 190 and not re.search(
                r"(?i)copyright|reprinted|untranslated|\[less than|\[text not|\[here follows", text)
            paragraphs.append({"ordinal": i, "text": text if allowed else "", "line": element.sourceline})
        docs[doc_id] = {"group": volume + "/" + doc_id, "volume": volume, "doc_id": doc_id,
                        "sender_key": identity, "paragraphs": paragraphs}
    return docs


def paragraph_window(parts, start):
    values = parts[start:start + 6]
    if len(values) != 6 or not all(p["text"] for p in values):
        return None
    total = sum(len(words(p["text"])) for p in values)
    return values if 180 <= total <= 900 else None


def target_sentence(parts):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    candidates = []
    for j in range(1, len(parts) - 1):
        doc = from_string(parts[j]["text"])
        units = segment(doc)
        if len(units) < 2 or doc.text != parts[j]["text"]:
            continue
        for unit in units:
            if 12 <= len(words(unit.text)) <= 40 and unit.text.endswith((".", "!", "?")):
                candidates.append((text_hash(unit.text), [j, unit.start, unit.end], unit.text))
    if candidates:
        _, target, sentence = min(candidates)
        return target, sentence
    return None


def old_changes(new, old):
    edits = []
    for i, (a, b) in enumerate(zip(new, old, strict=True)):
        if a["text"] == b["text"]:
            continue
        if not all(row["text"] for row in (a, b)):
            return []
        for action, lo, hi, x, y in SequenceMatcher(None, a["text"], b["text"], autojunk=False).get_opcodes():
            if action != "equal":
                edits.append({"paragraph": i, "reference": [lo, hi], "query": [x, y]})
    return edits


def qualification(run, contract):
    from scripts.text_elife_source_intake_v272 import overlap
    from scripts.text_short_insert_intake_v203 import fingerprints
    from scripts.writing_archive_author_intake import HISTORY, historical_texts

    counts, raw_corpus, corrections = defaultdict(Counter), [], []
    for volume in VOLUMES:
        current = source_rows(run / "raw" / f"{volume}.xml", volume)
        earlier = source_rows(run / "raw" / f"{volume}.prior.xml", volume)
        counts[volume]["official_docs"] = len(current)
        for doc_id in sorted(current, key=text_hash):
            row = current[doc_id]
            past = earlier.get(doc_id)
            selected, revised = None, None
            for i in range(max(0, len(row["paragraphs"]) - 5)):
                window = paragraph_window(row["paragraphs"], i)
                if window is None:
                    continue
                target = target_sentence(window)
                if target and selected is None:
                    selected = row | {"paragraphs": window, "target": target[0], "before": {"focus_text": target[1]}}
                if not past or revised is not None or len(past["paragraphs"]) != len(row["paragraphs"]):
                    continue
                old = paragraph_window(past["paragraphs"], i)
                if not old:
                    continue
                changes = old_changes(window, old)
                if 1 <= len(changes) <= 6 and len({edit["paragraph"] for edit in changes}) == 1 and changes[0]["paragraph"] in (1, 2, 3, 4):
                    revised = row | {"paragraphs": window, "earlier": old, "edits": changes}
            if selected is not None:
                raw_corpus.append(selected)
                counts[volume]["eligible_windows"] += 1
            if revised is not None:
                corrections.append(revised)
                counts[volume]["publisher_revision_windows"] += 1
    exposure = [(name, digest(path), None, historical_texts(path)) for name, path in HISTORY.items()]
    signatures = defaultdict(set)
    for row in raw_corpus + corrections:
        signatures[row["group"]].update(*(fingerprints(p["text"]) for p in row["paragraphs"]))
    screened, external, _ = overlap(exposure, signatures)
    tested = set()
    groups, index, used_senders, unknown = [], [], set(), []
    for volume in VOLUMES:
        pool = [row for row in raw_corpus if row["volume"] == volume]
        for row in pool:
            pats = signatures[row["group"]]
            if row["sender_key"] in used_senders or row["group"] in external or pats & tested:
                continue
            if counts[volume]["known"] < 4:
                counts[volume]["known"] += 1
                groups.append(row)
                used_senders.add(row["sender_key"])
                tested |= pats
            elif counts[volume]["unknown"] < 3:
                counts[volume]["unknown"] += 1
                unknown.append(row)
                used_senders.add(row["sender_key"])
                tested |= pats
            if counts[volume]["known"] >= 4 and counts[volume]["unknown"] >= 3:
                break
    excluded = {row["group"] for row in unknown}
    reserved = {row["group"] for row in groups} | excluded
    old_examples = []
    for row in corrections:
        volume = row["volume"]
        if row["group"] in reserved or row["group"] in external or counts[volume]["authentic_edits"] >= 3:
            continue
        reserved.add(row["group"])
        old_examples.append(row)
        counts[volume]["authentic_edits"] += 1
    current = source_rows(run / "raw" / f"{TYPO_VOLUME}.xml", TYPO_VOLUME)["d585"]
    earlier = source_rows(run / "raw" / f"{TYPO_VOLUME}.typo-parent.xml", TYPO_VOLUME)["d585"]
    current_parts, earlier_parts = (row["paragraphs"][3:6] for row in (current, earlier))
    actual = old_changes(current_parts, earlier_parts)
    require(len(actual) == 1 and "op- posing" in earlier_parts[0]["text"]
            and "opposing" in current_parts[0]["text"] and current["group"] not in reserved,
            "Targeted genuine archive correction differs")
    old_examples.append(current | {"paragraphs": current_parts, "earlier": earlier_parts, "edits": actual,
                                   "probe": contract["publisher_probe"]})
    counts[TYPO_VOLUME]["targeted_genuine_probe"] += 1
    index.extend(groups + old_examples)
    included = {row["group"] for row in index}
    for volume in VOLUMES:
        for row in (item for item in raw_corpus if item["volume"] == volume):
            if row["group"] in included or row["group"] in excluded:
                continue
            if counts[volume]["distractors"] >= 65:
                break
            counts[volume]["distractors"] += 1
            included.add(row["group"])
            index.append(row)
    require(len(groups) >= 8 and len(unknown) >= 6, "Insufficient fresh licensed sources")
    table = {row["group"]: {key: row[key] for key in ("volume", "doc_id", "paragraphs")} for row in index}
    return groups, unknown, old_examples, table, {"counts": {key: dict(value) for key, value in counts.items()},
        "prior_source_exposure": screened, "source_freeze_sha256": digest(run / "source-freeze.json"),
        "volume_pins": contract["volume_pins"], "publisher_probe": contract["publisher_probe"],
        "rights": "U.S. public domain and CC0 1.0"}


def grams(value):
    terms = [unicodedata.normalize("NFKC", match[0]).casefold().replace("’", "'") for match in WORD.finditer(value)]
    return [tuple(terms[i:i + NGRAM]) for i in range(len(terms) - NGRAM + 1)], len(terms)


def build_index(catalog):
    hits = defaultdict(set)
    for source, row in catalog.items():
        for j, para in enumerate(row["paragraphs"]):
            for key in grams(para["text"])[0]:
                hits[key].add((source, j))
    return {key: sorted(value) for key, value in hits.items() if len({source for source, _ in value}) <= 4}


def query_evidence(doc, catalog, index):
    from app.evidence.document_revision import differences, tokens

    body = []
    for i, (a, b) in enumerate(doc.paragraphs):
        source_map = doc.source_map[i]
        if source_map.get("quote_role", "body") != "body" or doc.text[a:b].lstrip().startswith(">"):
            continue
        body.append((a, b))
    if not 3 <= len(body) <= 24:
        return {"status": "out_of_scope", "source": None, "body_paragraphs": len(body), "matched": [], "edits": []}
    evidence = defaultdict(dict)
    denominator = 0
    for i, (a, b) in enumerate(body):
        fingerprints, n = grams(doc.text[a:b])
        denominator += n
        counts = defaultdict(set)
        for position, key in enumerate(fingerprints):
            for entry in index.get(key, ()):
                counts[entry].update(range(position, position + NGRAM))
        best_by_source = defaultdict(list)
        for (source, j), matches in counts.items():
            if len(matches) >= 20 and len(matches) * 100 >= 55 * n:
                best_by_source[source].append((len(matches), j))
        for source, choices in best_by_source.items():
            choices.sort(reverse=True)
            if len(choices) == 1 or choices[0][0] > choices[1][0]:
                evidence[source][i] = choices[0]
    ranked = []
    for source, items in evidence.items():
        ordered = sorted(items.items())
        if len(ordered) < 3 or any(a[1][1] >= b[1][1] for a, b in zip(ordered, ordered[1:])):
            continue
        coverage = sum(item[1][0] for item in ordered)
        if coverage >= 60:
            ranked.append((coverage, source, ordered))
    ranked.sort(reverse=True)
    if not ranked or ranked[0][0] * 100 < 65 * denominator:
        return {"status": "no_qualified_reference", "source": None, "body_paragraphs": len(body), "matched": [], "edits": []}
    if len(ranked) > 1 and (ranked[0][0] - ranked[1][0]) * 100 < 10 * denominator:
        return {"status": "ambiguous_reference", "source": None, "body_paragraphs": len(body), "matched": [], "edits": []}
    support, source, ordered = ranked[0]
    pairs = [(i, item[1]) for i, item in ordered]
    mapping = [(i, j, "fingerprint_anchor") for i, j in pairs]
    for (a, b), (c, d) in zip(pairs, pairs[1:]):
        if c - a == 2 and d - b == 2:
            mapping.append((a + 1, b + 1, "bounded_pair"))
        elif c - a == 2 and d - b == 1:
            mapping.append((a + 1, None, "bounded_insert"))
        elif c - a == 1 and d - b == 2:
            mapping.append((c, b + 1, "bounded_delete"))
    edits = []
    paragraphs = catalog[source]["paragraphs"]
    for i, j, relation in sorted(mapping, key=lambda item: (item[0], item[2], item[1] or 0)):
        lo, hi = body[i]
        if relation == "bounded_delete":
            length = len(paragraphs[j]["text"])
            record = [{"earlier": [0, length], "later": [lo, lo],
                       "earlier_changed_tokens": [[a, b] for _, a, b in tokens(paragraphs[j]["text"], [(0, length)])],
                       "later_changed_tokens": []}]
        elif j is None:
            record = [{"earlier": None, "later": [lo, hi], "earlier_changed_tokens": [],
                       "later_changed_tokens": [[a, b] for _, a, b in tokens(doc.text, [(lo, hi)])]}]
        else:
            archive = paragraphs[j]["text"]
            record = differences(tokens(archive, [(0, len(archive))]), tokens(doc.text, [(lo, hi)]))
        if record:
            edits.append({"query_paragraph": i, "source_paragraph": j, "source_ordinal": paragraphs[j]["ordinal"] if j is not None else None,
                          "relation": relation, "changes": record,
                          "query_envelope": [min(r["later"][0] for r in record), max(r["later"][1] for r in record)]})
    return {"status": "reference_bound", "source": source, "supported_words": support, "body_words": denominator,
            "body_paragraphs": len(body), "matched": [[i, j, relation] for i, j, relation in mapping], "edits": edits}


def method_freeze(run, snapshot):
    from scripts.text_elife_rewrites_v273 import GENERATORS, MLX_VERSIONS

    require(all(digest(snapshot / name) == pin for name, pin in BASELINE_ASSETS.items()), "Current model snapshot differs")
    config = generator_configuration()
    require(all(config[key]["license"] == "Apache-2.0" and digest(value[3] / "model.safetensors") == value[2]
                for key, value in GENERATORS.items()), "Generator model weights or terms differ")
    return {"schema": SCHEMA, "source_lock_sha256": digest(run / "source-freeze.json"),
            "source_receipt_sha256": digest(run / "source-receipt.json"),
            "source_assets": {name: digest(run / name) for name in ("originals.json", "unknown.json", "earlier.json", "index.json")},
            "generation_assets": config, "runtime": MLX_VERSIONS, "seeds": generation.SEEDS, "current": BASELINE_ASSETS,
            "prompt": generation.EDIT_PROMPT, "settings": generation.SETTINGS}


def body_text(paragraphs):
    return "\n\n".join(p["text"] if isinstance(p, dict) else p for p in paragraphs)


def offset(paragraphs, i):
    return sum(len(p["text"] if isinstance(p, dict) else p) + 2 for p in paragraphs[:i])


def generated_cases(run):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    def paired(row, kind, replacement):
        docs = [p["text"] for p in row["paragraphs"]]
        i, lo, hi = row["target"]
        start = offset(docs, i) + lo
        original = docs[i]
        updated = original[:lo] + replacement + original[hi:]
        a, b = (segment(from_string(text)) for text in (original, updated))
        expected = [replacement if [unit.start, unit.end] == [lo, hi] else unit.text for unit in a]
        if expected != [unit.text for unit in b]:
            return None
        docs[i] = updated
        return {"group": row["group"], "kind": kind, "text": body_text(docs), "source": row["group"],
                "truth": [start, start + len(replacement)]}

    sources = read(run / "originals.json")
    unknown = read(run / "unknown.json")
    packets = {name: read(run / f"generated-{name}.json") for name in generation.SEEDS}
    lock = digest(run / "freeze.json")
    require(all(row["freeze_sha256"] == lock and row["results"].keys() == {item["group"] for item in sources}
                for row in packets.values()), "Generator artifact differs")
    rows, counts = [], defaultdict(Counter)
    for record in sources:
        key = record["group"]
        rows.append({"group": key, "kind": "original", "text": body_text(record["paragraphs"]), "source": key, "truth": None})
        own_sentence = record["before"]["focus_text"]
        original_size = len(words(own_sentence))
        alternatives = []
        for j, para in enumerate(record["paragraphs"]):
            if j == record["target"][0]:
                continue
            for sentence in segment(from_string(para["text"])):
                n = len(words(sentence.text))
                if 12 <= n <= 40 and .7 <= n / original_size <= 1.4 and sentence.text.endswith((".", "!", "?")):
                    alternatives.append((abs(n - original_size), text_hash(sentence.text), sentence.text))
        if alternatives:
            human = paired(record, "human_swap", min(alternatives)[2])
            if human:
                rows.append(human)
        for family, packet in packets.items():
            counts[family]["attempted"] += 1
            attempt = packet["results"][key]
            require(attempt["input_sha256"] == text_hash(own_sentence), "Generation source hash differs")
            revised = valid_sentence(attempt, own_sentence, generated=True)
            item = paired(record, family, revised) if revised else None
            if item:
                rows.append(item)
                counts[family]["admitted"] += 1
    for row in read(run / "earlier.json"):
        item = {"group": row["group"], "kind": "earlier_publisher", "text": body_text(row["earlier"]),
                "source": row["group"], "truth": None, "publisher_edits": row["edits"]}
        rows.append(item)
        rows.append({"group": row["group"], "kind": "publisher_current_control", "text": body_text(row["paragraphs"]),
                     "source": row["group"], "truth": None})
    for row, known in zip(unknown, sources, strict=False):
        plain = body_text(row["paragraphs"])
        rows.append({"group": row["group"], "kind": "unknown", "text": plain, "source": None, "truth": None})
        quote = "\n\n".join("> " + p["text"] for p in known["paragraphs"][1:4])
        rows.append({"group": row["group"], "kind": "quote_only", "text": plain + "\n\n" + quote, "source": None,
                     "truth": None, "quoted_source": known["group"]})
    for row in sources[:4]:
        key = row["group"]
        rows.append({"group": key, "kind": "duplicate_ambiguity", "text": body_text(row["paragraphs"]),
                     "source": None, "truth": None, "duplicate_source": key})
    return rows, {key: dict(value) for key, value in counts.items()}


def extract_queries(rows, run, *, materialize):
    from app.detectors.text.extract import extract
    from app.detectors.text.segment import segment
    from scripts.writing_body_localization_transfer import mime
    from scripts.writing_paired_email_revision import encoded, native_contexts

    values, cases = {}, []
    for i, row in enumerate(rows):
        suffix = "eml" if row["kind"] == "quote_only" else "txt"
        data, text = row["text"].encode(), row["text"]
        if suffix == "eml":
            body, quote = text.split("\n\n> ", 1)
            quoted = quote.replace("\n\n> ", " ")
            data, exact_quote = mime(body, quoted)
            text = body + "\n\n" + exact_quote
        path = run / "fixtures" / f"{i:03d}.{suffix}"
        if materialize:
            with path.open("xb") as stream:
                stream.write(data)
        require(path.read_bytes() == data, "Frozen query bytes differ")
        doc = extract(path, "." + suffix)
        require(doc.text == text, "Query decoding differs")
        if suffix == "eml":
            require(sum(row["quote_role"] == "quoted" for row in doc.source_map) == 1, "Quote role was lost")
        items = []
        if row["kind"] in ("original", "human_swap", *generation.SEEDS):
            for group in native_contexts(segment(doc)):
                entry, exact = encoded(group)
                values[entry["sha256"]] = exact
                items.append(entry)
        cases.append(({key: value for key, value in row.items() if key != "text"} | {
            "file": path.name, "file_sha256": digest(path), "text_sha256": text_hash(doc.text), "windows": items}, doc))
    return cases, values


def exact_positions(text, ranges):
    return {i for a, b in ranges for i in range(a, b) if not text[i].isspace()}


def useful(text, ranges, truth):
    a, b = exact_positions(text, ranges), exact_positions(text, [truth])
    return len(a & b) * 10 >= 3 * len(a | b) and len(a & b) * 2 >= len(b)


def verify_publisher(catalog, case, doc, result):
    if case["kind"] != "earlier_publisher" or result["source"] != case["source"]:
        return None
    joined = body_text(catalog[case["source"]]["paragraphs"])
    source = catalog[case["source"]]["paragraphs"]
    for edit in case["publisher_edits"]:
        i = edit["paragraph"]
        x, y = edit["query"]
        u, v = edit["reference"]
        a = doc.paragraphs[i][0]
        b = offset(source, i)
        require(doc.text[a + x:a + y] != joined[b + u:b + v], "Publisher revision truth has no change")
        possible = [entry for entry in result["edits"] if entry["query_paragraph"] == i and entry["source_paragraph"] == i]
        if not any(any((item["later"][0] <= a + x <= item["later"][1]
                        if x == y else item["later"][0] < a + y and a + x < item["later"][1])
                       and (item["earlier"][0] <= u <= item["earlier"][1]
                            if u == v else item["earlier"][0] < v and u < item["earlier"][1])
                       for item in group["changes"]) for group in possible):
            return False
    return True


def analyze(cases, catalog, responses):
    index = build_index(catalog)
    results = []
    for case, doc in cases:
        if case["kind"] == "duplicate_ambiguity":
            source = case["duplicate_source"]
            subset = catalog | {source + "/duplicate": catalog[source]}
            result = query_evidence(doc, subset, build_index(subset))
        else:
            result = query_evidence(doc, catalog, index)
        truth = case["truth"]
        current = [entry["spans"] for entry in case["windows"] if responses[entry["sha256"]]["score"] >= .5][:6]
        edits = result["edits"]
        results.append(case | {"source_comparison": result,
            "source_correct": result["source"] == case["source"], "current_flag": bool(current),
            "current_useful": bool(truth) and any(useful(doc.text, spans, truth) for spans in current),
            "archive_useful": bool(truth) and result["source"] == case["source"] and
                              any(useful(doc.text, [entry["query_envelope"]], truth) for entry in edits),
            "publisher_all_change_coordinates_touched": verify_publisher(catalog, case, doc, result),
            "unrelated_difference_paragraphs": sum(not truth or not (entry["query_envelope"][0] < truth[1]
                and truth[0] < entry["query_envelope"][1]) for entry in edits) if truth else None})
    return results


def summary(outcomes, counts, catalog):
    group = defaultdict(list)
    for row in outcomes:
        group[row["kind"]].append(row)
    return {"indexed_sources": len(catalog), "generation_admission": counts, "outcomes": {kind: {
        "files": len(rows), "correct_source_or_abstention": sum(row["source_correct"] for row in rows),
        "bound_source": sum(row["source_comparison"]["source"] is not None for row in rows),
        "ambiguous_reference": sum(row["source_comparison"]["status"] == "ambiguous_reference" for row in rows),
        "any_difference": sum(bool(row["source_comparison"]["edits"]) for row in rows),
        "archive_useful": sum(row["archive_useful"] for row in rows), "current_useful": sum(row["current_useful"] for row in rows),
        "current_flag": sum(row["current_flag"] for row in rows),
        "publisher_all_change_coordinates_touched": sum(row["publisher_all_change_coordinates_touched"] is True for row in rows),
        "unrelated_difference_paragraphs": sum(row["unrelated_difference_paragraphs"] or 0 for row in rows),
    } for kind, rows in group.items()}, "authorship_confidence_contribution": 0}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("source-freeze", "qualify", "freeze", "generate", "lock-inputs", "score", "report"))
    cli.add_argument("run", type=Path)
    cli.add_argument("snapshot", type=Path)
    cli.add_argument("--family", choices=generation.SEEDS)
    args = cli.parse_args()
    os.umask(0o077)
    run, phase = args.run, args.phase
    contract = raw_contract(run)
    if phase == "source-freeze":
        write(run / "source-freeze.json", contract)
        return
    require(read(run / "source-freeze.json") == contract, "Source intake contract differs")
    if phase == "qualify":
        originals, unknown, old, index, receipt = qualification(run, contract)
        for name, item in zip(("originals", "unknown", "earlier", "index", "source-receipt"), (originals, unknown, old, index, receipt), strict=True):
            write(run / f"{name}.json", item)
        print(receipt["counts"], flush=True)
        return
    require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1", "Models must be offline")
    frozen = method_freeze(run, args.snapshot)
    if phase == "freeze":
        write(run / "freeze.json", frozen)
        return
    require(read(run / "freeze.json") == frozen, "Frozen method differs")
    if phase == "generate":
        require(args.family, "Choose one local generator")
        generation.generate(run, run, frozen, args.family)
        return
    materialize = phase == "lock-inputs"
    if materialize:
        (run / "fixtures").mkdir()
    rows, counts = generated_cases(run)
    cases, values = extract_queries(rows, run, materialize=materialize)
    binding = {"cases": [row for row, _ in cases], "inputs": sorted(values), "admission": counts,
               "freeze_sha256": digest(run / "freeze.json")}
    lock = run / "input-lock.json"
    if materialize:
        write(lock, binding)
        print({"cases": len(cases), "inputs": len(values), "admission": counts}, flush=True)
        return
    require(read(lock) == binding, "Frozen query inputs differ")
    scores = run / "primary.json"
    if phase == "score":
        from scripts.writing_center_deletion_transfer import score
        score(args.snapshot, values, scores, digest(lock))
        return
    measurements = read(scores)
    require(measurements["binding"] == digest(lock) and measurements["responses"].keys() == values.keys()
            and measurements["forwards"] == math.ceil(len(values) / 4), "Primary inference receipt differs")
    catalog = read(run / "index.json")
    outcomes = analyze(cases, catalog, measurements["responses"])
    write(run / "outcomes.json", outcomes)
    report = summary(outcomes, counts, catalog)
    write(run / "aggregate.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
