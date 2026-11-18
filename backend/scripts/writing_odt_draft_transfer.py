# SPDX-License-Identifier: Apache-2.0
"""Replay known edits through same-format and mixed OpenDocument draft comparison."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZipInfo

from app.detectors.text.segment import spans
from app.evidence.document_revision import ODT_COORDINATES, compare_with_sources
from app.storage.files import ODT_MIME

PANELS = {
    "710d36ef4b31438567ab4186b194d9e493c4fa7975eefea9b8e00d48452edb0e": "gao",
    "0eb90145de1102f4732edfea7275f5d53219b40a01a063885c19bb70caa1b76c": "bea_census",
}
SCHEMA = "chipmunk.writing-odt-draft-transfer/1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def odt(text, *, restyled=False):
    def paragraph(value):
        content = escape(value)
        if restyled:
            content = f"<text:span>{content}</text:span>"
        return f"<text:p>{content}</text:p>"

    body = "".join(map(paragraph, text.split("\n\n")))
    xml = ('<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
           'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" office:version="1.3">'
           f"<office:body><office:text>{body}</office:text></office:body></office:document-content>")
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(ZipInfo("mimetype"), ODT_MIME)
        archive.writestr(ZipInfo("content.xml"), xml)
    return buffer.getvalue()


def bound_tokens(comparison, text):
    ranges, cursor = [], 0
    for paragraph in text.split("\n\n"):
        end = cursor + len(paragraph)
        ranges.append((cursor, end))
        cursor = end + 2
    output = []
    for difference in comparison["changes"]:
        later = difference["later"]
        if later is None:
            continue
        start, end, excerpt = (later[key] for key in ("start", "end", "text"))
        if excerpt != text[start:end]:
            raise ValueError("An exported excerpt differs from the decoded submitted text")
        expected = [{"paragraph": index + 1, "body_index": index, "start": max(start, a) - a, "end": min(end, b) - a}
                    for index, (a, b) in enumerate(ranges)
                    if max(start, a) < min(end, b) or start == end and a <= start <= b]
        if later["paragraphs"] != expected:
            raise ValueError("An exported paragraph location differs from the excerpt")
        output.extend(later["changed_tokens"])
    return output


def measure(path, output):
    data = path.read_bytes()
    source_sha256 = sha(data)
    if source_sha256 not in PANELS:
        raise ValueError("The input must be one of the exact bound local-paraphrase panels")
    document = json.loads(gzip.decompress(data))
    rows = []
    for work in document["works"]:
        source, *revisions = work["cases"]
        old = source["text"]
        earlier = odt(old)
        control = odt(old, restyled=True)
        unchanged, _, _, _ = compare_with_sources(earlier, control, ".odt")
        rows.append({"group": work["group"], "editor": work["generator"], "earlier_odt_sha256": sha(earlier),
                     "restyled_odt_sha256": sha(control),
                     "revisions": [compare_revision(source, revision, earlier) for revision in revisions],
                     "restyled_control_changes": len(unchanged["changes"])})
    summary = {}
    for editor in document["parents"]:
        result = Counter()
        for row in rows:
            if row["editor"] == editor:
                result.update(groups=1, restyled_controls_with_changes=bool(row["restyled_control_changes"]))
                for revision in row["revisions"]:
                    if revision["kind"] == "ordinary":
                        result.update(human_revisions=1, human_revisions_located=bool(revision["changed_token_ranges"]))
                    else:
                        result.update(edited_pairs=1, known=len(revision["known_sentence_ranges"]),
                                      located=len(revision["detected_sentence_ranges"]),
                                      tokens_outside_known=len(revision["outside_known_ranges"]),
                                      tokens_in_repeated_source_sentences=revision["tokens_in_repeated_source_sentences"])
        summary[editor] = dict(result)
    receipt = {"schema": SCHEMA, "source_sha256": source_sha256, "panel": PANELS[source_sha256],
               "summary": summary, "rows": rows}
    with output.open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(summary, sort_keys=True))


def compare_revision(source, edit, earlier):
    old, new = source["text"], edit["text"]
    later = odt(new)
    native, old_text, new_text, _ = compare_with_sources(earlier, later, ".odt")
    mixed, _, mixed_new, _ = compare_with_sources(old.encode(), later, ".odt", earlier_extension=".txt")
    if (native["coordinate_space"] != ODT_COORDINATES or mixed["later"]["coordinate_space"] != ODT_COORDINATES
            or (old_text, new_text, mixed_new) != (old, new, new)
            or any(sha(case["text"].encode()) != case["native_text_sha256"] for case in (source, edit))):
        raise ValueError("The OpenDocument projections differ from the original labelled text or coordinates")
    known = [item["source"] for item in edit["truth"]]
    native_tokens, mixed_tokens = bound_tokens(native, new), bound_tokens(mixed, new)
    if native_tokens != mixed_tokens:
        raise ValueError("Same-format and mixed-format comparisons highlight different submitted tokens")
    matches = [any(max(a, start) < min(b, end) for start, end in native_tokens) for a, b in known]
    outside = [token for token in native_tokens if known and not any(a <= token[0] < token[1] <= b for a, b in known)]
    old_sentences = Counter(old[a:b] for a, b, _ in spans(old, [(0, len(old))]))
    new_ranges = spans(new, [(0, len(new))])
    new_sentences = Counter(new[a:b] for a, b, _ in new_ranges)
    repeated = [(a, b) for a, b, _ in new_ranges if 0 < old_sentences[new[a:b]] < new_sentences[new[a:b]]]
    repeated_tokens = sum(any(a <= start < end <= b for a, b in repeated) for start, end in outside)
    return {"kind": edit["kind"], "later_odt_sha256": sha(later), "known_sentence_ranges": known,
            "detected_sentence_ranges": [region for region, match in zip(known, matches, strict=True) if match],
            "changed_token_ranges": native_tokens, "outside_known_ranges": outside,
            "tokens_in_repeated_source_sentences": repeated_tokens}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    measure(args.panel, args.output)
