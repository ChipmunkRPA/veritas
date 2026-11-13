"""Replay native Word draft study decisions and source-range geometry without models."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.writing-word-boundary-transfer/1"
SCOPES = {"plos": 27, "journal": 15, "f1000": 5, "selection": 2}
POINTS = {"joint": 0.12844311183234092, "context_only": 0.17013514557368367, "original_context": math.log(2)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def odds(value):
    require(math.isfinite(value) and 0 <= value <= 1, "Invalid response")
    clipped = min(1 - 1e-7, max(1e-7, value))
    return clipped / (1 - clipped)


def evaluate(readings):
    values = [[float.fromhex(value) for value in response["responses_hex"]] for response in readings]
    require(len(values) == 2 and all(len(row) == 4 for row in values), "Invalid response order")
    require(all(len(row["token_counts"]) == 4 and all(type(t) is int and 0 < t <= 768 for t in row["token_counts"])
                for row in readings), "Invalid recorded complete-input token counts")
    context = min(math.log(odds(row[1]) / odds(row[0])) for row in values)
    sentence = min(math.log(odds(row[3]) / odds(row[2])) for row in values)
    return {"joint": min(context, sentence) > POINTS["joint"],
            "context_only": context > POINTS["context_only"],
            "old_context_comparator": context >= POINTS["original_context"]}


def geometry(row):
    identities = []
    for side in ("earlier", "later"):
        source = row["geometry"][side]
        spans = source["spans"]
        word = source["word"]
        require(len(spans) == len(word) == 3, "Expected three complete sentence ranges")
        starts = {}
        for (start, end), (number, index, local_start, local_end) in zip(spans, word, strict=True):
            require(0 <= local_start < local_end and 0 <= start < end and end - start == local_end - local_start
                    and number > 0 and index >= number - 1, "Word and body ranges differ")
            base = start - local_start
            require(base >= 0 and starts.setdefault((number, index), base) == base, "Word paragraph base differs")
        require(all(before[1] < after[0] for before, after in zip(spans, spans[1:])), "Sentences overlap")
        ids = [tuple(item[:2]) for item in word]
        require(len(starts) == 2 and ids[0] <= ids[1] <= ids[-1]
                and ids[-1] == (ids[0][0] + 1, ids[0][1] + 1), "Word paragraph break differs")
        identities.append(ids)
    require(identities[0] == identities[1], "Word paragraph identities changed")
    if row["truth"] is not None:
        require(row["truth"] == row["geometry"]["later"]["spans"][1], "Measured sentence missed target")


def review(receipt):
    require(receipt["schema"] == SCHEMA and {k: float.fromhex(v) for k, v in receipt["internal_operating_points_hex"].items()} == POINTS,
            "Unexpected study or operating points")
    require(receipt["inputs_order"] == ["earlier_context", "later_context", "earlier_sentence", "later_sentence"]
            and receipt["anchors_order"] == ["paragraph", "body_index", "start", "end"], "Unexpected input order")
    rows = receipt["cases"]
    require(Counter(row["panel"] for row in rows) == SCOPES, "Unexpected panel sizes")
    candidates = {}
    previous = 0
    held = Counter()
    for row in rows:
        key = (row["panel"], row["group"], row["role"])
        require(key not in candidates, "Duplicate case")
        geometry(row)
        decisions = evaluate(row["readings"])
        require(decisions == row["decisions"], "Saved response direction differs")
        if row["panel"] == "plos":
            held[row["role"]] += 1
            for name, positive in decisions.items():
                held[(row["role"], name)] += positive
        else:
            require(row["previous_txt"] == [reading["responses_hex"] for reading in row["readings"]],
                    "Previously measured TXT responses differ")
            previous += 1
        candidates[key] = row
    require({(x["doi"].rsplit(".", 1)[-1]) for x in receipt["plos_sources"]} ==
            {group for panel, group, role in candidates if panel == "plos" and role == "ordinary"},
            "Source DOI or ordinary control roster changed")
    app = receipt["app"]
    require(app["simulated_paid_jobs"] == len(app["cases"]) and app["unpaid_jobs"] == 0, "Paid sample counts differ")
    for delivery in app["cases"]:
        record = candidates[delivery["panel"], delivery["group"], delivery["role"]]
        expected = "joint_increase" if record["decisions"]["joint"] else "no_joint_increase"
        require(delivery["status"] == expected and delivery["exact_word_sentence_locations"] == 6, "Paid result differs")
        require(delivery["source_hashes"] == [record["geometry"][side]["file_sha256"] for side in ("earlier", "later")],
                "Paid source differs")
        require(all(delivery[field] is True for field in ("pdf_json_exact", "previous_opt_out_replayed", "assessment_unchanged",
                                                          "saved_response_arithmetic_replayed")), "Paid verification incomplete")
    browser = app["browser"]
    require(browser["page_errors"] == 0 and browser["native_word_sentence_locations"] == 6
            and browser["pdf_json_downloads_match"] and browser["plain_view"], "Browser verification incomplete")
    return {"prior_txt_batches_bit_exact": previous, "paid_cases": len(app["cases"]),
            "held": {role: {"pairs": held[role], **{name: held[role, name] for name in
                      ("joint", "old_context_comparator", "context_only")}} for role in ("granite", "olmo", "ordinary")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    try:
        result = review(json.loads(args.receipt.read_bytes()))
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"Review failed. {error}") from None
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
