"""Reopen the original MIME and independently check the measured HTML source positions."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from email import policy
from email.parser import BytesParser
from hashlib import sha256
from html import unescape
from pathlib import Path
from xml.etree import ElementTree

from app.detectors.text.extract import extract


def verify(run, plain, previous):
    lock = json.loads((run / "freeze.json").read_bytes())
    cases = json.loads((run / "cases.json").read_bytes())
    assert sha256((run / "cases.json").read_bytes()).hexdigest() == lock["cases"]
    result = json.loads((run / "result.json").read_bytes())
    known = json.loads((previous / "freeze.json").read_bytes())["requests"]
    decisions = {(row["source"], row["layout"]): row for row in json.loads((previous / "decisions.json").read_bytes())}
    records = {row["name"]: row for row in result["records"]}
    checked = Counter()
    for case in cases:
        path = run / "html" / case["name"]
        encoded = path.read_bytes()
        assert sha256(encoded).hexdigest() == case["html"]
        if case["control"] is not None:
            assert case["name"] not in records
            checked["excluded_deliveries"] += 1
            continue
        record = records.pop(path.name)
        exact = extract(path, ".eml", allow_ocr=False)
        plain_bytes = (plain / path.name).read_bytes()
        assert sha256(plain_bytes).hexdigest() == case["plain"]
        message = BytesParser(policy=policy.default).parsebytes(encoded)
        body, attachment = message.iter_parts()
        assert record["source"] == "1.1" and body.get_content_type() == "text/html"
        assert attachment.get_content_disposition() == "attachment" and attachment.get_filename() == "ignore.bin"
        checked[body["Content-Transfer-Encoding"]] += 1
        raw = body.get_content()
        markup = ElementTree.fromstring(raw.removeprefix("<!doctype html>"))
        body_node = markup.find("body")
        assert body_node is not None and body_node[-1].tag == "blockquote"
        blocks = list(body_node)[:-1] + list(body_node[-1])
        roles = ["body"] * (len(body_node) - 1) + ["quoted"] * len(body_node[-1])
        parts = [" ".join("".join(block.itertext()).split()) for block in blocks]
        assert "\n\n".join(parts) == exact.text
        valid_body = []
        for part, role, block in zip(parts, roles, exact.source_map, strict=True):
            start, stop = block["start"], block["end"]
            assert block["mime_part"] == "1.1" and block["quote_role"] == role and exact.text[start:stop] == part
            if role == "body":
                valid_body.append({"start": start, "end": stop})
            output, markup_end = start, 0
            for a, b, source_start, source_stop in block["source_ranges"]:
                assert a == output and b > a and markup_end <= source_start < source_stop <= len(raw)
                piece = unescape(re.sub(r"</?(?:b|em|i|span|strong)>", "", raw[source_start:source_stop]))
                target = exact.text[a:b]
                assert piece == target or target == " " and piece.isspace()
                output, markup_end = b, source_stop
                checked["verified_mime_ranges"] += 1
            assert output == stop
        assert valid_body == record["body"]
        names = tuple(map(int, path.stem.split("-")))
        assert known[decisions[names]["request"]] == [item["sha256"] for item in record["selected"]]
        for item in record["selected"]:
            fragments = []
            for span in item["spans"]:
                a, b = span["start"], span["end"]
                assert any(x["start"] <= a < b <= x["end"] for x in valid_body)
                fragments.append(exact.text[a:b])
                checked["verified_context_ranges"] += 1
            assert sha256(" ".join(fragments).encode()).hexdigest() == item["sha256"]
            checked["verified_contexts"] += 1
        checked["verified_deliveries"] += 1
    assert not records and checked["verified_deliveries"] == 308 and checked["excluded_deliveries"] == 36
    return {"schema": lock["schema"], "design": lock["design"], "by_layout": result["by_layout"],
            "controls": result["controls"], "independent": checked,
            "freeze_sha256": sha256((run / "freeze.json").read_bytes()).hexdigest(),
            "exact_request_replay_without_new_model_calls": True}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "plain", "previous", "output"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    result = verify(args.run, args.plain, args.previous)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result["independent"]), flush=True)


if __name__ == "__main__":
    main()
