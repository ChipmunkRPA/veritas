"""Replay the frozen longer-passage comparison on archival eLife writing."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .review_writing_f1000_position_transfer import POINTS, check, control_matches, digest, generated_matches
from .review_writing_long_position import compare, read, select_case

SCHEMA = "chipmunk.writing-elife-long-position-transfer/1"
GENERATORS = {"qwen": 3845958, "granite": 2845115}
ORDINARY = ("before", "after")
ARCHIVE = "58e688ec3d4523eb1f3b03bed6bb440bf91e4786"
PRIOR = (
    Path(__file__).resolve().parents[2]
    / "docs/evaluation-panels/writing-long-position-transfer-2026-11-12.json"
)


def outcomes(cases):
    output = {}
    for name, point in POINTS.items():
        locations = []
        for case in cases:
            result = select_case(case, point, allow_punctuation_unit=True, allow_touching=True)
            index = result["position"]
            positions = case["positions"]
            if index is None and not result["current"]:
                continue
            locations.append(
                {
                    "group": case["group"],
                    "variant": case["variant"],
                    "known_edit": case["truth"],
                    "position": None if index is None else positions[index][:2],
                    "current": [
                        [positions[max(0, i - 1)][0], positions[min(len(positions) - 1, i + 1)][1]]
                        for i in result["current"]
                    ],
                }
            )
        output[name] = {
            "counts": compare(cases, point, allow_punctuation_unit=True, allow_touching=True)["elife"],
            "locations": locations,
        }
    return output


def diagnostics(cases):
    targets = []
    top = 0
    for case in cases:
        if case["truth"] is None:
            continue
        values = [focus for focus in case["review"] if focus["contrast"] is not None]
        actual = next(focus["contrast"] for focus in values if focus["span"] == case["truth"])
        targets.append(actual)
        top += sum(focus["contrast"] >= actual for focus in values) == 1
    return {
        "targets_measured": len(targets),
        "targets_unique_top": top,
        "maximum_target_contrast": max(targets),
        "targets_above_frozen_point": {
            name: sum(value > point for value in targets) for name, point in POINTS.items()
        },
    }


def verify(receipt):
    check(
        receipt["schema"] == SCHEMA and receipt["prior_sha256"] == digest(PRIOR),
        "The frozen prior trial differs",
    )
    check(receipt["source"]["frozen_points"] == POINTS, "The frozen operating points differ")
    cases = receipt["cases"]
    source = receipt["source"]["sources"]
    admission = receipt["admission"]
    pins = receipt["commitments"]
    check(
        set(pins) == {"generation_freeze", "score_freeze", "raw_cases", "measurements"}
        and all(re.fullmatch(r"[a-f0-9]{64}", value) for value in pins.values()),
        "Malformed private run commitments",
    )
    check(
        admission["detector_runs"] == 0
        and admission["case_sha256"] == pins["raw_cases"]
        and admission["freeze_sha256"] == pins["generation_freeze"],
        "The source admission freeze differs",
    )
    by_case = {(case["group"], case["variant"]): case for case in cases}
    dispositions = {(row["group"], row["family"]): row for row in admission["dispositions"]}
    check(
        len(source) == 12
        and len({row["group"] for row in source}) == 12
        and len(by_case) == len(cases)
        and len(dispositions) == len(admission["dispositions"]) == 24,
        "Missing or repeated source views",
    )
    groups = set()
    admitted = dict.fromkeys(GENERATORS, 0)
    expected = set()
    for i, row in enumerate(source):
        group = row["group"]
        archive_name(row)
        for side in ORDINARY:
            key = group, side
            expected.add(key)
            check(
                by_case[key]["truth"] is None and control_matches(by_case[key], row["views"][side]),
                "Recorded ordinary control differs from its source geometry",
            )
        for family, seed in GENERATORS.items():
            key = group, family
            disposition = dispositions[key]
            check(
                re.fullmatch(r"[a-f0-9]{64}", disposition["generator_packet_sha256"]),
                "Invalid generation packet commitment",
            )
            if not disposition["admitted"]:
                check(
                    key not in by_case and disposition["selected_seed"] is None,
                    "An excluded rewrite was scored",
                )
                continue
            expected.add(key)
            check(
                disposition["selected_seed"] in range(seed + i * 3, seed + i * 3 + 3)
                and generated_matches(by_case[key], row["views"]["before"]),
                "The generated edit is unbound",
            )
            groups.add(group)
            admitted[family] += 1
    check(
        expected == set(by_case)
        and len(cases) == admission["total_views"]
        and len(groups) == admission["held_distinct"] >= 8
        and admission["generator_counts"] == admitted
        and min(admitted.values()) >= 6,
        "The frozen admission rules differ",
    )
    all_centers = [focus for case in cases for focus in case["review"]]
    current_hashes = {row["input_sha256"] for case in cases for row in case["native_current"]}
    position_hashes = {
        row["input_sha256"] for focus in all_centers for row in focus["responses"] if row is not None
    }
    check(
        len(current_hashes | position_hashes) == receipt["unique_measured_inputs"]
        and len(all_centers) == receipt["eligible_centers"]
        and sum(focus["contrast"] is None for focus in all_centers) == receipt["incomplete_centers"],
        "Saved input coverage differs",
    )
    overflow = receipt["overflow_current_inputs"]
    check(
        len({row["input_sha256"] for row in overflow}) == len(overflow)
        and all(
            row["input_sha256"] in current_hashes - position_hashes
            and type(row["tokens"]) is int
            and row["tokens"] > 512
            for row in overflow
        ),
        "Invalid incumbent-only overflow records",
    )
    actual = outcomes(cases)
    counts = actual["long_document"]["counts"]
    advance = len(counts["additional_useful_works"]) >= 2 and len(counts["additional_ordinary_groups"]) <= 1
    check(
        actual == receipt["outcomes"] and receipt["advance"] is advance, "Saved outcomes differ from replay"
    )
    check(receipt["diagnostics"] == diagnostics(cases), "Saved target response diagnostics differ")
    return actual


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def archive_name(row):
    group, version = row["group"], row["version"]
    check(
        re.fullmatch(r"\d{5}", group) and type(version) is int and version in {2, 3},
        "Invalid publisher work identity",
    )
    name = f"elife-{group}-v{version}.xml"
    check(row["xml"]["archive_path"] == name, "Invalid publisher archive path")
    return name


def acquire_native(source, cache):
    check(source["archive_commit"] == ARCHIVE, "Unexpected publisher archive commit")
    cache.mkdir(parents=True, exist_ok=True)
    opener = build_opener(NoRedirect)
    for row in source["sources"]:
        group, name = row["group"], archive_name(row)
        for path, url, cap in (
            (
                cache / name,
                f"https://raw.githubusercontent.com/elifesciences/elife-article-xml/{ARCHIVE}/articles/{name}",
                450_000,
            ),
            (
                cache / f"{group}-versions.json",
                f"https://api.elifesciences.org/articles/{group}/versions",
                250_000,
            ),
        ):
            if path.exists():
                continue
            request = Request(url, headers={"User-Agent": "ChipmunkVerify-Source-Replay/1.0"})
            with opener.open(request, timeout=20) as response:
                raw = response.read(cap + 1)
            check(0 < len(raw) <= cap, "Publisher source exceeded the byte limit")
            with path.open("xb") as stream:
                stream.write(raw)


def verify_native(source, cache):
    from lxml import etree

    from app.detectors.text.extract import text_hash

    from .writing_f1000_revision_spans import LICENSE, public_view, views
    from .writing_reference_revision_sources import blocks

    verified = 0
    for row in source["sources"]:
        archive = row["xml"]
        path = cache / archive_name(row)
        check(0 < path.stat().st_size <= 450_000, "Archived XML exceeds the source bound")
        raw = path.read_bytes()
        blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        check(
            len(raw) == archive["bytes"]
            and digest(path) == archive["sha256"]
            and blob == archive["git_blob_sha1"],
            "Historical XML bytes differ from the archived source",
        )
        history = cache / f"{row['group']}-versions.json"
        check(0 < history.stat().st_size <= 250_000, "Publisher history exceeds the source bound")
        check(digest(history) == row["history_sha256"], "Publisher version history differs")
        official = [item for item in read(history)["versions"] if item["version"] == row["version"]]
        check(
            len(official) == 1
            and official[0]["versionDate"] == row["published"]
            and official[0]["status"] == "vor"
            and official[0]["copyright"]["license"] == "CC-BY-4.0"
            and official[0]["doi"].lower() == row["doi"].lower(),
            "Version of record provenance differs",
        )
        root = etree.fromstring(raw, parser=etree.XMLParser(resolve_entities=False, no_network=True))
        meta = root.find("front/article-meta")
        names = [
            " ".join(" ".join(node.itertext()).split())
            for node in meta.findall("contrib-group/contrib[@contrib-type='author']/name")
        ]
        terms = {
            node.get("{http://www.w3.org/1999/xlink}href", "").replace("http://", "https://").rstrip("/")
            for node in meta.findall("permissions/license")
        }
        check(
            names == row["authors"]
            and terms == {LICENSE}
            and row["license"] == LICENSE + "/"
            and meta.findtext("article-id[@pub-id-type='doi']", "").lower() == row["doi"].lower(),
            "Publisher XML attribution or license differs",
        )
        body = root.find("body")
        source_blocks = blocks(body, True)
        earlier = row["views"]["before"]
        matches = [
            entry
            for variants in views(body).values()
            for entry in variants
            if all(entry[key] == earlier[key] for key in ("block", "target", "paragraph_range"))
        ]
        check(
            len(matches) == 1 and public_view(matches[0]) == earlier,
            "Original excerpt or input positions differ",
        )
        donor = row["ordinary_donor"]
        donor_block = source_blocks[donor["block"]]
        text = donor_block[slice(*donor["range"])]
        check(
            donor["block"] != earlier["block"]
            and text_hash(donor_block) == donor["block_sha256"]
            and text_hash(text) == donor["text_sha256"],
            "Ordinary source donor differs",
        )
        a, b = earlier["target"]
        view = matches[0]
        control = {
            **view,
            "text": view["text"][:a] + text + view["text"][b:],
            "target_text": text,
            "target": [a, a + len(text)],
        }
        check(public_view(control) == row["views"]["after"], "Ordinary control does not reconstruct")
        verified += 2
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--native-cache", type=Path)
    parser.add_argument("--download-native", action="store_true")
    args = parser.parse_args()
    if args.download_native and args.native_cache is None:
        parser.error("--download-native requires --native-cache")
    receipt = read(args.receipt)
    result = {"outcomes": verify(receipt)}
    if args.native_cache:
        if args.download_native:
            acquire_native(receipt["source"], args.native_cache)
        result["verified_native_controls"] = verify_native(receipt["source"], args.native_cache)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
