"""Admit fixed pre-ChatGPT eLife articles for a fresh passage transfer cohort."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

from scripts import text_elife_source_intake_v272 as july
from scripts.video_appearance_corrected_v112 import scoped

NAME = "text-elife-transfer-intake-v286"
ROOT = july.prior.ROOT
DATA = july.DATA.with_name(NAME)
REVISION = "bdfb477443c50c8c7ca015ad19fb7b7874c672f5"
MONTHS = ("may", "june", "august", "september", "october")
ARCHIVES = tuple(f"https://elifesciences.org/archive/2022/{month}" for month in MONTHS)
MIN_GROUPS = 15
JULY = july.prior.PANELS / "text-elife-source-intake-complete-v272.json"
JULY_SHA = "e494c33e29a893abc16a1845f028bedcf9a65b5a8ce9f963323c267c006627a2"
REWRITES = july.prior.PANELS / "text-elife-rewrites-complete-v273.json"
REWRITES_SHA = "5437db6e274c3fc87eab1aa6424e7ed17f4d2a84a243dde84045fc16e40e7b7f"
WORKS = (
    ("79813", "2022-05-30", "Avraham N Bayer, Fred L Homa"),
    ("79446", "2022-05-24", "Amelia J Ralowicz, Michael B Hoppa"),
    ("79593", "2022-05-20", "Lauren C Radlinski, Andreas J Bäumler"),
    ("79310", "2022-05-19", "Grzegorz Kudla, Marcin Plech"),
    ("79125", "2022-05-19", "Magdalena Shumanska, Ivan Bogeski"),
    ("78869", "2022-05-11", "Tarana Siddika, Ilka U Heinemann, Patrick O’Donoghue"),
    ("80416", "2022-06-30", "Danelle Devenport"),
    ("80373", "2022-06-30", "Bilal Cakir, In-Hyun Park"),
    ("80552", "2022-06-28", "Paola Kučan Brlić, Ilija Brizić"),
    ("80232", "2022-06-20", "Alexis Cooper, Benedikt Berninger"),
    ("80030", "2022-06-15", "Chuqi Liu, Gui Xue"),
    ("79651", "2022-06-09", "Qianjiang Hu, Melanie Königshoff"),
    ("79839", "2022-06-01", "Kathleen S Rockland"),
    ("82028", "2022-08-30", "Mingyu Shin, Jiwon Shim"),
    ("82070", "2022-08-23", "Prathyusha Konda, Srinivas R Viswanathan"),
    ("81890", "2022-08-22", "Ricardo A Feldman"),
    ("81963", "2022-08-19", "Anne Schmidt"),
    ("81670", "2022-08-11", "Olga Ponomarova"),
    ("81520", "2022-08-05", "Tathagata Biswas, Jaya Krishnan, Nicolas Rohner"),
    ("81077", "2022-08-03", "Ken Duffy"),
    ("82610", "2022-09-30", "Erin Brintnell, Art Poon"),
    ("82258", "2022-09-27", "Faye Smith, Timothy D Griffiths"),
    ("82873", "2022-09-23", "Pablo Barbeito, Francesc R Garcia-Gonzalo"),
    ("82635", "2022-09-20", "Samuel Brudner, Thierry Emonet"),
    ("82453", "2022-09-14", "Leigh C Walker, Paulo Pinares-Garcia, Andrew J Lawrence"),
    ("82391", "2022-09-13", "Maggie R Wagner"),
    ("82747", "2022-09-07", "Łukasz Bola"),
    ("82221", "2022-09-02", "Elisabet Romero"),
    ("83488", "2022-10-25", "Eugene V Koonin, Mart Krupovic"),
    ("83482", "2022-10-19", "Ivan Maslov, Jelle Hendrix"),
    ("83076", "2022-10-14", "Abdou Akkouche, Emilie Brasset"),
    ("83308", "2022-10-13", "Hongyu Bao, Hongda Huang"),
    ("83296", "2022-10-11", "Sayantan Datta, William C Ratcliff"),
    ("83254", "2022-10-07", "Ralf HJM Kurvers, Lysanne Snijders"),
    ("83121", "2022-10-07", "Magnus Hallas-Møller, Katja S Johansen"),
)
require, verify, sha = july.require, july.verify, july.sha
checked = july.prior.checked_bytes
normalized = july.normalized


def filename(source_id):
    return f"elife-{source_id}-v1.xml"


def name_key(value):
    return "".join(char for char in unicodedata.normalize("NFKD", value.casefold()) if char.isalnum())


def related_dois(article):
    values = set()
    for element in article.findall("front/article-meta/related-article"):
        if element.get("related-article-type") == "commentary-article":
            link = element.get(july.XLINK, "").lower().removeprefix("https://doi.org/")
            if re.fullmatch(r"10\.\d{4,9}/\S+", link):
                values.add(link)
    return sorted(values)


def metadata(article, work):
    source_id, date, display_names = work
    doi = f"10.7554/eLife.{source_id}"
    meta = article.find("front/article-meta")
    verify(
        article.tag == "article" and article.get("article-type") == "article-commentary" and meta is not None,
        "wrong_article_type",
    )
    kind = meta.find("article-categories/subj-group[@subj-group-type='display-channel']/subject")
    verify(
        normalized(kind) == "Insight"
        and normalized(meta.find("article-id[@pub-id-type='doi']")).lower() == doi.lower()
        and normalized(meta.find("volume")) == "11"
        and normalized(meta.find("elocation-id")) == f"e{source_id}",
        "wrong_work",
    )
    dates = set()
    for entry in meta.findall("pub-date"):
        parts = [entry.findtext(key, "") for key in ("year", "month", "day")]
        if all(value.isdecimal() for value in parts):
            dates.add(tuple(map(int, parts)))
    verify(dates == {tuple(map(int, date.split("-")))}, "publication_date_mismatch")

    authors = [
        f"{normalized(name.find('given-names'))} {normalized(name.find('surname'))}".strip()
        for name in meta.findall("contrib-group/contrib[@contrib-type='author']/name")
    ]
    verify(list(map(name_key, authors)) == list(map(name_key, display_names.split(", "))), "author_mismatch")
    title = normalized(meta.find("title-group/article-title"))
    verify(bool(title), "missing_title")

    licenses = meta.findall("permissions/license")
    verify(len(licenses) == 1, "missing_or_ambiguous_license")
    license_element = licenses[0]
    notice = " ".join(normalized(line) for line in license_element.findall("license-p"))
    verify(
        july.canonical_license(license_element.get(july.XLINK)) == july.LICENSE and notice,
        "license_not_cc_by_4",
    )
    verify(
        all(
            july.canonical_license(link.get(july.XLINK)) == july.LICENSE
            for link in license_element.iter("ext-link")
        ),
        "conflicting_license_link",
    )
    return {
        "doi": doi,
        "title": title,
        "authors": authors,
        "publication_date": date,
        "license_url": july.LICENSE,
        "license_notice_sha256": sha(notice.encode()),
        "related_research_dois": related_dois(article),
    }


def protocol():
    july_receipt = json.loads(checked(JULY, JULY_SHA))
    ids = {row[0] for row in WORKS}
    require(
        len(ids) == len(WORKS) == 35
        and not ids.intersection(row["source_id"] for row in july_receipt["works"])
        and july_receipt["publisher_revision"] == REVISION,
        "Expected independent fixed article IDs",
    )
    require(
        sha(Path(july.__file__).read_bytes()) == july_receipt["code_sha256"],
        "Exact previously admitted body projection and passage rules required",
    )
    files = (Path(__file__), Path(july.__file__), ROOT / f"docs/{NAME}-method.md")
    return {
        "schema": "chipmunk.text-elife-transfer-roster/286",
        "publisher_revision": REVISION,
        "publisher_archives": ARCHIVES,
        "prior_receipts_sha256": {"july_v272": JULY_SHA, "rewrites_v273": REWRITES_SHA},
        "minimum_target_source_groups": MIN_GROUPS,
        "maximum_bytes_per_xml": july.MAX_XML,
        "code_sha256": {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in files},
        "works": [
            {"source_id": source_id, "date": date, "authors": names.split(", ")}
            for source_id, date, names in WORKS
        ],
    }


def earlier_july(directory):
    receipt = json.loads(checked(JULY, JULY_SHA))
    authors, related, texts = set(), set(), []
    for work in receipt["works"]:
        raw = july.prior.text.read_file(directory / filename(work["source_id"]), july.MAX_XML)
        require(sha(raw) == work["xml_sha256"], "Exact earlier July publisher bytes required")
        article = ET.fromstring(raw)
        projection = list(july.body_paragraphs(article.find("body")))
        require(
            sha(july.json_bytes(projection)) == work["body_projection_sha256"], "Earlier July body changed"
        )
        texts.append("\n\n".join(part for part in projection if part is not None))
        authors.update(map(name_key, work["authors"]))
        related.update(related_dois(article))
    return authors, related, texts


def earlier_rewrites():
    receipt = json.loads(checked(REWRITES, REWRITES_SHA))
    path = DATA.with_name("text-elife-rewrites-v273") / "private-cases.jsonl"
    body = checked(path, receipt["private_cases_sha256"])
    rows = list(july.prior.json_lines(body))
    texts = list(
        july.prior.validated_texts(
            row for row in rows if row["arm"] == "paragraph_ai" and row["status"] == "constructed"
        )
    )
    require(len(rows) == 15 and len(texts) == 6, "Expected exact retained v273 exposure inputs")
    return receipt["private_cases_sha256"], texts


def selected_groups(inspections, historical, duplicates, old_authors, old_related):
    identifiers = defaultdict(set)
    for record, selection, _ in inspections:
        if selection is not None:
            for name in record["authors"]:
                identifiers["author", name_key(name)].add(record["source_id"])
            for doi in record["related_research_dois"]:
                identifiers["doi", doi].add(record["source_id"])
    repeated = {work for sources in identifiers.values() if len(sources) > 1 for work in sources}
    selected = []
    for record, selection, _ in inspections:
        if selection is None:
            continue
        work = record["source_id"]
        if work in historical or work in duplicates:
            record["status"] = (
                "historical_fragment_overlap" if work in historical else "within_roster_fragment_overlap"
            )
        elif old_authors.intersection(map(name_key, record["authors"])):
            record["status"] = "prior_july_author_overlap"
        elif old_related.intersection(record["related_research_dois"]):
            record["status"] = "prior_july_research_doi_overlap"
        elif work in repeated:
            record["status"] = "within_roster_author_or_research_doi_overlap"
        else:
            record.update({key: value for key, value in selection.items() if key != "text"})
            record["status"] = "admitted"
            selected.append(selection)
    return selected


def run(directory, previous, cache, gohumanize, lock):
    require(lock == json.loads(july.json_bytes(protocol())), "Expected exact pre-acquisition roster and code")
    old_authors, old_related, old_texts = earlier_july(previous)
    rewrite_sha, rewrite_texts = earlier_rewrites()
    inspect = scoped(july.inspect, {"metadata": metadata})
    inspected = [inspect(directory, work, REVISION) for work in WORKS]
    signatures = {
        row["source_id"]: fragments for row, selection, fragments in inspected if selection is not None
    }

    def packets():
        yield from july.exposure_packets(cache, gohumanize)
        yield "july_v272_all_publisher_bodies", JULY_SHA, len(old_texts), old_texts
        yield "v273_constructed_rewrites", rewrite_sha, len(rewrite_texts), rewrite_texts

    evidence, historical, duplicates = july.overlap(packets(), signatures)
    selected = selected_groups(inspected, historical, duplicates, old_authors, old_related)
    journal = b"".join(map(july.json_bytes, selected))
    records = [row for row, _, _ in inspected]
    return journal, {
        "schema": "chipmunk.text-elife-transfer-intake/286",
        "status": "complete_intake" if len(selected) >= MIN_GROUPS else "complete_intake_below_target",
        "roster_sha256": sha(july.json_bytes(lock)),
        "publisher_revision": REVISION,
        "publisher_archives": ARCHIVES,
        "publisher_terms_url": july.TERMS,
        "planned_works": len(WORKS),
        "admitted_source_groups": len(selected),
        "admitted_groups_by_month": dict(
            Counter(row["publication_date"][:7] for row in records if row["status"] == "admitted")
        ),
        "dispositions": dict(Counter(row["status"] for row in records)),
        "works": records,
        "exposure_packets": evidence,
        "private_originals_sha256": sha(journal),
        "new_neural_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    commands.add_parser("roster")
    commands.add_parser("urls").add_argument("roster", type=Path)
    intake = commands.add_parser("intake")
    for name in ("roster", "xml_dir", "july_xml_dir", "gohumanize"):
        intake.add_argument(name, type=Path)
    intake.add_argument("--research-cache", type=Path, default=Path("/private/tmp"))
    intake.add_argument("--retain", action="store_true")
    args = parser.parse_args()
    if args.mode == "roster":
        print(json.dumps(protocol(), indent=2, ensure_ascii=False))
    elif args.mode == "urls":
        require(
            json.loads(args.roster.read_bytes()) == json.loads(july.json_bytes(protocol())),
            "Use the exact pre-acquisition roster",
        )
        for source_id, _, _ in WORKS:
            name = filename(source_id)
            print(
                f"{name}\thttps://raw.githubusercontent.com/elifesciences/elife-article-xml/{REVISION}/articles/{name}"
            )
    else:
        for folder in (args.xml_dir, args.july_xml_dir):
            require(folder.is_dir() and not folder.is_symlink(), "Expected direct owner-only XML directory")
            july.prior.archive.safe_cache(folder)
        journal, receipt = run(
            args.xml_dir,
            args.july_xml_dir,
            args.research_cache,
            args.gohumanize,
            json.loads(args.roster.read_bytes()),
        )
        if args.retain:
            july.prior.archive.safe_cache(DATA)
            july.prior.archive.retain(DATA / "private-originals.jsonl", journal)
            july.prior.archive.retain(DATA / "aggregate.json", july.json_bytes(receipt))
        print(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
