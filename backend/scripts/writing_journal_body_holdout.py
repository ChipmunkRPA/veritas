"""Freeze and independently replay exact Wellcome and Gates article body revisions."""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

from lxml import etree

from app.detectors.text.extract import text_hash
from scripts.prepare_writing_option_order_transfer import benchmark_texts
from scripts.text_elife_source_intake_v272 import overlap
from scripts.text_short_insert_intake_v203 import fingerprints
from scripts.writing_agreement_revision_sources import read, require, retrieve, save
from scripts.writing_focus_v3 import digest
from scripts.writing_reference_revision_sources import LINK, excerpt, license_match

SCHEMA = "chipmunk.writing-journal-body-holdout/1"
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend/data/independent-evaluation"
JOURNALS = {
    "wellcome": ("2398-502X", "Wellcome Open Research", "wellcomeopenresearch.org", "wellcomeopenres"),
    "gates": ("2572-4754", "Gates Open Research", "gatesopenresearch.org", "gatesopenres"),
}
QUERY = {"filter": "from-pub-date:2018-01-01,until-pub-date:2019-12-31", "rows": 650,
         "sort": "published", "order": "asc", "select": "DOI,link,license,published,container-title"}
HISTORY = {
    "previous-agreement": DATA / "writing-agreement-ordinary-edit-2026-11-11/previous-agreement.jsonl",
    "llmtrace": DATA.parent / "text-upgrade/sources/llmtrace-test.jsonl",
    "elife-v301": DATA / "text-elife-editorial-controls-v301/private-controls.jsonl",
    "elife-v286": DATA / "text-elife-transfer-intake-v286/private-originals.jsonl",
    "elife-v272": DATA / "text-elife-source-intake-v272/private-originals.jsonl",
    "opai-v257": DATA / "text-opai-bench-intake-v257/private-cases.jsonl",
    "f1000-previous": DATA / "writing-reference-revision-transfer-2026-11-11/expanded/originals.json",
    "publisher-abstract-pilot": DATA / "writing-publisher-revision-replication-2026-11-11/holdout/originals.json",
}
PLAN = {
    "schema": SCHEMA, "journals": JOURNALS, "query": QUERY, "pool_per_journal": 40, "maximum_per_journal": 12,
    "rights": "Require CC BY 4.0 in Crossref and both version-specific publisher XML witnesses.",
    "source": "First 40 version-2 DOI hashes per journal in a complete dated Crossref catalog. Verify both online dates, "
              "identical original authors and exact article DOI. XML cap is 2.5 MB per witness. No source substitution.",
    "geometry": "Use the prior five-sentence authored body comparator after omitting marked bibliography references. "
                "Require two unchanged preceding and following sentences and a material 12–40-word interior revision. "
                "Freeze the source order and reject 12-token overlap with retained history and admitted works. "
                "Exclude authors from earlier source pairs and across admitted works.",
}


def roster(journal, catalog):
    _, title, host, stem = JOURNALS[journal]
    document = read(catalog)["message"]
    rows = document["items"]
    require(len(rows) == document["total-results"] <= QUERY["rows"], "Incomplete dated journal catalog")
    by_doi = {}
    for row in rows:
        doi = row["DOI"]
        if (not re.fullmatch(rf"10\.12688/{stem}\.[0-9]+\.2", doi)
                or row["container-title"] != [title]
                or not any(license_match(lic["URL"]) for lic in row.get("license", []))):
            continue
        xml = [link["URL"] for link in row.get("link", [])
               if link["content-type"] == "application/xml"
               and re.fullmatch(rf"https://{re.escape(host)}/articles/[0-9]+-[0-9]+/v2/xml", link["URL"])]
        published = row["published"]["date-parts"][0]
        if len(xml) == 1 and len(published) == 3:
            require(doi not in by_doi, "Duplicate version DOI")
            by_doi[doi] = {"before": xml[0].replace("/v2/", "/v1/"), "after": xml[0], "published": published}
    require(len(by_doi) >= PLAN["pool_per_journal"], "Insufficient complete publisher version metadata")
    return {doi: by_doi[doi] for doi in sorted(by_doi, key=text_hash)[:PLAN["pool_per_journal"]]}


def history_pins():
    return {name: digest(path) for name, path in HISTORY.items()}


def freeze(run):
    directory = run / "publisher"
    directory.mkdir(parents=True, exist_ok=True)
    method = run / "source-plan.json"
    if not method.exists():
        save(method, PLAN)
    require(read(method) == {**PLAN, "journals": {k: list(v) for k, v in JOURNALS.items()}}, "Source plan differs")
    pools, catalogs = {}, {}
    for journal, (issn, *_) in JOURNALS.items():
        url = f"https://api.crossref.org/journals/{issn}/works?{urlencode(QUERY)}"
        path = retrieve(directory, f"{journal}-catalog.json", url)
        pools[journal], catalogs[journal] = roster(journal, path), digest(path)
    result = {"source_plan_sha256": digest(method), "code_sha256": digest(Path(__file__)),
              "history_sha256": history_pins(), "catalog_sha256": catalogs, "pool": pools}
    path = run / "catalog-freeze.json"
    initial = run / "catalog-freeze-initial.json"
    if initial.exists():
        previous = read(initial)
        require({k: v for k, v in previous.items() if k != "code_sha256"}
                == {k: v for k, v in result.items() if k != "code_sha256"},
                "A correction to the historical input reader changed the initial roster")
        result["initial_catalog_freeze_sha256"] = digest(initial)
    if not path.exists():
        save(path, result)
    require(read(path) == result, "The prospectively frozen catalog or code differs")
    return result


def local_names(journal, doi):
    stem = f"{journal}-{doi.split('.')[-2]}"
    return {side: f"{stem}-{side}.xml" for side in ("before", "after")}


def fetch(run, lock):
    directory = run / "publisher"

    def journal_xml(journal):
        for doi, row in lock["pool"][journal].items():
            for side, name in local_names(journal, doi).items():
                if (directory / name).exists() or (directory / (name + ".omitted.json")).exists():
                    continue
                for attempt in range(3):
                    try:
                        retrieve(directory, name, row[side])
                        break
                    except HTTPError as error:
                        if error.code in {403, 404, 410}:
                            save(directory / (name + ".omitted.json"), {"reason": f"HTTP {error.code}"})
                            break
                        if attempt == 2 or error.code not in {429, 500, 502, 503, 504}:
                            raise
                        sleep(3 * (attempt + 1))
                    except (URLError, TimeoutError):
                        if attempt == 2:
                            raise
                        sleep(3 * (attempt + 1))
                    except ValueError:
                        save(directory / (name + ".omitted.json"), {"reason": "Bounded XML intake"})
                        break
                sleep(.25)
            print(f"Completed {journal} article {doi.split('.')[-2]}", flush=True)

    with ThreadPoolExecutor(max_workers=len(JOURNALS)) as pool:
        list(pool.map(journal_xml, JOURNALS))


def article(path, doi, title):
    root = etree.fromstring(path.read_bytes(), parser=etree.XMLParser(resolve_entities=False, no_network=True))
    front = root.find("front/article-meta")
    require(root.tag == "article" and root.get("{http://www.w3.org/XML/1998/namespace}lang") == "en"
            and root.findtext("front/journal-meta/journal-title-group/journal-title") == title
            and front is not None and front.findtext("article-id[@pub-id-type='doi']") == doi,
            "Publisher identity, version or language differs")
    rights = front.findall("permissions/license")
    require(len(rights) == 1 and license_match(rights[0].get(LINK)), "Exact reuse term is missing")
    online = front.find("pub-date[@pub-type='epub']")
    if online is None:
        online = front.find("pub-date[@date-type='pub']")
    require(online is not None, "Online version date is missing")
    published = [int(online.findtext(key)) for key in ("year", "month", "day")]
    names = [" ".join(" ".join(name.itertext()).split())
             for name in front.findall("contrib-group/contrib[@contrib-type='author']/name")]
    require(names and all(names), "Authors are missing")
    return root.find("body"), names, published


def old_author_names():
    prior = [row for name in ("f1000-previous", "publisher-abstract-pilot") for row in read(HISTORY[name])]
    names = (person for row in prior for person in row.get("attribution", row.get("authors", [])))
    return {" ".join(person if isinstance(person, list) else person.split()).casefold() for person in names}


def historical_texts(path):
    if path.suffix == ".json":
        for row in read(path):
            for side in ("before", "after"):
                yield row[side]["text"]
    else:
        yield from benchmark_texts(path)


def qualify(run, lock):
    directory = run / "publisher"
    ledger, options, signatures = {}, {}, {}
    counts = {journal: Counter() for journal in JOURNALS}
    for journal, (_, title, *_) in JOURNALS.items():
        for doi in sorted(lock["pool"][journal], key=text_hash):
            row = lock["pool"][journal][doi]
            group = f"{journal}-{doi.split('.')[-2]}"
            paths = {side: directory / name for side, name in local_names(journal, doi).items()}
            ledger[group] = {side: digest(path if path.exists() else path.with_suffix(".xml.omitted.json"))
                             for side, path in paths.items()}
            if not all(path.exists() for path in paths.values()):
                counts[journal]["omitted_xml"] += 1
                continue
            try:
                old = article(paths["before"], doi[:-1] + "1", title)
                new = article(paths["after"], doi, title)
            except (TypeError, ValueError, etree.XMLSyntaxError):
                counts[journal]["unsupported_metadata"] += 1
                continue
            if (old[1] != new[1] or not date(*old[2]) < date(*new[2]) < date(2020, 1, 1)
                    or new[2] != row["published"]):
                counts[journal]["author_or_date_mismatch"] += 1
                continue
            matches = list(excerpt(old[0], new[0], expanded=True))
            if not matches:
                counts[journal]["no_exact_body_edit"] += 1
                continue
            options[group] = [{"publisher": journal, "group": group, "before": before, "after": after,
                               "before_doi": doi[:-1] + "1", "after_doi": doi, "attribution": old[1],
                               "before_date": old[2], "after_date": new[2], "license": "CC BY 4.0"}
                              for before, after in matches]
            for index, (before, after) in enumerate(matches):
                signatures[group, index] = fingerprints(before["text"]) | fingerprints(after["text"])
    packets = [(name, lock["history_sha256"][name], None, historical_texts(path)) for name, path in HISTORY.items()]
    history, outside, _ = overlap(packets, signatures)
    retained, authors, selected = set(), old_author_names(), []
    for group, candidates in options.items():
        journal = candidates[0]["publisher"]
        if counts[journal]["selected"] >= PLAN["maximum_per_journal"]:
            counts[journal]["after_quota"] += 1
            continue
        for index, candidate in enumerate(candidates):
            contributors = {name.casefold() for name in candidate["attribution"]}
            signature = signatures[group, index]
            if (group, index) in outside or retained & signature or authors & contributors:
                continue
            selected.append(candidate)
            retained.update(signature)
            authors.update(contributors)
            counts[journal]["selected"] += 1
            break
        else:
            counts[journal]["source_overlap"] += 1
    require(all(sum(value.values()) == PLAN["pool_per_journal"] for value in counts.values()), "Incomplete roster")
    return selected, {"schema": SCHEMA, "source_code_sha256": digest(Path(__file__)),
                      "catalog_freeze_sha256": digest(run / "catalog-freeze.json"), "history": history,
                      "source_commitments": ledger, "dispositions": counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fetch", "prepare", "verify"))
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    lock = freeze(args.run)
    if args.phase == "fetch":
        fetch(args.run, lock)
        return
    selected, receipt = qualify(args.run, lock)
    originals, path = args.run / "originals.json", args.run / "source-receipt.json"
    if args.phase == "prepare":
        save(originals, selected)
        save(path, {**receipt, "originals_sha256": digest(originals)})
    else:
        require(read(originals) == selected and read(path) == {**receipt, "originals_sha256": digest(originals)},
                "Source replay differs")
    print({journal: dict(value) for journal, value in receipt["dispositions"].items()}, flush=True)


if __name__ == "__main__":
    main()
