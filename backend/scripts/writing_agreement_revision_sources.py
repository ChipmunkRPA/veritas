"""Acquire and qualify public pre-2020 body revisions for writing agreement."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import sleep, time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from lxml import html

from app.detectors.text.extract import _ocr_language
from scripts.prepare_writing_option_order_transfer import benchmark_texts, sentences
from scripts.text_elife_source_intake_v272 import overlap
from scripts.text_short_insert_intake_v203 import fingerprints
from scripts.writing_blind_rewrite_probe import distance, words

SITES = {
    "academia": "academia.stackexchange.com",
    "history": "history.stackexchange.com",
    "workplace": "workplace.stackexchange.com",
    "writers": "writing.stackexchange.com",
}
GUIDE = "18F/content-guide"
GUIDE_PAGES = (
    "content-types/technical-and-interface-writing",
    "our-approach/address-the-user",
    "our-approach/be-concise",
    "our-approach/content-principles",
    "our-approach/giving-and-receiving-feedback",
    "our-approach/make-content-web-friendly",
    "our-approach/plain-language",
    "our-approach/structure-the-content",
)
GUIDE_PIN = "e6ffba0de05d77f0673e9428a82b94a1116b1423"
BEFORE = "2020-01-01T00:00:00Z"
LIMIT = int(datetime.fromisoformat(BEFORE).timestamp())
START = int(datetime(2016, 1, 1, tzinfo=UTC).timestamp())
QUESTION_COUNT = 40
BODY_CANDIDATES_PER_PAGE = 4
COMMUNITY_QUOTA = 3
GUIDE_QUOTA = 4
CC_TERMS = {
    "CC BY-SA 3.0": "https://creativecommons.org/licenses/by-sa/3.0/",
    "CC BY-SA 4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
}


def sha(raw):
    if isinstance(raw, Path):
        return hashlib.sha256(raw.read_bytes()).hexdigest()
    return hashlib.sha256(raw.encode()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def plan():
    return {"schema": "chipmunk.writing-revision-source-plan/1", "sites": SITES, "questions_per_community": QUESTION_COUNT,
            "community_quota": COMMUNITY_QUOTA, "stack_question_start": START, "revision_cutoff": BEFORE,
            "guide_repository": GUIDE, "guide_pin": GUIDE_PIN, "guide_paths": list(GUIDE_PAGES),
            "guide_quota": GUIDE_QUOTA, "guide_commits_per_file": BODY_CANDIDATES_PER_PAGE,
            "rule": "Newest eligible body pair per question or page. First qualifying four-sentence excerpt with "
                    "three exact unchanged flanks, one interior sentence with a lexical revision, 68–134 words overall, "
                    "12–40 words in each interior sentence, exact before and after language and geometry, and no "
                    "normalized 12-token overlap with another selected source group or the retained history. "
                    "Groups ordered by ID in each community and fixed guide path order."}


def retrieve(directory, filename, url):
    path = directory / filename
    if not path.is_file():
        request = Request(url, headers={"User-Agent": "Chipmunk-Verify-public-revision-research/1.0"})
        with urlopen(request, timeout=15) as source:
            if source.headers.get("Content-Encoding") == "gzip":
                with gzip.GzipFile(fileobj=source) as decoded:
                    body = decoded.read(2_500_001)
            else:
                body = source.read(2_500_001)
        require(0 < len(body) <= 2_500_000, "Public response exceeded bounded intake")
        with path.open("xb") as out:
            out.write(body)
    return path


def endpoint(host, route, **params):
    return f"https://{host}/{route}?" + urlencode(params)


def fetch(run):
    frozen = run / "source-plan.json"
    if not frozen.exists():
        save(frozen, plan())
    require(read(frozen) == plan(), "Source plan differs")
    raw = run / "publisher"
    raw.mkdir(exist_ok=True)
    for site in SITES:
        query = endpoint("api.stackexchange.com", "2.3/questions", site=site, pagesize=QUESTION_COUNT, filter="default",
                         sort="votes", order="desc", fromdate=START, todate=LIMIT - 1)
        ids = [entry["question_id"] for entry in read(retrieve(raw, site + "-list.json", query))["items"]]
        require(0 < len(ids) <= QUESTION_COUNT and all(type(value) is int and value > 0 for value in ids), "Unexpected post list")
        for offset in range(0, len(ids), 10):
            joined = ";".join(str(value) for value in ids[offset:offset + 10])
            for page in range(1, 4):
                url = endpoint("api.stackexchange.com", f"2.3/posts/{joined}/revisions", site=site,
                               pagesize=100, page=page, filter="withbody")
                response = retrieve(raw, f"{site}-{offset}-{page}.json", url)
                result = read(response)
                retry_after = result.get("backoff", 0) - (time() - response.stat().st_mtime)
                while retry_after > 0:
                    sleep(min(retry_after, 30))
                    retry_after = result.get("backoff", 0) - (time() - response.stat().st_mtime)
                if not result["has_more"]:
                    break
            else:
                raise ValueError("Public revision roster exceeded the fixed pagination budget")
        print(f"Retrieved {site}: {len(ids)} posts", flush=True)
    terms = f"https://raw.githubusercontent.com/{GUIDE}/{GUIDE_PIN}/LICENSE.md"
    retrieve(raw, "guide-license.md", terms)
    for entry in GUIDE_PAGES:
        path, name = f"_pages/{entry}.md", entry.rsplit("/", 1)[1]
        url = endpoint("api.github.com", f"repos/{GUIDE}/commits", sha=GUIDE_PIN, path=path, until=BEFORE,
                       per_page=BODY_CANDIDATES_PER_PAGE)
        history = read(retrieve(raw, f"guide-{name}.json", url))
        require(isinstance(history, list), "Guide history unavailable")
        for record in history:
            revision = record["sha"]
            require(re.fullmatch("[0-9a-f]{40}", revision), "Unexpected publisher revision")
            retrieve(raw, f"guide-{name}-{revision}.md", f"https://raw.githubusercontent.com/{GUIDE}/{revision}/{path}")
    print("Retrieved guide revisions", flush=True)


def normalize(text):
    return " ".join(text.split())


def prose_html(value):
    tree = html.fragment_fromstring(value, create_parent=True)
    blocks = []
    for item in tree.xpath(".//p[not(ancestor::blockquote) and not(ancestor::li) and not(ancestor::aside)]"):
        if item.xpath(".//code | .//img | .//sup | .//math"):
            continue
        if block := normalize(item.text_content()):
            blocks.append(block)
    return blocks


def prose_markdown(raw):
    raw = re.sub(r"\A---\s*\n.*?\n---\s*\n", "", raw, flags=re.S)
    raw = re.sub(r"```.*?```|<!--.*?-->", "", raw, flags=re.S)
    blocks = []
    for block in re.split(r"\n\s*\n", raw):
        block = re.sub(r"\[([^\]\n]+)\]\([^\n)]+\)", r"\1", block)
        block = block.replace("**", "").replace("__", "")
        if any(re.match(r"^\s*(?:[#>*+`<|]|-\s|\d+\.\s|\{[\{%])", line) for line in block.splitlines()):
            continue
        if block := normalize(block):
            blocks.append(block)
    return blocks


def usable(window):
    value = " ".join(window)
    return (68 <= len(value.split()) <= 134
            and all(12 <= len(words(sentence)) <= 40 for sentence in window[1:3])
            and all(sentence[0].isupper() and sentence[-1] in ".!?" for sentence in window)
            and not any(mark in value for mark in ("http", "www.", "@", "<", ">", "`", "“", "”", '"'))
            and [sentence for _, _, sentence in sentences(value)] == window
            and _ocr_language(value).get("code") == "en")


def candidate_windows(before_blocks, after_blocks):
    earlier = [sentence for block in before_blocks for _, _, sentence in sentences(block)]
    later = [sentence for block in after_blocks for _, _, sentence in sentences(block)]
    for i in range(len(earlier) - 3):
        before = earlier[i:i + 4]
        for j in range(len(later) - 3):
            after = later[j:j + 4]
            for target in (1, 2):
                if not (before[:target] == after[:target] and before[target + 1:] == after[target + 1:]):
                    continue
                if before[target] == after[target] or not usable(before) or not usable(after):
                    continue
                change = distance(before[target], after[target])
                if .05 <= change <= .70:
                    yield {"before": before, "after": after, "target": target,
                           "before_sentence_index": i, "after_sentence_index": j}


def stack_groups(raw):
    for site, domain in SITES.items():
        records = {}
        paths = sorted(raw.glob(site + "-[0-9]*.json"))
        require(paths, f"No public revisions for {site}")
        for path in paths:
            for record in read(path)["items"]:
                if (record.get("body") and record.get("revision_type") == "single_user"
                        and record.get("creation_date", LIMIT) < LIMIT and record.get("revision_number")
                        and not record.get("is_rollback") and record.get("content_license") in CC_TERMS
                        and record.get("user", {}).get("user_id")):
                    records.setdefault(record["post_id"], []).append((record, path))
        for post_id, versions in sorted(records.items()):
            versions.sort(key=lambda item: (item[0]["revision_number"], item[0]["creation_date"], item[0]["revision_guid"]))
            pairs = []
            for (before, before_path), (after, after_path) in reversed(list(zip(versions, versions[1:]))):
                if not (before["revision_number"] < after["revision_number"] and before["creation_date"] < after["creation_date"]):
                    continue
                sources = {}
                for side, row, path in (("before", before, before_path), ("after", after, after_path)):
                    guid = row["revision_guid"]
                    sources[side] = {
                        "url": f"https://{domain}/revisions/{post_id}/{row['revision_number']}",
                        "revision": guid, "record": path.name, "date": datetime.fromtimestamp(row["creation_date"], UTC).isoformat(),
                        "publisher_record_sha256": sha(path), "body_sha256": sha(row["body"]),
                        "attribution": row["user"]["display_name"], "contributor_url": row["user"].get("link"),
                        "license": row["content_license"], "license_url": CC_TERMS[row["content_license"]],
                    }
                ids = {("stack_account", account) if (account := row["user"].get("account_id"))
                       else (site, row["user"]["user_id"]) for row in (before, after)}
                pairs.append((prose_html(before["body"]), prose_html(after["body"]), sources, ids))
            yield site, f"{site}-{post_id}", f"https://{domain}/questions/{post_id}", pairs


def guide_groups(raw):
    license_path = raw / "guide-license.md"
    terms = license_path.read_text()
    require("public domain" in terms.lower() and ("CC0" in terms or "Creative Commons Zero" in terms), "Guide terms changed")
    license_url = f"https://github.com/{GUIDE}/blob/{GUIDE_PIN}/LICENSE.md"
    for entry in GUIDE_PAGES:
        path, name = f"_pages/{entry}.md", entry.rsplit("/", 1)[1]
        journal = raw / f"guide-{name}.json"
        history = read(journal)
        pairs = []
        for later, earlier in zip(history, history[1:]):
            sources, blocks, authors = {}, [], set()
            for side, row in (("before", earlier), ("after", later)):
                revision = row["sha"]
                blob = raw / f"guide-{name}-{revision}.md"
                when = row["commit"]["committer"]["date"]
                require(when < BEFORE, "Guide commit is newer than cutoff")
                blocks.append(prose_markdown(blob.read_text()))
                author = row.get("author") or {}
                authors.add(("guide", author.get("id") or row["commit"]["author"]["name"]))
                sources[side] = {"url": f"https://github.com/{GUIDE}/blob/{revision}/{path}", "revision": revision,
                                 "record": journal.name, "date": when, "publisher_record_sha256": sha(journal),
                                 "body_sha256": sha(blob), "attribution": "18F", "license": "CC0-1.0",
                                 "license_url": license_url, "license_sha256": sha(license_path)}
            pairs.append((*blocks, sources, authors))
        yield "18F", f"guide-{name}", f"https://github.com/{GUIDE}/blob/{GUIDE_PIN}/{path}", pairs


def prepare(run, histories):
    require(read(run / "source-plan.json") == plan(), "Source plan differs")
    raw = run / "publisher"
    possible, groups, identities = {}, {}, {}
    for iterator in (stack_groups(raw), guide_groups(raw)):
        for publisher, group, url, pairs in iterator:
            groups[group] = (publisher, url)
            for before, after, receipts, authors in pairs:
                for value in candidate_windows(before, after):
                    index = group, sum(1 for key in possible if key[0] == group)
                    possible[index] = value, receipts, authors
                    identities[index] = fingerprints(" ".join(value["before"])) | fingerprints(" ".join(value["after"]))
    exposures, known, _ = overlap([(name, sha(path), None, benchmark_texts(path)) for name, path in histories], identities)
    admitted, selected, dispositions, used_authors, used_fingerprints = [], Counter(), Counter(), set(), set()
    for group, (publisher, url) in groups.items():
        quota = GUIDE_QUOTA if publisher == "18F" else COMMUNITY_QUOTA
        if selected[publisher] >= quota:
            dispositions[(publisher, "quota")] += 1
            continue
        for key, (value, receipts, authors) in possible.items():
            if key[0] == group and key not in known and not used_authors & authors and not used_fingerprints & identities[key]:
                admitted.append({"group": group, "publisher": publisher, "url": url, "revisions": receipts, **value})
                selected[publisher] += 1
                used_authors.update(authors)
                used_fingerprints.update(identities[key])
                break
        else:
            dispositions[(publisher, "no_disjoint_qualifying_revision")] += 1
            continue
        dispositions[(publisher, "admitted")] += 1
    require(len(admitted) >= 6 and len(selected) >= 3, "Too few cross-community source groups")
    save(run / "originals.json", admitted)
    save(run / "source-receipt.json", {
        "source_plan_sha256": sha(run / "source-plan.json"), "source_code_sha256": sha(Path(__file__)),
        "originals_sha256": sha(run / "originals.json"), "exposure_packets": exposures,
        "groups": [{key: value for key, value in row.items() if key not in {"before", "after"}}
                   | {part + "_sentences_sha256": [sha(s) for s in row[part]] for part in ("before", "after")}
                   for row in admitted],
        "dispositions": {publisher: {key: value for (scope, key), value in sorted(dispositions.items()) if scope == publisher}
                         for publisher in ("18F", *SITES)},
        "selected": dict(selected),
    })
    print(dict(selected), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fetch", "prepare"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--history", action="append", default=[])
    args = parser.parse_args()
    os.umask(0o077)
    if args.stage == "fetch":
        fetch(args.run)
        return
    entries = [item.split("=", 1) for item in args.history]
    require(entries and all(len(item) == 2 for item in entries), "Use one or more --history NAME=JSONL")
    prepare(args.run, [(name, Path(path)) for name, path in entries])


if __name__ == "__main__":
    main()
