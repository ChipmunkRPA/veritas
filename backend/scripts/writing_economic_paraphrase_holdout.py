# SPDX-License-Identifier: Apache-2.0
"""Reserve and test short local rewrites of previously unused official economic prose."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from app.bench.writing_entropy_prominence import compact_range
from scripts import text_elife_rewrites_v273 as editors
from scripts import writing_entropy_sources as history
from scripts import writing_multi_passage_panel as native
from scripts import writing_token_supervision_eval as comparison
from scripts import writing_token_supervision_transfer as prior

SCHEMA = "chipmunk.writing-economic-paraphrase-holdout/1"
COUNT = 24
MIN_WORKS = 20
POINT = prior.POINT
INDEXES = (
    *(f"https://www.bea.gov/news/archive?created_1={year}&field_related_product_target_id=All&title=&page={page}"
      for year in (6, 7, 8) for page in (0, 1)),
    *(f"https://www.bea.gov/news/blog/archive?created_1={year}&title=&page={page}"
      for year in (1, 2) for page in (0, 1, 2)),
)
CENSUS = ("https://www.census.gov/newsroom/press-releases/2019/acs-1year.html",
          "https://www.census.gov/newsroom/press-releases/2020/acs-5-year.html",
          "https://www.census.gov/newsroom/press-releases/2020/income-poverty.html",
          "https://www.census.gov/newsroom/press-releases/2019/income-poverty.html")
RIGHTS = {"bea": "https://www.bea.gov/help/faq/145", "census": "https://www2.census.gov/foia/ds_policies/ds027.pdf"}
SOURCE = re.compile(r"/news/(?:20(?:17|18|19)/|blog/20(?:17|18|19)-\d{2}-\d{2}/)[a-z0-9-]+$")
LIMIT = 1024 * 1024


def order(value):
    return native.digest(f"{SCHEMA}/{value}".encode())


def fragments(value):
    words = re.findall(r"\w+", value.casefold())
    return {native.digest(" ".join(words[i:i + 12]).encode()) for i in range(len(words) - 11)}


def fetch(address):
    request = Request(address, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html"})
    with urlopen(request, timeout=20) as response:
        if urlparse(response.geturl()).hostname != urlparse(address).hostname:
            raise ValueError("Publisher returned a different host")
        body = response.read(LIMIT + 1)
        if not body or len(body) > LIMIT:
            raise ValueError("Publisher response exceeds the intake size bound")
        return body


def source_family(address):
    if "census.gov" in address:
        return "census_survey" if "/acs-" in address else "census_income"
    slug = urlparse(address).path.rsplit("/", 1)[-1]
    if any(term in slug for term in ("personal-income", "consumer-spending", "consumption-expenditures")):
        return "consumer_income"
    if any(term in slug for term in ("international", "trade-gap", "current-account")):
        return "international"
    if any(term in slug for term in ("industry", "industries", "multinational", "corporate-profits")):
        return "industry"
    if any(term in slug for term in ("state", "county", "islands", "guam", "samoa", "local-area", "puerto-rico")):
        return "regional"
    return "national_accounts"


def bodies(address, raw):
    from lxml import html

    tree = html.fromstring(raw)
    publisher = "census" if "census.gov" in address else "bea"
    if publisher == "bea":
        paragraphs = tree.xpath('//div[contains(@class,"field--name-body")]/p')
    else:
        paragraphs = tree.xpath('//div[contains(@class,"uscb-text-media-text")]/p')
    title = " ".join(tree.xpath("string(//h1)").split())
    if not title or re.search(r"^access denied|page not found", title, re.I):
        return None
    parts = []
    for item in paragraphs:
        text = " ".join(item.text_content().split())
        if (len(text.split()) < 9 or re.search(r'["“”]|@|^\(?\d+[.)]|^(NOTE|CONTACT):', text)
                or any(urlparse(link).netloc and not urlparse(link).hostname.endswith((".bea.gov", ".census.gov"))
                       for link in item.xpath('.//a/@href') if urlparse(link).hostname)):
            continue
        if re.search(r"\btechnical notes\b|^definitions\b|^for more information", text, re.I):
            continue
        parts.append(text)
    return title, publisher, "\n\n".join(parts)


def find_texts(value):
    if isinstance(value, dict):
        for key, entry in value.items():
            if isinstance(entry, str) and key in {"text", "before", "after", "source_text", "response", "generation"}:
                yield entry
            elif isinstance(entry, (dict, list)):
                yield from find_texts(entry)
    elif isinstance(value, list):
        for entry in value:
            yield from find_texts(entry)


def reserve(output, freeze, prior_inputs):
    from lxml import html

    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    output.mkdir(mode=0o700)
    cache = output / "publisher"
    cache.mkdir(mode=0o700)
    _, prior_digests, receipts = history.exposures()
    seen = set()
    for path in prior_inputs:
        for entry in find_texts(native.read(path)):
            seen.update(fragments(entry))
    addresses = set(CENSUS)
    indexes = {}
    for address in INDEXES:
        raw = fetch(address)
        indexes[address] = native.digest(raw)
        for path in html.fromstring(raw).xpath("//a/@href"):
            if SOURCE.fullmatch(path):
                addresses.add(urljoin(address, path))

    def pull(address):
        try:
            return address, fetch(address)
        except (OSError, ValueError):
            return address, None

    counts, groups, rejected = Counter(), Counter(), Counter()
    inspected, selected = {}, []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for address, raw in pool.map(pull, sorted(addresses, key=order)):
            if raw is None:
                rejected["download"] += 1
                continue
            sha = native.digest(raw)
            inspected[address] = sha
            (cache / f"{sha}.html").write_bytes(raw)
            if sha in prior_digests or (entry := bodies(address, raw)) is None:
                rejected["source_or_markup"] += 1
                continue
            title, publisher, content = entry
            family = source_family(address)
            doc = from_string(content)
            sentences = segment(doc)
            tag = "bea_blog" if "/news/blog/" in address else f"{publisher}_release"
            # Keep the choice score-blind, and distribute original URLs across programme families.
            if groups[family] >= 6 or counts[tag] >= 12 or len(selected) >= COUNT:
                rejected["quota"] += 1
                continue
            for start in sorted(range(len(sentences) - prior.WIDTH + 1), key=lambda i: order(f"{address}/{i}")):
                opening, closing = sentences[start], sentences[start + prior.WIDTH - 1]
                excerpt = doc.text[opening.start:closing.end]
                clean = native.native(excerpt, ())
                signatures = fragments(excerpt)
                if (clean is None or clean["text"] != excerpt or len(clean["sentences"]) != prior.WIDTH
                        or seen & signatures or clean["native_text_sha256"] in prior_digests):
                    continue
                rows = segment(from_string(excerpt))
                choices = [(i, a, b) for i, (a, b) in enumerate(zip(rows, rows[1:]))
                           if 2 <= i < prior.WIDTH - 3 and 11 <= a.n_words <= 42 and 11 <= b.n_words <= 42
                           and a.paragraph == b.paragraph]
                if not choices:
                    continue
                i, a, b = min(choices, key=lambda row: order(f"{address}/target/{row[0]}"))
                expected = a.n_words + b.n_words
                donors = [(c, d) for c, d in zip(sentences, sentences[1:])
                          if (d.end <= opening.start or c.start >= closing.end) and c.paragraph == d.paragraph
                          and .7 <= (c.n_words + d.n_words) / expected <= 1.4]
                editor = min(editors.GENERATORS, key=lambda item: sum(row["editor"] == item for row in selected))
                donor = None
                for c, d in sorted(donors, key=lambda pair: order(f"{address}/ordinary/{pair[0].start}")):
                    replacement = doc.text[c.start:d.end]
                    if len(segment(from_string(replacement))) == 2 and not seen & fragments(replacement):
                        target = excerpt[:a.start] + replacement + excerpt[b.end:]
                        ordinary = native.native(target, ())
                        if ordinary is not None and len(ordinary["sentences"]) == prior.WIDTH:
                            donor = {"source_codepoints": [c.start, d.end],
                                     "replacement_sha256": native.digest(replacement.encode()), "text": ordinary["text"]}
                            break
                selected.append({"document": address, "group": native.digest(address.encode()), "publisher": publisher,
                                 "title": title, "family": family, "series": tag, "source_sha256": sha,
                                 "normalized_body_sha256": native.digest(doc.text.encode()),
                                 "source_codepoints": [opening.start, closing.end], "text": excerpt,
                                 "editor": editor, "edit_target_codepoints": [a.start, b.end],
                                 "sentence_index": i, "ordinary_edit": donor})
                seen.update(signatures)
                if donor:
                    seen.update(fragments(donor["text"]))
                groups[family] += 1
                counts[tag] += 1
                break
            else:
                rejected["no_fresh_continuous_excerpt"] += 1
    native.save(output / "reservation.json.gz", {
        "schema": SCHEMA, "sources": selected, "index_sha256": indexes, "inspected": inspected,
        "prior_receipts": receipts, "prior_input_sha256": {p.name: comparison.bind(p) for p in prior_inputs},
        "point": POINT, "freeze_sha256": comparison.bind(freeze), "rights": RIGHTS,
        "quotas": {"total": COUNT, "per_release_family": 6, "per_series": 12}, "unselected": dict(rejected),
    })
    for editor in editors.GENERATORS:
        native.save(output / f"{editor}.json.gz", {"sources": [row for row in selected if row["editor"] == editor]})
    if len(selected) < MIN_WORKS or len(groups) < 4:
        raise ValueError(f"Insufficient unrelated official works: {dict(groups)} from {dict(counts)}")
    print(f"Reserved {len(selected)} works from {dict(counts)} with {sum(bool(r['ordinary_edit']) for r in selected)} matched ordinary edits.")


def assemble(directory, output):
    reservation = native.read(directory / "reservation.json.gz")
    packets = {}
    parents = {}
    first = directory / "reservation.json.gz"
    for editor, (_, revision, weights, _) in editors.GENERATORS.items():
        seed = directory / f"{editor}.json.gz"
        reply = directory / f"{editor}-edits.json.gz"
        sources, answers = native.read(seed)["sources"], native.read(reply)
        if (sources != [row for row in reservation["sources"] if row["editor"] == editor]
                or (answers["source_plan_sha256"], answers["family"], answers["revision"], answers["weights_sha256"])
                != (comparison.bind(seed), editor, revision, weights)
                or [row["document"] for row in sources] != [row["document"] for row in answers["responses"]]):
            raise ValueError("Local editor receipt differs from the reserved text or pinned weights")
        packets.update({row["document"]: row["response"] for row in answers["responses"]})
        parents[editor] = {"seed_sha256": comparison.bind(seed), "reply_sha256": comparison.bind(reply),
                           "revision": revision, "weights_sha256": weights}
    works, failed = [], Counter()
    for row in reservation["sources"]:
        cases = []
        response = packets[row["document"]]
        pair = prior.edit_pair(row, response)
        if pair is None:
            failed[row["editor"]] += 1
            pair = [native.native(row["text"], ()) | {"kind": "ordinary", "changed_words": []}]
        cases.extend(pair)
        if donor := row["ordinary_edit"]:
            cases.append(native.native(donor["text"], ()) | {"kind": "ordinary", "changed_words": []})
        if len({case["native_text_sha256"] for case in cases}) != len(cases):
            raise ValueError("Duplicate delivery within one publisher work")
        works.append({"group": row["group"], "document": row["document"], "genre": row["family"],
                      "publisher": row["publisher"], "generator": row["editor"], "operation": "local_paraphrase",
                      "prior_project_exposure": False, "original_source_sha256": row["source_sha256"], "cases": cases})
    native.save(output, {"schema": SCHEMA, "role": "held", "works": works, "reservation_sha256": comparison.bind(first),
                         "freeze_sha256": reservation["freeze_sha256"], "parents": parents, "unusable": dict(failed)})
    print(f"Assembled {len(works)} works and {sum(len(w['cases']) for w in works)} deliveries with {dict(failed)} unusable rewrites.")


def summary(cases, seal, *, alternate=False):
    arm, view, mode = POINT.split("/")
    point = seal["points"][f"{arm}/{view}"][mode]

    def tally(samples):
        return {"incumbent": comparison.tally(samples, alternate=alternate),
                "medium_report": comparison.tally(samples, arm, view, point, alternate=alternate),
                "combined": comparison.tally(samples, arm, view, point, combined=True, alternate=alternate)}

    stats = tally(cases)
    for family in ("qwen", "smol2"):
        result = tally([case for case in cases if case["generator"] == family])
        for method, metrics in stats.items():
            metrics[family] = result[method]["all"]
    return stats


def run(plan, checkpoint, weights, freeze, baseline, output, device):
    roster, frozen, reference = (native.read(path) for path in (plan, freeze, baseline))
    fit_path = weights / "fit.json"
    fit = native.read(fit_path)
    if (comparison.bind(freeze) != roster["freeze_sha256"] or comparison.bind(fit_path) != frozen["fit_sha256"]
            or reference["plan_sha256"] != comparison.bind(plan) or fit["runtime"] != comparison.learner.configure(device)):
        raise ValueError("Held comparison differs from the earlier fitted and calibrated model")
    cases = comparison.measure(roster, checkpoint, weights, fit, device)
    expected = {(work["group"], case["native_text_sha256"]): case for work in reference["works"] for case in work["variants"]}
    for case in cases:
        measured = expected.pop((case["group"], case["native_sha256"]))
        selected, assessed = comparison.existing({"native_text_sha256": case["native_sha256"], "sentences": case["sentences"]},
                                                 measured)
        case["incumbent"] = [compact_range(case["sentences"], span) for span in selected]
        case["incumbent_assessed"] = assessed
    if expected:
        raise ValueError("Baseline deliveries do not match the reserved comparison")
    native.save(output, {key: value for key, value in roster.items() if key != "works"}
                | {"point": POINT, "cases": cases, "source_works": [{k: v for k, v in work.items() if k != "cases"}
                                                               for work in roster["works"]],
                   "plan_sha256": comparison.bind(plan), "baseline_sha256": comparison.bind(baseline),
                   "results": summary(cases, frozen)})


def freeze_matches(seal, path, expected, source):
    if comparison.bind(path) == expected:
        return True
    link = source.get("public_freeze_replay", {})
    if comparison.bind(path) != link.get("sha256") or expected != source.get("freeze_sha256"):
        return False
    actual = seal | {"implementation": link["local_implementation"]}
    payload = json.dumps(actual, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return native.digest(payload) == link["local_uncompressed_sha256"]


def review(freeze, result, sources=None):
    seal, receipt = native.read(freeze), native.read(result)
    source = native.read(sources) if sources else {}
    if (receipt.get("schema") != SCHEMA or receipt["point"] != POINT
            or not freeze_matches(seal, freeze, receipt["freeze_sha256"], source)):
        raise ValueError("Held results do not use the selected frozen point")
    cases = receipt["cases"]
    groups = {row["group"] for row in receipt["source_works"]}
    primary = Counter(case["group"] for case in cases if case["kind"] == "ordinary")
    mixed = Counter(case["group"] for case in cases if case["kind"] == "mixed")
    if (groups != set(primary) or mixed.keys() - groups
            or any(case["kind"] not in {"ordinary", "mixed"} or bool(case["truth"]) != (case["kind"] == "mixed") for case in cases)
            or any(count not in {1, 2} for count in primary.values()) or any(count != 1 for count in mixed.values())
            or len({case["native_sha256"] for case in cases}) != len(cases)):
        raise ValueError("The report lost a source group or conflated its ordinary controls")
    # Position checks require one record per key. Preserve source grouping for
    # reporting, and address individual clean and edited files by their hashes.
    comparison.verify_positions([case | {"group": case["native_sha256"]} for case in cases])
    if source:
        if (source["reservation_sha256"] != receipt["reservation_sha256"]
                or source["held_inputs_sha256"] != receipt["plan_sha256"] or source["editor_assets"] != receipt["parents"]
                or source["unusable"] != receipt["unusable"]):
            raise ValueError("Published coordinates refer to different frozen inputs")
        expected = {(work["group"], case["native_text_sha256"]): case
                    for work in source["assembled_works"] for case in work["deliveries"]}
        for case in cases:
            plan = expected.pop((case["group"], case["native_sha256"]))
            if (plan["kind"], plan["sentences"], plan["nonspace_characters"], [row["compact"] for row in plan["truth"]]) != (
                case["kind"], case["sentences"], case["extent"], case["truth"]
            ):
                raise ValueError("A displayed range differs from the committed source coordinates")
        if expected:
            raise ValueError("One or more frozen deliveries are missing the measured result")
    if summary(cases, seal) != receipt["results"] or summary(cases, seal, alternate=True) != receipt["results"]:
        raise ValueError("Recounted text positions differ from the retained results")
    for method, rows in receipt["results"].items():
        print(method, rows["all"], flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="stage", required=True)
    specs = {
        reserve: ("output", "freeze"), assemble: ("directory", "output"),
        run: ("plan", "checkpoint", "weights", "freeze", "baseline", "output"), review: ("freeze", "result"),
    }
    for command, fields in specs.items():
        sub = commands.add_parser(command.__name__)
        sub.set_defaults(call=command)
        for field in fields:
            sub.add_argument(field, type=Path)
        if command is review:
            sub.add_argument("--sources", type=Path)
        if command is reserve:
            sub.add_argument("prior_inputs", type=Path, nargs="+")
        if command is run:
            sub.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = vars(cli.parse_args())
    args.pop("stage")
    args.pop("call")(**args)
