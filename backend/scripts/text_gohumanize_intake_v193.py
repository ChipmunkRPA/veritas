"""Prepare licensed, source-bound writing controls for a later passage study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
TRIAL = BASE / "text-gohumanize-intake-v193"
OUTPUT = TRIAL / "preparation"
PUBLIC = ROOT / "docs/evaluation-panels/text-gohumanize-preparation-freeze-v193.json"
REVISION = "cc066df21227f5fdb73b513c330181f946ce82ef"
LICENSE = "CC-BY-4.0"
ASSETS = {
    "README.md": "f2412af4b33a48a57436fa76b5ff0ec2d46e1cf9bfb176210e21ac253acfe7e7",
    "federal-rights-audit.csv": "e62604be44860a80c65f2d82a870315f9f3158f9f8ea5c263982bcff69730519",
    "stats.json": "7fda30205feff42c2ffa7084732f984608dc2c51e2628c696ae9735d18f621b6",
    "test.jsonl": "e8142517c08744d123aa11ad585d0d7b44fb1c384e599b72100d0f656c717f98",
}
CODE = (
    "backend/scripts/text_gohumanize_intake_v193.py",
    "backend/tests/test_text_gohumanize_intake_v193.py",
    "docs/text-gohumanize-intake-v193-method.md",
)
HISTORY = {
    "writing_v3_v4_exposed_contexts": (
        "writing-v3/inputs.jsonl",
        "writing-v3-focus-freeze.json",
        "input_sha256",
    ),
    "writing_v5_selected_source_answers": (
        "writing-specialists-v5/plan.json",
        "writing-specialists-v5-original-source-audit.json",
        "plan_sha256",
    ),
    "writing_v7_prepared_exposures": (
        "writing-spans-v7/inputs.json",
        "writing-spans-v7-freeze.json",
        "input_sha256",
    ),
}
AGENCIES = {
    "Federal Reserve": ("Federal Reserve Board", "www.federalreserve.gov", 28),
    "National Institute of Standards and Technology": ("NIST", "www.nist.gov", 23),
    "Department of Energy": ("Department of Energy", "www.energy.gov", 3),
}
GENERATORS = {"gpt-4o-mini": 39, "deepseek-chat": 15}
RIGHTS = "Public-domain basis supported"
ARMS = ("human", "full_ai", "short_ai", "short_format", "long_ai", "long_format")
BANDS = {"short": (8, 16), "long": (35, 65)}
PAIRS, ARTICLES, HISTORY_NGRAM = 54, 30, 12
WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*")
END = re.compile(r"""[.!?]+["'”’»\)\]]*(?=\s|$)""")
NONFINAL = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|Sr|Jr|Sen|Rep|Gov|No|Nos|Fig|Figs|vs|etc|[A-Za-z])\.$"
    r"|\b(?:[A-Za-z]\.){2,}$",
    re.IGNORECASE,
)
WRAP = re.compile(r"(?<=\S) (?=\S)")


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def digest(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write(path, data):
    with path.open("xb") as stream:
        stream.write(data)


def load_assets(source):
    result = {}
    for name, expected in ASSETS.items():
        path = source / name
        require(path.is_file() and not path.is_symlink(), "Missing direct GoHumanize release file")
        content = path.read_bytes()
        require(digest(content) == expected, "GoHumanize release bytes differ from the pinned revision")
        result[name] = content
    require(
        re.search(rb"^license: cc-by-4\.0\s*$", result["README.md"], re.MULTILINE),
        "Exact publisher license declaration required",
    )
    return result


def check_rights(row, audit):
    agency, host, _ = AGENCIES[row["author"]]
    url = urlsplit(row["url"])
    require(
        row["id"] == audit["id"]
        and row["url"] == audit["source_url"]
        and audit["agency"] == agency
        and audit["rights_assessment"] == RIGHTS
        and row["title"] == row["url"]
        and url.scheme == "https"
        and url.netloc == host
        and url.path.startswith("/")
        and not url.fragment,
        "Exact ID, article URL and qualifying agency rights record required",
    )


def dataset(source):
    assets = load_assets(source)
    rows = [json.loads(line) for line in assets["test.jsonl"].splitlines()]
    stats = json.loads(assets["stats.json"])["splits"]["test"]
    selected = [row for row in rows if row["category"] == "modern"]
    require(
        len(rows) == stats["rows"] == 254
        and len({row["id"] for row in rows}) == len(rows)
        and len(selected) == stats["category"]["modern"] == stats["source"]["us-federal"] == PAIRS
        and len({row["url"] for row in selected}) == stats["articles"] == ARTICLES
        and Counter(row["generator"] for row in selected) == GENERATORS
        and Counter(row["author"] for row in selected)
        == {agency: values[2] for agency, values in AGENCIES.items()},
        "Exact released modern test roster, article grouping and generators required",
    )
    audits = list(csv.DictReader(io.StringIO(assets["federal-rights-audit.csv"].decode("utf-8"), newline="")))
    by_id = {row["id"]: row for row in audits}
    require(len(by_id) == len(audits) == 1057, "Unambiguous publisher rights audit required")
    for row in selected:
        require(
            row["source"] == "us-federal"
            and row["gutenberg_id"] == 0
            and row["id"] in by_id
            and row["input"].strip()
            and row["output"].strip()
            and row["input"] != row["output"]
            and row["input_words"] == len(row["input"].split())
            and row["output_words"] == len(row["output"].split()),
            "Exact unchanged human output and full released AI input required",
        )
        check_rights(row, by_id[row["id"]])
    selected.sort(key=lambda row: row["id"])
    urls = {url: f"article-{i:02d}" for i, url in enumerate(sorted({row["url"] for row in selected}))}
    roster = [
        {
            "pair": f"pair-{i:02d}",
            "article_group": urls[row["url"]],
            "publisher_row": row["id"],
            "article_url": row["url"],
            "agency": row["author"],
            "generator": row["generator"],
            "style": row["style"],
            "rights_assessment": RIGHTS,
            "human_sha256": digest(row["output"]),
            "full_ai_sha256": digest(row["input"]),
        }
        for i, row in enumerate(selected)
    ]
    return selected, roster


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    words: int

    @property
    def offsets(self):
        return [self.start, self.end]

    @property
    def doubled_center(self):
        return self.start + self.end


def sentences(text):
    spans = []
    start = 0
    for match in END.finditer(text):
        end = match.end()
        if match.group() == "." and end < len(text) and NONFINAL.search(text[max(start, end - 22) : end]):
            continue
        while start < end and text[start].isspace():
            start += 1
        words = len(WORD.findall(text[start:end]))
        if words:
            spans.append(Span(start, end, words))
        start = end
    return spans


def candidates(text, bounds):
    parts = sentences(text)
    for i in range(1, len(parts) - 1):
        words = 0
        for j in range(i, len(parts) - 1):
            words += parts[j].words
            if words > bounds[1]:
                break
            span = Span(parts[i].start, parts[j].end, words)
            if words >= bounds[0] and len(text) <= 2 * span.doubled_center <= 3 * len(text):
                yield span


def lexical(text):
    return tuple(WORD.findall(unicodedata.normalize("NFKC", text).casefold().replace("’", "'")))


def match_spans(human, ai, bounds):
    host = [
        (span, lexical(human[span.start : span.end]))
        for span in candidates(human, bounds)
        if WRAP.search(human[span.start : span.end])
    ]
    donor = [(span, lexical(ai[span.start : span.end])) for span in candidates(ai, bounds)]
    middle = len(human) * len(ai)

    def key(pair):
        h, d = pair
        h_center = h.doubled_center * len(ai)
        d_center = d.doubled_center * len(human)
        return (
            abs(h_center - d_center) + abs(h_center - middle) + abs(d_center - middle),
            abs(h.words - d.words),
            h.start,
            d.start,
            h.end,
            d.end,
        )

    return min(
        ((h, d) for h, h_words in host for d, d_words in donor if h_words != d_words), key=key, default=None
    )


def case(meta, arm, text, ai_spans, *, host=None, donor=None, formatting=None):
    return {
        **{key: meta[key] for key in ("pair", "article_group", "agency", "generator")},
        "case": f"{meta['pair']}-{arm}",
        "arm": arm,
        "status": "unavailable" if text is None else "constructed",
        "reason": "no_eligible_sentence_pair" if text is None else None,
        "text": text,
        "text_sha256": digest(text) if text is not None else None,
        "ai_spans": ai_spans,
        "host_range": host.offsets if host else None,
        "donor_range": donor.offsets if donor else None,
        "format_range": formatting,
    }


def construct(row, meta):
    human, ai = row["output"], row["input"]
    yield case(meta, "human", human, [])
    yield case(meta, "full_ai", ai, [[0, len(ai)]])
    for band, bounds in BANDS.items():
        pair = match_spans(human, ai, bounds)
        if pair is None:
            yield case(meta, f"{band}_ai", None, None)
            yield case(meta, f"{band}_format", None, None)
            continue
        host, donor = pair
        insert = ai[donor.start : donor.end]
        altered = human[: host.start] + insert + human[host.end :]
        yield case(
            meta,
            f"{band}_ai",
            altered,
            [[host.start, host.start + len(insert)]],
            host=host,
            donor=donor,
        )
        line_break = min(
            (match.start() for match in WRAP.finditer(human, host.start, host.end)),
            key=lambda index: (abs(2 * index + 1 - host.doubled_center), index),
        )
        formatted = human[:line_break] + "\n" + human[line_break + 1 :]
        yield case(meta, f"{band}_format", formatted, [], host=host, formatting=[line_break, line_break + 1])


def history_commitments():
    result = {}
    for name, (input_name, receipt, key) in HISTORY.items():
        public = ROOT / "docs/evaluation-panels" / receipt
        source = BASE / input_name
        published = json.loads(public.read_bytes())[key]
        if source.is_file():
            require(
                digest(source.read_bytes()) == published, "Known historical writing exposure bytes differ"
            )
        result[name] = {
            "status": "available" if source.is_file() else "not_locally_available",
            "committed_input_sha256": published,
            "public_receipt_sha256": digest(public.read_bytes()),
        }
    return result


def historical_texts(name, data):
    if name == "writing_v3_v4_exposed_contexts":
        for line in data.splitlines()[1:]:
            for focus in json.loads(line)["foci"]:
                yield focus["context"]
    elif name == "writing_v5_selected_source_answers":
        for group in json.loads(data)["groups"]:
            for source in group["sources"]:
                yield source["text"]
    else:
        for row in json.loads(data)["cases"]:
            yield row["text"]


def ngrams(words):
    for i in range(len(words) - HISTORY_NGRAM + 1):
        yield words[i : i + HISTORY_NGRAM]


def exposure_audit(rows, commitments):
    whole, excerpts = defaultdict(set), defaultdict(set)
    for i, row in enumerate(rows):
        for arm, key in (("human", "output"), ("full_ai", "input")):
            tokens = lexical(row[key])
            whole[tokens].add((i, arm))
            for grams in ngrams(tokens):
                excerpts[grams].add((i, arm))
    result = {}
    for name, commitment in commitments.items():
        if commitment["status"] != "available":
            result[name] = {"status": commitment["status"]}
            continue
        source = (BASE / HISTORY[name][0]).read_bytes()
        require(digest(source) == commitment["committed_input_sha256"], "Historical writing exposure changed")
        count, exact, fragment = 0, set(), set()
        for previous in historical_texts(name, source):
            count += 1
            tokens = lexical(previous)
            exact.update(whole.get(tokens, ()))
            for grams in ngrams(tokens):
                fragment.update(excerpts.get(grams, ()))
        result[name] = {
            "status": "checked",
            "historical_texts_compared": count,
            "exact_normalized_document_matches": dict(Counter(arm for _, arm in exact)),
            "at_least_one_normalized_12_word_match": dict(Counter(arm for _, arm in fragment)),
            "pairs_with_any_match": len({pair for pair, _ in exact | fragment}),
        }
    return result


def context(source):
    rows, roster = dataset(source)
    roster_data = json_bytes(roster)
    lock = {
        "schema": "chipmunk.text-gohumanize-preparation-freeze/193",
        "dataset": "gohumanize/gohumanize-open-humanizer-dataset",
        "dataset_revision": REVISION,
        "dataset_license": LICENSE,
        "publication_doi": "10.5281/zenodo.22843083",
        "dataset_files_sha256": ASSETS,
        "code_sha256": {name: digest((ROOT / name).read_bytes()) for name in CODE},
        "private_roster_sha256": digest(roster_data),
        "required_rights_assessment": RIGHTS,
        "planned_pairs": PAIRS,
        "article_groups": ARTICLES,
        "arms": list(ARMS),
        "replacement_word_bands": {band: list(bounds) for band, bounds in BANDS.items()},
        "historical_writing_exposure_commitments": history_commitments(),
        "model_calls_permitted": False,
        "fitting_permitted": False,
    }
    return lock, rows, roster, roster_data


def freeze(source):
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve any previous intake freeze or attempt")
    lock, _, _, roster = context(source)
    TRIAL.mkdir(mode=0o700)
    write(TRIAL / "private-roster.json", roster)
    write(TRIAL / "freeze.json", json_bytes(lock))
    write(PUBLIC, json_bytes(lock))
    return {
        "status": "frozen",
        "planned_pairs": PAIRS,
        "planned_cases": PAIRS * len(ARMS),
        "new_model_calls": 0,
    }


def verify(source):
    lock, rows, roster, roster_data = context(source)
    require(
        json_bytes(lock) == PUBLIC.read_bytes() == (TRIAL / "freeze.json").read_bytes()
        and roster_data == (TRIAL / "private-roster.json").read_bytes(),
        "Exact prospective freeze and private article roster required",
    )
    return lock, rows, roster


def build(source):
    lock, rows, roster = verify(source)
    cases = [item for row, meta in zip(rows, roster, strict=True) for item in construct(row, meta)]
    require(
        [item["arm"] for item in cases] == list(ARMS) * PAIRS
        and len({item["case"] for item in cases}) == len(cases),
        "Every planned case requires exactly one ordered outcome",
    )
    statuses = ("constructed", "unavailable")
    counts = {arm: dict.fromkeys(statuses, 0) for arm in ARMS}
    by_generator = {generator: {arm: dict.fromkeys(statuses, 0) for arm in ARMS} for generator in GENERATORS}
    groups = {arm: set() for arm in ARMS}
    for item in cases:
        counts[item["arm"]][item["status"]] += 1
        by_generator[item["generator"]][item["arm"]][item["status"]] += 1
        if item["status"] == "constructed":
            groups[item["arm"]].add(item["article_group"])
    journal = b"".join(json_bytes(item) for item in cases)
    result = {
        "schema": "chipmunk.text-gohumanize-preparation-result/193",
        "status": "complete",
        "freeze_sha256": digest(PUBLIC.read_bytes()),
        "private_cases_sha256": digest(journal),
        "planned_pairs": PAIRS,
        "article_groups": ARTICLES,
        "rights_verified_pairs": PAIRS,
        "planned_cases": len(cases),
        "constructed_cases": sum(count["constructed"] for count in counts.values()),
        "unavailable_cases": sum(count["unavailable"] for count in counts.values()),
        "by_arm": counts,
        "article_groups_with_constructed_case_by_arm": {arm: len(values) for arm, values in groups.items()},
        "by_generator": by_generator,
        "pairs_by_agency": Counter(item["agency"] for item in roster),
        "historical_writing_exposure_audit": exposure_audit(
            rows, lock["historical_writing_exposure_commitments"]
        ),
        "new_model_calls": 0,
        "training_started": False,
        "source_text_published": False,
    }
    return journal, result


def prepare(source):
    require(not OUTPUT.exists(), "Preserve any previous preparation attempt")
    journal, result = build(source)
    OUTPUT.mkdir(mode=0o700)
    write(OUTPUT / "private-cases.jsonl", journal)
    write(OUTPUT / "result.json", json_bytes(result))
    return result


def readout(source):
    journal, result = build(source)
    require(
        journal == (OUTPUT / "private-cases.jsonl").read_bytes()
        and json_bytes(result) == (OUTPUT / "result.json").read_bytes(),
        "Exact independently reconstructed preparation required",
    )
    return result


def preflight(source):
    lock, _, _, _ = context(source)
    return {
        "status": "ready_for_freeze",
        "planned_pairs": PAIRS,
        "article_groups": ARTICLES,
        "planned_cases": PAIRS * len(ARMS),
        "historical_manifests_available": sum(
            item["status"] == "available" for item in lock["historical_writing_exposure_commitments"].values()
        ),
        "new_model_calls": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "prepare", "readout"))
    parser.add_argument("source", type=Path, help="Directory holding the exact four licensed release assets")
    args = parser.parse_args()
    commands = {"preflight": preflight, "freeze": freeze, "prepare": prepare, "readout": readout}
    print(json.dumps(commands[args.command](args.source), indent=2, sort_keys=True, allow_nan=False))
