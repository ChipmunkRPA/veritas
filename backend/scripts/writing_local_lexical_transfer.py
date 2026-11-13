"""Replay a source-bound, model-free writing localization comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from fractions import Fraction
from math import comb
from pathlib import Path

SCHEMA = "chipmunk.writing-local-lexical-transfer/1"
ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/evaluation-panels"
NATIVE = ROOT / "backend/data/independent-evaluation"
PANELS = {
    "technical": ("writing-multiblock-independent-2026-11-12", "writing-multiblock-independent-2026-11-12-reviewed"),
    "scientific": ("writing-elife-exact-sentence-confirmation-2026-11-12", "writing-elife-exact-sentence-confirmation-2026-11-12"),
}
HUMAN = {"original", "ordinary", "human", "splice"}
ARMS = ("richness", "novelty", "joint")
WORD = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)*")
COMMON = frozenset({
    "a", "about", "across", "after", "against", "all", "also", "am", "an", "and", "any", "are", "as", "at", "be",
    "because", "been", "before", "being", "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
    "each", "either", "even", "every", "for", "from", "had", "has", "have", "having", "he", "her", "here", "hers",
    "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just", "like", "may", "me", "might", "mine",
    "more", "most", "must", "my", "neither", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "out", "over", "own", "same", "shall", "she", "should", "since", "so", "some", "still",
    "such", "than", "that", "the", "their", "theirs", "them", "then", "there", "these", "they", "this", "those",
    "though", "through", "to", "too", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "were",
    "what", "when", "where", "whether", "which", "while", "who", "whom", "why", "will", "with", "within", "without",
    "would", "yet", "you", "your", "yours",
})
METHOD = {
    "focus": "Use existing native sentence positions and both immediate neighbors. Require 12–65 lexical words in each of the three sentences, 50 lexical words elsewhere, and six distinct content terms in each sentence.",
    "richness": "Compare the fraction of distinct types expected in a uniform sample of 12 lexical words without replacement. Score the focus distance outside the two-neighbor range, in either direction.",
    "novelty": "Against the same remaining article vocabulary, count the share of each sentence's unique content terms absent from the remainder. Score the excess over the more novel neighbor. Content terms have four or more characters and exclude the fixed function words.",
    "joint": "Take the smaller positive departure on the two unit scales. Inspect the ablations independently.",
    "point": "Select at most one uniquely highest positive focus per document. Each arm uses the third highest genuine source-group maximum in the older technical selection partition, with strict excess.",
    "metric": "Require at least half of the exact edited non-whitespace characters and .30 intersection over union. Keep ordinary same-work substitutions as controls and count false alerts by source group.",
    "transfer": "Against current plus each supplement, report new useful groups, added genuine groups, wrong-only locations and measured scope separately for later Kubernetes and eLife works. The joint arm must add useful cases from both generator families in each held panel with no more than one extra genuine group in either panel.",
    "diagnostic": "A descriptive held equal-alarm comparison counts the most useful edits reachable with each observed rule and each false-group budget. Held points never enter the frozen decision.",
}


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def sha(data):
    if isinstance(data, Path):
        with data.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    return hashlib.sha256(data.encode()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    with path.open("x") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n")


def save_receipt(path, value):
    with path.open("x") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")


def panel_paths(name, native):
    public, directory = PANELS[name]
    directory = native / directory
    main = "cases.json" if name == "technical" else "plan.json"
    return DOCS / f"{public}.json", directory / main


def lock_paths(native):
    return {name: dict(zip(("public", "native"), map(sha, panel_paths(name, native)), strict=True)) for name in PANELS}


def freeze(run, native):
    commits = lock_paths(native)
    for name, links in commits.items():
        public = read(panel_paths(name, native)[0])
        field, file = (("private_commitments", "cases.json") if name == "technical" else ("private_sha256", "plan.json"))
        require(public[field][file] == links["native"], f"The {name} source ledger no longer matches its reviewed receipt")
    run.mkdir(mode=0o700)
    save(run / "freeze.json", {"schema": SCHEMA, "method": METHOD, "code": sha(Path(__file__)), "sources": commits})


def words(value):
    return [match.casefold().replace("’", "'") for match in WORD.findall(value) if any(c.isalpha() for c in match)]


def histogram(tokens):
    return sorted(Counter(tokens).values())


def richness(hist):
    length = sum(hist)
    samples = comb(length, 12)
    return Fraction(sum(samples - comb(length - count, 12) for count in hist), 12 * samples)


def measure(focuses, remainder, position):
    three = focuses[position - 1:position + 2]
    if not all(12 <= len(value) <= 65 for value in three) or sum(remainder.values()) < 50:
        return None
    content = [{term for term in value if len(term) >= 4 and term not in COMMON} for value in three]
    if min(map(len, content)) < 6:
        return None
    return [{"histogram": histogram(terms), "content": len(vocabulary),
             "novel": sum(not remainder[token] for token in vocabulary)}
            for terms, vocabulary in zip(three, content, strict=True)]


def source_overlap(body, truth, spans):
    marked = {index for start, end in spans for index in range(start, end) if not body[index].isspace()}
    actual = set() if truth is None else {index for index in range(*truth) if not body[index].isspace()}
    return [len(marked & actual), len(marked | actual), len(actual)]


def valid_position(body, units):
    return (all(0 <= a < b <= len(body) for a, b in units)
            and all(l[1] <= r[0] for l, r in zip(units, units[1:])))


def generate_rows(panel, native, *, fit=False, holdout=False):
    pub_path, local_path = panel_paths(panel, native)
    published, local = read(pub_path)["cases"], read(local_path)
    if panel == "scientific":
        local = local["cases"]
    for case, receipt in zip(local, published, strict=True):
        partition = case.get("partition", "held")
        if (partition == "fit" and not fit) or (partition == "held" and not holdout):
            continue
        kind = case.get("kind", case.get("role"))
        body, truth = case["text"], case["truth"]
        pin = sha(body)
        group = sha(case["group"])[:16]
        require(receipt.get("kind", receipt.get("role")) == kind and receipt["group"] == group
                and receipt.get("text_sha256", receipt.get("sha256")) == pin and receipt["truth"] == truth,
                "Retained case identity or truth differs from its published receipt")
        if panel == "technical":
            units = [row["focus"] for row in case["windows"]]
            incumbent = receipt["current"]
        else:
            units = [list(item) for item in sorted({tuple(span) for row in case["contexts"] for span in row["spans"]})]
            incumbent = [row["overlap"] for row in receipt["native"] if row["score"] >= .5][:6]
            require(len(units) == len(case["contexts"]), "Retained sentence count differs")
        require(valid_position(body, units) and (truth is None or truth in units), "Unbound sentence geometry")
        tokenized = [words(body[start:end]) for start, end in units]
        full_counts = Counter(token for line in tokenized for token in line)
        rows = []
        for i in range(1, len(units) - 1):
            three_counts = Counter(token for line in tokenized[i - 1:i + 2] for token in line)
            remainder = full_counts - three_counts
            measured = measure(tokenized, remainder, i)
            if measured is not None:
                rows.append({"span": units[i], "overlap": source_overlap(body, truth, [units[i]]),
                             "three": measured, "background_words": sum(remainder.values())})
        yield {"panel": panel, "partition": partition, "kind": kind, "sha256": pin, "group": group,
               "truth": truth, "current": incumbent, "units": len(units), "rows": rows}


def scores(measurement):
    for item in measurement["three"]:
        hist, content, novel = (item[key] for key in ("histogram", "content", "novel"))
        require(hist == sorted(hist) and all(isinstance(c, int) and c > 0 for c in hist) and 12 <= sum(hist) <= 65
                and 6 <= content <= len(hist) and 0 <= novel <= content, "Invalid lexical measurement")
    rich = [richness(item["histogram"]) for item in measurement["three"]]
    unseen = [Fraction(item["novel"], item["content"]) for item in measurement["three"]]
    richness_departure = max(Fraction(0), rich[1] - max(rich[0], rich[2]), min(rich[0], rich[2]) - rich[1])
    novel_departure = max(Fraction(0), unseen[1] - max(unseen[0], unseen[2]))
    return dict(zip(ARMS, (richness_departure, novel_departure, min(richness_departure, novel_departure)), strict=True))


def winners(case):
    values = [(row, scores(row)) for row in case["rows"]]
    winners = {}
    for arm in ARMS:
        ranked = sorted(((value[arm], index) for index, (_, value) in enumerate(values)), reverse=True)
        if ranked and ranked[0][0] > 0 and (len(ranked) == 1 or ranked[0][0] != ranked[1][0]):
            score, index = ranked[0]
            winners[arm] = (score, values[index][0])
    return winners


def fit_points(selection):
    by_arm = {arm: defaultdict(Fraction) for arm in ARMS}
    ordinary = [case for case in selection if case["kind"] in HUMAN]
    require(len({case["group"] for case in ordinary}) == 12, "Expected all 12 older genuine selection groups")
    for case in ordinary:
        top = winners(case)
        for arm in ARMS:
            score = top[arm][0] if arm in top else Fraction(0)
            by_arm[arm][case["group"]] = max(by_arm[arm][case["group"]], score)
    return {arm: str(sorted(by_arm[arm].values(), reverse=True)[2]) for arm in ARMS}


def useful(item):
    shared, union, target = item
    require(0 <= shared <= target <= union, "Impossible position overlap")
    return bool(target and 2 * shared >= target and 10 * shared >= 3 * union)


def compare(winners, cutoff):
    return [(True, useful(top[1]["overlap"])) if top is not None and top[0] > cutoff else (False, False)
            for top in winners]


def summarize(cases, cutoffs):
    by_id = {case["sha256"]: winners(case) for case in cases}
    require(len(by_id) == len(cases), "Repeated case bytes in panel")
    outcomes = {}
    current = [(bool(case["current"]), any(useful(row) for row in case["current"])) for case in cases]
    arms = {"current": current}
    curves = {}
    for arm, point in cutoffs.items():
        arm_winners = [by_id[case["sha256"]].get(arm) for case in cases]

        observed = compare(arm_winners, Fraction(point))
        arms[arm] = observed
        arms[f"current_plus_{arm}"] = [(a or c, b or d) for (a, b), (c, d) in zip(current, observed, strict=True)]
        diagnostics = [compare(arm_winners, value) for value in {Fraction(0), *(top[0] for top in arm_winners if top)}]
        curve = defaultdict(int)
        for attempt in diagnostics:
            false_groups = {case["group"] for case, row in zip(cases, attempt, strict=True) if case["kind"] in HUMAN and row[0]}
            correct = sum(row[1] for case, row in zip(cases, attempt, strict=True) if case["kind"] not in HUMAN)
            curve[len(false_groups)] = max(curve[len(false_groups)], correct)
        curves[arm] = [max((n for count, n in curve.items() if count <= limit), default=0) for limit in range(5)]

    for arm, rows in arms.items():
        genuine = set()
        added, added_families, added_false = set(), set(), set()
        baseline_false = {case["group"] for case, row in zip(cases, current, strict=True) if case["kind"] in HUMAN and row[0]}
        counts = defaultdict(Counter)
        for case, (flag, hit), (_, base_hit) in zip(cases, rows, current, strict=True):
            kind, group = case["kind"], case["group"]
            count = counts[kind]
            count.update({"files": 1, "flags": flag, "useful": hit and kind not in HUMAN,
                          "flags_on_ordinary_replacement": hit and kind in HUMAN,
                          "cue_scope_files": bool(case["rows"]),
                          "cue_scope_target": kind not in HUMAN and any(useful(row["overlap"]) for row in case["rows"]),
                          "wrong_only": kind not in HUMAN and flag and not hit})
            if kind in HUMAN and flag:
                genuine.add(group)
                if group not in baseline_false:
                    added_false.add(group)
            if kind not in HUMAN and hit and not base_hit:
                added.add(group)
                added_families.add(kind)
        outcomes[arm] = {"by_kind": {name: dict(counts[name]) for name in sorted(counts)}, "genuine_groups": len(genuine),
                         "added_genuine_groups": len(added_false), "added_useful_groups": len(added),
                         "added_families": sorted(added_families)}
    return {"fixed": outcomes, "descriptive_best_useful_at_up_to_0_to_4_genuine_groups": curves}


def evaluate(selection, held, cutoffs):
    outputs = {"selection": summarize(selection, cutoffs)}
    for panel in PANELS:
        cases = [row for row in held if row["panel"] == panel]
        output = summarize(cases, cutoffs)
        joint = output["fixed"]["current_plus_joint"]
        families = {row["kind"] for row in cases} - HUMAN
        output["advance"] = (joint["added_useful_groups"] >= 2 and set(joint["added_families"]) == families
                             and len(families) == 2 and joint["added_genuine_groups"] <= 1)
        outputs[panel] = output
    return outputs


def audit(packet):
    require(packet["freeze"]["schema"] == SCHEMA and packet["freeze"]["method"] == METHOD, "Unknown procedure")
    seen = set()
    partition_groups = defaultdict(set)
    for row in packet["selection"] + packet["held"]:
        key = row["sha256"]
        require(key not in seen and len(key) == 64 and len(row["group"]) == 16, "Repeated or invalid case")
        seen.add(key)
        partition_groups[row["panel"], row["partition"]].add(row["group"])
        windows = row["rows"]
        require(0 <= len(windows) <= row["units"] - 2
                and all(x["background_words"] >= 50 for x in windows)
                and all(len(x["three"]) == 3 and 0 <= x["span"][0] < x["span"][1] for x in windows)
                and all(left["span"][1] <= right["span"][0] for left, right in zip(windows, windows[1:])),
                "Impossible recorded sentence scope")
        for item in windows:
            useful(item["overlap"])
            if row["truth"] is None:
                require(item["overlap"][0] == item["overlap"][2] == 0, "Unexpected edit in an original")
            else:
                require((item["span"] == row["truth"]) == bool(item["overlap"][0]),
                        "Exact focus overlap differs from the edit")
    groups = list(partition_groups.values())
    require(len(groups) == 3 and all(not group & right for i, group in enumerate(groups) for right in groups[i + 1:]),
            "Source groups overlap")
    require(all(row["panel"] == "technical" and row["partition"] == "fit" for row in packet["selection"])
            and all(row["panel"] in PANELS and row["partition"] == "held" for row in packet["held"]), "Invalid case role")
    points = fit_points(packet["selection"])
    require(points == packet["points"], "Selection points differ")
    results = evaluate(packet["selection"], packet["held"], points)
    require(results == packet["results"], "Reported outcomes differ")
    return results


def execute(args):
    if args.phase == "review":
        packet = read(args.location)
        results = audit(packet)
        if args.native is not None:
            require(sha(Path(__file__)) == packet["freeze"]["code"], "The native replay uses different runner bytes")
            require(lock_paths(args.native) == packet["freeze"]["sources"], "Prior source commitments differ")
            for panel in PANELS:
                expected = [row for row in packet["selection"] + packet["held"] if row["panel"] == panel]
                require(list(generate_rows(panel, args.native, fit=True, holdout=True)) == expected, "Exact source replay differs")
        print(json.dumps(results, sort_keys=True))
        return
    native = args.native or NATIVE
    run = args.location
    if args.phase == "freeze":
        freeze(run, native)
        return
    frozen = read(run / "freeze.json")
    require(frozen == {"schema": SCHEMA, "method": METHOD, "code": sha(Path(__file__)), "sources": lock_paths(native)},
            "Frozen method or source bytes differ")
    if args.phase == "fit":
        cases = list(generate_rows("technical", native, fit=True))
        save(run / "selection.json", {"rows": cases, "points": fit_points(cases), "freeze": sha(run / "freeze.json")})
        return
    fitted = read(run / "selection.json")
    selection, points = fitted["rows"], fitted["points"]
    require(fitted["freeze"] == sha(run / "freeze.json") and fit_points(selection) == points, "Selection record differs")
    held = [row for panel in PANELS for row in generate_rows(panel, native, holdout=True)]
    results = evaluate(selection, held, points)
    packet = {"freeze": frozen, "selection": selection, "held": held, "points": points, "results": results}
    audit(packet)
    save_receipt(args.output or run / "receipt.json", packet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "fit", "held", "review"))
    parser.add_argument("location", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--output", type=Path)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
