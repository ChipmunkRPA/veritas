"""Recalculate the eLife writing results independently of detector inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.writing-elife-exact-sentence-confirmation/1"
ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "docs/evaluation-panels/writing-multiblock-independent-2026-11-12.json"
FAMILIES = ("granite", "olmo")
ROLES = ("original", "ordinary", *FAMILIES)
ARMS = ("current", "exact", "union")
WORDS = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)*")


def read(path):
    return json.loads(path.read_bytes())


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def useful(overlap):
    shared, all_chars, truth = overlap
    assert type(shared) is type(all_chars) is type(truth) is int
    assert 0 <= shared <= truth <= all_chars
    return truth > 0 and shared / all_chars >= .30 and shared / truth >= .50


def log_odds(p):
    assert 0 <= p <= 1
    value = min(1 - 1e-7, max(1e-7, p))
    return math.log(value / (1 - value))


def replay(packet):
    assert packet["schema"] == SCHEMA and packet["point"] == read(REFERENCE)["points"]["independent_focus"]
    reviewed, groups = [], {role: set() for role in ROLES}
    for row in packet["cases"]:
        role, group = row["role"], row["group"]
        assert role in groups and group not in groups[role] and re.fullmatch("[0-9a-f]{16}", group)
        assert re.fullmatch("[0-9a-f]{64}", row["sha256"])
        groups[role].add(group)
        native = [item for item in row["native"] if item["score"] >= .5][:6]
        exact = row["isolated"]
        assert all(0 <= item["score"] <= 1 for item in row["native"])
        spans = [tuple(item["span"]) for item in exact]
        assert spans == sorted(set(spans))
        scores = [log_odds(item["score"]) for item in exact]
        top = max(scores, default=-math.inf)
        chosen = exact[scores.index(top)] if top > packet["point"] and scores.count(top) == 1 else None
        current = {"flag": bool(native), "useful": any(useful(item["overlap"]) for item in native)}
        isolated = {"flag": chosen is not None, "useful": chosen is not None and useful(chosen["overlap"])}
        expected = {"current": current, "exact": isolated,
                    "union": {name: current[name] or isolated[name] for name in current}}
        assert all(row[arm] == expected[arm] for arm in ARMS)
        assert row["target_measured"] == any(useful(item["overlap"]) for item in exact)
        reviewed.append((role, group, expected, row["target_measured"]))
    assert len(groups["original"]) == 15 and groups["original"] == groups["ordinary"]
    for family in FAMILIES:
        totals = packet["generation_admission"][family]
        assert groups[family] <= groups["original"] and totals["groups"] == 15 and totals["attempted"] == 45
        assert totals.get("admitted", 0) == len(groups[family])

    base_groups, exact_groups = ({group for role, group, result, _ in reviewed if role in ROLES[:2] and result[arm]["flag"]}
                                 for arm in ("current", "exact"))
    for arm in ARMS:
        observed = packet["counts"][arm]
        assert observed["ordinary_groups_flagged"] == len(
            {"current": base_groups, "exact": exact_groups, "union": base_groups | exact_groups}[arm])
        for role in ROLES:
            subset = [result[arm] for measured_role, _, result, _ in reviewed if measured_role == role]
            assert observed["by_role"][role] == {
                "files": len(subset), "flags": sum(row["flag"] for row in subset),
                "at_known_location": sum(row["useful"] for row in subset),
                "wrong_only": sum(row["flag"] and not row["useful"] for row in subset if role in FAMILIES),
            }
    gained = [(role, group) for role, group, result, _ in reviewed
              if role in FAMILIES and result["exact"]["useful"] and not result["current"]["useful"]]
    checks = {"minimum_per_family": min(len(groups[family]) for family in FAMILIES) >= 6,
              "known_targets_measured": all(measured for role, _, _, measured in reviewed if role in FAMILIES),
              "added_groups": len({group for _, group in gained}),
              "added_families": sorted({role for role, _ in gained}),
              "added_ordinary_groups": len(exact_groups - base_groups)}
    assert packet["checks"] == checks
    assert packet["advanced"] == (checks["minimum_per_family"] and checks["known_targets_measured"]
                                  and checks["added_groups"] >= 2 and checks["added_families"] == list(FAMILIES)
                                  and checks["added_ordinary_groups"] <= 1)
    print(f"Replayed the published cutoff, three rule arms and all {len(reviewed)} held decisions.")


def overlaps(body, truth, spans):
    marked = {i for a, b in spans for i in range(a, b) if not body[i].isspace()}
    actual = set() if truth is None else {i for i in range(*truth) if not body[i].isspace()}
    return [len(marked & actual), len(marked | actual), len(actual)]


def select_generated(run, freeze, original, donors, frozen_hash):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment
    from scripts.writing_blind_rewrite_probe import valid_sentence

    selected, admission = {}, {}
    for family in FAMILIES:
        counts = Counter()
        for ordinal, (group, item) in enumerate(original.items()):
            body = item["text"]
            a, b = item["focus"]
            target = body[a:b]
            units = segment(from_string(body))
            index = next(i for i, unit in enumerate(units) if (unit.start, unit.end) == (a, b))
            unchanged = [unit.text for unit in units]
            record = read(run / f"generation-{family}/{group}.json")
            seed = freeze["seeds"][family] + ordinal * 3
            assert record["freeze_sha256"] == frozen_hash and record["input_sha256"] == sha(target)
            assert [entry["seed"] for entry in record["attempts"]] == list(range(seed, seed + 3))
            counts["groups"] += 1
            counts["attempted"] += len(record["attempts"])
            for attempt in record["attempts"]:
                replacement = valid_sentence(attempt, target, generated=True)
                if replacement is None or replacement in body or replacement == donors[group]["donor"]:
                    counts["unsuitable_attempts_before_selection"] += 1
                    continue
                text = body[:a] + replacement + body[b:]
                expected = unchanged[:index] + [replacement] + unchanged[index + 1:]
                if [unit.text for unit in segment(from_string(text))] == expected:
                    selected[family, group] = replacement
                    counts["admitted"] += 1
                    break
                counts["unsuitable_attempts_before_selection"] += 1
        admission[family] = dict(counts)
    return selected, admission


def native_review(packet, run, source_dir):
    freeze = read(run / "freeze.json")
    assert sha((run / "freeze.json").read_bytes()) == packet["freeze_sha256"]
    assert freeze["point"] == packet["point"] and freeze["reference_sha256"] == sha(REFERENCE.read_bytes())
    assert all(sha((run / name).read_bytes()) == value for name, value in packet["private_sha256"].items())
    plan = read(run / "plan.json")
    source = read(run / "source.json")
    assert plan["freeze_sha256"] == packet["freeze_sha256"] and plan["admission"] == packet["generation_admission"]
    primary = read(run / "primary.json")
    independent = read(run / "independent.json")
    for name, data in (("primary", primary), ("independent", independent)):
        assert data["binding"] == packet["private_sha256"]["plan.json"] and data["responses"].keys() == plan[name].keys()
        assert all(sha(value) == pin for pin, value in plan[name].items())
        assert all(entry["coverage"]["token_windows"] == 1 and entry["coverage"]["tokens"] > 0
                   for entry in data["responses"].values())
        assert packet["execution"][name] == {"forwards": data["forwards"], "inputs": len(data["responses"]), "seconds": data["seconds"]}
    original = source["articles"]
    for group, item in original.items():
        official = source_dir / f"elife-{group}-v2.xml"
        assert sha(official.read_bytes()) == item["after_sha256"]
    selected, admission = select_generated(run, freeze, original, source["ordinary"], packet["freeze_sha256"])
    assert admission == plan["admission"]
    assert selected.keys() == {(case["role"], case["group"]) for case in plan["cases"] if case["role"] in FAMILIES}

    windows, exact_spans = 0, 0
    for case, public in zip(plan["cases"], packet["cases"], strict=True):
        group, role, body, truth = (case[key] for key in ("group", "role", "text", "truth"))
        article = original[group]["text"]
        a, b = original[group]["focus"]
        assert public["group"] == sha(group)[:16] and public["sha256"] == case["sha256"] == sha(body)
        assert public["role"] == role and public["truth"] == truth
        if role == "original":
            assert body == article and truth is None
        else:
            changed = body[slice(*truth)]
            assert truth == [a, a + len(changed)] and body == article[:a] + changed + article[b:]
            if role == "ordinary":
                assert changed == source["ordinary"][group]["donor"]
            else:
                assert changed == selected[role, group]

        native = sorted({tuple(pair) for window in case["contexts"] for pair in window["spans"]})
        assert all(0 <= left < right <= len(body) for left, right in native)
        assert all(left[1] <= right[0] for left, right in zip(native, native[1:]))
        assert len(native) == len(case["contexts"])
        sentences = [body[left:right] for left, right in native]
        eligible = []
        for i, (row, exposed) in enumerate(zip(case["contexts"], public["native"], strict=True)):
            group_spans = native[max(i - 1, 0):i + 2]
            spans = [list(pair) for pair in group_spans]
            content = " ".join(sentences[max(i - 1, 0):i + 2])
            assert row["spans"] == spans == exposed["spans"] and row["pin"] == sha(content)
            assert plan["primary"][row["pin"]] == content and exposed["score"] == primary["responses"][row["pin"]]["score"]
            assert exposed["overlap"] == overlaps(body, truth, spans)
            windows += 1
            if (0 < i < len(native) - 1 and 12 <= len(WORDS.findall(sentences[i])) <= 65
                    and all(8 <= len(WORDS.findall(sentences[j])) <= 80 for j in (i - 1, i + 1))):
                eligible.append((list(native[i]), sentences[i]))
        for (span, content), row, exposed in zip(eligible, case["isolated"], public["isolated"], strict=True):
            assert row["span"] == span == exposed["span"] and row["pin"] == sha(content)
            assert plan["independent"][row["pin"]] == content and exposed["score"] == independent["responses"][row["pin"]]["score"]
            assert exposed["overlap"] == overlaps(body, truth, [span])
            exact_spans += 1
    print(f"Replayed {len(original)} official JATS bytes, {windows} native contexts and {exact_spans} exact focus spans.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--native", type=Path, nargs=2, metavar=("RUN", "PUBLISHER"))
    args = parser.parse_args()
    packet = read(args.receipt)
    replay(packet)
    if args.native:
        native_review(packet, *args.native)


if __name__ == "__main__":
    main()
