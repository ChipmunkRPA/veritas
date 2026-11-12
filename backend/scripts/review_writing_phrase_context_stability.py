"""Replay source anchors, exact score inputs and held decisions without loading weights."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

from app.detectors.text.extract import text_hash
from app.detectors.text.segment import spans
from scripts import writing_archive_author_intake as history
from scripts import writing_phrase_context_stability as trial
from scripts.video_appearance_native_v108 import read, require, write
from scripts.writing_email_phrase_persistence import WORDS
from scripts.writing_focus_v3 import digest


def replay_sources(sources):
    archive = trial.paths()[-1]
    witnesses = []
    for phase in ("calibration", "holdout"):
        for row in sources[phase]:
            source = row["source"]
            for index, side in enumerate(("before", "after")):
                commit = source["commit"] if "commit" in source else source[side]
                original = history.git(archive, "show", f"{commit}:{source['path']}")
                expected = source["blob_sha256"]
                require(text_hash(original) == (expected[index] if isinstance(expected, list) else expected),
                        "Archive source blob differs")
                passage = row[side]
                for anchor in passage["witness"]:
                    lo, hi = anchor["raw_span"]
                    raw = original[lo:hi]
                    require(text_hash(raw) == anchor["raw_sha256"]
                            and original.count("\n", 0, lo) + 1 == anchor["lines"][0]
                            and original.count("\n", 0, hi - 1) + 1 == anchor["lines"][1],
                            "Native archive range differs")
                start, stop = passage["focus"]
                require(passage["text"][start:stop] == passage["witness"][2]["text"]
                        and " ".join(item["text"] for item in passage["witness"]) == passage["text"],
                        "Exact archived phrase sentence differs")
                witnesses.append({"group": row["group"], "phase": phase, "side": side, "commit": commit,
                                  "path": source["path"], "raw_span": passage["witness"][2]["raw_span"],
                                  "raw_sha256": passage["witness"][2]["raw_sha256"]})
    return witnesses


def evaluate(row, packet, observed):
    text = row["text"]
    require(text_hash(text) == row["sha256"], "Fixture bytes differ")
    bounds = [(a, b) for a, b, _ in spans(text, [(0, len(text))])]
    require(len(bounds) == 5, "Independent native sentence partition differs")
    base, proposed = [], []
    candidate_count = 0
    for index, (start, end) in enumerate(bounds):
        words = list(WORDS.finditer(text[start:end]))
        candidates = [(i, j, [start + words[i].start(), start + words[j - 1].end()])
                      for i in range(4, len(words) - 11)
                      for j in (i + 8,)
                      if all(text[start + a.end():start + b.start()] == " " for a, b in zip(words[i:j - 1], words[i + 1:j]))]
        previous, following = bounds[max(0, index - 1):index], bounds[index + 1:index + 2]
        full = [*previous, bounds[index], *following]
        geometries = [full]
        if previous and following and candidates:
            geometries.extend(([previous[0], bounds[index]], [bounds[index], following[0]]))
        readings = []
        for members in geometries:
            model_input = " ".join(text[a:b] for a, b in members)
            observed.add(model_input)
            response = packet[model_input]
            require(response["input_sha256"] == text_hash(model_input) and 0 <= response["raw"] <= 1,
                    "Exact response input differs")
            if members == full:
                base.append(([list(item) for item in members], response["raw"]))
            if len(geometries) == 1:
                continue
            shift = sum(b - a + 1 for a, b in members if a < start)
            require(model_input[shift:shift + end - start] == text[start:end], "Native focus was not seen")
            characters = defaultdict(list)
            for lo, hi, score in response["offsets"]:
                require(0 <= lo < hi <= len(model_input) and math.isfinite(score), "Invalid token geometry")
                for character in range(max(0, lo - shift), min(end - start, hi - shift)):
                    characters[character].append(score)
            readings.append([math.fsum(math.fsum(characters[c]) / len(characters[c])
                                      for c in range(word.start(), word.end())) / len(word[0]) for word in words])
        if len(readings) != 3:
            continue
        for i, j, region in candidates:
            values = []
            for vector in readings:
                persistence = min(math.fsum(vector[i + 2:i + 6]) / 4, math.fsum(vector[i:j]) / 8)
                contrast = persistence - max(math.fsum(vector[i - 4:i]) / 4, math.fsum(vector[j:j + 4]) / 4)
                values.append((persistence, contrast))
            proposed.append((region, [values[0][0], min(values[1][0], values[2][0]), min(values[1][1], values[2][1])]))
            candidate_count += 1
    return base, proposed, candidate_count


def overlap_good(text, expected, regions):
    def mask(parts):
        value = 0
        for start, end in parts:
            require(0 <= start < end <= len(text), "Evidence interval differs from source")
            for index in range(start, end):
                if not text[index].isspace():
                    value |= 1 << index
        return value

    target = mask([expected])
    for region in regions:
        selected = mask(region)
        common = (target & selected).bit_count()
        if common * 10 >= (target | selected).bit_count() * 3 and common * 2 >= target.bit_count():
            return True
    return False


def evaluate_phase(run, rows, phase):
    artifact = run / f"{phase}-responses.json"
    response = read(artifact)
    require(response["fixture_sha256"] == digest(run / "fixtures.json"), "Fixture binding differs")
    packet, observed = response["responses"], set()
    results = [(row, *evaluate(row, packet, observed)) for row in rows if row["role"] == phase]
    require(set(packet) == observed and len(observed) == response["inputs"], "Unbound native input in response packet")
    return results, len(observed), response["forwards"]


def retrospective(rows):
    """Describe ranking at equal held human-group counts without changing frozen decisions."""
    arms = trial.ARMS[:-1]
    ranking = defaultdict(list)
    for row, native, proposals, _ in rows:
        options = [native]
        options.extend([([region], scores[index]) for region, scores in proposals] for index in range(len(arms) - 1))
        for name, measurements in zip(arms, options, strict=True):
            ordered = sorted(measurements, key=lambda item: item[1], reverse=True)
            if ordered and (len(ordered) == 1 or ordered[0][1] > ordered[1][1]):
                region, score = ordered[0]
                ranking[name].append((row, region, score))
    out = {}
    for name in arms:
        maxima = defaultdict(lambda: -math.inf)
        for row, _, score in ranking[name]:
            if row["kind"] not in trial.SEEDS:
                maxima[row["group"]] = max(maxima[row["group"]], score)
        ordered = sorted(maxima.values(), reverse=True)
        dispositions = {}
        for budget in (0, 1, 2, 5):
            point = ordered[budget] if budget < len(ordered) else -math.inf
            selected = [(row, region) for row, region, score in ranking[name] if score > point]
            dispositions[budget] = {
                "false_groups": len({row["group"] for row, _ in selected if row["kind"] not in trial.SEEDS}),
                "useful_locations": sum(row["kind"] in trial.SEEDS and overlap_good(row["text"], row["span"], [region])
                                        for row, region in selected),
            }
        out[name] = dispositions
    return out


def audit(run):
    sources = read(run / "sources.json")
    require(sources == trial.prepare(), "Prospective source selection differs")
    witnesses = replay_sources(sources)
    frozen = read(run / "freeze.json")
    require(frozen["source_sha256"] == digest(run / "sources.json")
            and frozen["impl_sha256"] == digest(Path(trial.__file__)), "Source or method commitment differs")
    fixtures = read(run / "fixtures.json")
    require(trial.build(run) == fixtures, "Frozen edits or native fixture text differ")
    prepared, points, result = fixtures["fixtures"], read(run / "points.json"), read(run / "result.json")
    require(points["fixture_sha256"] == digest(run / "fixtures.json")
            and points["response_sha256"] == digest(run / "calibration-responses.json"), "Frozen point evidence differs")
    cal, count_cal, forwards_cal = evaluate_phase(run, prepared, "calibration")
    held, count_hold, forwards_hold = evaluate_phase(run, prepared, "holdout")
    points = points["points"]
    names = trial.ARMS[1:-1]
    require(not ({item[0]["group"] for item in cal} & {item[0]["group"] for item in held}), "Group leaked between roles")
    for arm, name in enumerate(names):
        maxima = defaultdict(lambda: -math.inf)
        for row, _, candidates, _ in cal:
            maxima[row["group"]] = max(maxima[row["group"]], max(scores[arm] for _, scores in candidates))
        require(math.isclose(sorted(maxima.values())[-2], points[name], rel_tol=1e-12, abs_tol=1e-12),
                "Independent operating point differs")

    false_groups, gained, counts = defaultdict(set), set(), defaultdict(Counter)
    candidate_count, comparisons = 0, 0
    for original, (row, baseline, candidates, amount) in zip(result["cases"], held, strict=True):
        require((original["group"], original["kind"], original["fixture_sha256"], original["target"], original["candidate_count"])
                == (row["group"], row["kind"], row["sha256"], row["span"], amount), "Held result source key differs")
        candidate_count += amount
        selected = {"current": [region for region, score in baseline if score >= .5][:6]}
        proposals = {"current": [region for region, _ in baseline]}
        for index, name in enumerate(names):
            values = [scores[index] for _, scores in candidates]
            best = max(values, default=-math.inf)
            locations = [region for (region, _), value in zip(candidates, values, strict=True) if value == best]
            selected[name] = locations if best > points[name] and len(locations) == 1 else []
            proposals[name] = [region for region, _ in candidates]
            selected[name] = [[item] for item in selected[name]]
            proposals[name] = [[item] for item in proposals[name]]
        selected[trial.ARMS[-1]] = selected["current"] + selected["local_contrast"]
        proposals[trial.ARMS[-1]] = proposals["current"] + proposals["local_contrast"]
        kind = row["kind"]
        positive = kind in trial.SEEDS
        counts[kind]["cases"] += 1
        for name in trial.ARMS:
            has = positive and overlap_good(row["text"], row["span"], selected[name])
            possible = positive and overlap_good(row["text"], row["span"], proposals[name])
            require(original["outcomes"][name] == {"spans": selected[name], "useful": has, "geometry_allows_useful": possible},
                    "Independent location or result differs")
            comparisons += 1
            counts[kind][f"{name}_flag"] += int(bool(selected[name]))
            counts[kind][f"{name}_useful"] += int(has)
            counts[kind][f"{name}_possible"] += int(possible)
            if not positive and selected[name]:
                false_groups[name].add(row["group"])
        if original["outcomes"][trial.ARMS[-1]]["useful"] and not original["outcomes"]["current"]["useful"]:
            gained.add((row["group"], kind))
    require({name: len(false_groups[name]) for name in trial.ARMS} == result["false_groups"]
            and dict(counts) == result["counts"] and sorted(map(list, gained)) == result["added_useful_groups"],
            "Aggregate review counts differ")
    added_false = len(false_groups[trial.ARMS[-1]] - false_groups["current"])
    advance = (len({group for group, _ in gained}) >= 2 and {kind for _, kind in gained} == set(trial.SEEDS)
               and added_false <= 1 and all(counts[family]["cases"] >= 6 for family in trial.SEEDS))
    require(added_false == result["new_false_groups"] and advance == result["advance"], "Frozen advancement differs")
    checkpoint_assets = [{"file": name, "sha256": pin} for name, pin in frozen["checkpoint_assets"].items()]
    generators = {}
    for family, metadata in frozen["generators"].items():
        record = dict(metadata)
        configs = record.pop("local_config_sha256")
        record["local_config_files"] = [{"file": name, "sha256": pin} for name, pin in configs.items()]
        generators[family] = record
    return {"schema": trial.SCHEMA + "-audit", "method": frozen["method"], "archive_commit": sources["archive_commit"],
            "archive_license_sha256": sources["archive_license_sha256"], "intake_paths": sources["intake_paths"],
            "source_qualification": sources["source_qualification"], "prior_corpus": sources["prior_corpus"],
            "source_witnesses": witnesses, "operating_points": points, "checkpoint_revision": frozen["revision"],
            "checkpoint_assets": checkpoint_assets, "generators": generators,
            "reviewer_sha256": digest(Path(__file__)),
            "source_sentence_ranges_verified": len(witnesses) * 5, "calibration_works": len(sources["calibration"]),
            "held_works": len(sources["holdout"]), "calibration_exact_inputs": count_cal, "held_exact_inputs": count_hold,
            "native_cpu_forwards": forwards_cal + forwards_hold, "held_candidate_positions_verified": candidate_count,
            "held_dispositions_verified": comparisons, "admission": fixtures["counts"], "result": result,
            "retrospective_at_matched_false_group_budgets": retrospective(held),
            "commitments": {file: digest(run / file) for file in ("freeze.json", "sources.json", "fixtures.json",
                "generate-qwen.json", "generate-smol2.json", "calibration-responses.json", "holdout-responses.json", "result.json")}}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("run", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    result = audit(args.run)
    write(args.output, result)
    print(f"Verified {result['held_dispositions_verified']} held dispositions and "
          f"{result['source_sentence_ranges_verified']} archival sentence ranges.", flush=True)


if __name__ == "__main__":
    main()
