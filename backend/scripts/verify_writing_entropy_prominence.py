# SPDX-License-Identifier: Apache-2.0
"""Replay the writing entropy comparison and audit its original coordinates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from app.bench import writing_entropy_prominence as bench
from scripts import writing_entropy_prominence as trial
from scripts import writing_multi_passage_panel as sources


def public_roster(path):
    native = sources.read(path)
    works = []
    for work in native["works"]:
        variants = [{key: value for key, value in row.items() if key != "text"} for row in work["variants"]]
        works.append(work | {"variants": variants})
    return native | {"native_plan_sha256": sources.digest(path.read_bytes()), "works": works}


def matching(predicted, truth):
    states = {0}
    for lo, hi in predicted:
        eligible = []
        for label, (start, stop) in enumerate(truth):
            width = min(hi, stop) - max(lo, start)
            if width > 0 and width * 2 >= stop - start and width * 10 >= 3 * (hi - lo + stop - start - width):
                eligible.append(1 << label)
        states |= {used | active for used in states for active in eligible if not used & active}
    return max(bits.bit_count() for bits in states)


def validate(held, roster):
    if held["plan_sha256"] != roster["native_plan_sha256"]:
        raise ValueError("Held results refer to a different input plan")
    variants = {(work["document"], row["version"]): (work, row) for work in roster["works"] for row in work["variants"]}
    for case in held["cases"]:
        work, row = variants.pop((case["document"], case["role"]))
        if (case["group"] != work["group_sha256"] or case["genre"] != work["genre"]
                or case["native_sha256"] != row["native_text_sha256"] or case["sentences"] != row["sentences"]
                or case["truth"] != [span["compact"] for span in row["truth"]]):
            raise ValueError("Held label or source geometry differs from the roster")
        for start, stop, count, relative, curvature in case["measurements"]:
            if (not 0 <= start < stop <= len(row["sentences"]) or stop - start not in bench.WIDTHS
                    or not bench.TOKEN_BOUNDS[0] <= count < bench.TOKEN_BOUNDS[-1]
                    or not 0 <= relative <= bench.UNITS or type(curvature) is not int):
                raise ValueError("Invalid source interval or entropy reading")
    if variants:
        raise ValueError("One or more frozen deliveries are missing")


def recount(held, freeze, outcome):
    def compact(positions, interval):
        first, stop = interval
        return [positions[first][2], positions[stop - 1][3]]

    for label, points in freeze["points"].items():
        counters = defaultdict(lambda: [0, 0, 0, 0])
        for case in held["cases"]:
            positions = case["sentences"]
            baseline = [compact(positions, choice) for choice in case["existing"]]
            old = matching(baseline, case["truth"])
            options = bench.selections(case, freeze["scales"], points)
            for arm, (option, _) in options.items():
                added = [compact(positions, option)] if option else []
                new = matching(baseline + added, case["truth"])
                for genre in ("all", case["genre"]):
                    item = counters[arm, genre]
                    if case["role"] == "ordinary":
                        item[0] += bool(baseline)
                        item[1] += bool(added) and not baseline
                    else:
                        item[2] += old
                        item[3] += new - old
        for (arm, genre), counts in counters.items():
            expected = outcome["outcomes"][label][arm][genre]
            if counts != [expected.get(key, 0) for key in ("existing_flags", "extra_flags", "existing_hits", "extra_hits")]:
                raise ValueError("Independent overlap count differs from the published result")


def native_offsets(held, native_path, tokenizer_path):
    from transformers import AutoTokenizer

    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    plan = sources.read(native_path)
    by_case = {(work["document"], row["version"]): row for work in plan["works"] for row in work["variants"]}
    windows = 0
    for case in held["cases"]:
        raw = by_case[case["document"], case["role"]]["text"]
        doc = from_string(raw)
        prefix = sources._prefix(doc.text)
        geometry = [[s.start, s.end, prefix[s.start], prefix[s.end], s.paragraph, s.n_words] for s in segment(doc)]
        if geometry != case["sentences"] or sources.digest(doc.text.encode()) != case["native_sha256"]:
            raise ValueError("Native extraction or sentence positions cannot be reproduced")
        tokenization = tokenizer(doc.text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = tokenization["offset_mapping"]
        if len(offsets) != case["token_count"]:
            raise ValueError("Token counts differ from the exact native positions")
        for first, stop, count, *_ in case["measurements"]:
            lo, hi = geometry[first][0], geometry[stop - 1][1]
            inside = [i for i, (a, b) in enumerate(offsets) if max(a, lo) < min(b, hi) and prefix[min(b, hi)] > prefix[max(a, lo)]]
            if not inside or inside[-1] - inside[0] + 1 != count:
                raise ValueError("The comparison includes different token positions")
            windows += 1
    return windows


def verify(older, frozen, held, result, roster, native=None, tokenizer=None):
    lock, captured, observed, public = (sources.read(path) for path in (frozen, held, result, roster))
    validate(captured, public)
    with TemporaryDirectory() as scratch:
        fitted = Path(scratch) / "locked.json"
        replayed = Path(scratch) / "outcome.json"
        if lock["older_sha256"] != sources.digest(older.read_bytes()) or lock["implementation"] != trial.implementation():
            raise ValueError("Calibration inputs or inference code differ")
        trial.confirm(held, frozen, replayed)
        if replayed.read_bytes() != result.read_bytes():
            raise ValueError("Frozen decision replay differs")
        if native is not None:
            trial.freeze(older, native, fitted)
            if fitted.read_bytes() != frozen.read_bytes():
                raise ValueError("Native calibration replay differs")
    recount(captured, lock, observed)
    if native and tokenizer:
        print("Native token windows checked", native_offsets(captured, native, tokenizer))
    print("Independent overlap recount and held source binding passed for", len(captured["cases"]), "deliveries")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("older", "frozen", "held", "result", "roster"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--native", type=Path)
    cli.add_argument("--tokenizer", type=Path)
    verify(**vars(cli.parse_args()))
