"""Compare original and typography-normalized response on short writing edits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.writing-typography-transfer/1"
VIEWS = ("native", "normalized")
ARMS = ("native", "normalized", "persistent", "relative")
PROTOCOL = {
    "scope": "One eligible interior sentence in one submitted English excerpt; no earlier draft enters scoring.",
    "transform": "NFKC, casefold, ASCII quotes and dashes, decimal digits replaced with zeros, collapsed whitespace.",
    "geometry": "Center of 12–65 words and two adjacent sentences of at least three words each in the same paragraph.",
    "responses": "Exact-sentence native, normalized, lower of both, and lower center odds advantage over the stronger immediate neighbor in each view.",
    "selection": "Second-highest genuine group maximum over paired historic journal versions; strictly exceed this point. Report groups outside the measurement scope.",
    "baseline": "First six native primary three-sentence contexts whose response is at least 0.5.",
    "localization": "One displayed range must meet 30% non-whitespace IoU and 50% known edited-character recall.",
    "advancement": "At least three additional works across both generators, at most one added genuine work flag, and at least 90% assessment.",
    "replication": "Optionally hold out eLife controls and rewrites at the same fixed points. Crop each exact case to a centered three-sentence native body excerpt. Use unchanged and same-paper genuine substitution controls.",
}
TYPOGRAPHY = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "—": "-", "–": "-", "−": "-", "…": "..."})
DECIMAL = re.compile(r"\d")
WORD = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)*")
HASH = re.compile(r"[0-9a-f]{64}")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def serialize(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    with path.open("x", encoding="utf8") as stream:
        stream.write(serialize(value))


def canonical(text):
    text = unicodedata.normalize("NFKC", text).casefold().translate(TYPOGRAPHY)
    return " ".join(DECIMAL.sub("0", text).split())


def native_case(text, family, group, cohort, truth):
    from app.detectors.text.extract import from_string
    from app.detectors.text.pipeline import windows
    from app.detectors.text.segment import segment

    document = from_string(text)
    require(document.text == text, "The exact plain-text geometry changed")
    units = segment(document)
    require(3 <= len(units) <= 12 and len(text) <= 4000, "Unsupported short source geometry")
    require(truth is None or any([unit.start, unit.end] == truth for unit in units), "Known edit is outside native sentence scope")
    return {"cohort": cohort, "group": group, "family": family, "text": text, "text_sha256": sha(text),
            "units": [{"span": [s.start, s.end], "paragraph": s.paragraph, "words": len(WORD.findall(s.text))}
                      for s in units], "truth": truth, "context": windows(units, 3),
            "native": [s.text for s in units], "normalized": [canonical(s.text) for s in units]}


def retained_sources(selection, forum, uswds, elife=None):
    from scripts.writing_blind_rewrite_probe import valid_sentence
    from scripts.writing_uswds_revision_transfer import bind_case

    cases = []
    for row in read(selection):
        if row["role"] in {"before", "edited"}:
            cases.append(native_case(row["text"], row["role"], row["group"], "selection", None))
    for cohort, directory in (("forum", forum), ("uswds", uswds)):
        generated = {arm: read(directory / f"{arm}.json")["rows"] for arm in ("granite", "olmo")}
        for row in read(directory / "roster.json"):
            for role in ("before", "after"):
                bound = bind_case(row, role, row[role + "_focus"])
                cases.append(native_case(bound["text"], role, row["group"], cohort, None))
            for arm, replies in generated.items():
                focus = row["before_focus"]
                original = focus[row["target"]]
                reply = replies[row["group"]]
                require(reply["input_sha256"] == sha(original), "Generator input does not bind its source")
                value = next((item for attempt in reply["attempts"]
                              if (item := valid_sentence(attempt, original, generated=True))), None)
                if value is not None:
                    rewritten = list(focus)
                    rewritten[row["target"]] = value
                    bound = bind_case(row, arm, rewritten)
                    cases.append(native_case(bound["text"], arm, row["group"], cohort, bound["truth"]))
    require(Counter(row["cohort"] for row in cases) == {"selection": 42, "forum": 30, "uswds": 21},
            "Unexpected source scope")
    if elife is not None:
        cases.extend(elife_cases(elife))
    return cases


def elife_cases(file):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    raw = read(file)["cases"]
    require(Counter(row["role"] for row in raw) == {"original": 15, "ordinary": 15, "granite": 13, "olmo": 11},
            "Unknown eLife source roles")
    targets = {row["group"]: row["truth"] for row in raw if row["role"] == "ordinary"}
    out = []
    for row in raw:
        units = segment(from_string(row["text"]))
        target = row["truth"] or targets[row["group"]]
        centers = [i for i, unit in enumerate(units) if unit.start == target[0]]
        if len(centers) != 1 or centers[0] in {0, len(units) - 1}:
            continue
        index = centers[0]
        left, center, right = units[index - 1:index + 2]
        if len({left.paragraph, center.paragraph, right.paragraph}) != 1:
            continue
        synthetic = row["role"] in {"granite", "olmo"}
        if synthetic and [center.start, center.end] != target:
            continue
        truth = [center.start - left.start, center.end - left.start] if synthetic else None
        out.append(native_case(row["text"][left.start:right.end], row["role"], row["group"], "elife", truth))
    return out


def freeze(run, selection, forum, uswds, elife=None):
    save(run / "roster.json", retained_sources(selection, forum, uswds, elife))
    commitments = {"selection": digest(selection)}
    for cohort, directory in (("forum", forum), ("uswds", uswds)):
        commitments[cohort] = {file: digest(directory / file) for file in ("roster.json", "granite.json", "olmo.json")}
    if elife is not None:
        commitments["elife"] = digest(elife)
    save(run / "freeze.json", {"schema": SCHEMA, "protocol": PROTOCOL, "source_commitments": commitments,
                              "runner_sha256": digest(Path(__file__)), "roster_sha256": digest(run / "roster.json")})


def verify_freeze(run):
    lock = read(run / "freeze.json")
    require(lock["protocol"] == PROTOCOL and lock["runner_sha256"] == digest(Path(__file__))
            and lock["roster_sha256"] == digest(run / "roster.json"), "Frozen procedure or sources changed")
    return lock


def score(run, role, primary, independent):
    import torch
    from scripts.score_text_upgrade import sequence_predictor
    from scripts.writing_body_localization_transfer import SECONDARY_ASSETS
    from scripts.writing_focus_v3 import BASELINE_ASSETS

    verify_freeze(run)
    require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1", "Use local model mode")
    if role != "selection":
        require(read(run / "points.json") == points(bound_rows(run, "selection")), "Freeze points before held inference")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    selected = [row for row in read(run / "roster.json") if in_role(row, role)]
    out = {}
    for mode, snapshot, assets, keys in (("tropa", primary, BASELINE_ASSETS, ("context",)),
                                          ("modernbert", independent, SECONDARY_ASSETS, VIEWS)):
        require(all(digest(snapshot / name) == pin for name, pin in assets.items()), "Pinned checkpoint differs")
        requests = sorted({t for row in selected for key in keys for t in row[key]}, key=sha)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        token_lengths = tokenizer(requests, padding=False, truncation=False, return_length=True)["length"]
        require(all(0 < size <= 512 for size in token_lengths), "A model input would exceed the complete token limit")
        predict = sequence_predictor(mode, snapshot, "cpu", 4, max_len=512)
        replies = predict(requests)
        require(len(requests) == len(replies) and all(math.isfinite(x) and 0 <= x <= 1 for x in replies),
                "Invalid local checkpoint response")
        out[mode] = {sha(t): p for t, p in zip(requests, replies, strict=True)}
        del predict
        print(role, mode, len(requests), flush=True)
    save(run / f"{role}.json", {"freeze_sha256": digest(run / "freeze.json"), "responses": out})


def in_role(row, role):
    return row["cohort"] == role or (role == "held" and row["cohort"] in {"forum", "uswds"})


def bound_rows(run, role):
    packet = read(run / f"{role}.json")
    require(packet["freeze_sha256"] == digest(run / "freeze.json"), "Score packet uses a different freeze")
    result = []
    for row in read(run / "roster.json"):
        if not in_role(row, role):
            continue
        values = {}
        for key in ("context", *VIEWS):
            readings = packet["responses"]["tropa" if key == "context" else "modernbert"]
            values[key] = [[sha(t), readings[sha(t)]] for t in row[key]]
        spaces = sorted(i for i, char in enumerate(row["text"]) if char.isspace())
        result.append({k: row[k] for k in ("cohort", "group", "family", "text_sha256", "units", "truth")}
                      | {"characters": len(row["text"]), "spaces": spaces, **values})
    return result


def odds(value):
    p = min(1 - 1e-7, max(1e-7, value))
    return math.log(p / (1 - p))


def measurements(row):
    values = [[odds(value) for _, value in row[view]] for view in VIEWS]
    units = row["units"]
    choices = {arm: [] for arm in ARMS}
    for i in range(1, len(units) - 1):
        if (len({u["paragraph"] for u in units[i - 1:i + 2]}) != 1
                or not 12 <= units[i]["words"] <= 65 or min(units[i - 1]["words"], units[i + 1]["words"]) < 3):
            continue
        raw, transformed = (series[i] for series in values)
        relative = min(series[i] - max(series[i - 1], series[i + 1]) for series in values)
        for arm, response in zip(ARMS, (raw, transformed, min(raw, transformed), relative), strict=True):
            choices[arm].append((response, i))
    return {arm: min(candidates, key=lambda item: (-item[0], item[1]), default=None)
            for arm, candidates in choices.items()}


def points(rows):
    maxima = {arm: {} for arm in ARMS}
    for row in rows:
        for arm, best in measurements(row).items():
            if best is not None:
                previous = maxima[arm].get(row["group"], -math.inf)
                maxima[arm][row["group"]] = max(previous, best[0])
    require(all(len(values) >= 18 for values in maxima.values()), "Insufficient genuine source groups")
    return {"maxima": maxima, "points": {arm: sorted(values.values())[-2] for arm, values in maxima.items()}}


def baseline(row):
    units = row["units"]
    return [[units[max(0, i - 1)]["span"][0], units[min(len(units) - 1, i + 1)]["span"][1]]
            for i, (_, response) in enumerate(row["context"]) if response >= .5][:6]


def result(rows, cutoff):
    totals = defaultdict(lambda: defaultdict(Counter))
    additional = {arm: set() for arm in ARMS}
    added_genuine = {arm: set() for arm in ARMS}
    families = {arm: set() for arm in ARMS}
    records = []
    for row in rows:
        spaces = set(row["spaces"])
        truth = set(range(*row["truth"])) - spaces if row["truth"] else set()
        choices = measurements(row)
        statuses = {}
        for arm, best in {"baseline": None, **choices}.items():
            spans = baseline(row) if arm == "baseline" else (
                [row["units"][best[1]]["span"]] if best and best[0] > cutoff[arm] else [])
            tested = arm == "baseline" or best is not None
            selected = [set(range(*span)) - spaces for span in spans]
            useful = bool(truth) and any(len(mark & truth) * 2 >= len(truth)
                                         and len(mark & truth) * 10 >= 3 * len(mark | truth) for mark in selected)
            flagged = bool(spans)
            statuses[arm] = {"assessed": tested, "spans": spans, "useful": useful, "response": best[0] if best else None}
            bucket = "edited" if truth else "genuine"
            for stratum in (bucket, row["cohort"] + "/" + bucket, row["cohort"] + "/" + row["family"]):
                totals[stratum][arm].update(files=1, assessed=tested, flags=flagged, useful=useful,
                                            wrong_only=bool(truth) and flagged and not useful)
            if arm in ARMS and useful and not statuses["baseline"]["useful"]:
                additional[arm].add(row["group"])
                families[arm].add(row["family"])
            if arm in ARMS and not truth and flagged and not statuses["baseline"]["spans"]:
                added_genuine[arm].add(row["group"])
        records.append({"cohort": row["cohort"], "group": row["group"], "family": row["family"], "results": statuses})
    decision = {arm: {"added_groups": len(additional[arm]), "families": sorted(families[arm]),
                      "additional_genuine_groups": len(added_genuine[arm]),
                      "passes": len(additional[arm]) >= 3 and families[arm] == {"granite", "olmo"}
                      and len(added_genuine[arm]) <= 1 and totals["edited"][arm]["assessed"] >= .9 * totals["edited"][arm]["files"]}
                for arm in ARMS}
    return {"totals": totals, "decisions": decision, "rows": records}


def verify(path):
    receipt = read(path)
    require(receipt["schema"] == SCHEMA and receipt["freeze"]["protocol"] == PROTOCOL
            and receipt["freeze"]["runner_sha256"] == digest(Path(__file__)), "Unexpected frozen method")
    fit, held = receipt["selection"], receipt["held"]
    groups = {role: {r["group"] for r in rows} for role, rows in (("fit", fit), ("held", held))}
    require(len(groups["fit"]) == 21 and len(groups["held"]) == 14 and not groups["fit"] & groups["held"], "Source roles overlap")
    require(all(row["truth"] is None for row in fit), "Only genuine controls set points")
    for row in fit + held:
        n = len(row["units"])
        require(all(len(row[key]) == n and all(HASH.fullmatch(h) and 0 <= score <= 1 for h, score in row[key])
                    for key in ("context", *VIEWS)), "Invalid native coverage")
        require(all(0 <= u["span"][0] < u["span"][1] <= row["characters"] for u in row["units"]), "Invalid source range")
    fitted = points(fit)
    require(fitted == receipt["points"] and result(held, fitted["points"]) == receipt["result"], "Outcome or points differ")
    if "replication" in receipt:
        replication = receipt["replication"]
        rows = replication["cases"]
        require(len({r["group"] for r in rows}) >= 10 and all(r["cohort"] == "elife" for r in rows)
                and not (groups["fit"] | groups["held"]) & {r["group"] for r in rows}, "Invalid eLife source roles")
        require(result(rows, fitted["points"]) == replication["result"], "Replication outcomes differ")
    return receipt["result"]["decisions"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "selection", "points", "held", "elife", "report", "review"))
    parser.add_argument("path", type=Path)
    for value in ("selection", "forum", "uswds", "elife", "primary", "independent", "output"):
        parser.add_argument(f"--{value}", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    if args.action == "freeze":
        freeze(args.path, args.selection, args.forum, args.uswds, args.elife)
    elif args.action in {"selection", "held", "elife"}:
        score(args.path, args.action, args.primary, args.independent)
    elif args.action == "points":
        verify_freeze(args.path)
        save(args.path / "points.json", points(bound_rows(args.path, "selection")))
    elif args.action == "report":
        lock = verify_freeze(args.path)
        fit, held = (bound_rows(args.path, role) for role in ("selection", "held"))
        fitted = read(args.path / "points.json")
        require(fitted == points(fit), "Genuine points differ")
        output = {"schema": SCHEMA, "freeze": lock, "selection": fit, "held": held,
                  "points": fitted, "result": result(held, fitted["points"])}
        if "elife" in lock["source_commitments"]:
            rows = bound_rows(args.path, "elife")
            output["replication"] = {"cases": rows, "result": result(rows, fitted["points"])}
        save(args.output, output)
        print(verify(args.output))
    else:
        print(verify(args.path))


if __name__ == "__main__":
    main()
