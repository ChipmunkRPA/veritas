"""Score the frozen v210 paired river crossover with the unchanged speech route."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

from scripts import audio_demand_transfer_v208 as serving
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = serving.ROOT, serving.BASE, serving.PANEL
TRIAL = BASE / "audio-demand-crossover-score-v214"
PUBLIC = PANEL / "audio-demand-crossover-score-freeze-v214.json"
PREP = PANEL / "audio-demand-crossover-complete-v210.json"
PREP_FREEZE = PANEL / "audio-demand-crossover-freeze-v210.json"
INPUT_TRIAL = BASE / "audio-demand-crossover-v210"
INPUT = INPUT_TRIAL / "bind-00"
METHOD = ROOT / "docs/audio-demand-crossover-score-v214-method.md"
MODULE = "scripts.audio_demand_crossover_score_v214"
CONDITIONS = ("matched_clean", "river_early_0db", "river_late_0db")
ORIGINS = ("genuine", "synthetic")
CODE = ("backend/scripts/audio_demand_crossover_score_v214.py",
        "backend/tests/test_audio_demand_crossover_score_v214.py", "docs/audio-demand-crossover-score-v214-method.md",
        "backend/scripts/audio_demand_crossover_v210.py", "backend/scripts/audio_demand_transfer_v208.py",
        "backend/scripts/video_appearance_runtime_v108.py")
read, write, sha, require = serving.read, serving.write, serving.sha, serving.require


def waveform(row):
    return scoped(serving.waveform, {"INPUT": INPUT})(row)


def offsets(index):
    require(type(index) is int and 0 <= index < 24, "A fixed paired roster position is required")
    return {"early": (4 + 6 * index) * 16000, "late": (4 + 6 * (index + 24)) * 16000}


def cases():
    """Bind the published v210 receipt and every planned saved production waveform."""
    serving.prior.pilot.require_model_free()
    frozen, result, published = read(PREP_FREEZE), read(INPUT / "output/result.json"), read(PREP)
    profile = INPUT / "profile.json"
    journal = INPUT / "output/private-cases.jsonl"
    serving.prior.verify_data_guard(read(profile), INPUT, INPUT / "output/result.json",
                                    frozen["code_sha256"]["backend/scripts/audio_demand_crossover_v210.py"])
    require(frozen == read(INPUT_TRIAL / "freeze.json") and tuple(frozen["conditions"]) == CONDITIONS
            and frozen["schema"] == "chipmunk.audio-demand-crossover-freeze/210"
            and frozen["planned_pairs"] == 24 and result["schema"] == "chipmunk.audio-demand-crossover-result/210"
            and result["status"] == "complete" and result["new_neural_calls"] == 0
            and all(published.get(key) == value for key, value in result.items())
            and result["freeze_sha256"] == sha(PREP_FREEZE) and result["private_cases_sha256"] == sha(journal)
            and published["profile_sha256"] == sha(profile)
            and all(sha(ROOT / name) == digest for name, digest in frozen["code_sha256"].items()),
            "Exact published guarded v210 preparation required")
    rows = serving.records(journal)
    require(len(rows) == 144 and result["planned_pairs"] == 24 and result["planned_cases"] == 144
            and result["outcomes"] == dict(Counter(row["status"] for row in rows)),
            "Every planned paired preparation outcome is required")
    distributions = defaultdict(Counter)
    for index in range(24):
        pair = f"pair-{index:02d}"
        chunk = rows[index * 6:(index + 1) * 6]
        require([(r["pair"], r["origin"], r["condition"], r["case"]) for r in chunk]
                == [(pair, origin, condition, f"{pair}-{origin}-{condition}")
                    for origin in ORIGINS for condition in CONDITIONS], "Exact ordered paired source roster required")
        require(all(row["status"] in {"prepared", "unavailable"} for row in chunk)
                and len({row["status"] for row in chunk}) == 1, "Paired preparation status differs")
        for row in chunk:
            distributions[row["origin"], row["condition"]][row["status"]] += 1
            if row["status"] == "prepared":
                waveform(row)
        if chunk[0]["status"] == "prepared":
            require(len({row["samples"] for row in chunk}) == 1
                    and 40000 <= chunk[0]["samples"] <= 64000,
                    "Paired foreground prefix extent differs")
            for origin_index in (0, 3):
                clean, early, late = chunk[origin_index:origin_index + 3]
                require(clean["gain"] == early["gain"] == late["gain"]
                        and clean["foreground_prefix_pcm_sha256"] == early["foreground_prefix_pcm_sha256"] == late["foreground_prefix_pcm_sha256"]
                        and clean["background_sample_offset"] is None
                        and early["background_sample_offset"] == offsets(index)["early"]
                        and late["background_sample_offset"] == offsets(index)["late"]
                        and abs(early["measured_snr_db"]) <= .1 and abs(late["measured_snr_db"]) <= .1,
                        "Paired matched-clean or source-window commitment differs")
            require(chunk[1]["background_pcm_sha256"] == chunk[4]["background_pcm_sha256"]
                    and chunk[2]["background_pcm_sha256"] == chunk[5]["background_pcm_sha256"],
                    "Both classes must share the exact river excerpts")
    require(all(dict(distributions[origin, condition]) == published["by_origin_and_condition"][origin][condition]
                for origin in ORIGINS for condition in CONDITIONS), "Published preparation coverage differs")
    return rows


def identity():
    serving.prior.pilot.require_model_free()
    rows = cases()
    serving.prior.verify()
    assets = serving.prior.pilot.ASSETS
    require(all(sha(serving.prior.pilot.WEIGHTS / name) == digest for name, digest in assets.items()),
            "Exact frozen local speech assets required")
    serving.prior.pilot.require_model_free()
    asset_digest = hashlib.sha256(json.dumps(assets, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema": "chipmunk.audio-demand-crossover-score-freeze/214",
            "v210_preparation_sha256": sha(PREP), "v210_freeze_sha256": sha(PREP_FREEZE),
            "route_freeze_sha256": sha(serving.prior.PUBLIC), "asset_set_commitment_sha256": asset_digest,
            "code_sha256": {name: sha(ROOT / name) for name in CODE}, "planned_pairs": 24,
            "planned_cases": len(rows), "conditions": list(CONDITIONS), "scoring_batches": 3,
            "calibration_change_permitted": False, "holdout_access_permitted": False}, rows


def freeze():
    require(not PUBLIC.exists() and not TRIAL.exists(), "Preserve any earlier scoring attempt")
    locked, _ = identity()
    TRIAL.mkdir(mode=0o700)
    write(TRIAL / "freeze.json", locked)
    write(PUBLIC, locked)
    return locked


def verify():
    locked, rows = identity()
    require(read(PUBLIC) == read(TRIAL / "freeze.json") == locked, "Exact frozen crossover scoring required")
    return rows


def audit_case(row, case):
    require(all(row[key] == case[key] for key in ("case", "pair", "origin", "condition")), "Outcome roster differs")
    serving.prior.pilot.check_case({**row, "slot": case["case"]},
                                  {**case, "slot": case["case"],
                                   "status": "qualified" if case["status"] == "prepared" else "unavailable"})


def worker(batch, output):
    require(type(batch) is int and 0 <= batch < len(CONDITIONS), "One fixed batch is required")
    rows = verify()
    stage = TRIAL / f"extract-{batch:02d}"
    require(output == stage / "output" and (stage / "resources.jsonl").is_file() and (stage / "worker.log").is_file()
            and not (stage / "profile.json").exists()
            and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
            "Exact guarded offline scoring worker required")
    selected = rows[batch::len(CONDITIONS)]
    serving.prior.pilot.require_model_free()
    output.mkdir(mode=0o700)
    journal, forwards = output / "private-observations.jsonl", 0

    def count(_module, _args):
        nonlocal forwards
        forwards += 1

    with ExitStack() as stack, journal.open("x") as stream:
        score = serving.start(stack, count)
        for case in selected:
            before = forwards
            outcome = score(waveform(case)) if case["status"] == "prepared" else serving.prior.pilot.unavailable()
            row = {key: case[key] for key in ("case", "pair", "condition", "origin")}
            row.update(outcome, pella_forwards=forwards - before)
            audit_case(row, case)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
    write(output / "result.json", {"status": "complete", "batch": batch, "condition": CONDITIONS[batch],
          "freeze_sha256": sha(PUBLIC), "private_observations_sha256": sha(journal), "new_neural_calls": forwards})


def completed_batch(batch, cases_):
    require(type(batch) is int and 0 <= batch < len(CONDITIONS), "One fixed batch is required")
    stage = TRIAL / f"extract-{batch:02d}"
    journal, result_path = stage / "output/private-observations.jsonl", stage / "output/result.json"
    rows, result = serving.records(journal), read(result_path)
    count = sum(row["pella_forwards"] for row in rows)
    serving.prior.pilot.speech.qualification_profile(stage, count, {}, Path(__file__).name, "extract", batch)
    require(result["status"] == "complete" and result["batch"] == batch and result["condition"] == CONDITIONS[batch]
            and result["private_observations_sha256"] == sha(journal) and result["freeze_sha256"] == sha(PUBLIC)
            and result["new_neural_calls"] == count and len(rows) == 48, "Exact complete scored batch required")
    for row, case in zip(rows, cases_[batch::len(CONDITIONS)], strict=True):
        audit_case(row, case)
    return rows, sha(stage / "profile.json"), count


def run(batch):
    require(type(batch) is int and 0 <= batch < len(CONDITIONS), "One fixed batch is required")
    sources = verify()
    for previous in range(batch):
        completed_batch(previous, sources)

    def launch(argv, **kwargs):
        output = TRIAL / f"extract-{batch:02d}/output"
        require(argv[3:] == ["scripts.video_appearance_native_v108", "extract", str(output), str(batch)],
                "Exact owned scoring command required")
        return subprocess.Popen([*argv[:3], MODULE, "_worker", str(batch), str(output)], **kwargs)

    proxy = SimpleNamespace(Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired)
    scoped(runtime.run, {"TRIAL": TRIAL, "verify": verify, "subprocess": proxy,
                         "prerequisites": lambda *_: True, "__file__": __file__})("extract", batch)


def summarize(rows):
    require(len(rows) == 144, "Every planned source and condition needs a result")
    expected = {(f"pair-{i:02d}", origin, condition) for i in range(24) for origin in ORIGINS for condition in CONDITIONS}
    require({(row["pair"], row["origin"], row["condition"]) for row in rows} == expected,
            "Exact unique planned crossover outcomes required")
    require(all(row["route_status"] in {"ok", "not_applicable", "preparation_unavailable"}
                and row["decision"] in serving.prior.pilot.DIRECTIONS
                and ((row["route_status"] == "ok") == (row["decision"] != "unavailable")) for row in rows),
            "Recorded direction and availability must agree")
    by_key = {(row["pair"], row["origin"], row["condition"]): row for row in rows}
    outcomes, transitions, joint = {}, {}, {}
    for origin in ORIGINS:
        outcomes[origin], transitions[origin] = {}, {}
        for condition in CONDITIONS:
            group = [by_key[f"pair-{i:02d}", origin, condition] for i in range(24)]
            counts = Counter(row["decision"] for row in group)
            outcomes[origin][condition] = {"planned": len(group), "scored": sum(r["route_status"] == "ok" for r in group),
                "speech_gate_rejected": sum(r["route_status"] == "not_applicable" for r in group),
                "preparation_unavailable": sum(r["route_status"] == "preparation_unavailable" for r in group),
                "directions": {key: counts[key] for key in serving.prior.pilot.DIRECTIONS},
                "false_ai": counts["ai"] if origin == "genuine" else None,
                "missed_ai_over_planned": len(group) - counts["ai"] if origin == "synthetic" else None,
                "wrong_original": counts["original"] if origin == "synthetic" else None}
        for first, second in (("matched_clean", "river_early_0db"), ("matched_clean", "river_late_0db"),
                              ("river_early_0db", "river_late_0db")):
            paired = [(by_key[f"pair-{i:02d}", origin, first], by_key[f"pair-{i:02d}", origin, second]) for i in range(24)]
            transitions[origin][f"{first}_to_{second}"] = {
                "planned_pairs": len(paired), "both_scored": sum(a["route_status"] == b["route_status"] == "ok" for a, b in paired),
                "directions": dict(Counter(f"{a['decision']}->{b['decision']}" for a, b in paired)),
                "route_statuses": dict(Counter(f"{a['route_status']}->{b['route_status']}" for a, b in paired))}
    for condition in CONDITIONS:
        pairs = [(by_key[f"pair-{i:02d}", "genuine", condition], by_key[f"pair-{i:02d}", "synthetic", condition])
                 for i in range(24)]
        joint[condition] = {"planned_pairs": len(pairs),
                            "both_scored": sum(a["route_status"] == b["route_status"] == "ok" for a, b in pairs),
                            "genuine_to_synthetic_directions": dict(Counter(f"{a['decision']}->{b['decision']}" for a, b in pairs))}
    return {"outcomes": outcomes, "within_foreground_transitions": transitions, "cross_class_shared_excerpt": joint}


def readout():
    sources = verify()
    rows, profiles, calls = [], [], 0
    for batch in range(len(CONDITIONS)):
        current, profile, count = completed_batch(batch, sources)
        rows.extend(current)
        profiles.append(profile)
        calls += count
    return {"schema": "chipmunk.audio-demand-crossover-score-result/214", "status": "complete",
            "freeze_sha256": sha(PUBLIC), "new_neural_calls": calls, "profile_sha256": profiles,
            **summarize(rows), "scope": "Consumed paired Turkish foregrounds and one river tape. "
            "Differences cannot isolate source class or establish population performance; no calibration or defaults changed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("freeze")
    sub.add_parser("readout")
    for action in ("run", "_worker"):
        command = sub.add_parser(action)
        command.add_argument("batch", type=int, choices=range(len(CONDITIONS)))
        if action == "_worker":
            command.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "run":
        run(args.batch)
    elif args.action == "_worker":
        worker(args.batch, args.output)
    else:
        print(json.dumps({"freeze": freeze, "readout": readout}[args.action](), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
