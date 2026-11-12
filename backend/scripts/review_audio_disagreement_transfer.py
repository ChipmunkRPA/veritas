"""Independently replay the selective voice comparison without model inference."""

from __future__ import annotations

import argparse
import math
from array import array
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np

from scripts import audio_disagreement_transfer as trial


def pcm(raw):
    values = array("h")
    values.frombytes(raw)
    return values


def deliveries(raw, cuts):
    signal = pcm(raw)
    clean_power = sum(value * value for value in signal)
    body = []
    for cut in cuts:
        background = pcm(cut)
        level = math.sqrt(clean_power / sum(value * value for value in background)) * 10**.25
        body.append([a + level * b for a, b in zip(signal, background, strict=True)])
    gain = min(1., 31129 / max(abs(value) for stream in [signal, *body] for value in stream))
    return [array("h", (round(value * gain) for value in stream)).tobytes() for stream in [signal, *body]]


def preemphasis(raw):
    values = np.asarray(pcm(raw), dtype="<f4") * np.float32(1 / 32768)
    delayed = np.concatenate((np.zeros(1, dtype="<f4"), values[:-1]))
    filter_output = values - delayed * np.float32(.97)
    return sha256(np.concatenate((filter_output, filter_output[:16600])).tobytes()).hexdigest()


def direction(old, score, point):
    if old == "original" or score is None:
        return old
    return "ai" if point < score else "inconclusive"


def review(args):
    run = args.output
    lock, roster = trial.checked(args)
    parents, cuts, commitments = trial.inputs(args)
    require, digest, read = trial.require, trial.sha, trial.read
    require(commitments == lock["source_receipts"] and [row for _, row in cuts] == lock["rooms"], "Source lineage changed")
    point_file, result_file = run / "points.json", run / "result.json"
    frozen, result = read(point_file), read(result_file)
    require(frozen["freeze_sha256"] == result["freeze_sha256"] == digest(run / "freeze.json")
            and result["points_sha256"] == digest(point_file), "Study or decision point receipt differs")
    fitted_point = read(args.paired / "points.json")["points"]["paired_correction"]["point"]
    rows_by_panel = {}
    total, graph_inputs = 0, 0
    for index, panel in enumerate(trial.PANELS):
        mode_files = {mode: run / f"{panel}-{mode}.json" for mode in ("voice", "secondary")}
        hashes = {mode: digest(path) for mode, path in mode_files.items()}
        require(hashes == result["score_ledgers_sha256"][panel], "Observation journal changed")
        if panel == "calibration":
            require(hashes == frozen["calibration_ledgers_sha256"], "Calibration journal changed")
        captured = {mode: read(path) for mode, path in mode_files.items()}
        selected = [(raw, row) for raw, row in cuts if (row["group"] == 0) == (index == 0)]
        expected = []
        for original, source in parents[index]:
            exact = deliveries(original, [wave for wave, _ in selected])
            conditions = [(None, 0), *((row["meeting"], row["annotated_voices"]) for _, row in selected)]
            expected.extend((view, source["lane"], source["group_id"], room, voices, False)
                            for view, (room, voices) in zip(exact, conditions, strict=True))
        expected.extend((raw, "background", row["meeting"], row["meeting"], row["annotated_voices"], True) for raw, row in selected)
        observed = []
        for row, voice, secondary, exact in zip((r for r in roster if r["panel"] == panel), captured["voice"]["rows"],
                                                captured["secondary"]["rows"], expected, strict=True):
            raw, lane, group, room, voices, isolated = exact
            require((lane, group, room, voices, isolated) == (row["lane"], row["group_id"], row["meeting"],
                     row["annotated_voices"], row["isolated"]), "Foreground or room role changed")
            require(sha256(raw).hexdigest() == digest(run / "pcm" / f"{row['id']:03d}.pcm") == row["pcm_sha256"],
                    "Independent matched delivery differs")
            for entry in (voice, secondary):
                require((entry["id"], entry["pcm_sha256"]) == (row["id"], row["pcm_sha256"]), "Observation is for a different input")
            scored = voice["current"] != "unassessed"
            require(all((value is not None) == scored for value in (voice["raw"], voice["corrected"], secondary["secondary_score"]))
                    and all(math.isfinite(value) for value in (voice["raw"], voice["corrected"], secondary["secondary_score"])
                            if value is not None)
                    and direction(voice["current"], voice["corrected"], fitted_point) == voice["paired"], "Admission or paired point differs")
            require(secondary["input_sha256"] == (preemphasis(raw) if scored else None), "Independent graph input changed")
            graph_inputs += scored
            observed.append({**row, **voice, **secondary})
        require(sum(row["secondary_score"] is not None for row in observed) == captured["secondary"]["runtime"]["independent_forwards"],
                "Independent forward count differs")
        total += len(observed)
        rows_by_panel[panel] = observed
    controls = [row for row in rows_by_panel["calibration"] if row["lane"] in ("genuine", "background")
                and row["secondary_score"] is not None]
    contested = [r for r in controls if (r["current"], r["paired"]) == ("ai", "inconclusive")]
    standalone = sorted(r["secondary_score"] for r in controls)[-1]
    dispute_point = sorted(r["secondary_score"] for r in contested)[-1] if contested else standalone
    require(frozen["points"] == {"independent": standalone, "disagreement": dispute_point, "genuine_or_background_views": len(controls),
            "disputed_control_views": len(contested), "disputed_control_groups": len({r["group_id"] for r in contested})},
            "Independently reconstructed points differ")
    cell_count = 0
    transitions = {}
    held_subgroups = {}
    for panel, rows in rows_by_panel.items():
        observations = []
        restored, lost = Counter(), Counter()
        for row in rows:
            current, paired, value = row["current"], row["paired"], row["secondary_score"]
            compound = paired
            if current == "ai" and paired == "inconclusive" and dispute_point < value:
                compound = "ai"
                restored[row["condition"], row["lane"]] += 1
            if current == "ai" and compound != "ai":
                lost[row["condition"], row["lane"]] += 1
            observations.append({"current": current, "paired": paired, "independent": direction(current, value, standalone),
                                 "disagreement": compound})
            if panel == "confirmation" and row["meeting"] is not None:
                room = f"held_room_{row['group']}"
                group = held_subgroups.setdefault(room, {}).setdefault(row["condition"], {}).setdefault(row["lane"], Counter())
                group["planned"] += 1
                for arm, verdict in observations[-1].items():
                    group[f"{arm}_ai"] += verdict == "ai"
                    group[f"{arm}_unassessed"] += verdict == "unassessed"
        transitions[panel] = {"restored": {"/".join(k): v for k, v in restored.items()},
                              "lost": {"/".join(k): v for k, v in lost.items()}}
        for cell, measures in result["results"][panel].items():
            kind = {"genuine": "genuine", "background": "background"}
            entries = [(kind.get(row["lane"], "synthetic"), observed) for row, observed in zip(rows, observations, strict=True)
                       if ("isolated" if row["isolated"] else row["condition"]) == cell]
            for arm, saved in measures.items():
                require(dict(Counter(f"{truth}_{observed[arm]}" for truth, observed in entries)) == saved,
                        "Independent outcome count differs")
                cell_count += 1
    require(total == sum(result["views"].values()) and cell_count == 32, "Incomplete held or calibration denominator")
    counts = result["results"]["confirmation"]
    changes = transitions["confirmation"]

    def flags(cell, arm, truth):
        return counts[cell][arm].get(truth + "_ai", 0)

    no_more = all(flags(cell, "disagreement", truth) <= flags(cell, "paired", truth)
                  for cell, truth in (("clean", "genuine"), ("isolated", "background")))
    qualification = {
        "three_voice_restoration": any(f"3_voices/{lane}" in changes["restored"] for lane in ("clone", "stock", "expressive")),
        "three_voice_no_lost_current": not any(f"3_voices/{lane}" in changes["lost"] for lane in ("clone", "stock", "expressive")),
        "three_voice_no_false_increase_vs_current": flags("3_voices", "disagreement", "genuine") <= flags("3_voices", "current", "genuine"),
        "no_added_clean_or_background_false_flags": no_more,
        "noisy_false_reduction_vs_current": sum(flags(cell, "current", "genuine") - flags(cell, "disagreement", "genuine")
                                               for cell in ("2_voices", "3_voices")) > 0,
        "no_noisy_net_synthetic_loss_vs_current": sum(flags(cell, "disagreement", "synthetic") - flags(cell, "current", "synthetic")
                                                     for cell in ("2_voices", "3_voices")) >= 0,
    }
    require(qualification == result["advancement"] and result["advances"] == all(qualification.values()), "Advancement differs")
    trial.write(run / "audit.json", {"exact_pcm": total, "independent_graph_inputs": graph_inputs, "outcome_cells": cell_count,
                "result_sha256": digest(result_file), "advancement": qualification, "transitions": transitions,
                "held_subgroups": held_subgroups})
    print(f"Rebuilt {total} waveforms, {graph_inputs} graph inputs and {cell_count} result cells", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("output", "paired", "annotations", "rooms", "assets", "weights"):
        parser.add_argument(field, type=Path)
    review(parser.parse_args())
