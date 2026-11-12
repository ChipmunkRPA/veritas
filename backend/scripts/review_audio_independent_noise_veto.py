"""Recheck the independent noisy-voice veto without loading either checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

from scripts.review_audio_spectra_background_transfer import graph_sha

PUBLIC = Path(__file__).resolve().parents[2] / "docs/evaluation-panels"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def check(value, message):
    if not value:
        raise ValueError(message)


def commitments(lock, selection, held, delivery):
    for label, folder, public, stage in (
        ("selection", selection, "audio-spectra-recovery-comparison", {"freeze": "freeze", "confirm": "confirmation"}),
        ("held", held, "audio-spatial-interference", {"freeze": "freeze", "scores": "scores"}),
        ("delivery", delivery, "audio-phase-corroboration", {"freeze": "freeze", "scores": "scores"}),
    ):
        prior = read(PUBLIC / f"{public}-2026-11-11.json")
        actual = {name: digest(folder / f"{name}.json") for name in stage}
        check(actual == lock["source_bindings"][label] == {k: prior[f"{v}_sha256"] for k, v in stage.items()},
              f"The source chain changed for {label}")
        if label != "delivery":
            check(read(folder / "freeze.json")["roster_sha256"] == digest(folder / "roster.json"), "The source roster changed")


def recheck_point(lock, selection):
    roster = {row["id"]: row for row in read(selection / "roster.json") if row["role"] == "prospective"}
    result = read(selection / "confirm.json")
    clean, mixed = [], []
    eligible = []
    for record in result["rows"]:
        source = roster.pop(record["id"])
        check(all(record[key] == source[key] for key in ("id", "group", "family", "pcm_sha256")),
              "The selection reading changed source")
        if record["decision"] != "ai":
            continue
        exact = (selection / "pcm" / f'{record["id"]:03}.pcm').read_bytes()
        check(len(exact) == 96_000 and hashlib.sha256(exact).hexdigest() == source["pcm_sha256"]
              and graph_sha(exact) == record["model_input_sha256"], "The frozen selection source input changed")
        eligible.append((source, record["candidate"]))
        if source["family"] != "genuine":
            (clean if source["condition"] == "clean" else mixed).append(record["candidate"])
    check(not roster and len(clean) >= 6 and len(mixed) >= 8, "Incomplete independent selection")
    points = sorted(set(clean + mixed))
    allowed = [point for point in points if all(value >= point for value in clean)
               and sum(value < point for value in mixed) <= 1]
    check(bool(allowed) and max(allowed) == lock["point"] and len(eligible) == lock["selection_ai_controls"],
          "The measured veto point differs from the frozen selection")
    summary = {}
    for source, response in eligible:
        kind = "clean" if source["condition"] == "clean" else "noisy"
        origin = "genuine" if source["family"] == "genuine" else "synthetic"
        cell = summary.setdefault(f"{kind}/{origin}", Counter())
        cell.update({"current_ai": 1, "withheld": int(response < max(allowed))})
    return summary


def native_and_transformed(held, delivery):
    rows = {row["id"]: row for row in read(held / "roster.json") if row["role"] == "transfer"}
    primary = {row["id"]: row for row in read(held / "scores.json")["rows"] if row["id"] in rows}
    deliveries = read(delivery / "scores.json")["observed"]
    views = {(i, "native"): (row, primary[i]["baseline"]["decision"], row["mono_sha256"], None)
             for i, row in rows.items() if row["mono_sha256"] == primary[i]["mono_sha256"]}
    for row in deliveries:
        if row["view"] != "parent":
            key = row["id"], row["view"]
            check(key not in views and row["id"] in rows, "The additional delivery is duplicated or unrelated")
            views[key] = (rows[row["id"]], row["baseline"]["decision"], row["sha256"], row)
    return views


def disjoint_sources(selection, held):
    a = [r for r in read(selection / "roster.json") if r["role"] == "prospective" and r["condition"] == "clean"]
    b = [r for r in read(held / "roster.json") if r["role"] == "transfer" and r["placement"] == "clean"]
    selection_groups, held_groups = ({r["group"] for r in panel} for panel in (a, b))
    earlier = {r["meeting"] for r in read(selection / "freeze.json")["prospective_backgrounds"]}
    later = set(read(held / "freeze.json")["publisher_mics"])
    check((len(selection_groups), len(held_groups), len(earlier), len(later)) == (17, 16, 2, 2)
          and not (selection_groups & held_groups or earlier & later)
          and not ({r[k] for r in a for k in ("pcm_sha256", "source_pcm_sha256")} & {r["mono_sha256"] for r in b}),
          "A foreground group, clean waveform or recording was shared across roles")


def exact(held, delivery, row, codec, commitment):
    i = row["id"]
    if codec == "native":
        return (held / "mono" / f"{i:03}.pcm").read_bytes()
    suffix = {"opus": "opus", "restored": "wav"}[codec]
    path = delivery / "delivery" / f"{i:03}-{codec}.{suffix}"
    check(commitment["timeline"] == [{"start_sample": 0, "end_sample": 48_000, "source_pts": 0}]
          and digest(path) == commitment["delivery_sha256"], "Transformed delivery file or clock changed")
    return subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file", "-i", str(path.resolve()),
                           "-map", "0:a:0", "-af", "aformat=sample_fmts=s16:channel_layouts=mono,aresample=16000,"
                           "atrim=end_sample=48000", "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"],
                          capture_output=True, timeout=15, check=True).stdout


def audit(run, selection, held, delivery):
    lock, scores, public = (read(run / f"{name}.json") for name in ("freeze", "scores", "aggregate"))
    check(public["freeze_sha256"] == scores["freeze_sha256"] == digest(run / "freeze.json")
          and public["scores_sha256"] == digest(run / "scores.json") and public["point"] == lock["point"],
          "The frozen measurement chain changed")
    commitments(lock, selection, held, delivery)
    disjoint_sources(selection, held)
    check(recheck_point(lock, selection) == public["selection"], "The selection aggregate changed")
    views = native_and_transformed(held, delivery)
    expected = {(r["id"], r["delivery"]): r for r in lock["held"]}
    responses = {(r["id"], r["delivery"]): r for r in scores["rows"]}
    check(len(views) == len(expected) == len(responses) == len(lock["held"]) == len(scores["rows"]) == public["held_views"] == 212
          and views.keys() == expected.keys() == responses.keys(), "The complete held roster differs")
    grouped, cleared_groups, cleared_rooms = {}, set(), set()
    graphs, clean_losses = 0, 0
    for key, (source, baseline, hashed, commitment) in views.items():
        input_row, response = expected[key], responses[key]
        i, codec = key
        check((baseline, hashed) == (input_row["baseline"], input_row["pcm_sha256"])
              and all(input_row[k] == source[k] for k in ("id", "group", "family", "placement", "room", "mode")),
              "A held source or incumbent response differs")
        waveform = exact(held, delivery, source, codec, commitment)
        check(len(waveform) == 96_000 and hashlib.sha256(waveform).hexdigest() == hashed,
              "A held delivered waveform differs")
        reading = response["candidate"]
        if baseline == "ai":
            check(isinstance(reading, float) and math.isfinite(reading)
                  and response["model_input_sha256"] == graph_sha(waveform), "Missing or changed model input")
            graphs += 1
        else:
            check(reading is None and response["model_input_sha256"] is None, "A model call escaped the AI-only rule")
        withheld = reading is not None and reading < lock["point"]
        kind, origin = source["placement"], "genuine" if source["family"] == "genuine" else "synthetic"
        cell = grouped.setdefault(f"{codec}/{kind}/{origin}", Counter())
        cell.update({"planned": 1, f"current_{baseline}": 1, "withheld": int(withheld),
                     "remaining_ai": int(baseline == "ai" and not withheld)})
        if withheld and kind == "center" and origin == "genuine" and codec == "native":
            cleared_groups.add(source["group"])
            cleared_rooms.add(source["room"])
        clean_losses += int(withheld and kind == "clean")
    check(grouped == public["cells"] and len(cleared_groups) == public["center_corrected_groups"]
          and len(cleared_rooms) == public["center_corrected_rooms"] and graphs + 1 == scores["forwards"] == public["forwards"],
          "Held paired counts, corrected groups, or model forward accounting changed")
    checks = {"center_genuine_cleared": grouped["native/center/genuine"]["withheld"] >= 3,
              "center_genuine_groups": len(cleared_groups) >= 2,
              "center_rooms": len(cleared_rooms) == 2,
              "clean_retained": clean_losses == 0,
              "center_synthetic_retained": grouped["native/center/synthetic"]["withheld"] <= 2,
              "offset_synthetic_retained": grouped["native/offset/synthetic"]["withheld"] <= 1,
              "delivery_synthetic_retained": all(grouped[f"{codec}/center/synthetic"]["withheld"] <= 1
                                                for codec in ("opus", "restored"))}
    advance = all(checks.values())
    check(checks == public["checks"] and advance == public["advance"], "The deployment decision changed")
    return {"held_input_waveforms_replayed": len(views), "independent_model_inputs_rebuilt": graphs,
            "held_delivery_waveforms_decoded": sum(codec != "native" for _, codec in views),
            "decision_cells_reproduced": len(grouped), "advance": advance}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("run", "selection", "held", "delivery"):
        parser.add_argument(path, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(audit(arguments.run, arguments.selection, arguments.held, arguments.delivery), indent=2, sort_keys=True))
