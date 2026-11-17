"""Compare short speech findings on continuously presented 48 kHz carriers."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from app.evidence import audio_long_review_binding as long
from app.evidence import audio_short_review_binding as short
from app.evidence import audio_two_second_binding as binding
from scripts import audio_two_second_transfer as previous

SCHEMA = "chipmunk.audio-two-second-carrier/1"
DELIVERIES = {"wav": ("-c:a", "pcm_s16le"),
              "m4a": ("-c:a", "aac", "-b:a", "96k", "-movie_timescale", "48000", "-use_editlist", "1")}
SCOPE = "overlap_-10dB"
ROLE = ("reencoded", "transfer")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def encode(source, path):
    profile = path.suffix.lstrip(".")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", str(source),
                    "-ar", "48000", *DELIVERIES[profile], str(path)], check=True, capture_output=True, timeout=30)
    return digest(path.read_bytes())


def group_roster(root, exclude, *, reserve=()):
    freeze_file = root / "freeze.json"
    freeze = json.loads(freeze_file.read_bytes())
    by_family = defaultdict(list)
    for origin in freeze["origins"]:
        if origin["group"] not in exclude:
            by_family[origin["family"]].append(origin)
    if len(by_family) != 7 or any(len(rows) < 2 for rows in by_family.values()):
        raise ValueError("Expected separate sources across six generator families and ordinary recordings")
    def rank(row):
        return row["group"] in reserve, digest(f"{SCHEMA}/{row['id']}".encode())

    ordinary = by_family.pop(None)
    chosen = [row for _, rows in sorted(by_family.items()) for row in sorted(rows, key=rank)[:2]]
    edited_readers = {row["group"] for row in chosen}
    chosen.extend(sorted((row for row in ordinary if row["group"] not in edited_readers), key=rank)[:12])
    selected = {row["group"] for row in chosen}
    assert len(chosen) == len(selected) == 24
    sources = {row["id"] for row in chosen}
    rows = [row for row in freeze["cases"] if row["condition"] == SCOPE and row["id"].split("/", 1)[0] in sources
            and row["kind"] in {"edited", "ordinary", "background"}]
    for row in rows:
        path = root / "pcm" / f"{row['pcm_sha256']}.pcm"
        assert path.stat().st_size == 2 * row["samples"] and digest(path.read_bytes()) == row["pcm_sha256"]
    assert Counter(row["kind"] for row in rows) == {"edited": 12, "ordinary": 12, "background": 24}
    return {"plan_sha256": digest(freeze_file.read_bytes()), "groups": sorted(selected), "cases": rows}


def freeze(first, second):
    reserve = {row["group"] for row in json.loads((second / "freeze.json").read_bytes())["origins"]}
    reencoded = group_roster(first, set(), reserve=reserve)
    transfer = group_roster(second, set(reencoded["groups"]))
    assert not ({r["pcm_sha256"] for r in reencoded["cases"]} & {r["pcm_sha256"] for r in transfer["cases"]})
    assert not ({r["room"] for r in reencoded["cases"]} & {r["room"] for r in transfer["cases"]})
    paths = (Path(__file__), Path(binding.__file__), Path(short.__file__))
    from app.detectors.audio import decode, extended_short_voice_review, short_voice_review

    paths += tuple(Path(mod.__file__) for mod in (decode, extended_short_voice_review, short_voice_review))
    return {"schema": SCHEMA, "code": {path.name: digest(path.read_bytes()) for path in paths},
            "points": {"wav": short.REVIEW_POINT, "m4a": short.AAC_REVIEW_POINT},
            "encoding": DELIVERIES, "roles": dict(zip(ROLE, (reencoded, transfer), strict=True))}


def ranges(row):
    pcm_size, codec = row["samples"], row["codec"]
    excerpts = binding.summarize_readings(row["excerpts"], pcm_size, schema=binding.PRESENTED_SCHEMA, codec=codec)
    additions = [(area["start_sample"], area["end_sample"])
                 for area in binding.regions(excerpts, schema=binding.PRESENTED_SCHEMA)]
    initial = row["current_windows"]
    if long.focused_intervals(pcm_size):
        initial = long.focused_regions(pcm_size, initial)
    else:
        initial = [r for r in initial if r["direction"] == "ai"] if long.status(initial) == "corroborated" else []
    baseline = [(area["start_sample"], area["end_sample"]) for area in initial]
    return baseline, sorted(set(baseline + additions)), additions


def recount(rows):
    table = defaultdict(Counter)
    match = previous.prior.matches
    for row in rows:
        tally = table[row["role"]]
        base, union, extra = ranges(row)
        kind, truth = row["kind"], row["truth"]
        tally[kind] += 1
        tally[f"{kind}_assessed"] += any(entry["responses"] for entry in row["excerpts"])
        tally[f"{kind}_baseline_flagged"] += bool(base)
        tally[f"{kind}_combined_flagged"] += bool(union)
        tally[f"{kind}_candidate_flagged"] += bool(extra)
        if truth:
            tally["labels"] += len(truth)
            first, both = match(base, truth), match(union, truth)
            tally["useful_baseline"] += len(first)
            tally["useful_combined"] += len(both)
            tally["useful_candidate"] += len(match(extra, truth))
            tally["added_source"] += bool(both - first)
            tally["lost_locations"] += len(first - both)
            tally["new_misplaced_deliveries"] += bool(extra) and not match(extra, truth)
            if both - first:
                tally[f"gain_{row['family']}"] += len(both - first)
    return dict(table)


def score(plan_path, roots, carrier, output):
    from app.core.canonical import canonical_json
    from app.detectors.audio.decode import decode_audio
    from app.detectors.audio.extended_short_voice_review import scan
    from app.engine.boundary import Deadline
    from scripts.audio_joint_generator_transfer import panels, prior
    from scripts.audio_radar_transfer import adapter_result
    from scripts.run_audio_specialists_v15 import load_voice

    plan = json.loads(plan_path.read_bytes())
    assert json.loads(json.dumps(freeze(*roots))) == plan
    weights = Path("models/weights/audio-upgrade")
    assets = prior.read(panels.PRIMARY)["incumbent"]["assets"]
    voice = load_voice({"weights": weights, "assets": assets})
    activity = prior.runtime(weights / "silero_vad.jit")
    rows = []
    with output.with_suffix(".jsonl").open("x") as log, tempfile.TemporaryDirectory() as work:
        for role, root in zip(ROLE, roots, strict=True):
            for case in plan["roles"][role]["cases"]:
                path = Path(work) / f"delivery.{carrier}"
                pin = encode(root / "pcm" / f"{case['pcm_sha256']}.pcm", path)
                decoded = decode_audio(path, source_sha256=pin)
                mapping = decoded.mapping
                extent = binding.presented_samples(mapping)
                assert extent == case["samples"], (case["id"], carrier, mapping["samples"])
                pcm = decoded.pcm[:extent]
                excerpts = scan(pcm, activity, voice, Deadline(None))
                codec = mapping["source_codec"]
                saved = json.loads(canonical_json(excerpts))
                assert binding.summarize_readings(saved, extent, schema=binding.PRESENTED_SCHEMA, codec=codec) == (
                    binding.summarize_readings(excerpts, extent, schema=binding.PRESENTED_SCHEMA, codec=codec))
                windows = [{"start_sample": a, "end_sample": b,
                            "direction": adapter_result(pcm[a:b], voice, activity)["decision"]}
                           for a, b in (long.focused_intervals(extent) or long.intervals(extent))]
                row = case | {"role": role, "file_sha256": pin, "decoded_pcm_sha256": mapping["pcm_sha256"],
                              "codec": codec, "bit_rate": mapping.get("source_bit_rate"),
                              "decoded_samples": mapping["samples"], "native_ticks": mapping["source_duration_pts"],
                              "current_windows": windows, "excerpts": saved}
                rows.append(row)
                log.write(json.dumps(row, separators=(",", ":")) + "\n")
                log.flush()
                path.unlink()
                if len(rows) % 8 == 0:
                    print(carrier, len(rows), "/ 96", flush=True)
    prior.save(output, {"plan": plan, "model_sha256": assets, "carrier": carrier, "rows": rows, "counts": recount(rows)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="command", required=True)
    prepare = modes.add_parser("freeze")
    for key in ("first", "second", "output"):
        prepare.add_argument(key, type=Path)
    measured = modes.add_parser("score")
    for key in ("plan", "first", "second", "output"):
        measured.add_argument(key, type=Path)
    measured.add_argument("carrier", choices=DELIVERIES)
    verify = modes.add_parser("verify")
    verify.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        with args.output.open("x") as stream:
            json.dump(freeze(args.first, args.second), stream, sort_keys=True, indent=2)
    elif args.command == "score":
        score(args.plan, (args.first, args.second), args.carrier, args.output)
    else:
        result = previous.prior.read(args.receipt)
        counts = recount(result["rows"])
        assert counts == result["counts"]
        print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
