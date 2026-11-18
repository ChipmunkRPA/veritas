"""Replay matched source groups through standalone two-second review on Ogg Opus."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from app.evidence import audio_long_review_binding as long
from app.evidence import audio_short_review_binding as short
from app.evidence import audio_two_second_binding as binding
from scripts import audio_two_second_carrier_transfer as prior

PROFILE = ("-c:a", "libopus", "-b:a", "64k", "-vbr", "off", "-frame_duration", "20", "-application", "audio")
POLICIES = {"lossless_point": None, "aac_point": "aac"}
MEDIA_FIELDS = ("id", "kind", "group", "family", "room", "samples", "truth", "pcm_sha256")
STUDY = "chipmunk.two-second-opus/1"
RULE = {"minimum_useful_matched_retention": .8, "minimum_useful_same_carrier_additions": 2,
        "maximum_new_ordinary_flags_vs_same_carrier": 1, "maximum_room_only_flags": 0,
        "required_nonzero_scope_transfer": True}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())


def write(path, data):
    body = json.dumps(data, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    with path.open("xb") as stream:
        stream.write(gzip.compress(body, mtime=0) if path.suffix == ".gz" else body)


def references(wav_path, aac_path):
    wav, aac = read(wav_path), read(aac_path)
    if {row["role"] for row in wav["rows"]} != set(prior.ROLE):
        raise ValueError("The matched study must contain two source roles")
    truth = {(row["role"], row["id"]): row for row in aac["rows"]}
    by_role, source_sets = defaultdict(list), defaultdict(set)
    for row in wav["rows"]:
        comparison = truth[row["role"], row["id"]]
        if any(row[key] != comparison[key] for key in MEDIA_FIELDS):
            raise ValueError("The two previous carriers do not reference identical source material")
        by_role[row["role"]].append({key: row[key] for key in MEDIA_FIELDS})
        for key in ("room", "group", "pcm_sha256"):
            source_sets[row["role"], key].add(row[key])
    if any(source_sets[prior.ROLE[0], key] & source_sets[prior.ROLE[1], key]
           for key in ("room", "group", "pcm_sha256")):
        raise ValueError("The retained source roles share a room, reader, or exact input")
    if any(Counter(row["kind"] for row in values) != {"edited": 12, "ordinary": 12, "background": 24}
           for values in by_role.values()):
        raise ValueError("Complete disjoint 48-case cohorts required")
    return wav, aac, by_role


def freeze(wav_path, aac_path, roots):
    from app.detectors.audio import decode, extended_short_voice_review, ogg_opus_reference

    wav, _, by_role = references(wav_path, aac_path)
    for role, root in zip(prior.ROLE, roots, strict=True):
        if digest((root / "freeze.json").read_bytes()) != wav["plan"]["roles"][role]["plan_sha256"]:
            raise ValueError("Source clock settlement differs from the earlier held evaluation")
        for row in by_role[role]:
            original = root / "pcm" / f"{row['pcm_sha256']}.pcm"
            raw = original.read_bytes()
            if len(raw) != 2 * row["samples"] or digest(raw) != row["pcm_sha256"]:
                raise ValueError("The retained signed PCM differs from its receipt")
    return {
        "schema": STUDY, "procedures": {"encoding": PROFILE, "points": {"lossless_point": short.REVIEW_POINT,
          "aac_point": short.AAC_REVIEW_POINT}, "rule": RULE, "packet_duration_samples": 960, "packet_bytes": 160},
        "implementation": {Path(module.__file__).name: digest(Path(module.__file__).read_bytes())
                           for module in (binding, decode, extended_short_voice_review, ogg_opus_reference, short)},
        "runner_sha256": digest(Path(__file__).read_bytes()),
        "references": {"wav": digest(wav_path.read_bytes()), "aac": digest(aac_path.read_bytes())},
        "assets": wav["model_sha256"], "roles": dict(by_role),
        "ffmpeg": subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, text=True).stdout.splitlines()[0],
    }


def encode(path, original):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", str(original),
                    "-ar", "48000", *PROFILE, "-fflags", "+bitexact", str(path)],
                   check=True, capture_output=True, timeout=30)
    return digest(path.read_bytes())


def measurements(path, case, original):
    from app.detectors.audio.decode import decode_audio
    from app.detectors.audio.ogg_opus_reference import PACKET_BYTES, PRE_SKIP, native_clock

    digest_ = encode(path, original)
    clock = native_clock(path, packet_bytes=PACKET_BYTES, presented=3 * case["samples"])
    decoded = decode_audio(path, source_sha256=digest_)
    n = case["samples"]
    if (clock["pre_skip"] != PRE_SKIP or len(decoded.pcm) != n or any(
                decoded.mapping[key] != value for key, value in {
                    "source_codec": "opus", "source_time_base": "1/48000", "source_start_pts": 0,
                    "source_channels": 1, "source_sample_rate": 48000,
                    "timeline": [{"start_sample": 0, "end_sample": n, "source_pts": 0}],
                }.items())):
        raise ValueError("The Opus page+packet clock and voice windows differ")
    return decoded.pcm, {"file_sha256": digest_, "decoded_pcm_sha256": decoded.mapping["pcm_sha256"], "clock": clock}


def rows_span(seed, *, codec):
    pcm_size = seed["samples"]
    excerpts = binding.summarize_readings(seed["excerpts"], pcm_size, schema=binding.PRESENTED_SCHEMA, codec=codec)
    extra = [(row["start_sample"], row["end_sample"])
             for row in binding.regions(excerpts, schema=binding.PRESENTED_SCHEMA)]
    source = seed["current_windows"]
    if long.focused_intervals(pcm_size):
        baseline = long.focused_regions(pcm_size, source)
    else:
        baseline = [row for row in source if row["direction"] == "ai"] if long.status(source) == "corroborated" else []
    return [(row["start_sample"], row["end_sample"]) for row in baseline], extra


def summary(rows, wav, aac):
    against = {
        fmt: {(row["role"], row["id"]): row for row in sequence["rows"]}
        for fmt, sequence in (("wav", wav), ("aac", aac))
    }
    tables = defaultdict(Counter)
    for row in rows:
        key = row["role"], row["id"]
        truth = row["truth"]
        match = prior.previous.prior.matches
        references = {}
        for fmt, receipt in against.items():
            baseline, extras = rows_span(receipt[key], codec=receipt[key]["codec"])
            references[fmt] = match(baseline + extras, truth) if truth else None
        for name, codec in POLICIES.items():
            baseline, extra = rows_span(row, codec=codec)
            base, combined = (match(values, truth) for values in (baseline, baseline + extra)) if truth else (set(), set())
            kind = row["kind"]
            tally = tables[row["role"], name]
            tally.update(planned=1, **{kind: 1, f"{kind}_assessed": any(entry["responses"] for entry in row["excerpts"]),
                                      f"{kind}_baseline_flagged": bool(baseline), f"{kind}_combined_flagged": bool(baseline or extra),
                                      f"{kind}_added_flag": bool(extra and not baseline)})
            if truth:
                tally.update(labeled=len(truth), baseline_useful=len(base), combined_useful=len(combined),
                             added_useful=len(combined - base), placed_extra=len(match(extra, truth)),
                             wrongly_located_supplement=int(bool(extra and not match(extra, truth))))
                for fmt, locations in references.items():
                    tally.update({f"{fmt}_useful": len(locations), f"{fmt}_retained_useful": len(combined & locations)})
                if combined - base:
                    tally[f"gain_{row['family']}"] += len(combined - base)
    formatted = {"/".join(role): dict(counts) for role, counts in tables.items()}
    decisions = {}
    for name in POLICIES:
        available = [counter for (role, key), counter in tables.items() if key == name]
        tests = {"two_complete_roles": len(available) == 2 and all(row["planned"] == 48 for row in available),
                 "useful_supplement": sum(x["added_useful"] for x in available) >= RULE["minimum_useful_same_carrier_additions"],
                 "gain_in_both_roles": len(available) == 2 and all(x["added_useful"] for x in available),
                 "location_retention": all(x[f"{fmt}_retained_useful"] >= RULE["minimum_useful_matched_retention"] * x[f"{fmt}_useful"]
                                          for x in available for fmt in against),
                 "ordinary_flags": sum(x["ordinary_added_flag"] for x in available) <= RULE["maximum_new_ordinary_flags_vs_same_carrier"],
                 "room_false_flags": not any(x["background_combined_flagged"] for x in available)}
        decisions[name] = {"tests": tests, "advances": all(tests.values())}
    return {"counts": formatted, "advancement": decisions}


def run(plan_path, wav_path, aac_path, roots, output):
    import torch

    from app.core.canonical import canonical_json
    from app.detectors.audio.extended_short_voice_review import scan
    from app.engine.boundary import Deadline
    from scripts.audio_joint_generator_transfer import prior as incumbent
    from scripts.audio_radar_transfer import adapter_result
    from scripts.run_audio_specialists_v15 import load_voice

    plan = read(plan_path)
    if plan != json.loads(json.dumps(freeze(wav_path, aac_path, roots))):
        raise ValueError("The source media or procedure changed after freezing")
    wav, aac, _ = references(wav_path, aac_path)
    weights = Path("models/weights/audio-upgrade")
    voice = load_voice({"weights": weights, "assets": plan["assets"]})
    torch.set_num_threads(2)
    activity = incumbent.runtime(weights / "silero_vad.jit")
    rows, counts = [], Counter()

    def count(*_):
        counts["voice_forwards"] += 1

    hook = voice.register_forward_pre_hook(count)
    journal_path = output.with_suffix(".jsonl")
    try:
        with tempfile.TemporaryDirectory(prefix="opus-voice-") as scratch, journal_path.open("x") as journal:
            for role, root in zip(prior.ROLE, roots, strict=True):
                for case in plan["roles"][role]:
                    path = Path(scratch) / "delivery.ogg"
                    pcm, clock = measurements(path, case, root / "pcm" / f"{case['pcm_sha256']}.pcm")
                    excerpts = json.loads(canonical_json(scan(pcm, activity, voice, Deadline(None))))
                    windows = [{"start_sample": a, "end_sample": b, "direction": adapter_result(pcm[a:b], voice, activity)["decision"]}
                               for a, b in (long.focused_intervals(len(pcm)) or long.intervals(len(pcm)))]
                    rows.append({**case, "role": role, **clock, "excerpts": excerpts, "current_windows": windows})
                    journal.write(json.dumps(rows[-1], separators=(",", ":")) + "\n")
                    journal.flush()
                    path.unlink()
                    if len(rows) % 4 == 0:
                        print(len(rows), len(plan["roles"][role]), dict(counts), flush=True)
    finally:
        hook.remove()
    write(output, {"plan": plan, "rows": rows, "resource_counts": dict(counts), **summary(rows, wav, aac)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "score", "verify"):
        mode = modes.add_parser(name)
        modes_fields = ("wav", "aac")
        if name != "verify":
            modes_fields += ("first", "second")
        if name in {"score", "verify"}:
            modes_fields += (("plan",) if name == "score" else ("receipt",))
        if name != "verify":
            modes_fields += ("output",)
        for key in modes_fields:
            mode.add_argument(key, type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        write(args.output, freeze(args.wav, args.aac, (args.first, args.second)))
    elif args.command == "score":
        run(args.plan, args.wav, args.aac, (args.first, args.second), args.output)
    else:
        wav, aac, _ = references(args.wav, args.aac)
        result = read(args.receipt)
        checked = summary(result["rows"], wav, aac)
        if any(result[key] != value for key, value in checked.items()):
            raise ValueError("The recorded summary differs")
        print(json.dumps(checked, sort_keys=True))


if __name__ == "__main__":
    main()
