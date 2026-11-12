"""Compare short edited speech before and after a fixed communication chain."""

from __future__ import annotations

import argparse
import math
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from app.detectors.audio.decode import decode_audio
from scripts import audio_remote_neighbor_replication as source

SCHEMA = "chipmunk.audio-communication-transfer/1"
RATE = 16000
SAMPLES = 6 * RATE
WINDOWS = ((RATE, 3 * RATE), (3 * RATE, 5 * RATE))
CHAIN = "highpass=f=300,lowpass=f=3400,acompressor=threshold=0.1:ratio=3:attack=10:release=80"
CODEC = ("-c:a", "libopus", "-b:a", "24000", "-application", "voip", "-vbr", "off")
STEMS = ("roster.json", "freeze.json", "current.json", "result.json")
ARMS = ("original", "communication", "both")
read, write, sha, digest, require = source.read, source.write, source.sha, source.digest, source.require
OLD = source.OLD.old


def original_rows(parent):
    roster, freeze, scores, receipt = (read(parent / name) for name in STEMS)
    require(len(roster) == 52 and all(sha(parent / f"{key}.json") == value for key, value in receipt["commitments"].items()),
            "Source panel or score ledger changed")
    require(freeze["current_assets"] == OLD.voice.ASSETS and scores["freeze"] == sha(parent / "freeze.json"),
            "Source model or freeze changed")
    for row, score in zip(roster, scores["records"], strict=True):
        pcm = (parent / "pcm" / f"{row['id']}.pcm").read_bytes()
        require(len(pcm) == 2 * SAMPLES and row["id"] == score["id"]
                and digest(pcm) == row["pcm_sha256"] == score["pcm_sha256"], "Original source PCM changed")
        validate_windows(score["windows"], pcm)
        yield row, pcm, score["windows"]


def validate_windows(windows, pcm):
    require([(row["start"], row["end"]) for row in windows] == list(WINDOWS)
            and all(digest(pcm[2 * a:2 * b]) == row["pcm_sha256"]
                    for row, (a, b) in zip(windows, WINDOWS, strict=True)), "Window or sample binding changed")


def transform(raw, path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-i", "pipe:0",
                    "-af", CHAIN, *CODEC, str(path)], input=raw, capture_output=True, check=True, timeout=20)
    encoded_sha = sha(path)
    decoded = decode_audio(path, source_sha256=encoded_sha, max_seconds=7)
    require(decoded.mapping["source_codec"] == "opus" and decoded.mapping["samples"] == SAMPLES
            and decoded.mapping["timeline"] == [{"start_sample": 0, "end_sample": SAMPLES, "source_pts": 0}],
            "Delivery changed the native six-second analysis clock")
    return encoded_sha, (decoded.pcm * 32768).astype("<i2").tobytes()


def prepare(run, parent):
    rows = list(original_rows(parent))
    run.mkdir(mode=0o700)
    (run / "deliveries").mkdir()
    items = []
    for index, (row, pcm, _) in enumerate(rows):
        path = run / "deliveries" / f"{index:02d}.opus"
        encoded, delivered = transform(pcm, path)
        require(delivered != pcm, "Expected delivery processing to change the PCM")
        path.with_suffix(".pcm").write_bytes(delivered)
        items.append({"id": row["id"], "file": path.name, "source_pcm_sha256": digest(pcm),
                      "encoded_sha256": encoded, "delivered_pcm_sha256": digest(delivered)})
    write(run / "inputs.json", items)
    write(run / "freeze.json", {"schema": SCHEMA, "chain": CHAIN, "codec": CODEC, "windows": WINDOWS,
                                "cutoff": read(parent / "freeze.json")["points"]["current"], "model_assets": OLD.voice.ASSETS,
                                "script_sha256": sha(Path(__file__)), "inputs_sha256": sha(run / "inputs.json"),
                                "sources": {name: sha(parent / name) for name in STEMS},
                                "advance": "Preserve all useful original locations and remove at least one false genuine group with both delivery views."})
    print(f"Froze {len(items)} clock-verified Opus deliveries before new scoring", flush=True)


def panel(run, parent):
    lock = read(run / "freeze.json")
    require(lock["script_sha256"] == sha(Path(__file__)) and lock["inputs_sha256"] == sha(run / "inputs.json")
            and lock["sources"] == {name: sha(parent / name) for name in STEMS}, "Frozen experiment inputs changed")
    for item, (row, raw, old) in zip(read(run / "inputs.json"), original_rows(parent), strict=True):
        path = run / "deliveries" / item["file"]
        new = path.with_suffix(".pcm").read_bytes()
        require(item["id"] == row["id"] and item["source_pcm_sha256"] == digest(raw)
                and len(new) == len(raw) and item["delivered_pcm_sha256"] == digest(new)
                and item["encoded_sha256"] == sha(path), "Delivered source changed")
        yield row, raw, old, new, path


def score(run, parent, weights):
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.run_audio_specialists_v15 import load_voice

    OLD.spectra.incumbent.network_free()
    require(all(sha(weights / name) == pin for name, pin in OLD.voice.ASSETS.items()), "Incumbent weights changed")
    model = load_voice({"weights": str(weights), "assets": OLD.voice.ASSETS})
    vad = SpeechActivity(weights / "silero_vad.jit")
    results = []
    for index, (row, _, _, data, _) in enumerate(panel(run, parent), 1):
        values = np.frombuffer(data, "<i2").astype(np.float32) / 32768
        spans = vad.spans(values)
        windows = []
        for start, end in WINDOWS:
            value = None
            if OLD.window_in_speech(spans, start / RATE, end / RATE):
                measured, = model.predict([values[start:end]])
                require(math.isfinite(measured), "Invalid incumbent response")
                value = OLD.calibration.logit(measured)
            windows.append({"start": start, "end": end, "score": value, "pcm_sha256": digest(data[2 * start:2 * end])})
        results.append({"id": row["id"], "windows": windows})
        if index % 13 == 0:
            print(f"Scored {index}/52 exact audio deliveries", flush=True)
    write(run / "scores.json", {"freeze": sha(run / "freeze.json"), "rows": results})


def best(windows):
    found = max((w for w in windows if w["score"] is not None), key=lambda w: w["score"], default=None)
    return [None, None] if found is None else [found["score"], [found["start"], found["end"]]]


def summarize(run, parent, audit=False):
    old_api = source.OLD
    score = read(run / "scores.json")
    require(score["freeze"] == sha(run / "freeze.json"), "Model outputs belong to a different freeze")
    cutoff = read(run / "freeze.json")["cutoff"]
    rows = []
    for proof, (row, _, original, delivered, path) in zip(score["rows"], panel(run, parent), strict=True):
        windows = proof["windows"]
        require(row["id"] == proof["id"], "Score row is from a different source")
        validate_windows(windows, delivered)
        if audit:
            decoded = decode_audio(path, source_sha256=sha(path), max_seconds=7)
            require((decoded.pcm * 32768).astype("<i2").tobytes() == delivered, "Codec replay changed submitted samples")
        paired = [{**b, "score": min(a["score"], b["score"]) if a["score"] is not None and b["score"] is not None else None}
                  for a, b in zip(original, windows, strict=True)]
        rows.append(row | {"responses": dict(zip(ARMS, map(best, (original, windows, paired)), strict=True))})
    subsets = {"all": rows, **{family: [r for r in rows if r["family"] == family] for family in source.FAMILIES}}
    table = {name: {"inputs": dict(Counter(r["arm"] for r in group)),
                    "fixed": {arm: old_api.performance(group, arm, cutoff) for arm in ARMS},
                    "equal_false_groups": {str(n): {arm: old_api.equal_alarm(group, arm, n) for arm in ARMS} for n in (0, 1, 2)}}
             for name, group in subsets.items()}
    original, _, both = (table["all"]["fixed"][arm] for arm in ARMS)
    result = {"schema": SCHEMA, "fixed_point": cutoff, "tables": table, "new_classifier_windows":
              sum(w["score"] is not None for row in score["rows"] for w in row["windows"]),
              "passes_advancement": both["useful_ranges"] >= original["useful_ranges"]
              and both["false_groups"] < original["false_groups"],
              "commitments": {name: sha(run / f"{name}.json") for name in ("freeze", "inputs", "scores")}}
    if audit:
        require(result == read(run / "aggregate.json"), "Result replay changed")
        print("Re-decoded 52 deliveries and checked 208 original and delivered model-window commitments", flush=True)
    else:
        write(run / "aggregate.json", result)
        print(table["all"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "score", "report", "audit"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--parent", type=Path, default=source.BASE / "audio-remote-neighbor-replication-2026-11-12")
    parser.add_argument("--weights", type=Path, default=source.ROOT / "models/weights/audio-upgrade")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.run, args.parent)
    elif args.phase == "score":
        score(args.run, args.parent, args.weights)
    else:
        summarize(args.run, args.parent, args.phase == "audit")


if __name__ == "__main__":
    main()
