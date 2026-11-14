# SPDX-License-Identifier: Apache-2.0
"""Compare sample-exact mute evidence with half-second energy bins."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


SCHEMA = "chipmunk.audio-digital-silence-transfer/1"
CASES = ("unmodified", "mute_120ms", "mute_240ms", "foreground_muted", "quiet_attenuation", "room_only")


def _mix(voice, room, *, mute):
    import numpy as np

    level = np.linalg.norm(voice) / max(1., np.linalg.norm(room)) * 10 ** .25
    background = room.astype(np.float64) * level
    sound = voice.astype(np.float64) + background
    scale = min(1., 32000 / max(1., np.max(np.abs(sound))))
    if mute is not None:
        a, b = mute
        sound[a:b] = background[a:b]
    return np.rint(sound * scale).astype("<i2")


def inputs(panel, parent):
    import numpy as np

    from scripts.audio_sine_resynthesis_transfer import native_rows

    freeze = parent["freeze"]
    timeline = [{"start_sample": 0, "end_sample": 96000, "source_pts": 0}]
    seen = set()
    for binding, pcm in native_rows(panel, freeze):
        if binding["role"] == "background_only":
            yield "room_only", binding["side"], pcm, None, timeline, "contained_within_room"
            continue
        if binding["side"] != "original" or binding["condition"] != "clean":
            continue
        group = binding["id"]
        host = freeze["pairs"][group]
        room = np.frombuffer((panel / "background" / (host["room"] + ".pcm")).read_bytes(), dtype="<i2")
        require(group not in seen and pcm.shape == room.shape == (96000,), "A matched audio group differs")
        seen.add(group)
        control = _mix(pcm, room, mute=None)
        start = 27040 + 160 * (group % 13)
        yield "unmodified", group, pcm, None, timeline, "dry"
        yield "unmodified", group, control, None, timeline, "competing"
        for n, label in ((1920, "mute_120ms"), (3840, "mute_240ms")):
            for signal, condition in ((pcm, "dry"), (control, "competing")):
                altered = signal.copy()
                altered[start:start + n] = 0
                yield label, group, altered, [start, start + n], timeline, condition
        yield "foreground_muted", group, _mix(pcm, room, mute=(start, start + 3840)), None, timeline, "competing"
        quiet = control.copy()
        quiet[start:start + 3840] = np.rint(quiet[start:start + 3840] * .025)
        yield "quiet_attenuation", group, quiet, None, timeline, "competing"
    require(len(seen) == 36, "Expected all 36 native SINE source crops")


def measure(panel, parent):
    import numpy as np

    from app.evidence.audio_background import _candidates, measurements
    from app.evidence.audio_silence import measure as silence

    rows = []
    for role, group, wave, truth, timeline, condition in inputs(panel, parent):
        decoded = np.asarray(wave, dtype=np.float32) / 32768
        baseline = [[row["start_sample"], row["end_sample"]] for row in _candidates(measurements(decoded))
                    if row["category"] == "near_silence"]
        result = silence(decoded, timeline)
        rows.append({"case": role, "source": group, "condition": condition, "truth": truth, "pcm_sha256": digest(wave.tobytes()),
                     "baseline": baseline, "digital_silence": result})
    return rows


def review(receipt):
    rows = receipt["rows"]
    require(receipt["schema"] == SCHEMA and len(rows) == 296, "Incomplete source roster")
    expected = {(case, group, condition) for group in range(36) for case in CASES[:-1]
                for condition in (("competing",) if case in ("foreground_muted", "quiet_attenuation") else ("dry", "competing"))}
    found = {(row["case"], row["source"], row["condition"]) for row in rows if row["case"] != "room_only"}
    rooms = [row["source"] for row in rows if row["case"] == "room_only"]
    require(found == expected and len(set(rooms)) == len(rooms) == 8, "Matched cases are duplicated or missing")
    counts = {}
    for row in rows:
        result, truth = row["digital_silence"], row["truth"]
        if truth:
            start = 27040 + 160 * (row["source"] % 13)
            length = 1920 if row["case"] == "mute_120ms" else 3840
            require(truth == [start, start + length], "An authored mute differs from its exact sample range")
        key = row["case"] + "/" + row["condition"]
        group = counts.setdefault(key, Counter())
        picks = [[part["start_sample"], part["end_sample"]] for part in result["selected"]]
        require(result["checked_samples"] == 96000 and result["affects_overall_confidence"] is False
                and bool(truth) == row["case"].startswith("mute_")
                and re.fullmatch(r"[0-9a-f]{64}", row["pcm_sha256"])
                and result["qualifying_intervals"] >= len(picks)
                and all(0 <= a < b <= 96000 for a, b in picks + row["baseline"]),
                "Invalid case coverage or reported range")
        group["total"] += 1
        group["baseline_flags"] += bool(row["baseline"])
        group["selected_flags"] += bool(picks)
        group["baseline_exact"] += truth in row["baseline"]
        group["selected_exact"] += truth in picks
    return {k: dict(v) for k, v in counts.items()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("receipt", type=Path)
    cli.add_argument("--sine-parent", type=Path)
    cli.add_argument("--panel", type=Path)
    args = cli.parse_args()
    if args.panel is None:
        result = json.loads(args.receipt.read_text())
        require(review(result) == result["outcome"], "Digital-silence results do not replay")
    else:
        source = args.sine_parent.read_bytes()
        parent = json.loads(source)
        require(json.loads((args.panel / "freeze.json").read_text()) == parent["freeze"], "Native panel differs from the published source plan")
        from app.evidence import audio_silence, audio_silence_binding

        result = {"schema": SCHEMA, "source_plan_sha256": digest(source), "runner_sha256": digest(Path(__file__).read_bytes()),
                  "method_sha256": digest(Path(audio_silence.__file__).read_bytes()),
                  "binding_sha256": digest(Path(audio_silence_binding.__file__).read_bytes()),
                  "rows": measure(args.panel, parent)}
        result["outcome"] = review(result)
        with args.receipt.open("x") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
    print(json.dumps(result["outcome"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
