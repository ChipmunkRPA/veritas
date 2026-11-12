"""Test the frozen target mask on source-qualified same-speaker clone edits."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.detectors.audio.decode import read_wav
from scripts import audio_enrolled_target_transfer as prior
from scripts import audio_same_speaker_clone_sources as source

RATE = prior.RATE
SOURCE = source.BASE / "audio-same-speaker-clone-sources-final-2026-11-12"
PREVIOUS = source.BASE / "audio-enrolled-target-2026-11-12"
SCHEMA = "chipmunk.audio-same-speaker-enrollment-transfer/1"
EDIT = 12_800
read, write, sha, digest, require = prior.read, prior.write, prior.sha, prior.digest, prior.require
METHOD = {
    "comparison": "Reapply the earlier frozen target mask unchanged with a genuine same-speaker reference, a sham from a distinct reader, and the exact submitted mono mixture as acoustic baseline. Use the earlier four distinct generic reference speakers unchanged. No detector scores or fitting across test cases.",
    "sources": "Use every source-qualified speaker group in its prospective role and reserved meeting. Each 6-second host follows a 2-second same-host control reserve. Match an ordinary same-host splice, an ordinary different-reader splice and a publisher-linked clone splice at the same 0.8-second sample range. Other voice and sham use different same-role speakers. Donor selection uses only fixed salted grid order and signal energy.",
    "rooms": "Rerender native speech from the reserved AMI rooms at -5 dB over the entire waveform. Exclude exact previously used capture intervals, including all prior enrollment trial cuts. Choose the first eligible host interval in its fixed salted order that permits at least 400 ms of two annotated overlapping voices in the reserved room. All matched arms share background, peak scaling and exact samples outside the edit.",
    "decision": "Reuse prior held acoustic gates unchanged: true enrollment beats sham in at least two-thirds of genuine same-host edits, median exact-edit SNR improves by at least 0.5 dB in each held room on genuine same-host edits, and median synthetic foreground retention in the exact edit is at least -3 dB in both held rooms. This stage has no authenticity outcome or false-alarm operating point.",
}


def active_positions(values, key):
    positions = prior.inputs.positions(len(values), EDIT, key)
    scaled = values * 32768
    return (a for a in positions if prior.inputs.signal_eligible(scaled[a:a+EDIT]))


def target_and_room(base, slot, room, rooms, blocked):
    for start in active_positions(base, slot+"-target"):
        try:
            noise, background = prior.room_cut(rooms[room], blocked, room, len(base), (start, start+EDIT))
        except ValueError as error:
            if str(error) != "No verified room interval with local overlapping words":
                raise
        else:
            return start, noise, background
    raise ValueError(f"No active target interval intersects the frozen background requirement: {slot}")


def render():
    receipt = read(SOURCE / "result.json")
    require(receipt["proof_source_minimum_met"] and receipt["new_neural_calls"] == 0
            and all(sha(SOURCE / f"{key}.json") == pin for key, pin in receipt["private_commitments"].items()), "Source qualification differs")
    old = read(PREVIOUS / "freeze.json")
    require(old["runner_sha256"] == sha(Path(prior.__file__)) and old["roster_sha256"] == sha(PREVIOUS / "roster.json"), "Prior acoustic method differs")
    records = [row for row in read(SOURCE / "sources.json") if row["status"] == "qualified"]
    catalog, _, _, _ = prior.inputs.catalogues()
    families = defaultdict(list)
    for row in records:
        families[row["role"]].append(row)
    identities = {row["host"]["voice"] for row in records}
    generic = {catalog[key]["canonical_speaker"] for key in old["generic"]}
    require(len(identities) == 14 and len(generic) == 4 and not identities & generic, "Generic bank shares an evaluated target speaker")
    rooms, blocked = prior.word_room_sources()
    for old_row in read(PREVIOUS / "roster.json"):
        bg = old_row["background"]
        size = (PREVIOUS / "pcm" / f"{old_row['id']}-original.pcm").stat().st_size//2
        start = bg["capture_start_sample"]
        blocked[bg["meeting"]].append((start, start+size))
    references = {}
    for key in old["generic"]:
        data = (PREVIOUS / "references" / f"{key}.pcm").read_bytes()
        require(digest(data) == old["references"][key] and len(data) == 8*RATE, "Exact generic enrollment changed")
        references[key] = data

    def checked(row, arm):
        raw = (SOURCE / "wav" / f"{row['slot']}-{arm}.wav").read_bytes()
        require(digest(raw) == row[arm]["wav_sha256"], "Qualified source WAV differs")
        return read_wav(raw).astype(np.float64)

    for row in records:
        samples = checked(row, "enrollment")
        prior.selected_frames(samples)
        references[row["enrollment"]["original_sha256"]] = prior.reference_pcm(samples)

    output = []
    for role, cases in families.items():
        for i, row in enumerate(cases):
            slot, room = row["slot"], row["future_room"]
            other = cases[(i+1) % len(cases)]
            sham = cases[(i+2) % len(cases)]
            host, other_host, cloned = checked(row, "host"), checked(other, "host"), checked(row, "clone")
            base, reserve, other_reserve = host[2*RATE:], host[:2*RATE], other_host[:2*RATE]
            variants = (("ordinary_same_host", reserve), ("ordinary_other_voice", other_reserve), ("synthetic", cloned))
            ranges = {arm: next(active_positions(pcm, slot+"-"+arm), None) for arm, pcm in variants}
            require(all(x is not None for x in ranges.values()), f"A frozen role source cannot supply an active edit: {slot}")
            a, noise, background = target_and_room(base, slot, room, rooms, blocked)
            b = a+EDIT
            noise *= np.linalg.norm(base*32768)*10**.25/np.linalg.norm(noise)
            variants = {arm: prior.inputs.splice(base, pcm, a, ranges[arm], EDIT)*32768 for arm, pcm in variants}
            gain = min(1., 31_129/max(float(np.max(abs(pcm)+abs(noise))) for pcm in variants.values()))
            residual = np.rint(gain*noise)
            background_pcm = prior.long_room.pcm(residual)
            for arm, pcm in variants.items():
                foreground = np.rint(gain*pcm)
                original = prior.long_room.pcm(foreground+residual)
                foreground_pcm = prior.long_room.pcm(foreground)
                output.append(({"id": f"{slot}-{arm}", "group": slot, "role": role, "arm": arm,
                                "family": "clone" if arm == "synthetic" else None, "edit": [a, b], "truth": [a, b] if arm == "synthetic" else None,
                                "target_reference": row["enrollment"]["original_sha256"], "sham_reference": sham["enrollment"]["original_sha256"],
                                "background": background, "donor_start_sample": ranges[arm], "host_start_sample": 2*RATE,
                                "original_pcm_sha256": digest(original), "foreground_pcm_sha256": digest(foreground_pcm),
                                "background_pcm_sha256": digest(background_pcm)}, original, foreground_pcm, background_pcm))
    require(len(output) == 42 and len(references) == 18, "Same-speaker acoustic inputs are incomplete")
    return output, references, old["generic"]


def freeze(run):
    views, references, generic = render()
    run.mkdir(mode=0o700)
    (run / "references").mkdir()
    (run / "pcm").mkdir()
    for pin, data in references.items():
        (run / "references" / f"{pin}.pcm").write_bytes(data)
    for row, *data in views:
        for arm, raw in zip(("original", "foreground", "background"), data, strict=True):
            (run / "pcm" / f"{row['id']}-{arm}.pcm").write_bytes(raw)
    write(run / "roster.json", [row for row, *_ in views])
    write(run / "freeze.json", {"schema": SCHEMA, "method": METHOD, "runner_sha256": sha(Path(__file__)),
          "prior_runner_sha256": sha(Path(prior.__file__)), "source_receipt_sha256": sha(SOURCE / "result.json"),
          "roster_sha256": sha(run / "roster.json"), "generic": generic, "references": {key: digest(data) for key, data in references.items()}})
    print("Frozen 42 new exact matched waveforms and 18 enrollment inputs", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "measure", "audit"))
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run
    if args.phase == "freeze":
        freeze(run)
        return
    lock, rows = read(run / "freeze.json"), read(run / "roster.json")
    require((lock["runner_sha256"], lock["prior_runner_sha256"], lock["source_receipt_sha256"], lock["roster_sha256"])
            == tuple(sha(path) for path in (Path(__file__), Path(prior.__file__), SOURCE / "result.json", run / "roster.json")), "Acoustic inputs changed")
    started = time.monotonic()
    observed = prior.measure(args, lock, rows)
    result = prior.summarize(rows, observed)
    if args.phase == "measure":
        write(run / "measurements.json", observed)
        write(run / "result.json", {"schema": SCHEMA, "method": METHOD, **result,
              "commitments": {key: sha(run / f"{key}.json") for key in ("freeze", "roster", "measurements")},
              "seconds": round(time.monotonic()-started, 3)})
        print(result["acoustic_gate"], flush=True)
    else:
        expected = read(run / "result.json")
        require(read(run / "measurements.json") == observed and all(expected[key] == val for key, val in result.items()), "Acoustic readback differs")
        views, references, generic = render()
        require([row for row, *_ in views] == rows and {key: digest(raw) for key, raw in references.items()} == lock["references"]
                and lock["generic"] == generic, "Source mixture or reference readback differs")
        print(f"Reproduced {len(rows)} native inputs, {len(references)} enrollments, and acoustic gates", flush=True)


if __name__ == "__main__":
    main()
