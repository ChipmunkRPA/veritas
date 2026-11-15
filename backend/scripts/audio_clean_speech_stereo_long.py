# SPDX-License-Identifier: Apache-2.0
"""Test separate stereo channels against an earlier copy under competing speech."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from app.evidence import audio_clean_speech_stereo as channel_check
from scripts import audio_clean_speech_aac_long_transfer as prior

select = channel_check.select
overlap = channel_check.overlap
old = prior.previous
SCHEMA = "chipmunk.audio-clean-speech-stereo-long/1"
BASE = old.PANELS / "audio-clean-speech-aac-long-2026-11-15.json.gz"
CARRIERS = ("retained_stereo", "asymmetric_stereo")
ARMS = ("downmix", "left", "right", "agreement", "reference_selected")
RANGE_SAMPLES = 11_520
CLOCK = {"sample_rate": 48000, "channels": 2, "time_base": "1/48000",
         "presentation_end_exclusive_pts": 3 * prior.SAMPLES}
REASONS = {"reference_match_not_supported", "consistent_frequency_balance_change", "local_change",
           "local_support_not_available", "no_corroborated_local_change"}
METHOD = {
    "clock": "Twelve seconds of verified native 48 kHz AAC-LC stereo. Read both channels and the default mono "
             "downmix at the same 16 kHz source-tick clock. Discard samples beyond the declared presentation.",
    "channel": "Use the existing clean-earlier-copy comparison on the downmix and both native channels. Prefer one "
               "channel only if it supports comparison in both halves and has at least .04 higher reference "
               "correlation on both halves. Otherwise require both channels to flag overlapping ranges.",
    "agreement": "Agreement requires both native channels to flag ranges with at least half their duration in common. "
                 "Keep the input range from the channel with the lower start tick, using the downmix only to break "
                 "a tie when it supports that same start.",
    "layout": "Replay the retained stereo delivery and a second stereo mix adding different recorded speech to one "
              "channel. Alternate the contaminated channel across source groups in sorted order.",
}


def stereo_pcm(path, stereo_bytes, expected_hash=None):
    from app.detectors.audio.decode import aac48_presentation, decode_audio

    if stereo_bytes is not None:
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "s16le", "-ar", "16000",
                        "-ac", "2", "-i", "pipe:0", "-ar", "48000", "-map_metadata", "-1", "-threads", "1",
                        "-c:a", "aac", "-profile:a", "aac_low", "-b:a", "96k", "-fflags", "+bitexact",
                        "-flags:a", "+bitexact", str(path)], input=stereo_bytes, check=True,
                       stdout=subprocess.DEVNULL, timeout=25)
    pin = old.sha(path.read_bytes())
    old.require(expected_hash is None or pin == expected_hash, "Retained AAC delivery changed")
    clock = aac48_presentation(path)
    old.require(clock == CLOCK, "Stereo presentation clock changed")
    decoded = decode_audio(path, source_sha256=pin, max_seconds=13, retain_stereo=True)
    extent = len(decoded.pcm)
    old.require(decoded.stereo_pcm is not None and decoded.stereo_pcm.shape == (extent, 2)
                and 0 <= extent - prior.SAMPLES <= 342 and decoded.mapping["timeline"] == [
                    {"start_sample": 0, "end_sample": extent, "source_pts": 0}], "Stereo decoded samples differ")
    views = {"downmix": (decoded.pcm[:prior.SAMPLES] * 32768).astype("<i2")}
    views["left"], views["right"] = decoded.stereo_pcm[:prior.SAMPLES].T
    return views, {"encoded_sha256": pin, "decoded_samples": extent, "native_clock": clock}


def asymmetric(submitted, interference, *, reverse):
    import numpy as np

    native, borrowed = (np.frombuffer(pcm, dtype="<i2").astype(np.float64) for pcm in (submitted, interference))
    level = np.linalg.norm(native) / max(1., np.linalg.norm(borrowed))
    weak = .35 * native + 1.20 * level * borrowed
    weak *= min(1., 32760 / max(1., np.max(np.abs(weak))))
    channels = (weak, native) if reverse else (native, weak)
    return np.rint(np.column_stack(channels)).astype("<i2").tobytes()


def check(first, views):
    import numpy as np

    earlier = np.frombuffer(first, dtype="<i2")
    _, readings, support = channel_check.observe(earlier, views)
    for name, pcm in views.items():
        readings[name]["pcm_sha256"] = old.sha(pcm.astype("<i2", copy=False).tobytes())
    return {"readings": readings, "support": support, "selected": select(readings, support)}


def outcomes(rows):
    report = defaultdict(Counter)
    for row in rows:
        role = prior.category(row["kind"])
        for carrier, entry in row["carriers"].items():
            old_result = entry["readings"]["downmix"]
            base_hit = old.location(old_result["range"], row["truth"])
            for arm in ARMS:
                channel = arm if arm in entry["readings"] else entry["selected"][arm]
                found = entry["readings"][channel]["range"] if channel else None
                useful = old.location(found, row["truth"])
                report[f"{carrier}/{arm}/{role}"].update(
                    files=1, flagged=bool(found), useful=useful,
                    gained=useful and not base_hit, lost=base_hit and not useful,
                    late_useful=useful and row["half"] == "late",
                    reference_admitted=bool(channel and entry["readings"][channel]["reason"] != "reference_match_not_supported"),
                )
    return dict(report)


def corroboration_diagnostic(rows):
    """Probe the other channel on this already-scored panel."""
    result = {}
    for carrier in CARRIERS:
        by_point = {}
        for point in (.05, .10, .15, .20, .25, .30):
            counts = Counter()
            for row in rows:
                entry = row["carriers"][carrier]
                choice = entry["selected"]["reference_selected"]
                if choice in ("left", "right"):
                    other = "right" if choice == "left" else "left"
                    if any((part["upper_correlation"] or 0.) < point for part in entry["support"][other]):
                        choice = entry["selected"]["agreement"]
                if choice:
                    reading = entry["readings"][choice]
                    span = reading["range"]
                    kind = prior.category(row["kind"])
                    counts.update({f"{kind}/useful": old.location(span, row["truth"]),
                                   f"{kind}/flagged": bool(span),
                                   f"{row['kind']}/admitted": reading["reason"] != "reference_match_not_supported"})
            by_point[f"{point:.2f}"] = dict(counts)
        result[carrier] = by_point
    return result


def baseline():
    receipt = json.loads(gzip.decompress(BASE.read_bytes()))
    prior.review(receipt)
    return {(r["bank"], r["group"], r["kind"]): r for r in receipt["rows"]}


def run(icsi, ami):
    known = baseline()
    rows = []
    with TemporaryDirectory(prefix="chipmunk-stereo-reference-") as folder:
        path = Path(folder) / "submitted.m4a"
        for bank, source in (("icsi", icsi), ("ami", ami)):
            joined = list(prior.pairs(bank, source))
            interferers = {row["group"]: pcm[0] for row, pcm in joined if row["kind"] == "unrelated"}
            order = {group: slot for slot, group in enumerate(sorted(interferers))}
            for row, (first, submitted) in joined:
                group = row["group"]
                check_row = known[bank, group, row["kind"]]
                old.require((old.sha(first), old.sha(submitted)) == (
                    check_row["earlier_sha256"], check_row["submitted_sha256"]), "Earlier or submitted input changed")
                old.encode48(submitted, path, "aac48_stereo")
                retained = check_row["readings"]["aac48_stereo"]
                first_views, first_clock = stereo_pcm(path, None, retained["encoded_sha256"])
                second_views, second_clock = stereo_pcm(path, asymmetric(submitted, interferers[group], reverse=bool(order[group] % 2)))
                first_check, second_check = check(first, first_views), check(first, second_views)
                old.require(all(first_check["readings"]["downmix"][key] == retained[key] for key in ("range", "reason")),
                            "Existing mono comparison changed")
                rows.append(row | {"bank": bank, "carriers": {
                    "retained_stereo": first_check | first_clock,
                    "asymmetric_stereo": second_check | second_clock,
                }})
                if len(rows) % 20 == 0:
                    print(f"Compared both channels of {len(rows)} stereo pairs.", flush=True)
    return {"schema": SCHEMA, "method": METHOD, "prior_sha256": old.sha(BASE.read_bytes()),
            "rows": rows, "outcomes": outcomes(rows)}


def review(document):
    rows = document["rows"]
    if document["schema"] != SCHEMA or document["method"] != METHOD or document["prior_sha256"] != old.sha(BASE.read_bytes()):
        raise ValueError("The stereo comparison refers to a different trial")
    known = baseline()
    keys = {(r["bank"], r["group"], r["kind"]) for r in rows}
    if len(keys) != len(rows) or keys != known.keys():
        raise ValueError("The held recordings changed")
    for row in rows:
        source = known[row["bank"], row["group"], row["kind"]]
        if any(row[key] != source[key] for key in ("companion", "half", "truth")):
            raise ValueError("The known edit or companion moved")
        if set(row["carriers"]) != set(CARRIERS):
            raise ValueError("A stereo layout is missing")
        retained = row["carriers"]["retained_stereo"]
        prior_delivery = source["readings"]["aac48_stereo"]
        if (retained["encoded_sha256"] != prior_delivery["encoded_sha256"]
                or any(retained["readings"]["downmix"][key] != prior_delivery[key] for key in ("range", "reason"))):
            raise ValueError("The prior stereo delivery or downmix decision changed")
        for entry in row["carriers"].values():
            if (entry["native_clock"] != CLOCK or not 0 <= entry["decoded_samples"] - prior.SAMPLES <= 342
                    or not re.fullmatch(r"[0-9a-f]{64}", entry["encoded_sha256"])):
                raise ValueError("The stereo delivery lacks a verified playback clock")
            readings, support = entry["readings"], entry["support"]
            if set(readings) != {"downmix", "left", "right"} or set(support) != {"left", "right"}:
                raise ValueError("A stereo measurement is missing")
            for reading in readings.values():
                span = reading["range"]
                if (not re.fullmatch(r"[0-9a-f]{64}", reading["pcm_sha256"]) or reading["reason"] not in REASONS
                        or (reading["reason"] == "local_change") != (span is not None)):
                    raise ValueError("Invalid stereo range disposition")
                if span is not None and (len(span) != 2 or any(type(tick) is not int for tick in span)
                                         or not 0 <= span[0] < span[1] <= prior.SAMPLES
                                         or span[1] - span[0] != RANGE_SAMPLES):
                    raise ValueError("A stereo range is outside the frozen grid")
            for halves in support.values():
                if len(halves) != 2:
                    raise ValueError("Reference support must cover both halves")
                for half in halves:
                    upper = half["upper_correlation"]
                    active, matching = half["active_windows"], half["matching_windows"]
                    if (half["checked_windows"] != 73 or not 0 <= matching <= active <= 73
                            or (upper is None) != (active == 0) or (upper is not None and not -1 <= upper <= 1)
                            or type(half["admitted"]) is not bool
                            or (half["admitted"] and not (active >= 36 and matching >= 18 and upper >= .30))):
                        raise ValueError("Invalid channel support for the reference")
            if entry["selected"] != select(readings, support):
                raise ValueError("A selected interval lacks its native channel witness")
    actual = outcomes(rows)
    if actual != document["outcomes"]:
        raise ValueError("The held measurements do not reproduce")
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--icsi", type=Path)
    parser.add_argument("--ami", type=Path)
    parser.add_argument("--diagnostic", action="store_true", help="Recount the later two-channel support probe")
    args = parser.parse_args()
    if args.icsi and args.ami:
        receipt = run(args.icsi, args.ami)
        review(receipt)
        with args.receipt.open("xb") as out:
            out.write(gzip.compress(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode(), mtime=0))
    elif not args.icsi and not args.ami:
        receipt = json.loads(gzip.decompress(args.receipt.read_bytes()))
    else:
        parser.error("Provide both pinned source banks for a native run")
    results = review(receipt)
    print(json.dumps(corroboration_diagnostic(receipt["rows"]) if args.diagnostic else results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
