# SPDX-License-Identifier: Apache-2.0
"""Qualify local blind phone identity and audio release coordinates on retained native video."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

from app.detectors.audio.decode import decode_audio
from app.detectors.video.decode import probe_frames
from scripts import grid_phone_alignment_v161 as prior
from scripts import grid_phone_alignment_v162 as safe
from scripts import grid_reserve_word_clocks as grid
from scripts import grid_single_clip_release_timing as raw
from scripts import review_video_lipsync_local_delay_transfer as audio
from scripts.video_common_sources_v89 import read, require, sha, write

BASE = grid.planning.BASE
RATE, BIN = 16000, 40
NATIVE = BASE / "video-native-sources-v92"
CREMA = BASE / "video-common-acquisition-v91/asset-plan.json"
GRID = BASE / "grid-reserve-word-clocks-2026-11-12"
PAIRED = BASE / "grid-paired-closure-panel-2026-11-12"
PHONE = prior.SITE / "pocketsphinx/model/en-us"
LEXICAL = {
    "MTI": ("maybe tomorrow it will be cold", ("B", "B")),
    "ITS": ("i think i've seen this before", ("B",)),
    "IWL": ("i would like a new alarm clock", ()),
    "TIE": ("that is exactly what happened", ("P",)),
    "IEO": ("it's eleven o'clock", ()),
    "ITH": ("i think i have a doctor's appointment", ("P",)),
    "DFA": ("don't forget a jacket", ()),
}
INITIAL = {"bin": "B", "blue": "B", "by": "B", "b": "B", "place": "P", "please": "P"}
METHOD = {
    "identity": "Compare blind all-phone output B/P order with published full CREMA-D phrase classes. Known transcript is a ceiling and cannot establish blind identity.",
    "anchors": "GRID publisher initial B/P word windows only. A first-difference impulse peak and independent positive log-spectral flux peak must agree within 15 ms. Both must stand out from competing peaks. These are signal landmarks with 25 ms uncertainty, without manual phoneme-release truth.",
    "raw_comparator": "Frozen high-pass untyped transients from the preceding single-file evaluation.",
    "timed_comparators": "Known-transcript and blind allphone B/P interval ends within 50 ms of the presealed raw landmark. Interval ends are model coordinates, not independently established releases.",
    "scope": "Verify native PTS and canonical PCM. For an existing audio-only shift copy, only map a source anchor if the full uncertainty slice is exactly copied and both video clocks still agree.",
    "gate": {
        "minimum_positive_exact_phrases": 4,
        "maximum_negative_control_flags": 1,
        "minimum_reference_anchors": 5,
        "minimum_blind_timed_anchors": 3,
        "minimum_blind_shifted_anchors": 2,
    },
}


def pcm_record(path, digest, *, lexical=False):
    if lexical:
        # For phrase identity, use PCM in decode order. FLV packet times are rounded to milliseconds.
        decoded = decode_audio(path, source_sha256=digest, max_seconds=8)
        pcm = np.rint(decoded.pcm * 32768).astype("<i2")
        origin = None
    else:
        pcm, origin, overlap = audio.source_pcm(path, digest)
        require(not overlap, "A complete nonoverlapping canonical audio clock is required")
    table = probe_frames(path, timeout_s=15)
    require(
        not (table.rotation or table.interpolated or table.truncated), "A native picture clock is required"
    )
    times = [str(value * table.tb) for value in table.pts]
    require(
        all(Fraction(a) < Fraction(b) for a, b in zip(times, times[1:], strict=False)),
        "Picture clock must increase",
    )
    return pcm, {
        "file_sha256": digest,
        "pcm_sha256": audio.sha(pcm.tobytes()),
        "samples": len(pcm),
        "origin": str(origin) if origin is not None else None,
        "pts": list(table.pts),
        "time_base": table.time_base,
        "picture_times": times,
    }


def anchors(pcm, window):
    low, high = window
    start, stop = max(low - 320, 400), min(high - 160, low + 1920, len(pcm) - 400)
    if stop - start < 480:
        return None
    ticks = np.arange(start, stop, BIN)
    values = pcm.astype(float) / 32768
    edges = np.diff(values, prepend=0)
    impulses, fluxes = [], []
    taper = np.hanning(160)
    for tick in ticks:
        impulses.append(float(np.sqrt(np.mean(edges[tick : tick + BIN] ** 2))))
        before = np.abs(np.fft.rfft(values[tick - 160 : tick] * taper))
        after = np.abs(np.fft.rfft(values[tick : tick + 160] * taper))
        diff = np.maximum(np.log1p(20 * after[12:60]) - np.log1p(20 * before[12:60]), 0)
        fluxes.append(float(diff.mean()))

    def distinct_peak(series):
        index = int(np.argmax(series))
        primary = series[index]
        alternative = max((v for i, v in enumerate(series) if abs(i - index) >= 8), default=0)
        if primary < max(1.5 * float(np.median(series)), 1.12 * alternative, 1e-5):
            return None
        return int(ticks[index])

    impulse, flux = distinct_peak(impulses), distinct_peak(fluxes)
    if impulse is None or flux is None or abs(impulse - flux) > 240:
        return None
    point = (impulse + flux) // 2
    return {
        "signal_sample": point,
        "uncertainty_samples": 400,
        "impulse_sample": impulse,
        "flux_sample": flux,
        "native_pcm_sha256": audio.sha(pcm[point - 400 : point + 400].tobytes()),
    }


def binding():
    paths = [
        Path(__file__),
        Path(raw.__file__),
        Path(safe.__file__),
        Path(prior.__file__),
        prior.WHEEL,
        PHONE / "en-us-phone.lm.bin",
        PHONE / "en-us.lm.bin",
        PHONE / "cmudict-en-us.dict",
        *sorted((PHONE / "en-us").iterdir()),
        CREMA,
        NATIVE / "qualified-plan.json",
        GRID / "private-word-clocks.json",
        BASE / "grid-single-clip-release-2026-11-12/freeze.json",
        PAIRED / "freeze.json",
        PAIRED / "media.json",
    ]
    return {str(p.relative_to(grid.planning.ROOT)): sha(p) for p in paths}


def intake(run):
    run.mkdir(mode=0o700)
    rules = binding()
    require(
        rules[str(prior.WHEEL.relative_to(grid.planning.ROOT))] == prior.WHEEL_SHA,
        "Published PocketSphinx wheel checksum differs",
    )
    source, _ = grid.plan()
    record = read(CREMA)
    roles = [
        item
        for item in record["assets"]
        if item["source_family"] == "recorded_audiovisual_speech"
        and item["provisional_role"] == "confirmation"
    ]
    require(
        len(roles) == len({item["group"] for item in roles}) == 12, "Exact CREMA confirmation roster required"
    )
    cases = []
    for row in roles:
        file = Path(record["media_root"]) / row["file"]
        require(sha(file) == row["sha256"], "CREMA publisher hash differs")
        phrase, phones = LEXICAL[row["source_path"].split("_")[1]]
        _, receipt = pcm_record(file, row["sha256"], lexical=True)
        cases.append(
            {
                "bank": "crema_lexical",
                "id": row["index"],
                "group": row["group"],
                "file": str(file),
                "words": phrase.split(),
                "lexical_phones": phones,
                "native": receipt,
                "anchors": [],
            }
        )
    previous = read(BASE / "grid-single-clip-release-2026-11-12/freeze.json")
    roster = {item["ordinal"] for item in previous["held"]}
    paired = read(PAIRED / "media.json")
    require(sha(PAIRED / "freeze.json") == paired["freeze_sha256"], "Existing paired source roles differ")
    shifted = {row["ordinal"]: row for row in paired["cases"] if row["role"] == "audio_late_ordinary"}
    for row in read(GRID / "private-word-clocks.json")["records"]:
        ordinal = row["ordinal"]
        if ordinal not in roster:
            continue
        src = source[ordinal]["input"]
        file, digest = Path(src["source_file"]), src["source_sha256"]
        pcm, receipt = pcm_record(file, digest)
        require(
            receipt["origin"] == "0" and receipt["samples"] == 47648 and len(receipt["pts"]) == 75,
            "GRID native clocks differ",
        )
        expected = []
        for word in row["words"]:
            if word["word"] in INITIAL:
                window = word["source_pcm_samples"]
                expected.append(
                    {
                        "phone": INITIAL[word["word"]],
                        "word_window_samples": window,
                        "anchor": anchors(pcm, window),
                    }
                )
        info = {
            "bank": "grid_signal",
            "id": ordinal,
            "group": src["planned"]["source"]["publisher_name"],
            "file": str(file),
            "words": [item["word"] for item in row["words"]],
            "native": receipt,
            "anchors": expected,
        }
        cases.append(info)
        if ordinal not in shifted:
            continue
        edited = shifted[ordinal]
        newer, shifted_native = pcm_record(Path(edited["path"]), edited["sha256"])
        require(
            shifted_native["picture_times"] == receipt["picture_times"]
            and shifted_native["origin"] == receipt["origin"],
            "Paired clocks differ",
        )
        delta = edited["audio_shift"] * 640
        first, last = (k * 640 for k in edited["span"])
        require(
            np.array_equal(newer[first:last], pcm[first - delta : last - delta])
            and np.array_equal(newer[:first], pcm[:first])
            and np.array_equal(newer[last:], pcm[last:]),
            "Exact prior edit or copy support differs",
        )
        mapped = []
        for item in expected:
            anchor = item["anchor"]
            if anchor is None:
                continue
            point = anchor["signal_sample"] + delta
            pad = anchor["uncertainty_samples"]
            if first <= point - pad < point + pad <= last:
                require(
                    audio.sha(newer[point - pad : point + pad].tobytes()) == anchor["native_pcm_sha256"],
                    "Moved anchor waveform differs",
                )
                mapped.append({**item, "anchor": {**anchor, "signal_sample": point}, "shift_samples": delta})
        cases.append(
            {
                **info,
                "bank": "grid_shift",
                "file": edited["path"],
                "native": shifted_native,
                "anchors": mapped,
            }
        )
    require(
        Counter(item["bank"] for item in cases) == {"crema_lexical": 12, "grid_signal": 12, "grid_shift": 4},
        "Complete source bank required",
    )
    write(run / "freeze.json", {"method": METHOD, "bindings": rules, "roster": cases})
    print(
        "Frozen signal anchors",
        Counter(item["bank"] for item in cases for event in item["anchors"] if event["anchor"] is not None),
        flush=True,
    )


def decoder_factory():
    sys.path.insert(0, str(prior.SITE))
    from pocketsphinx import Decoder

    def build(blind):
        decoder = Decoder(
            hmm=str(PHONE / "en-us"),
            dict=str(PHONE / "cmudict-en-us.dict"),
            lm=None if blind else str(PHONE / "en-us.lm.bin"),
            samprate=RATE,
            frate=100,
            loglevel="ERROR",
        )
        if blind:
            decoder.add_allphone_file("held-phones", str(PHONE / "en-us-phone.lm.bin"))
            decoder.activate_search("held-phones")
        return decoder

    return build


def align(build, pcm, words, blind):
    decoder = build(blind)
    wire = pcm.tobytes()
    if blind:
        prior.decode(decoder, wire)
        rows = [
            {
                "phone": segment.word.upper(),
                "sample_range": [segment.start_frame * 160, (segment.end_frame + 1) * 160],
            }
            for segment in decoder.seg() or ()
            if segment.word.upper() in ("B", "P")
        ]
    else:
        decoder.set_align_text(" ".join(words))
        prior.decode(decoder, wire)
        decoder.set_alignment()
        prior.decode(decoder, wire)
        observed, phones = safe.copy_alignment(decoder.get_alignment(), len(pcm))
        require(
            [item[0] for item in observed] == words, "Known transcript alignment returned different words"
        )
        rows = [phone for phone in phones if phone["phone"] in ("B", "P")]
    require(
        all(0 <= a < b <= len(pcm) for a, b in (row["sample_range"] for row in rows)),
        "Phone coordinate escapes audio",
    )
    return rows


def score(run):
    lock = read(run / "freeze.json")
    require(
        lock["method"] == METHOD and lock["bindings"] == binding(), "Frozen method or source bindings changed"
    )
    make = decoder_factory()
    rows = []
    for item in lock["roster"]:
        pcm, receipt = pcm_record(
            Path(item["file"]), item["native"]["file_sha256"], lexical=item["bank"] == "crema_lexical"
        )
        require(receipt == item["native"], "Native data changed before scoring")
        measured = {}
        for label in ("known", "blind"):
            try:
                measured[label] = align(make, pcm, item["words"], blind=label == "blind")
            except (RuntimeError, ValueError) as error:
                measured[label] = {"unavailable": type(error).__name__}
        transients = raw.audio_candidates(pcm) if len(pcm) > 160 else []
        for event in item["anchors"]:
            point = (event["anchor"] or {}).get("signal_sample")
            if point is None:
                continue
            event["raw_nearest_error_samples"] = min(
                (abs(point - c["anchor"]) for c in transients), default=None
            )
            for label, observations in measured.items():
                matches = (
                    [r for r in observations if r["phone"] == event["phone"]]
                    if isinstance(observations, list)
                    else []
                )
                event[label + "_nearest_error_samples"] = min(
                    (abs(point - r["sample_range"][1]) for r in matches), default=None
                )
        result = {k: item[k] for k in ("id", "bank", "anchors")}
        if item["bank"] == "crema_lexical":
            result["lexical_phones"] = item["lexical_phones"]
        result["observed"] = measured
        rows.append(result)
        print(
            item["bank"],
            item["id"],
            {k: len(v) if isinstance(v, list) else None for k, v in measured.items()},
            flush=True,
        )
    write(run / "scores.json", {"freeze_sha256": sha(run / "freeze.json"), "rows": rows})


def report(run):
    scorecard = read(run / "scores.json")
    require(scorecard["freeze_sha256"] == sha(run / "freeze.json"), "Score freeze differs")
    lexical, timing = {}, {}
    for arm in ("known", "blind"):
        rows = [row for row in scorecard["rows"] if row["bank"] == "crema_lexical"]
        pair = [(row["lexical_phones"], row["observed"][arm]) for row in rows]
        lexical[arm] = {
            "measured": sum(isinstance(v, list) for _, v in pair),
            "positive_exact": sum(
                bool(w) and isinstance(v, list) and w == [x["phone"] for x in v] for w, v in pair
            ),
            "negative_nonempty": sum(not w and isinstance(v, list) and bool(v) for w, v in pair),
        }
    for bank in ("grid_signal", "grid_shift"):
        anchors = [item for row in scorecard["rows"] if row["bank"] == bank for item in row["anchors"]]
        usable = [item for item in anchors if item["anchor"] is not None]
        timing[bank] = {"lexical_windows": len(anchors), "signal_landmarks": len(usable)}
        for arm in ("raw", "known", "blind"):
            errors = [x[arm + "_nearest_error_samples"] for x in usable]
            timing[bank][arm] = {
                "anchors_with_nearby_event": sum(v is not None and v <= 800 for v in errors),
                "anchors_with_any_matching_event": sum(v is not None for v in errors),
            }
    gate = METHOD["gate"]
    checks = {
        "positive_identity": lexical["blind"]["positive_exact"] >= gate["minimum_positive_exact_phrases"],
        "negative_identity": lexical["blind"]["negative_nonempty"] <= gate["maximum_negative_control_flags"],
        "reference_coverage": timing["grid_signal"]["signal_landmarks"] >= gate["minimum_reference_anchors"],
        "timed_coverage": timing["grid_signal"]["blind"]["anchors_with_nearby_event"]
        >= gate["minimum_blind_timed_anchors"],
        "shifted_coverage": timing["grid_shift"]["blind"]["anchors_with_nearby_event"]
        >= gate["minimum_blind_shifted_anchors"],
    }
    write(
        run / "aggregate.json",
        {
            "method": METHOD,
            "lexical": lexical,
            "timing": timing,
            "gates": checks,
            "advances": all(checks.values()),
            "freeze_sha256": scorecard["freeze_sha256"],
            "scores_sha256": sha(run / "scores.json"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("intake", "score", "report"))
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    {"intake": intake, "score": score, "report": report}[args.phase](args.run)
