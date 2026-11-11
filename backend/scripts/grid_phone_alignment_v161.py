"""Evaluate isolated PocketSphinx word and phone support on the existing GRID pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import wave
import zipfile
from collections import Counter
from fractions import Fraction
from pathlib import Path
from statistics import median

from scripts import acquire_grid_transfer_v130 as acquisition
from scripts import grid_transfer_plan_v143 as planning

ROOT, BASE, PANEL = planning.ROOT, planning.BASE, planning.PANEL
ASSETS = BASE / "articulation-timing-assets-v161"
TRIAL = BASE / "grid-phone-alignment-v161"
PUBLIC = PANEL / "grid-phone-alignment-freeze-v161.json"
WHEEL = ASSETS / "pocketsphinx-5.0.4-cp312-cp312-macosx_10_13_universal2.whl"
WHEEL_SHA = "833d0ba73affdec98883e0828dbacb69ab4932f4e739062076ed679e54878f21"
ALIGNMENTS = ASSETS / "alignments.zip"
SITE = ASSETS / "site"
SOURCE_RATE, PCM_RATE, FEATURE_RATE = 25_000, 16_000, 100
NONWORDS = frozenset({"sil", "sp", "<sil>", "<s>", "</s>"})
read, write, sha, require = planning.read, planning.write, planning.sha, planning.require


def assets():
    publisher_path = acquisition.METADATA / "grid-record.json"
    require(sha(publisher_path) == acquisition.RECORD_SHA, "Exact publisher metadata required")
    publisher = read(publisher_path)
    item = next(item for item in publisher["files"] if item["key"] == "alignments.zip")
    require(publisher["metadata"]["license"]["id"] == "cc-by-4.0", "Publisher data licence required")
    with ALIGNMENTS.open("rb") as stream:
        actual_md5 = hashlib.file_digest(stream, lambda: hashlib.md5(usedforsecurity=False)).hexdigest()
    require(
        ALIGNMENTS.stat().st_size == item["size"] and item["checksum"] == f"md5:{actual_md5}",
        "Exact publisher word-alignment archive required",
    )
    require(sha(WHEEL) == WHEEL_SHA, "Exact publisher wheel required")
    with zipfile.ZipFile(WHEEL) as wheel:
        payload = {n for n in wheel.namelist() if n.startswith("pocketsphinx/") and not n.endswith("/")}
        installed = {str(p.relative_to(SITE)) for p in (SITE / "pocketsphinx").rglob("*") if p.is_file()}
        require(installed == payload, "Isolated package must have the complete exact wheel payload")
        require(
            all(
                (SITE / n).resolve().is_relative_to(SITE) and (SITE / n).read_bytes() == wheel.read(n)
                for n in payload
            ),
            "Installed package bytes differ from the reviewed wheel",
        )
    return {
        "wheel_sha256": WHEEL_SHA,
        "publisher_word_alignments_sha256": sha(ALIGNMENTS),
        "publisher_metadata_sha256": acquisition.RECORD_SHA,
    }


def publisher_words(payload):
    bounds = []
    for line in payload.decode("ascii").splitlines():
        start, end, name = line.split()
        start, end = int(start), int(end)
        require(0 <= start < end and (not bounds or start == bounds[-1][1]), "Invalid publisher interval")
        bounds.append((start, end, name.lower()))
    words = [(name, start, end) for start, end, name in bounds if name not in NONWORDS]
    require(len(words) == 6 and bounds[0][0] == 0, "Expected six-word GRID prompt")
    return words, bounds[-1][1]


def selected_sources():
    sources = [s for s in planning.load_completed()["sources"] if s["role"] == "pilot"]
    require(len(sources) == 16, "All sixteen existing pilot originals required")
    out = []
    with zipfile.ZipFile(ALIGNMENTS) as archive:
        names = Counter(archive.namelist())
        for source in sources:
            row = source["input"]
            plan = row["planned"]
            speaker = Path(plan["source"]["publisher_name"]).stem
            stem = Path(plan["member"]["name"]).stem
            require(
                re.fullmatch(r"s\d{1,2}", speaker) and re.fullmatch(r"[a-z0-9]+", stem),
                "Invalid source identity",
            )
            name = f"alignments/{speaker}/{stem}.align"
            if not names[name]:
                out.append({"unavailable": "publisher_word_reference_unmatched"})
                continue
            require(
                names[name] == 1 and archive.getinfo(name).file_size <= 2048,
                "Unique word-alignment member required",
            )
            words, duration = publisher_words(archive.read(name))
            path = Path(row["wav_file"])
            require(
                path.resolve().is_relative_to(planning.ARCHIVE) and sha(path) == row["wav_file_sha256"],
                "Qualified whole original WAV required",
            )
            require(row["audio_origin_seconds"] == "0", "Expected qualified zero audio origin")
            out.append(
                {
                    "wav_path": str(path),
                    "wav_sha256": row["wav_file_sha256"],
                    "words": words,
                    "publisher_end_sample": duration,
                }
            )
    return out


def identity():
    return {
        "schema": "chipmunk.grid-phone-alignment-freeze/161",
        "implementation_sha256": sha(Path(__file__)),
        "method_sha256": sha(ROOT / "docs/grid-phone-alignment-v161-method.md"),
        "private_inputs_sha256": sha(TRIAL / "private-inputs.json"),
        "source_plan_sha256": sha(planning.PUBLIC),
        "source_rate": SOURCE_RATE,
        "pcm_rate": PCM_RATE,
        "feature_frame_rate": FEATURE_RATE,
        "host": [platform.system(), platform.machine(), platform.python_version()],
        **assets(),
    }


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve prior attempts")
    assets()
    inputs = selected_sources()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "private-inputs.json", inputs)
    frozen = identity()
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)


def word_edits(reference, observed):
    previous = list(range(len(observed) + 1))
    for i, token in enumerate(reference, 1):
        current = [i]
        for j, other in enumerate(observed, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (token != other)))
        previous = current
    return previous[-1]


def decode(decoder, pcm):
    decoder.start_utt()
    decoder.process_raw(pcm, no_search=False, full_utt=True)
    decoder.end_utt()


def normalized(name):
    return re.sub(r"\(\d+\)$", "", name.lower())


def phoneme_row(phone, sample_count):
    # These are acoustic model frame boundaries, not independently measured release samples.
    start = phone.start * (PCM_RATE // FEATURE_RATE)
    end = (phone.start + phone.duration) * (PCM_RATE // FEATURE_RATE)
    require(0 <= start < end <= sample_count, "Phone alignment falls outside the admitted PCM")
    return {"phone": phone.name.upper(), "sample_range": [start, end], "release_sample": None}


def assess(row, create_decoder):
    path = Path(row["wav_path"])
    require(sha(path) == row["wav_sha256"], "Input WAV bytes changed")
    with wave.open(str(path), "rb") as reader:
        require(
            (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) == (1, 2, PCM_RATE),
            "Exact mono signed 16-bit 16 kHz PCM required",
        )
        samples = reader.getnframes()
        pcm = reader.readframes(samples)
    duration_error = abs(Fraction(samples, PCM_RATE) - Fraction(row["publisher_end_sample"], SOURCE_RATE))
    require(len(pcm) == 2 * samples and duration_error <= Fraction(1, 25), "Publisher and PCM extent differ")
    expected = [token for token, _, _ in row["words"]]
    blind = create_decoder()
    decode(blind, pcm)
    hypothesis = blind.hyp()
    words = [normalized(word) for word in hypothesis.hypstr.split()] if hypothesis else []
    result = {
        "blind_exact_words": words == expected,
        "word_edits": word_edits(expected, words),
        "words": len(expected),
    }
    try:
        result.update(align_words(row, create_decoder(), pcm, samples, expected))
    except (ValueError, RuntimeError) as error:
        result["forced_alignment_unavailable"] = type(error).__name__
    return result


def align_words(row, forced, pcm, samples, expected):
    forced.set_align_text(" ".join(expected))
    decode(forced, pcm)
    forced.set_alignment()
    decode(forced, pcm)
    alignment = [w for w in forced.get_alignment() if normalized(w.name) not in NONWORDS]
    require([normalized(w.name) for w in alignment] == expected, "Forced alignment lost an expected word")
    errors = []
    for (_, ref_start, ref_end), word in zip(row["words"], alignment, strict=True):
        errors.extend(
            (
                abs(Fraction(ref_start, SOURCE_RATE) - Fraction(word.start, FEATURE_RATE)),
                abs(Fraction(ref_end, SOURCE_RATE) - Fraction(word.start + word.duration, FEATURE_RATE)),
            )
        )
    phones = [phoneme_row(phone, samples) for word in alignment for phone in word]
    return {"word_boundary_abs_ms": [float(1000 * v) for v in errors], "phones": phones}


def run():
    frozen = identity()
    require(frozen == read(PUBLIC) == read(TRIAL / "freeze.json"), "Exact predeclared run required")
    (TRIAL / "run").mkdir(exist_ok=False)
    sys.path.insert(0, str(SITE))
    from pocketsphinx import Decoder

    model = SITE / "pocketsphinx/model/en-us"

    def create_decoder():
        return Decoder(
            hmm=str(model / "en-us"),
            dict=str(model / "cmudict-en-us.dict"),
            lm=str(model / "en-us.lm.bin"),
            samprate=PCM_RATE,
            frate=FEATURE_RATE,
            logfn="/dev/null",
        )

    observations = []
    journal = TRIAL / "run/observations.jsonl"
    with journal.open("x") as output:
        for row in read(TRIAL / "private-inputs.json"):
            observed = row
            if "unavailable" not in row:
                try:
                    observed = assess(row, create_decoder)
                except (ValueError, RuntimeError) as error:
                    observed = {"unavailable": type(error).__name__}
            observations.append(observed)
            output.write(json.dumps(observed, allow_nan=False) + "\n")
            output.flush()
    usable = [x for x in observations if "phones" in x]
    blind = [x for x in observations if "word_edits" in x]
    errors = [v for row in usable for v in row["word_boundary_abs_ms"]]
    result = {
        "schema": "chipmunk.grid-phone-alignment-result/161",
        "freeze_sha256": sha(PUBLIC),
        "private_observations_sha256": sha(journal),
        "planned_originals": len(observations),
        "publisher_word_reference_unmatched": sum(
            row.get("unavailable") == "publisher_word_reference_unmatched" for row in observations
        ),
        "other_unavailable": sum(
            "unavailable" in row and row["unavailable"] != "publisher_word_reference_unmatched"
            for row in observations
        ),
        "forced_alignment_completed": len(usable),
        "blind_prompts_assessed": len(blind),
        "blind_exact_prompts": sum(row["blind_exact_words"] for row in blind),
        "blind_word_edits": sum(row["word_edits"] for row in blind),
        "blind_reference_words": sum(row["words"] for row in blind),
        "word_boundaries_compared": len(errors),
        "median_abs_word_boundary_error_ms": median(errors) if errors else None,
        "maximum_abs_word_boundary_error_ms": max(errors, default=None),
        "candidate_bilabial_events": sum(
            p["phone"] in {"M", "B", "P"} for row in usable for p in row["phones"]
        ),
        "scope": "Word-level reference comparison and unqualified acoustic phone boundaries on original GRID pilot audio.",
    }
    write(TRIAL / "run/result.json", result)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    (freeze if args.action == "freeze" else run)()
