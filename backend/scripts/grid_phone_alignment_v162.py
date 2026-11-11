"""Copy native alignment fields before their owning iterator advances."""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

from scripts import grid_phone_alignment_v161 as earlier
from scripts.video_appearance_corrected_v112 import scoped

TRIAL = earlier.BASE / "grid-phone-alignment-v162"
PUBLIC = earlier.PANEL / "grid-phone-alignment-freeze-v162.json"
read, write, sha, require = earlier.read, earlier.write, earlier.sha, earlier.require
FAILED = earlier.TRIAL / "run/observations.jsonl"


def identity():
    original = earlier.identity()
    require(
        original == read(earlier.PUBLIC) == read(earlier.TRIAL / "freeze.json"), "Frozen v161 trial required"
    )
    require(
        FAILED.stat().st_size == 0 and not FAILED.with_name("result.json").exists(),
        "Retain failed native trial",
    )
    require(
        sha(TRIAL / "private-inputs.json") == original["private_inputs_sha256"],
        "Identical original roster required",
    )
    return {
        "schema": "chipmunk.grid-phone-alignment-freeze/162",
        "implementation_sha256": sha(Path(__file__)),
        "method_sha256": sha(earlier.ROOT / "docs/grid-phone-alignment-v162-method.md"),
        "regression_test_sha256": sha(earlier.ROOT / "backend/tests/test_grid_phone_alignment_v162.py"),
        "v161_freeze_sha256": sha(earlier.PUBLIC),
        "v161_empty_observation_sha256": sha(FAILED),
        "private_inputs_sha256": original["private_inputs_sha256"],
        "scope": "Exact earlier inputs and decoding, copy each word and its children during native iteration.",
    }


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve prior attempts")
    TRIAL.mkdir(exist_ok=False)
    (TRIAL / "private-inputs.json").write_bytes((earlier.TRIAL / "private-inputs.json").read_bytes())
    body = identity()
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)


def copy_alignment(alignment, samples):
    words, phones = [], []
    for word in alignment:
        name = earlier.normalized(word.name)
        if name in earlier.NONWORDS:
            continue
        words.append((name, word.start, word.start + word.duration))
        phones.extend(earlier.phoneme_row(phone, samples) for phone in word)
    return words, phones


def align_words(row, forced, pcm, samples, expected):
    forced.set_align_text(" ".join(expected))
    earlier.decode(forced, pcm)
    forced.set_alignment()
    earlier.decode(forced, pcm)
    words, phones = copy_alignment(forced.get_alignment(), samples)
    require([name for name, _, _ in words] == expected, "Forced alignment lost an expected word")
    errors = []
    for (_, ref_start, ref_end), (_, start, end) in zip(row["words"], words, strict=True):
        errors.extend(
            float(1000 * abs(Fraction(ref, earlier.SOURCE_RATE) - Fraction(observed, earlier.FEATURE_RATE)))
            for ref, observed in ((ref_start, start), (ref_end, end))
        )
    return {"word_boundary_abs_ms": errors, "phones": phones}


def run():
    def write_bound(path, result):
        result["schema"] = "chipmunk.grid-phone-alignment-result/162"
        write(path, result)

    scoped(
        earlier.run,
        {
            "TRIAL": TRIAL,
            "PUBLIC": PUBLIC,
            "identity": identity,
            "write": write_bound,
            "assess": scoped(earlier.assess, {"align_words": align_words}),
        },
    )()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    (freeze if args.action == "freeze" else run)()
