"""Different framehash clock bases must bind the same actual pictures and instants."""

import pytest

from scripts.review_video_native_inserts_v96 import parse_framehash


def packet(base="1/12800", pts=(0, 512, 1024), digests=("a" * 64, "b" * 64, "c" * 64)):
    header = f"#format: frame checksums\n#hash: SHA256\n#tb 0: {base}\n"
    return header + "\n".join(f"0, {p}, {p}, 512, 9216, {h}" for p, h in zip(pts, digests, strict=True))


def test_actual_sha256_and_rational_pts_are_independent_of_timebase_units():
    expected = ["a" * 64, "b" * 64, "c" * 64]
    assert parse_framehash(packet(), expected, "1/25", 64, 48) == expected
    assert parse_framehash(packet("1/25", (0, 1, 2)), expected, "1/25", 64, 48) == expected


@pytest.mark.parametrize(
    "changed",
    [
        packet(pts=(0, 512, 1025)),
        packet(digests=("a" * 64, "a" * 64, "c" * 64)),
        packet().replace("9216", "9215"),
        packet().replace("a" * 64, "a" * 32),
        packet().replace("#tb 0: 1/12800", ""),
    ],
)
def test_corrupt_time_picture_size_digest_or_missing_clock_cannot_pass(changed):
    with pytest.raises(ValueError):
        parse_framehash(changed, ["a" * 64, "b" * 64, "c" * 64], "1/25", 64, 48)
