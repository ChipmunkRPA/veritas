import base64
import csv
import hashlib
from fractions import Fraction
from types import SimpleNamespace

import httpx
import pytest

from scripts import video_vcd2_genuine_v195 as source


def test_index_rejects_inconsistent_publisher_paths(tmp_path):
    file = tmp_path / "index.csv"
    listing = tmp_path / "list.txt"
    stem = "a" * 32
    path = f"benchmarking/mkv/TH/{stem}.mkv"
    with file.open("w", newline="") as stream:
        rows = csv.writer(stream)
        rows.writerows([["relative_path", "group"], [path, "TH"]])
    listing.write_text(path + "\n")
    assert (
        source.checked_index(tmp_path, file.name, listing.name, source.BENCH_PATH, 1)[stem]["group"] == "TH"
    )
    listing.write_text(path.replace("/TH/", "/TH-BB/") + "\n")
    with pytest.raises(ValueError, match="file list disagree"):
        source.checked_index(tmp_path, file.name, listing.name, source.BENCH_PATH, 1)


def test_roster_rejects_altered_pinned_csv_before_reading_rows(tmp_path):
    first, _ = next(iter(source.PINS.items()))
    (tmp_path / first).write_text("relative_path\n")
    with pytest.raises(ValueError, match=f"Pin mismatch for {first}"):
        source.roster(tmp_path)


def table():
    # The first picture is at a nonzero absolute PTS. The clock changes step
    # inside the planned intervention and no index/FPS conversion is valid.
    pts = (13, 20, 29, 42, 48, 70, 74, 96, 106)
    return SimpleNamespace(
        pts=pts, tb=Fraction(1, 10), time_base="1/10", truncated=False, interpolated=(), rotation=0
    )


def test_intervention_uses_relative_picture_clock():
    probe = table()
    clock = source.clock_record(probe, [{"pts": value} for value in probe.pts])
    assert clock["ordinary_frame_interval"] == [4, 6]
    assert clock["ordinary_picture_pts"] == [48, 74]
    assert clock["actual_picture_span_s_relative"] == ["7/2", "61/10"]


@pytest.mark.parametrize("defect", ["interpolation", "packet", "duplicate"])
def test_clock_rejects_incomplete_provenance(defect):
    probe = table()
    packets = [{"pts": value} for value in probe.pts]
    if defect == "interpolation":
        probe.interpolated = (3,)
    elif defect == "packet":
        packets[4]["pts"] += 1
    else:
        probe.pts = probe.pts[:4] + (probe.pts[3],) + probe.pts[5:]
    with pytest.raises(ValueError, match="picture clock|picture PTS"):
        source.clock_record(probe, packets)


@pytest.mark.parametrize("defect", [None, "etag", "digest", "length"])
def test_transfer_binds_official_head_to_exact_local_bytes(tmp_path, monkeypatch, defect):
    body = b"Example test object"
    etag = '"test-object-version"'
    md5 = base64.b64encode(hashlib.md5(body, usedforsecurity=False).digest()).decode("ascii")
    headers = {
        "ETag": etag,
        "Content-Length": str(len(body)),
        "Content-Type": "application/octet-stream",
        "Content-MD5": md5 if defect != "digest" else "wrong",
    }
    url = source.AZURE + "test-metadata-only"

    def handle(request):
        assert str(request.url) == url
        if request.method == "HEAD":
            return httpx.Response(200, headers=headers)
        assert request.headers["If-Match"] == etag
        get_headers = {**headers, "ETag": '"different"' if defect == "etag" else etag}
        return httpx.Response(
            200, headers=get_headers, stream=httpx.ByteStream(body if defect != "length" else body[:-1])
        )

    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(handle), **kw))
    path = tmp_path / "object.partial"
    if defect:
        with pytest.raises(ValueError, match="HEAD object|Content-MD5|length changed"):
            source.download(url, path)
    else:
        result = source.download(url, path)
        assert path.read_bytes() == body and result["bytes"] == len(body)
        assert result["etag"] == etag and result["sha256"] == hashlib.sha256(body).hexdigest()
