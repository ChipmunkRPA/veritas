"""Exact remote-object refusals and real partial gzip-index continuation."""

import gzip
import hashlib
import io
import json
import tarfile

import pytest

from scripts.ranged_gzip_v21 import RangeFile, extract_part, sha


class Response:
    def __init__(self, body, start, stop, total, fault=None):
        self.status_code = 200 if fault == "not_partial" else 206
        self.headers = {"ETag": '"fixed"', "Content-Range": f"bytes {start}-{stop}/{total}"}
        if fault == "etag":
            self.headers["ETag"] = '"changed"'
        elif fault == "range":
            self.headers["Content-Range"] = f"bytes {start + 1}-{stop}/{total}"
        elif fault == "encoding":
            self.headers["Content-Encoding"] = "gzip"
        self.body = body[:-1] if fault == "truncated" else body + b"x" if fault == "too_long" else body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, _size):
        yield self.body


class Session:
    def __init__(self, data, fault=None):
        self.data, self.fault, self.ranges = data, fault, []

    def get(self, _url, *, headers, **kwargs):
        assert headers["If-Match"] == '"fixed"' and kwargs["allow_redirects"] is False
        start, stop = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        self.ranges.append((start, stop))
        return Response(self.data[start:stop + 1], start, stop, len(self.data), self.fault)


def remote(data, *, fault=None, block_size=1024):
    return RangeFile("https://archive.example/data.gz", len(data), '"fixed"', io.StringIO(), block_size=block_size, session=Session(data, fault))


def test_ranges_cache_and_cross_block_reads_preserve_every_byte():
    data = bytes(range(256)) * 23
    value = remote(data)
    assert value.read(1030) == data[:1030]
    value.seek(1000)
    assert value.read(80) == data[1000:1080]
    value.seek(-17, 2)
    assert value.read(100) == data[-17:]
    assert value.read(1) == b""
    assert all(stop - start + 1 <= 1024 for start, stop in value.session.ranges)


@pytest.mark.parametrize("fault", ["not_partial", "etag", "range", "encoding", "truncated", "too_long"])
def test_changed_or_incomplete_remote_ranges_are_not_accepted(fault):
    value = remote(b"a" * 5000, fault=fault)
    with pytest.raises(ValueError):
        value.read(10)
    receipt = json.loads(value.journal.getvalue())
    assert receipt["complete"] is False


def test_unbounded_and_outside_object_reads_cannot_download_whole_archive():
    value = remote(b"a" * 5000)
    for size in (-1, 33 * 1024**2):
        with pytest.raises(ValueError):
            value.read(size)
    for offset in (-1, 5001):
        with pytest.raises(ValueError):
            value.seek(offset)
    assert value.session.ranges == []


def authored_archive():
    import numpy as np

    raw, content = io.BytesIO(), {}
    rng = np.random.default_rng(219)
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT, pax_headers={"comment": "authored checkpoint test"}) as archive:
        for n in range(12):
            name = "training/" + "long-category-" * 9 + f"/{n}.png"
            data = rng.integers(0, 256, 240000, dtype=np.uint8).tobytes()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 1000.25
            archive.addfile(info, io.BytesIO(data))
            content[name] = data
    return gzip.compress(raw.getvalue(), mtime=0), content


def test_real_partial_gzip_checkpoints_resume_tar_pax_and_all_selected_bytes(tmp_path):
    data, content = authored_archive()
    wanted, objects, previous = set(content), tmp_path / "objects", None
    complete, phase_count, last_offset, last_selected = False, 0, -1, 0
    while not complete:
        part = tmp_path / f"part-{phase_count:03d}"
        value = remote(data, block_size=65536)
        receipt = extract_part(value, wanted, objects, part, previous=previous, previous_digest=sha(previous) if previous else None, max_transfer=500000, spacing=65536)
        complete = receipt["all_wanted_found"]
        assert receipt["completed_phase"] and receipt["headers_this_phase"] > 0
        assert receipt["gzip_crc_validation_disabled_after_index_import"] is (previous is not None)
        assert receipt["next_tar_offset"] > last_offset and len(receipt["selected"]) > last_selected
        last_offset, last_selected = receipt["next_tar_offset"], len(receipt["selected"])
        previous = part / "checkpoint.json"
        phase_count += 1
        assert phase_count <= len(content)
    assert phase_count > 1
    for name, expected in content.items():
        path = objects / (hashlib.sha256(expected).hexdigest() + ".png")
        assert path.read_bytes() == expected, name
    assert len(receipt["selected"]) == len(content)


def test_modified_checkpoint_is_refused_before_resume(tmp_path):
    data, content = authored_archive()
    first = tmp_path / "first"
    extract_part(remote(data, block_size=65536), set(content), tmp_path / "objects", first, max_transfer=500000, spacing=65536)
    checkpoint = first / "checkpoint.json"
    committed = sha(checkpoint)
    checkpoint.write_text(checkpoint.read_text() + " ")
    with pytest.raises(ValueError, match="committed checkpoint"):
        extract_part(remote(data), set(content), tmp_path / "objects", tmp_path / "second", previous=checkpoint, previous_digest=committed)
