"""Cache hits must preserve every requested byte and never weaken validation."""

import gzip
import hashlib
import io
import tarfile

import pytest

from scripts.ranged_gzip_cache_v21 import CachedRangeFile
from scripts.ranged_gzip_v21 import RangeFile, extract_part, sha


class Response:
    def __init__(self, data, start, stop, etag):
        self.status_code = 206
        self.headers = {"ETag": etag, "Content-Range": f"bytes {start}-{stop}/{len(data)}"}
        self.content = data[start:stop + 1]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, _size):
        yield self.content


class Session:
    def __init__(self, data, etag='"fixed"'):
        self.data, self.etag, self.calls = data, etag, 0

    def get(self, _url, **kwargs):
        start, stop = map(int, kwargs["headers"]["Range"].removeprefix("bytes=").split("-"))
        self.calls += 1
        return Response(self.data, start, stop, self.etag)


def make(cls, data, session):
    return cls("https://archive.example/file.gz", len(data), '"fixed"', io.StringIO(), block_size=1024, session=session)


def test_cached_backward_seeks_match_original_bytes_and_avoid_repeat_requests():
    data = bytes(range(256)) * 40
    first, second = Session(data), Session(data)
    plain, cached = make(RangeFile, data, first), make(CachedRangeFile, data, second)
    for offset in (0, 2048, 4096, 100, 2150, 4200, 100, 2048):
        plain.seek(offset)
        cached.seek(offset)
        assert plain.read(100) == cached.read(100) == data[offset:offset + 100]
    assert second.calls == 3 and first.calls > second.calls
    assert cached.cache_hits > 0


def test_cache_miss_still_rejects_changed_object_etag():
    data = b"a" * 4096
    value = make(CachedRangeFile, data, Session(data, '"changed"'))
    with pytest.raises(ValueError, match="ETag"):
        value.read(10)


def test_lru_memory_bound_does_not_lose_correct_bytes_after_eviction():
    data = bytes(range(256)) * 40
    value = make(CachedRangeFile, data, Session(data))
    value.cache_limit = 2048
    for offset in (0, 2048, 4096, 0):
        value.seek(offset)
        assert value.read(100) == data[offset:offset + 100]
        assert sum(len(v) for v in value.blocks.values()) <= 2048


def test_cached_reader_continues_an_original_reader_checkpoint_without_changing_members(tmp_path):
    import numpy as np

    source, expected = io.BytesIO(), {}
    rng = np.random.default_rng(217)
    with tarfile.open(fileobj=source, mode="w") as archive:
        for i in range(4):
            content = rng.integers(0, 256, 240000, dtype=np.uint8).tobytes()
            name = f"training/{i}.png"
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
            expected[name] = content
    data = gzip.compress(source.getvalue(), mtime=0)
    first = make(RangeFile, data, Session(data))
    checkpoint_dir = tmp_path / "first"
    extract_part(first, set(expected), tmp_path / "objects", checkpoint_dir, max_transfer=500000, spacing=65536)
    checkpoint = checkpoint_dir / "checkpoint.json"
    second = make(CachedRangeFile, data, Session(data))
    result = extract_part(second, set(expected), tmp_path / "objects", tmp_path / "second", previous=checkpoint, previous_digest=sha(checkpoint), spacing=65536)
    assert result["all_wanted_found"] and len(result["selected"]) == 4
    for content in expected.values():
        assert (tmp_path / "objects" / (hashlib.sha256(content).hexdigest() + ".png")).read_bytes() == content
