import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.acquire_genuine_parquet_v8 import PinnedRows


def corpus(tmp_path):
    path = tmp_path / "original.parquet"
    rows = [
        {
            "id": f"speaker-{i}",
            "speaker_id": i,
            "text": f"text {i}",
            "audio": {"bytes": f"original-{i}".encode(), "path": "ignored-source-path"},
        }
        for i in range(3)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=1)
    index = [
        {
            "id": r["id"],
            "speaker_id": r["speaker_id"],
            "text": r["text"],
            "row_index": i + 100,
            "parquet": "original.parquet",
        }
        for i, r in enumerate(rows)
    ]
    return PinnedRows({"original.parquet": path}, index), index


def test_original_row_group_boundaries_and_global_offsets_preserve_identity(tmp_path):
    reader, index = corpus(tmp_path)
    for i in (2, 0, 1):
        raw, origin = reader.get(index[i])
        assert raw == f"original-{i}".encode()
        assert origin["parquet_row_group"] == i
        assert origin["parquet_group_row"] == 0
        assert origin["parquet_local_row"] == i


def test_wrong_source_speaker_cannot_pass_original_corpus_lookup(tmp_path):
    reader, index = corpus(tmp_path)
    with pytest.raises(ValueError, match="identity/speaker/text"):
        reader.get({**index[0], "speaker_id": 99})


def test_out_of_range_global_offset_is_refused(tmp_path):
    reader, index = corpus(tmp_path)
    with pytest.raises(ValueError, match="outside original file"):
        reader.get({**index[0], "row_index": 103})
