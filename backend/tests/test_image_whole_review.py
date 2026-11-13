from __future__ import annotations

import copy
import hashlib
import io
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from app.engine.customer_assessment import customer_assessment
from app.engine.detector import Profile
from app.engine.exhibits import ExhibitWriter
from app.engine.stages import image as image_stage
from app.evidence import image_png_layout
from app.evidence import image_whole_review as review
from app.storage.objects import ObjectStore

FIXTURE = json.loads((Path(__file__).parent / "fixtures/image_scope_cases_v50.json").read_text())


def _jpeg(path):
    Image.new("RGB", (1024, 1400)).save(path, format="JPEG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("case", ["flagged", "no_flag", "skipped", "bad_binding", "reclaimed", "still_low"])
def test_review_only_publishes_hash_bound_positive_crops(tmp_path, monkeypatch, case):
    original = tmp_path / "file.jpg"
    sha = _jpeg(original)
    fake_venv = tmp_path / "python"
    fake_venv.symlink_to(original)
    monkeypatch.setattr(review.settings, "image_whole_review_python", fake_venv)
    monkeypatch.setattr(review.settings, "data_dir", tmp_path)
    monkeypatch.setattr(review.settings, "models_dir", tmp_path)
    monkeypatch.setattr(review.settings, "min_free_bytes", 0)
    monkeypatch.setattr(review, "MEMORY_RESERVE", 1)
    memory = {"reclaimed": [0, 1], "still_low": [0, 0]}.get(case, [1])
    probe = iter(memory)
    monkeypatch.setattr(review.psutil, "virtual_memory", lambda: SimpleNamespace(available=next(probe)))
    events = []

    @contextmanager
    def lease(models, *, prefer, activation_bytes):
        assert models == [] and prefer == "cpu" and activation_bytes == 1
        events.append("lease")
        try:
            yield
        finally:
            events.append("released")

    monkeypatch.setattr(review, "get_model_manager", lambda: SimpleNamespace(lease=lease))

    def subprocess_result(args, **kwargs):
        assert args[0] == str(fake_venv) and kwargs["timeout"] == review.TIMEOUT_S
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert args[-3] == str(original)
        assert case not in {"skipped", "still_low"}
        events.append("worker")
        output = Path(args[-1])
        regions = []
        if case in {"flagged", "bad_binding"}:
            for number in range(1, 4):
                crop = output / f"crop-{number}.png"
                Image.new("RGB", (528, 528), (number, 0, 0)).save(crop)
                regions.append({"region": number, "bounds": [10, 25 * number, 500, 500],
                                "sha256": hashlib.sha256(crop.read_bytes()).hexdigest()})
        record = {"source_sha256": sha if case != "bad_binding" else "0" * 64, "checked_views": 3,
                  "flagged": bool(regions), "regions": regions}
        (output / "result.json").write_text(json.dumps(record))

    monkeypatch.setattr(review.subprocess, "run", subprocess_result)
    store = ObjectStore(tmp_path / "store")
    writer = ExhibitWriter("analysis", store).bind(review.CREATOR)
    ids = review.add_exhibits(original, sha, writer, remaining=0 if case == "skipped" else None)
    data = {"input": {"sha256": sha}, "exhibits": [x.model_dump(mode="json") for x in writer.exhibits if x.id in ids]}
    projected = review.from_result(data)
    assert projected["source_sha256"] == sha and projected["authorship_confidence_contribution"] == 0
    assert projected["status"] == {"bad_binding": "skipped", "reclaimed": "no_flag", "still_low": "skipped"}.get(case, case)
    if case in {"reclaimed", "still_low"}:
        assert events == (["lease", "worker", "released"] if case == "reclaimed" else ["lease", "released"])
    else:
        assert "lease" not in events
    assert len(projected["crops"]) == (3 if case == "flagged" else 0)
    for crop in projected["crops"]:
        assert hashlib.sha256(store.read_bytes(crop["sha256"])).hexdigest() == crop["sha256"]
        assert crop["bounds"] == {"x": 10, "y": 25 * crop["region"], "w": 500, "h": 500}
    data["input"]["sha256"] = "changed"
    assert review.from_result(data) is None


def test_conflicting_supplement_changes_only_public_headline():
    result = copy.deepcopy(FIXTURE["base"])
    result["signals"][1]["metrics"]["regions"] = 0
    source = "0" * 64
    result["input"] = {"sha256": source}
    table = {"created_by": "render", "title": review.TABLE, "kind": "table", "columns": review.COLUMNS}
    result["exhibits"] = [{**table, "rows": [["no_flag", "", ""]]}]
    original = customer_assessment(result)
    assert original["direction"] == "original" and original["confidence_pct"] == 97
    result["exhibits"] = [
        {**table, "rows": [["flagged", str(n), "10, 20, 400, 500"] for n in range(1, 4)]},
        *[{"created_by": "render", "kind": "image", "title": f"{review.CROP} {n}", "id": f"crop-{n}",
           "sha256": f"{n:064x}"} for n in range(1, 4)],
    ]
    stored = copy.deepcopy(result)
    assessment = customer_assessment(result)
    assert assessment["direction"] == "inconclusive" and assessment["confidence_pct"] is None
    assert assessment["headline"] == "Inconclusive: image checks disagree"
    assert assessment["component_estimate"]["confidence_pct"] == 97
    assert assessment["component_estimate"]["interval_pct"] == original["interval_pct"]
    assert result == stored


def test_configured_full_png_explicitly_reports_scope_skip(tmp_path, monkeypatch):
    store = ObjectStore(tmp_path / "objects")
    with io.BytesIO() as buffer:
        Image.new("RGB", (1024, 1024)).save(buffer, "PNG")
        source = store.put_bytes(buffer.getvalue())
    monkeypatch.setattr(review.settings, "image_whole_review_python", tmp_path / "not-needed")
    monkeypatch.setattr(review.settings, "image_whole_png_review_enabled", False)
    monkeypatch.setattr(review.settings, "environment", "dev")
    monkeypatch.setattr(image_stage, "_legacy_artifacts", lambda _: {})
    context = SimpleNamespace(analysis_id="png", exhibits=ExhibitWriter("png", store), store=store,
                              deadline=SimpleNamespace(remaining=lambda: None), profile=Profile.FULL, state={},
                              sessions=lambda: nullcontext(SimpleNamespace(commit=lambda: None)),
                              outputs={"ingest": {"mime": "image/png", "sha256": source.sha256}})
    rendered = image_stage.render(context)
    projected = review.from_result({**rendered, "input": {"sha256": source.sha256}})
    assert projected["status"] == "skipped" and projected["interpretation"] == review.SCOPE
    assert not projected["crops"] and not rendered["artefacts"]


@pytest.mark.parametrize("mode,expected", [("RGB", review.PNG_LAYOUT), ("RGBA", review.PNG_SCOPE)])
def test_opted_in_png_rejects_outside_scope_without_worker(tmp_path, monkeypatch, mode, expected):
    source = tmp_path / "layout.png"
    Image.new(mode, (1024, 1024)).save(source, "PNG")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(review.settings, "image_whole_review_python", source)
    monkeypatch.setattr(review.settings, "image_whole_png_review_enabled", True)
    monkeypatch.setattr(review.settings, "models_dir", tmp_path)
    monkeypatch.setattr(review.settings, "data_dir", tmp_path)
    monkeypatch.setattr(review.settings, "min_free_bytes", 0)
    monkeypatch.setattr(review, "MEMORY_RESERVE", 0)
    monkeypatch.setattr(review.subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected worker"))
    writer = ExhibitWriter("png", ObjectStore(tmp_path / "objects")).bind(review.CREATOR)
    ids = review.add_exhibits(source, sha, writer, remaining=240)
    result = {"input": {"sha256": sha}, "exhibits": [x.model_dump(mode="json") for x in writer.exhibits if x.id in ids]}
    projected = review.from_result(result)
    assert projected["status"] == "skipped" and projected["interpretation"] == expected
    assert not projected["crops"] and projected["authorship_confidence_contribution"] == 0


@pytest.mark.parametrize("width,repeats", [(25, image_png_layout.MAX_WORDS + 1), (4920, 2)])
def test_png_layout_bounds_recognized_word_work(width, repeats):
    header = b"level\tblock_num\tpar_num\tline_num\tleft\ttop\twidth\theight\tconf\ttext\n"
    word = f"5\t1\t1\t1\t0\t0\t{width}\t96\t99\tWord\n".encode()
    with pytest.raises(ValueError, match="word measurement exceeds its size limit"):
        image_png_layout._word_boxes(header + word * repeats)
    with pytest.raises(ValueError, match="output exceeds its size limit"):
        image_png_layout._word_boxes(b"x" * (image_png_layout.MAX_TSV_BYTES + 1))
