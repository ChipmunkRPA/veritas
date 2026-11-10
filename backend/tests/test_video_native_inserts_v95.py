"""Actual lossless RGB/clock round trips, source correspondence and bounded retime buffering."""

import hashlib
import json
import shutil
from fractions import Fraction

import numpy as np
import pytest

from app.detectors.video.decode import DecodedFrame, FrameTable
from scripts import video_insert_runtime_v95 as runtime
from scripts import video_native_inserts_v95 as inserts


def table(rate=Fraction(25), frames=100, origin=0):
    rate = Fraction(rate)
    return FrameTable(
        time_base=f"1/{rate.numerator}",
        pts=tuple(origin + i * rate.denominator for i in range(frames)),
        key_frames=(0,),
        pict_types="I" * frames,
        width=64,
        height=48,
        coded_width=64,
        coded_height=48,
    )


def test_observed_rational_clock_survives_equivalent_timebases_and_six_second_center_cap():
    first = table(Fraction(30000, 1001), 300, 3000)
    second = FrameTable.from_json(
        {**first.to_json(), "time_base": "1/60000", "pts": [2 * p for p in first.pts]}
    )
    recipe = inserts.recipe([first, second, first])
    assert recipe["frames"] == 180 and recipe["native_rate"] == "30000/1001"
    assert recipe["source_origin_seconds"] == "1/10" and recipe["edit_frames"] == [60, 120]
    assert recipe["edit_seconds"] == ["1001/500", "1001/250"]
    assert recipe["representative_frames"] == {"before": 59, "during": 89, "after": 120}


@pytest.mark.parametrize(
    "change",
    [{"width": 65}, {"pts": [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]}, {"truncated": True}, {"interpolated": [1]}],
)
def test_mismatched_or_inferred_native_support_is_refused(change):
    first = table(frames=10)
    other = FrameTable.from_json({**first.to_json(), **change})
    with pytest.raises(ValueError):
        inserts.recipe([first, other, first])


def test_insufficient_before_and_after_context_is_not_padded():
    short = table(frames=8)
    with pytest.raises(ValueError, match="support"):
        inserts.recipe([short] * 3)


def test_temporal_step_restores_exact_original_after_boundary_and_closes_source_reader(monkeypatch):
    source_table = table(frames=30)
    source_rgb = [np.full((48, 64, 3), i, np.uint8) for i in range(30)]
    original = {
        "file": "authored.mp4",
        "frame_table": source_table.to_json(),
        "input_sha256": "fixture",
        "rgb_sha256": [hashlib.sha256(a.tobytes()).hexdigest() for a in source_rgb],
    }
    closed = []

    def decode(_path, _table, indices, **_kwargs):
        try:
            for i in indices:
                yield DecodedFrame(i, source_table.pts[i], source_rgb[i])
        finally:
            closed.append(True)

    monkeypatch.setattr(inserts, "decode_frames", decode)
    monkeypatch.setattr(inserts, "sha", lambda _: "fixture")
    construction = inserts.recipe([source_table] * 3)
    frames = list(inserts.source_frames(original, original, construction, "ordinary_time_step"))
    a, b = construction["edit_frames"]
    assert [int(f[0, 0, 0]) for f in frames] == [i + 2 if a <= i < b else i for i in range(30)]
    assert len(closed) == 1
    stream = inserts.source_frames(original, original, construction, "original")
    next(stream)
    stream.close()
    assert len(closed) == 2
    original["rgb_sha256"][0] = "changed"
    with pytest.raises(ValueError, match="source RGB changed"):
        list(inserts.source_frames(original, original, construction, "original"))
    assert len(closed) == 3


@pytest.mark.parametrize(
    "step",
    [
        Fraction(1, 25),
        Fraction(1, 30),
        Fraction(1, 50),
        Fraction(1, 60),
        Fraction(1001, 24000),
        Fraction(1001, 30000),
        Fraction(1001, 60000),
    ],
)
def test_real_rgb_mp4_roundtrip_preserves_every_pixel_and_fractional_native_pts(tmp_path, step):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg unavailable")
    frames = np.random.default_rng(47).integers(0, 256, (13, 48, 64, 3), dtype=np.uint8)
    path = tmp_path / "native.mp4"
    expected = inserts.encode_rgb(path, iter(frames), 64, 48, step)
    native, representatives = inserts.verify_rgb(path, expected, step, [0, 6, 12])
    assert len(native) == 13
    assert all(np.array_equal(representatives[i], frames[i]) for i in (0, 6, 12))
    old = path.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        inserts.encode_rgb(path, iter(frames), 64, 48, step)
    assert path.read_bytes() == old
    with pytest.raises(ValueError, match="presentation times"):
        inserts.verify_rgb(path, expected, step * 2)
    changed = list(expected)
    changed[6] = "bad"
    with pytest.raises(ValueError, match="RGB"):
        inserts.verify_rgb(path, changed, step)


def test_classical_controls_do_not_claim_ai_and_remain_native_uint8():
    image = np.arange(64 * 48 * 3, dtype=np.uint8).reshape(48, 64, 3)
    for condition in ("ordinary_light", "ordinary_blur"):
        result = inserts.transform(image, condition)
        assert result.dtype == np.uint8 and result.shape == image.shape and not np.array_equal(result, image)
    assert np.array_equal(inserts.transform(image, "original"), image)


def test_existing_constructor_attempt_cannot_be_implicitly_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "TRIAL", tmp_path)
    path = tmp_path / "group-000-complete.json"
    path.write_text("preserved\n")
    with pytest.raises(FileExistsError):
        runtime.run("build", 0)
    assert path.read_text() == "preserved\n" and not (tmp_path / "group-000-profile").exists()
    for index in (True, -1, 90):
        with pytest.raises(ValueError):
            runtime.run("build", index)


def test_complete_authored_group_preserves_context_snapshots_and_rejects_changed_plan(tmp_path, monkeypatch):
    sources = tmp_path / "sources"
    sources.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    trial = tmp_path / "trial"
    trial.mkdir()
    base = np.random.default_rng(95).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    original = [((base.astype(np.uint16) + i) % 256).astype(np.uint8) for i in range(60)]
    rows = []
    for name, color in [("original", None), ("first", (255, 0, 0)), ("second", (0, 255, 0))]:
        frames = [f.copy() for f in original]
        if color:
            for frame in frames:
                frame[16:32, 24:40] = color
        path = sources / (name + ".mp4")
        hashes = inserts.encode_rgb(path, iter(frames), 64, 48, Fraction(1, 25))
        native, _ = inserts.verify_rgb(path, hashes, Fraction(1, 25))
        rows.append(
            {
                "file": path.name,
                "frame_table": native.to_json(),
                "input_sha256": inserts.sha(path),
                "rgb_sha256": hashes,
            }
        )
    construction = inserts.recipe([FrameTable.from_json(r["frame_table"]) for r in rows])
    identity = {"authored_fixture": True}
    plan = {
        "identity": identity,
        "media_root": str(output),
        "groups": [
            {"sources": rows, "construction": construction, "group": "authored-fixture", "role": "fusion_fit"}
        ],
    }
    inserts.write(trial / "plan.json", plan)
    inserts.write(
        trial / "freeze.json", {"identity": identity, "private_plan_sha256": inserts.sha(trial / "plan.json")}
    )
    monkeypatch.setattr(inserts, "TRIAL", trial)
    monkeypatch.setattr(inserts, "MEDIA", output)
    monkeypatch.setattr(inserts, "SOURCES", sources)
    monkeypatch.setattr(inserts, "checked", lambda: None)
    monkeypatch.setattr(inserts, "identity", lambda: identity)
    inserts.build(0)
    receipt = json.loads((trial / "group-000-complete.json").read_text())
    assert len(receipt["cases"]) == 6 and receipt["new_neural_calls"] == 0
    for case in receipt["cases"]:
        assert case["encoded_duration_seconds"] == "12/5"
        if case["condition"] == "original":
            assert case["actual_changed_frames"] == []
        else:
            assert case["actual_changed_frames"] == list(range(20, 40))
        assert set(case["representatives"]) == {"before", "during", "after"}
        for label, index in [("before", 19), ("after", 40)]:
            image = inserts.cv2.imread(str(output / "group-000" / case["representatives"][label]["file"]))
            assert np.array_equal(inserts.cv2.cvtColor(image, inserts.cv2.COLOR_BGR2RGB), original[index])
    before = (output / "group-000/original.mp4").read_bytes()
    plan["groups"][0]["construction"]["edit_frames"] = [21, 40]
    (trial / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Frozen construction changed"):
        inserts.build(0)
    assert (output / "group-000/original.mp4").read_bytes() == before
