from types import SimpleNamespace

import numpy as np
import pytest

from app.detectors.image import dfdc
from app.detectors.video.adapters import frames
from app.detectors.video.adapters.common import NO_FACE_REASON, SampledFrame
from app.engine.boundary import Deadline, DetectorTimeout, planned_allowance
from app.engine.detector import Profile, RunContext, StageOutputs
from app.engine.manifest import DEFAULT_MANIFEST, load_manifest
from app.engine.models import FootprintStore, ManagedModel, MiB, ModelManager
from scripts import video_frame_check_batch_one as lab


class RecordedExhibits:
    def __init__(self):
        self.series_calls = []

    def series(self, series, **metadata):
        self.series_calls.append((series, metadata))
        return "face-frame-series"


def context(members=(), deadline=None):
    sampled = [
        SampledFrame(index, index * 9, pts_ms / 1000, pts_ms, np.full((220, 240, 3), fill, dtype=np.uint8))
        for index, (pts_ms, fill) in enumerate([(31, 25), (480, 80), (1030, 140), (1215, 220)])
    ]
    faces = [
        [{"x": 11, "y": 13, "w": 88, "h": 76}, {"x": 160, "y": 120, "w": 35, "h": 33}],
        [],
        [{"x": 12, "y": 14, "w": 140, "h": 107}],
        [{"x": 13, "y": 15, "w": 100, "h": 102}],
    ]
    entries = {
        model_id: SimpleNamespace(obj=member, revision=f"revision-{index}")
        for index, (model_id, member) in enumerate(zip(lab.FrameDFDCB7BatchOneDetector.MODEL_IDS, members))
    }
    return RunContext(
        analysis_id="authored",
        media=None,
        profile=Profile.FULL,
        device="cpu",
        dtype="float32",
        lease=SimpleNamespace(get=entries.__getitem__),
        outputs=StageOutputs(frames=sampled, faces=faces),
        deadline=Deadline() if deadline is None else deadline,
        exhibits=RecordedExhibits(),
        log=None,
    )


def test_one_item_allowance_covers_both_original_members(tmp_path):
    candidate = lab.FrameDFDCB7BatchOneDetector()
    original = frames.FrameDFDCB7Detector()
    manifest = load_manifest(DEFAULT_MANIFEST)

    def no_load(model_id):
        pytest.fail(f"Capacity planning attempted to load {model_id}")

    manager = ModelManager(
        budget_bytes=4608 * MiB,
        models=[ManagedModel.from_manifest(manifest.model(model_id)) for model_id in candidate.MODEL_IDS],
        loader_for=no_load,
        store=FootprintStore(tmp_path / "footprints.json"),
    )
    allowance, plan = planned_allowance(candidate, context(), manager)
    assert allowance == 800 * MiB
    assert tuple(use.model_id for use in plan) == original.MODEL_IDS
    assert tuple(use.max_items for use in plan) == (1, 1)
    assert all(use.unit == "380 px face crop" for use in plan)
    assert planned_allowance(original, context(), manager)[0] == 6400 * MiB
    assert candidate.spec.id == original.spec.id
    assert candidate.spec.version != original.spec.version
    assert candidate.spec.requirements.max_batch == 1


def test_real_predict_call_is_bounded_and_preserves_ordered_evidence(monkeypatch):
    calls, forwards = [], []

    class RecordedClassifier(dfdc.DFDCClassifier):
        def predict(self, items, batch=dfdc.BATCH):
            calls.append((batch, [(int(rgb[0, 0, 0]), face.w) for rgb, face in items]))
            return super().predict(items, batch=batch)

    def first_member(tensor):
        forwards.append(("first", tuple(tensor.shape)))
        return tensor[:, 0, 190, 190:191] * 0

    def second_member(tensor):
        forwards.append(("second", tuple(tensor.shape)))
        red = tensor[:, 0, 190, 190:191]
        return (red.gt(1).float() - red.lt(-1).float()) * 100

    members = (first_member, second_member)
    monkeypatch.setattr(lab, "DFDCClassifier", RecordedClassifier)
    monkeypatch.setattr(frames, "DFDCClassifier", RecordedClassifier)
    monkeypatch.setattr(lab.settings, "dfdc_min_face_px", 120)
    ctx = context(members)
    result = lab.FrameDFDCB7BatchOneDetector().run(ctx)

    assert calls == [(1, [(25, 88)]), (1, [(140, 140)]), (1, [(220, 100)])]
    assert forwards == [(member, (1, 3, 380, 380)) for member in ("first", "second") * 3]
    assert [unit.unit_ref for unit in result.unit_scores] == ["frame:0", "frame:2", "frame:3"]
    assert [unit.raw for unit in result.unit_scores] == [0.25, 0.5, 0.75]
    assert [unit.quality["face_px"] for unit in result.unit_scores] == [88.0, 140.0, 100.0]
    assert ctx.exhibits.series_calls[0][0] == [
        {"name": "p", "points": [(0.031, 0.25), (1.03, 0.5), (1.215, 0.75)]}
    ]
    assert result.aggregate.quality == {"face_frames": 3.0, "small_face_fraction": 2 / 3}

    calls.clear()
    forwards.clear()
    baseline_ctx = context(members)
    assert result == frames.FrameDFDCB7Detector().run(baseline_ctx)
    assert ctx.exhibits.series_calls == baseline_ctx.exhibits.series_calls
    assert calls == [(8, [(25, 88), (140, 140), (220, 100)])]
    assert forwards == [(member, (3, 3, 380, 380)) for member in ("first", "second")]


@pytest.mark.parametrize("expire_after", [1, 3])
def test_deadline_cuts_off_the_next_call_or_evidence(monkeypatch, expire_after):
    now, calls = [0.0], []

    class ExpiringClassifier:
        def __init__(self, *, members, device):
            pass

        def predict(self, items, *, batch):
            calls.append(batch)
            if len(calls) == expire_after:
                now[0] = 2.0
            return [0.25]

    monkeypatch.setattr(lab, "DFDCClassifier", ExpiringClassifier)
    ctx = context((object(), object()), Deadline(1, clock=lambda: now[0]))
    with pytest.raises(DetectorTimeout):
        lab.FrameDFDCB7BatchOneDetector().run(ctx)
    assert calls == [1] * expire_after
    assert ctx.exhibits.series_calls == []


def test_no_face_remains_not_applicable():
    ctx = context()
    ctx.outputs.faces = [[] for _ in ctx.outputs.frames]
    result = lab.FrameDFDCB7BatchOneDetector().applicable(ctx)
    assert not result.ok
    assert result.reason == NO_FACE_REASON
