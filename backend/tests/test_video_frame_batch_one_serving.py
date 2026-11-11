from dataclasses import replace

import pytest
from test_video_frame_check_batch_one import context

from app.detectors.image import dfdc
from app.detectors.video.adapters import frames, frames_batch
from app.detectors.video.adapters.common import NO_FACE_REASON
from app.engine.boundary import Deadline, DetectorTimeout
from app.engine.capacity import plan
from app.engine.detector import Profile
from app.engine.fusion import registry as fusion
from app.engine.manifest import DEFAULT_MANIFEST, lint
from app.engine.models import FootprintStore, MiB
from app.engine.registry import DetectorRegistry, build_registry

TARGET = "vid.frames.dfdc_b7"


def original_registry(registry):
    manifest = registry.manifest.model_copy(deep=True)
    entry = manifest.detector(TARGET)
    entry.adapter = "app.detectors.video.adapters.frames:FrameDFDCB7Detector"
    entry.version = "0.3.0"
    return DetectorRegistry(manifest, registry.policy)


def test_default_registry_admits_face_check_under_same_real_capacity_plan(tmp_path, monkeypatch):
    current = build_registry(DEFAULT_MANIFEST, policy="commercial")
    original = original_registry(current)
    adapter = current.instance(TARGET)
    assert type(adapter) is frames_batch.FrameDFDCB7BatchOneDetector
    assert type(original.instance(TARGET)) is frames.FrameDFDCB7Detector
    spec = original.instance(TARGET).spec
    assert adapter.spec == replace(
        spec, version="0.3.1", requirements=replace(spec.requirements, max_batch=1)
    )
    assert not any(issue.level == "error" and issue.subject == TARGET for issue in lint(current.manifest))

    store = FootprintStore(tmp_path / "footprints.json")
    options = {"device": "cpu", "budget_bytes": 4608 * MiB, "store": store}
    for profile in (Profile.QUICK, Profile.FULL, Profile.DEEP):
        before = plan(profile, registry=original, **options)
        after = plan(profile, registry=current, **options)
        prior = next(row for row in before.detectors if row.detector_id == TARGET)
        candidate = next(row for row in after.detectors if row.detector_id == TARGET)
        assert not prior.can_run and "memory budget" in prior.refusal
        assert candidate.can_run
        assert (prior.activation_bytes, candidate.activation_bytes) == (6400 * MiB, 800 * MiB)
        assert candidate.models == prior.models
        assert candidate.weight_bytes == 510 * MiB
        assert not store.path.exists()

    # Exercise the actual fusion compatibility inputs without opening any weights.
    monkeypatch.setattr(fusion, "models_installed", lambda manifest, model_ids: True)
    old_hashes = fusion.loaded_detector_hashes(original, "video", Profile.FULL)
    assert fusion.loaded_detector_hashes(current, "video", Profile.FULL) == old_hashes
    assert old_hashes[TARGET] == "651bc0cae350"
    assert (
        fusion.select_for_registry(current, "video", Profile.FULL).as_json()
        == fusion.select_for_registry(original, "video", Profile.FULL).as_json()
    )


def test_authored_production_forwards_match_eight_face_scores_and_evidence(monkeypatch):
    observed = []

    class RecordedClassifier(dfdc.DFDCClassifier):
        def predict(self, items, batch=dfdc.BATCH):
            observed.append(("batch", batch, [(int(rgb[0, 0, 0]), face.w) for rgb, face in items]))
            return super().predict(items, batch=batch)

    def first(tensor):
        observed.append(("first", tuple(tensor.shape)))
        return tensor[:, 0, 190, 190:191] * 0

    def second(tensor):
        observed.append(("second", tuple(tensor.shape)))
        red = tensor[:, 0, 190, 190:191]
        return (red.gt(1).float() - red.lt(-1).float()) * 100

    monkeypatch.setattr(frames, "DFDCClassifier", RecordedClassifier)
    monkeypatch.setattr(frames_batch, "DFDCClassifier", RecordedClassifier)
    monkeypatch.setattr(frames_batch.settings, "dfdc_min_face_px", 120)
    active, original = (context((first, second)) for _ in range(2))
    detector = build_registry(DEFAULT_MANIFEST, policy="commercial").instance(TARGET)
    result = detector.run(active)
    assert [row for row in observed if row[0] == "batch"] == [
        ("batch", 1, [(25, 88)]),
        ("batch", 1, [(140, 140)]),
        ("batch", 1, [(220, 100)]),
    ]
    assert [row for row in observed if row[0] != "batch"] == [
        (member, (1, 3, 380, 380)) for member in ("first", "second") * 3
    ]
    assert [unit.unit_ref for unit in result.unit_scores] == ["frame:0", "frame:2", "frame:3"]
    assert [unit.raw for unit in result.unit_scores] == [0.25, 0.5, 0.75]
    assert result == frames.FrameDFDCB7Detector().run(original)
    assert active.exhibits.series_calls == original.exhibits.series_calls
    assert active.exhibits.series_calls[0][0] == [
        {"name": "p", "points": [(0.031, 0.25), (1.03, 0.5), (1.215, 0.75)]}
    ]
    assert result.model_revisions == dict(zip(detector.MODEL_IDS, ("revision-0", "revision-1"), strict=True))

    active.outputs.faces = [[] for _ in active.outputs.frames]
    assert detector.applicable(active).reason == NO_FACE_REASON


@pytest.mark.parametrize("expire_after", [1, 3])
def test_production_deadline_never_publishes_partial_series(monkeypatch, expire_after):
    now, calls = [0.0], []

    class ExpiringClassifier:
        def __init__(self, *, members, device):
            pass

        def predict(self, items, *, batch):
            calls.append((len(items), batch))
            if len(calls) == expire_after:
                now[0] = 2.0
            return [0.25]

    monkeypatch.setattr(frames_batch, "DFDCClassifier", ExpiringClassifier)
    ctx = context((object(), object()), Deadline(1, clock=lambda: now[0]))
    detector = build_registry(DEFAULT_MANIFEST, policy="commercial").instance(TARGET)
    with pytest.raises(DetectorTimeout):
        detector.run(ctx)
    assert calls == [(1, 1)] * expire_after
    assert ctx.exhibits.series_calls == []
