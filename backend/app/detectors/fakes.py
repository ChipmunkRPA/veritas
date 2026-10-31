"""Model-free fake detectors for tests and `make e2e-fake` (ARCHITECTURE §9.2). Deterministic; no torch, no weights.

Also provides an in-memory `MediaHandle` and `ExhibitWriter` for model-free unit runs. They use the same call
signatures as `app.engine.media.StoredMedia` and `app.engine.exhibits.ExhibitWriter`, so the fakes also run through
the staged pipeline.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.engine.detector import (
    ActivationUse,
    Applicability,
    Detector,
    DetectorResult,
    DetectorSpec,
    Hypothesis,
    InputUnit,
    Modality,
    Profile,
    Requirements,
    RunContext,
    UnitScore,
)
from app.engine.models import GiB, ManagedModel

HEAVY_MODEL_ID = "fake.heavy_30gb"
FAKE_MODELS = (ManagedModel(id=HEAVY_MODEL_ID, fp32_bytes=30 * GiB, activation_bytes_per_item=0),)
_REFERENCE = "fake detector: no reference distribution"
_REASONING = "deterministic fake used for model-free tests"


def fake_loader(model_id: str, device: str, dtype: str) -> dict[str, str]:
    return {"model_id": model_id, "device": device, "dtype": dtype}


def _spec(det_id: str, name: str, modality: Modality, unit: InputUnit, family: str, hypotheses: tuple[Hypothesis, ...],
          produces: tuple[str, ...] = (), *, model_ids: tuple[str, ...] = (), timeout_s: float = 5.0) -> DetectorSpec:
    return DetectorSpec(id=det_id, version="1.0.0", name=name, modality=modality, input_unit=unit, family=family,
                        hypotheses=hypotheses, produces=produces,
                        requirements=Requirements(model_ids=model_ids, timeout_s=timeout_s, lane="cpu"),
                        profiles=frozenset(Profile))


# --- in-memory media and exhibits --------------------------------------------------------------------------

@dataclass
class InMemoryMedia:
    sha256: str
    kind: str
    data: bytes = b""
    pixels: np.ndarray | None = None
    path: Path | None = None

    @classmethod
    def from_array(cls, pixels: np.ndarray, *, kind: str = "image", path: Path | None = None) -> InMemoryMedia:
        arr = np.ascontiguousarray(pixels)
        return cls(sha256=hashlib.sha256(arr.tobytes()).hexdigest(), kind=kind, data=arr.tobytes(), pixels=arr, path=path)

    def original_bytes(self) -> bytes:
        return self.data

    def native_pixels(self, tile: tuple[int, int, int, int] | None = None) -> np.ndarray:
        if self.pixels is None:
            raise ValueError("media has no decoded pixels")
        if tile is None:
            return self.pixels
        x, y, w, h = tile
        return self.pixels[y:y + h, x:x + w]

    def analysis_copy(self) -> tuple[np.ndarray, float]:
        return self.native_pixels(), 1.0


@dataclass
class InMemoryExhibitWriter:
    """Records exhibits as plain dicts, with the call signatures of `app.engine.exhibits.ExhibitWriter`."""

    exhibits: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _add(self, kind: str, payload: dict[str, Any]) -> str:
        exhibit_id = f"ex-{len(self.exhibits) + 1:03d}"
        self.exhibits[exhibit_id] = {"kind": kind, **payload}
        return exhibit_id

    def heatmap(self, values: Any, *, semantics: str, title: str, mask: Any = None, **meta: Any) -> str:
        return self._add("heatmap", {"array": values, "mask": mask, "semantics": semantics, "title": title, **meta})

    def image(self, data: bytes, *, role: str, media_type: str, title: str, **meta: Any) -> str:
        return self._add("image", {"data": data, "role": role, "media_type": media_type, "title": title, **meta})

    def regions(self, regions: list[dict[str, Any]], *, title: str, **meta: Any) -> str:
        return self._add("regions", {"regions": regions, "title": title, **meta})

    def series(self, series: list[dict[str, Any]], *, x_unit: str, title: str, **meta: Any) -> str:
        return self._add("series", {"series": series, "x_unit": x_unit, "title": title, **meta})

    def segments(self, segments: list[dict[str, Any]], *, lane: str, title: str, **meta: Any) -> str:
        return self._add("segments", {"segments": segments, "lane": lane, "title": title, **meta})

    def table(self, columns: list[str], rows: list[list[Any]], *, title: str, **meta: Any) -> str:
        return self._add("table", {"columns": columns, "rows": rows, "title": title, **meta})

    def add(self, kind: str, *, title: str, media_ref: str, **fields: Any) -> str:
        return self._add(kind, {"title": title, "media_ref": media_ref, **fields})


def _duration_s(ctx: RunContext, default: float = 4.0) -> float:
    probe = ctx.outputs.probe
    for key in ("duration_s", "duration"):  # unit tests pass duration_s; the staged pipeline passes the ffprobe facts
        if isinstance(probe, dict) and probe.get(key) is not None:
            return float(probe[key])
    return default


def _ok(scores: list[UnitScore], observed: str, *, aggregate: UnitScore | None = None, **kw: Any) -> DetectorResult:
    return DetectorResult(status="ok", unit_scores=scores, aggregate=aggregate or (scores[0] if scores else None),
                          observed=observed, reference=_REFERENCE, reasoning=_REASONING, **kw)


def pause() -> None:
    """`VERITAS_FAKE_DELAY_S`: a standard fake sleeps first, so `make e2e-fake` can watch a running analysis (0 by default)."""
    if settings.fake_delay_s > 0:
        time.sleep(settings.fake_delay_s)


# --- fakes -------------------------------------------------------------------------------------------------

class FakeWholeImage(Detector):
    """Raw score from `<file>.fake.json` ({"raw": x} or {"<detector id>": x}), else the mean pixel intensity / 255."""

    spec = _spec("fake.whole_image", "Fake whole-image detector", Modality.IMAGE, InputUnit.IMAGE, "learned-whole-image",
                 (Hypothesis.H1_WHOLE_CONTENT_SYNTHESIS,))

    def run(self, ctx: RunContext) -> DetectorResult:
        pause()
        raw, source = self._sidecar(ctx), "fixture sidecar"
        if raw is None:
            pixels, _scale = ctx.media.analysis_copy()
            raw, source = float(np.asarray(pixels, dtype=np.float64).mean() / 255.0), "mean intensity"
        score = UnitScore("global", round(raw, 6), "probability")
        return _ok([score], f"{source}: {raw:.3f}", metrics={"raw": raw})

    def _sidecar(self, ctx: RunContext) -> float | None:
        path = getattr(ctx.media, "path", None)
        if path is None:
            return None
        sidecar = Path(path).with_name(Path(path).name + ".fake.json")
        if not sidecar.is_file():
            return None
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        value = data.get(self.spec.id, data.get("raw"))
        return None if value is None else float(value)


class FakeLocaliser(Detector):
    """64×64 residual map with a known square (x, y, w, h = 16, 16, 16, 16); rows 0–3 are marked unreliable. The map is registered
    like a real localiser's: its grid spans the image (`coordinate_space` 64², scale native width / 64), so regions segmented from
    it map to original pixels."""

    SQUARE = (16, 16, 16, 16)
    spec = _spec("fake.localiser", "Fake localiser", Modality.IMAGE, InputUnit.IMAGE, "forensic",
                 (Hypothesis.H2_LOCALISED_MANIPULATION,), ("heatmap",))

    def run(self, ctx: RunContext) -> DetectorResult:
        pause()
        x, y, w, h = self.SQUARE
        heat = np.zeros((64, 64), np.float32)
        heat[y:y + h, x:x + w] = 1.0
        mask = np.ones((64, 64), bool)
        mask[:4, :] = False
        pixels, scale = ctx.media.analysis_copy()
        native_width = round(pixels.shape[1] * scale)
        exhibit = ctx.exhibits.heatmap(heat, semantics="residual", title="Fake localiser response map", mask=mask,
                                       value_range=(0.0, 1.0), media_ref="original",
                                       coordinate_space={"width": 64, "height": 64, "scale_to_original": native_width / 64})
        score = UnitScore("global", 1.0, "score", quality={"area_px": float(w * h)})
        return _ok([score], "one square region with elevated response", metrics={"max": 1.0, "area_px": float(w * h)},
                   exhibit_ids=[exhibit])


class FakeFaceDetector(Detector):
    """One deterministic face box (x, y, w, h = 40, 32, 48, 48) on images ≥ 64 px."""

    BOX = {"x": 40.0, "y": 32.0, "w": 48.0, "h": 48.0, "score": 0.99}
    spec = _spec("fake.faces", "Fake face detector", Modality.IMAGE, InputUnit.IMAGE, "face-detection", (), ("regions",))

    def applicable(self, ctx: RunContext) -> Applicability:
        pixels, _scale = ctx.media.analysis_copy()
        return Applicability(True) if min(pixels.shape[:2]) >= 64 else Applicability(False, "image < 64 px")

    def run(self, ctx: RunContext) -> DetectorResult:
        pause()
        box = {k: self.BOX[k] for k in ("x", "y", "w", "h")}
        exhibit = ctx.exhibits.regions([{"region": {"kind": "bbox", "bbox": box}, "score": self.BOX["score"], "label": "face 1"}],
                                       title="Faces (fake detector)")
        return _ok([UnitScore("face:0", self.BOX["score"], "score")], "one face", metrics={"n_faces": 1.0},
                   exhibit_ids=[exhibit])


class FakeTracker(Detector):
    """One face track `track:0` ("Face A") with a detected box every 0.5 s. On the staged pipeline the frame indices and pts come
    from the frame table of the low-resolution stream (`outputs.frames.table`), as the real tracker's do; without a table the
    box times are milliseconds over the probe duration (default 4 s)."""

    STEP_S = 0.5
    spec = _spec("fake.tracker", "Fake face tracker", Modality.VIDEO, InputUnit.FRAME_BATCH, "face-tracking", (),
                 ("face_track",))

    def run(self, ctx: RunContext) -> DetectorResult:
        pause()
        table = getattr(ctx.outputs.frames, "table", None)
        if table is not None and len(table):
            times = table.times()
            indices = sorted({int(i) for i in np.searchsorted(times, np.arange(0.0, float(times[-1]) + 1e-9, self.STEP_S))
                              if i < len(times)})
            stamps = [(i, int(table.pts[i])) for i in indices]
        else:
            stamps = [(k, round(k * self.STEP_S * 1000)) for k in range(int(_duration_s(ctx) / self.STEP_S) + 1)]
        frames = [{"pts": pts, "frame_index": index, "bbox": {"x": 40.0 + 2 * k, "y": 32.0, "w": 64.0, "h": 64.0}, "det_score": 0.99}
                  for k, (index, pts) in enumerate(stamps)]
        track = {"track_id": "track:0", "first_pts": frames[0]["pts"], "last_pts": frames[-1]["pts"], "n_frames": len(frames),
                 "identity": "A"}
        exhibit = ctx.exhibits.add("face_track", title="Face A (fake)", media_ref="frame", track=track, frames=frames)
        return _ok([UnitScore("track:0", 1.0, "score")], "one face track", metrics={"n_tracks": 1.0},
                   exhibit_ids=[exhibit])


class FakeTrackScorer(Detector):
    """0.5 s windows on track:0: 0.9 inside 1.0–2.0 s, 0.1 elsewhere; aggregate = mean of the top 30 % windows."""

    WINDOW_S = 0.5
    FLAGGED = (1.0, 2.0)
    spec = _spec("fake.track_scorer", "Fake track scorer", Modality.VIDEO, InputUnit.TRACK_CLIP, "temporal-face",
                 (Hypothesis.H2_LOCALISED_MANIPULATION,), ("series",))

    def applicable(self, ctx: RunContext) -> Applicability:
        return Applicability(True) if ctx.outputs.tracks else Applicability(False, "no face tracks")

    def run(self, ctx: RunContext) -> DetectorResult:
        pause()
        n = max(1, round(_duration_s(ctx) / self.WINDOW_S))
        scores = []
        for k in range(n):
            start, end = k * self.WINDOW_S, (k + 1) * self.WINDOW_S
            raw = 0.9 if self.FLAGGED[0] <= start and end <= self.FLAGGED[1] else 0.1
            scores.append(UnitScore(f"track:0@window={k}", raw, "probability", quality={"start_s": start, "end_s": end}))
        top = sorted(s.raw for s in scores)[-max(1, math.ceil(0.3 * n)):]
        aggregate = UnitScore("track:0", float(np.mean(top)), "probability")
        flagged = [s for s in scores if s.raw >= 0.5]
        exhibit = ctx.exhibits.series([{"name": "p_window", "track_id": "track:0",
                                        "points": [(s.quality["start_s"], s.raw) for s in scores]}],
                                      x_unit="s", title="Face A window scores (fake)")
        observed = (f"elevated windows {flagged[0].quality['start_s']:.1f}–{flagged[-1].quality['end_s']:.1f} s"
                    if flagged else "no elevated windows")
        return _ok(scores, observed, aggregate=aggregate, metrics={"flagged_fraction": len(flagged) / n},
                   exhibit_ids=[exhibit])


class FakeLipSync(Detector):
    """0.5 s windows on track:0: sync offset 0 frames with p 0.1, and −5 frames (sound earlier) with p 0.85 inside 1.0–2.0 s;
    confidence 0.8 throughout. The offset and confidence series follow the lip-sync adapters' names (`offset_frames`,
    `confidence`)."""

    WINDOW_S = 0.5
    FLAGGED = (1.0, 2.0)
    OFFSET_FRAMES = -5.0
    spec = _spec("fake.lipsync", "Fake lip-sync detector", Modality.VIDEO, InputUnit.TRACK_CLIP, "av-sync",
                 (Hypothesis.H2_LOCALISED_MANIPULATION,), ("series",))

    def applicable(self, ctx: RunContext) -> Applicability:
        return Applicability(True) if ctx.outputs.tracks else Applicability(False, "no face tracks")

    def run(self, ctx: RunContext) -> DetectorResult:
        pause()
        n = max(1, round(_duration_s(ctx) / self.WINDOW_S))
        scores, offsets = [], []
        for k in range(n):
            start, end = k * self.WINDOW_S, (k + 1) * self.WINDOW_S
            flagged = self.FLAGGED[0] <= start and end <= self.FLAGGED[1]
            scores.append(UnitScore(f"track:0@window={k}", 0.85 if flagged else 0.1, "probability", quality={"start_s": start, "end_s": end}))
            offsets.append((start + self.WINDOW_S / 2, self.OFFSET_FRAMES if flagged else 0.0))
        exhibit = ctx.exhibits.series(
            [{"name": "offset_frames", "hypothesis": Hypothesis.H2_LOCALISED_MANIPULATION.value, "track_id": "track:0", "points": offsets},
             {"name": "confidence", "hypothesis": Hypothesis.H2_LOCALISED_MANIPULATION.value, "track_id": "track:0",
              "points": [(t, 0.8) for t, _ in offsets]}],
            x_unit="s", title="Sync offset and confidence per 1 s window, Face A (fake)", thresholds=[0.0])
        return _ok(scores, f"offset {self.OFFSET_FRAMES:+.0f} frames inside {self.FLAGGED[0]:.1f}–{self.FLAGGED[1]:.1f} s",
                   aggregate=UnitScore("track:0", max(s.raw for s in scores), "probability"), exhibit_ids=[exhibit])


class FakeVoice(Detector):
    """Windows every 2 s over the audio duration (each up to 2 s, the tail included), constant raw score 0.2."""

    HOP_S = 2.0
    spec = _spec("fake.voice", "Fake voice detector", Modality.AUDIO, InputUnit.AUDIO_WINDOW, "voice",
                 (Hypothesis.H2_LOCALISED_MANIPULATION,), ("series",))

    def applicable(self, ctx: RunContext) -> Applicability:
        return Applicability(True) if ctx.outputs.audio is not None else Applicability(False, "no audio track")

    def run(self, ctx: RunContext) -> DetectorResult:
        pause()
        duration = _duration_s(ctx)
        n = max(1, math.ceil(duration / self.HOP_S))
        scores = [UnitScore(f"window:{k}", 0.2, "probability", quality={"start_s": k * self.HOP_S, "end_s": min(duration, (k + 1) * self.HOP_S)})
                  for k in range(n)]
        exhibit = ctx.exhibits.series([{"name": "voice", "points": [(s.quality["start_s"], s.raw) for s in scores]}],
                                      x_unit="s", title="Voice windows (fake)")
        return _ok(scores, f"{n} voice windows", aggregate=UnitScore("global", 0.2, "probability"), exhibit_ids=[exhibit])


class FakeText(Detector):
    """Raw score = share of words longer than 6 characters; needs at least 15 words."""

    MIN_WORDS = 15
    spec = _spec("fake.text", "Fake text detector", Modality.TEXT, InputUnit.DOCUMENT, "text",
                 (Hypothesis.H1_WHOLE_CONTENT_SYNTHESIS,))

    @staticmethod
    def _words(ctx: RunContext) -> list[str]:
        document = ctx.outputs.document
        text = document if isinstance(document, str) else bytes(ctx.media.original_bytes()).decode("utf-8", "replace")
        return text.split()

    def applicable(self, ctx: RunContext) -> Applicability:
        n = len(self._words(ctx))
        return Applicability(True) if n >= self.MIN_WORDS else Applicability(False, f"fewer than {self.MIN_WORDS} words")

    def run(self, ctx: RunContext) -> DetectorResult:
        pause()
        words = self._words(ctx)
        raw = sum(len(w) > 6 for w in words) / len(words)
        return _ok([UnitScore("global", raw, "probability")], f"{len(words)} words", metrics={"words": float(len(words))})


class FakeDecisive(Detector):
    """Provenance fake emitting a D-15 override (decisive_ai)."""

    OVERRIDE = "decisive_ai"
    spec = _spec("fake.provenance_decisive", "Fake provenance (decisive)", Modality.IMAGE, InputUnit.FILE, "provenance",
                 (Hypothesis.H3_PROVENANCE_INTEGRITY,), ("table",))

    def run(self, ctx: RunContext) -> DetectorResult:
        pause()
        return _ok([UnitScore("global", 1.0, "score")], f"fixture provenance override: {self.OVERRIDE}",
                   provenance_override=self.OVERRIDE)


class FakeSelfDeclared(FakeDecisive):
    OVERRIDE = "self_declared_ai"
    spec = _spec("fake.provenance_self_declared", "Fake provenance (self-declared)", Modality.IMAGE, InputUnit.FILE,
                 "provenance", (Hypothesis.H3_PROVENANCE_INTEGRITY,), ("table",))


class FakeTrustedCapture(FakeDecisive):
    OVERRIDE = "trusted_capture"
    spec = _spec("fake.provenance_trusted_capture", "Fake provenance (trusted capture)", Modality.IMAGE, InputUnit.FILE,
                 "provenance", (Hypothesis.H3_PROVENANCE_INTEGRITY,), ("table",))


class FakeFailing(Detector):
    """Raises with a file-system path in the message (the boundary must sanitise it)."""

    spec = _spec("fake.failing", "Fake failing detector", Modality.IMAGE, InputUnit.IMAGE, "learned-whole-image",
                 (Hypothesis.H1_WHOLE_CONTENT_SYNTHESIS,))

    def run(self, ctx: RunContext) -> DetectorResult:
        raise RuntimeError("fake detector failure while reading /opt/veritas/data/weights/fake.pth")


class FakeSlow(Detector):
    """Loops with cooperative deadline checks until the budget is spent."""

    STEP_S = 0.02
    MAX_S = 30.0
    spec = _spec("fake.slow", "Fake slow detector", Modality.IMAGE, InputUnit.IMAGE, "learned-whole-image",
                 (Hypothesis.H1_WHOLE_CONTENT_SYNTHESIS,), timeout_s=0.25)

    def run(self, ctx: RunContext) -> DetectorResult:
        end = time.monotonic() + self.MAX_S
        while time.monotonic() < end:
            ctx.deadline.check("fake slow loop")
            time.sleep(self.STEP_S)
        return _ok([], "finished")


class FakeSlowBlocking(Detector):
    """Blocks without deadline checks, like a torch op that cannot be pre-empted; only the watchdog can end it."""

    BLOCK_S = 1.0
    spec = _spec("fake.slow_blocking", "Fake blocking detector", Modality.IMAGE, InputUnit.IMAGE, "learned-whole-image",
                 (Hypothesis.H1_WHOLE_CONTENT_SYNTHESIS,), timeout_s=0.25)

    def run(self, ctx: RunContext) -> DetectorResult:
        time.sleep(self.BLOCK_S)
        return _ok([], "finished late")


class FakeHeavy(Detector):
    """Needs `fake.heavy_30gb` (30 GB declared) to trigger the memory-budget refusal."""

    spec = _spec("fake.heavy", "Fake heavy detector", Modality.IMAGE, InputUnit.IMAGE, "learned-whole-image",
                 (Hypothesis.H1_WHOLE_CONTENT_SYNTHESIS,), model_ids=(HEAVY_MODEL_ID,))

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return (ActivationUse(HEAVY_MODEL_ID, 1, "fake-object", "no tensor computation"),)

    def run(self, ctx: RunContext) -> DetectorResult:
        model = ctx.lease.get(HEAVY_MODEL_ID)
        return _ok([UnitScore("global", 0.5, "probability")], f"loaded {model['model_id']}")


STANDARD_FAKES: tuple[type[Detector], ...] = (
    FakeWholeImage, FakeLocaliser, FakeFaceDetector, FakeTracker, FakeTrackScorer, FakeLipSync, FakeVoice, FakeText,
    FakeDecisive, FakeSelfDeclared, FakeTrustedCapture,
)
FAULT_FAKES: tuple[type[Detector], ...] = (FakeFailing, FakeSlow, FakeSlowBlocking, FakeHeavy)
ALL_FAKES: tuple[type[Detector], ...] = STANDARD_FAKES + FAULT_FAKES
