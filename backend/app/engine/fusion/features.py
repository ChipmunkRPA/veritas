"""Stage 1 features (ML §4.2; ARCHITECTURE §4.4): z⁺ / z⁻, missing masks, context one-hots.

- z_j = clip(raw logit, −8, 8), or clip(logit p, −8, 8) when a detector reports only a probability (`logit_clip8`).
  Heuristic forensics keep the v0.3 [0.40, 0.75] cap as their input transform (`heuristic_cap`, ML §4.6), so absent
  anomalies stay near-neutral.
- A not-applicable detector (undersize, no face) has z_j := 0 and m_j = 1. A detector that failed, timed out, was skipped
  or gave no usable score is `inactive` at runtime: every column 0, so it adds nothing (D-46; `runtime.not_run` caps the
  decision).
- Context terms are one-hot buckets with the reference level omitted: no face or face_px < 120; JPEG quality lossless or
  none; min_side 512–1024.
- Interaction groups G = {has_face, jpeg_q<70, video_frame, screenshot_like} (D-44: A5 context terms are terms the runtime
  computes; `screenshot_like` is ConditionProfile `screenshot_likelihood` ≥ 0.5).
- Context gates (D-41 / H-3 "context-gate"): a detector may carry a gate, a conjunction of literals `term` or `!term`.
  Outside its gate every column of the detector is 0 (no z, no missing indicator), so it neither contributes nor counts
  as missing.

Coefficient names (the artefact's `features` and `coef` keys): `bias`, `ctx:<term>`, `z+:<det>`, `z-:<det>`,
`v+:<det>|<group>`, `v-:<det>|<group>`, `miss:<det>`.
"""
from __future__ import annotations

import functools
import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

Z_CLIP = 8.0
HEURISTIC_CAP = (0.40, 0.75)
TRANSFORMS = ("logit_clip8", "heuristic_cap")
CONTEXT_TERMS = ("has_face", "face_px:120-256", "face_px>=256", "jpeg_q>=90", "jpeg_q:70-89", "jpeg_q<70",
                 "min_side<512", "min_side>1024", "upscaled", "has_camera_exif", "video_frame", "screenshot_like")
GROUPS = ("has_face", "jpeg_q<70", "video_frame", "screenshot_like")
SIZE_FACE_TERMS = ("has_face", "face_px:120-256", "face_px>=256", "min_side<512", "min_side>1024")  # never additive for images (D-45)
METADATA_TERMS = ("has_camera_exif",)  # never an additive term, group or gate literal of an image fit (D-50)
FACE_TERMS = ("has_face", "face_px:120-256", "face_px>=256")  # a gate on these needs a completed face stage (D-49)
SCREENSHOT_LIKE_MIN = 0.5  # D-44
PLATT_CONTEXTS = ("photo_noface", "photo_face", "lowq", "video", "audio")
MODALITIES = ("image", "video", "audio", "document")


def fp_guarded(fn):
    """Run a fusion entry point with numpy's floating-point warnings off.

    numpy 2.2 built against Apple Accelerate raises divide-by-zero / overflow / invalid flags in `matmul` on finite
    inputs (verified 2026-09-12: the product equals `einsum` exactly and is finite). Guarded functions check their
    outputs for finiteness instead, so a real overflow still fails loudly. Each call gets a fresh `errstate`, so guarded
    functions may call each other.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return fn(*args, **kwargs)

    return wrapper


@dataclass(frozen=True)
class Context:
    """What the fuser knows about an input before any detector speaks (a subset of the ConditionProfile)."""

    modality: str = "image"
    face_px: int | None = None  # largest face side in pixels; None or 0 = no face found
    jpeg_q: int | None = None  # estimated JPEG quality; None = lossless or not JPEG
    min_side: int | None = None
    upscaled: bool = False
    has_camera_exif: bool = False
    screenshot_like: bool = False

    def __post_init__(self) -> None:
        if self.modality not in MODALITIES:
            raise ValueError(f"unknown modality {self.modality!r}")

    @property
    def has_face(self) -> bool:
        return self.face_px is not None and self.face_px > 0

    def terms(self) -> dict[str, float]:
        face, q, side, px = self.has_face, self.jpeg_q, self.min_side, self.face_px or 0
        on = {
            "has_face": face,
            "face_px:120-256": face and 120 <= px < 256,
            "face_px>=256": face and px >= 256,
            "jpeg_q>=90": q is not None and q >= 90,
            "jpeg_q:70-89": q is not None and 70 <= q < 90,
            "jpeg_q<70": q is not None and q < 70,
            "min_side<512": side is not None and side < 512,
            "min_side>1024": side is not None and side > 1024,
            "upscaled": self.upscaled,
            "has_camera_exif": self.has_camera_exif,
            "video_frame": self.modality == "video",
            "screenshot_like": self.screenshot_like,
        }
        return {term: 1.0 if value else 0.0 for term, value in on.items()}

    def platt_context(self) -> str:
        """Calibration context K (ML §4.4); the first matching rule wins."""
        if self.modality in ("audio", "video"):
            return self.modality
        lowq = (self.jpeg_q is not None and self.jpeg_q < 70) or self.screenshot_like or (
            self.min_side is not None and self.min_side < 512)
        if lowq:
            return "lowq"
        return "photo_face" if self.has_face else "photo_noface"


def face_gated(literals: Sequence[str] | None) -> bool:
    """Whether a gate names a face term, in either polarity (D-49: the detector counts as face-dependent)."""
    return any(lit.lstrip("!") in FACE_TERMS for lit in literals or ())


def gate_literals(literals: Sequence[str]) -> list[tuple[str, bool]]:
    """Parse a gate: `term` requires the context term, `!term` requires its absence; a gate needs at least one literal."""
    if isinstance(literals, str) or not literals:
        raise ValueError("a gate is a non-empty list of context-term literals")
    parsed = [(lit[1:], False) if lit.startswith("!") else (lit, True) for lit in literals]
    if any(term not in CONTEXT_TERMS for term, _ in parsed) or len({t for t, _ in parsed}) != len(parsed):
        raise ValueError(f"gate literals must be distinct members of {CONTEXT_TERMS}, optionally prefixed with '!'")
    return parsed


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def logit_clip(p: float) -> float:
    q = min(max(float(p), 1e-12), 1.0 - 1e-12)
    return max(-Z_CLIP, min(Z_CLIP, math.log(q / (1.0 - q))))


def transform_score(prob: float | None, raw_logit: float | None = None, transform: str = "logit_clip8") -> float | None:
    """z for one detector output, or None when the detector produced no usable score (→ missing)."""
    if transform not in TRANSFORMS:
        raise ValueError(f"unknown transform {transform!r}")
    if _finite(prob) and not 0.0 <= prob <= 1.0:
        raise ValueError(f"probability {prob!r} outside [0, 1]")
    if transform == "heuristic_cap":
        if not _finite(prob):
            return None
        lo, hi = HEURISTIC_CAP
        return logit_clip(min(max(float(prob), lo), hi))
    if _finite(raw_logit):
        return max(-Z_CLIP, min(Z_CLIP, float(raw_logit)))
    return logit_clip(prob) if _finite(prob) else None


@dataclass(frozen=True)
class FeatureSpace:
    """The column layout of the stacker design matrix; `names[i]` is the coefficient of column i."""

    detectors: tuple[str, ...]
    context_terms: tuple[str, ...] = CONTEXT_TERMS
    groups: tuple[str, ...] = GROUPS
    gates: tuple[tuple[str, tuple[str, ...]], ...] = ()  # (detector, literals); see `gate_literals`

    def __post_init__(self) -> None:
        if not self.detectors:
            raise ValueError("a feature space needs at least one detector")
        if len(set(self.detectors)) != len(self.detectors):
            raise ValueError("duplicate detector ids")
        if any(t not in CONTEXT_TERMS for t in self.context_terms) or len(set(self.context_terms)) != len(self.context_terms):
            raise ValueError(f"context terms must be distinct members of {CONTEXT_TERMS}")
        if any(g not in GROUPS for g in self.groups) or len(set(self.groups)) != len(self.groups):
            raise ValueError(f"groups must be distinct members of {GROUPS}")
        gated = [det for det, _ in self.gates]
        if len(set(gated)) != len(gated) or any(det not in self.detectors for det in gated):
            raise ValueError("gates need distinct detectors of the feature space")
        for _, literals in self.gates:
            gate_literals(literals)

    def gate_open(self, j: int, terms: Mapping[str, float]) -> bool:
        """Whether detector j is inside its gate for these context terms (always, without a gate)."""
        literals = dict(self.gates).get(self.detectors[j])
        return literals is None or all((terms[term] == 1.0) == wanted for term, wanted in gate_literals(literals))

    @property
    def block(self) -> int:
        """Columns per detector: z⁺, z⁻, v⁺ per group, v⁻ per group, miss."""
        return 3 + 2 * len(self.groups)

    def offset(self, j: int) -> int:
        return 1 + len(self.context_terms) + j * self.block

    @property
    def names(self) -> tuple[str, ...]:
        out = ["bias", *(f"ctx:{t}" for t in self.context_terms)]
        for det in self.detectors:
            out += [f"z+:{det}", f"z-:{det}", *(f"v+:{det}|{g}" for g in self.groups),
                    *(f"v-:{det}|{g}" for g in self.groups), f"miss:{det}"]
        return tuple(out)

    def row(self, z: Sequence[float | None] | np.ndarray, ctx: Context, inactive: Collection[str] = ()) -> np.ndarray:
        """One design row; the columns of `inactive` detectors are all 0, like a closed gate (D-46)."""
        if len(z) != len(self.detectors):
            raise ValueError(f"expected {len(self.detectors)} detector scores, got {len(z)}")
        terms = ctx.terms()
        g = [terms[name] for name in self.groups]
        x = [1.0, *(terms[t] for t in self.context_terms)]
        for j, value in enumerate(z):
            if (self.gates and not self.gate_open(j, terms)) or self.detectors[j] in inactive:
                x += [0.0] * self.block
                continue
            missing = value is None or not math.isfinite(value)
            zp = 0.0 if missing else max(float(value), 0.0)
            zm = 0.0 if missing else min(float(value), 0.0)
            x += [zp, zm, *(zp * gi for gi in g), *(zm * gi for gi in g), 1.0 if missing else 0.0]
        return np.asarray(x, dtype=np.float64)

    def matrix(self, z: np.ndarray, contexts: Sequence[Context]) -> np.ndarray:
        """Design matrix for n items; z is n × J with NaN for a missing detector."""
        z = np.asarray(z, dtype=np.float64)
        if z.shape != (len(contexts), len(self.detectors)):
            raise ValueError(f"z has shape {z.shape}, expected {(len(contexts), len(self.detectors))}")
        if not contexts:
            return np.zeros((0, len(self.names)))
        return np.vstack([self.row(z[i], contexts[i]) for i in range(len(contexts))])
