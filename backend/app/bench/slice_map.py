"""Slice validation map (PRD EA-02; BUILD_SPEC H-7).

- **Leaf slices** are the runtime slice ids (`abstain.image_slice`: `image/<jpeg>/<shorter side>/<largest face>`; video
  `video/<height>/<duration>/<largest face>` and audio `audio/<duration>` since H-9). A leaf
  is measured when both classes have ≥ `MIN_CLASS_N` items. Otherwise it carries the AUC of its nearest measured parent
  (`image/<jpeg>/<side>`, then `image/<jpeg>`, then `image`) and names that parent in `measured_at`. A leaf without
  benchmark items is absent, so inputs like it stay outside the validated envelope. Support is direct, pooled, or
  unsupported; a leaf lacking either class stays unsupported even when a parent AUC is available for context.
- **Field conditions** name the common field inputs. The clean ones (JPEG q ≥ 90 at ≥ 1 MP without a face, JPEG q 70–89,
  lossless, face present) are predicates over the evaluated items; social recompression and screenshots are measured on
  the robustness transforms. Each follows the same rule: measured at n ≥ `MIN_CLASS_N` per class, else merged with the
  nearest measured parent condition.
- **Envelope fraction:** the share of field items whose recorded slice id lies inside a slice map (AUC ≥ 0.70).
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from app.bench import metrics as mt
from app.engine.fusion import abstain

MIN_CLASS_N = abstain.MIN_SUPPORT_CLASS_N
SOCIAL_TRANSFORMS = ("social_ig", "social_x")
SCREENSHOT_TRANSFORMS = ("screenshot", "screenshot_jpeg")


class SliceItem(Protocol):
    y: int
    width: int | None
    height: int | None
    jpeg_q: int | None
    face_px: int | None


class Scored(Protocol):
    items: Sequence[SliceItem]
    s: np.ndarray


def parents(slice_id: str) -> list[str]:
    """Nearest first: `image/q70-89/side512-1024/noface` → `image/q70-89/side512-1024`, `image/q70-89`, `image`."""
    parts = slice_id.split("/")
    return ["/".join(parts[:k]) for k in range(len(parts) - 1, 0, -1)]


def _measure(s: np.ndarray, y: np.ndarray, min_n: int) -> dict[str, Any]:
    n_real, n_fake = int((y == 0).sum()), int((y == 1).sum())
    return {"n_real": n_real, "n_fake": n_fake, "auc": mt.auc(s, y) if min(n_real, n_fake) >= min_n else None}


def _resolve(own: Mapping[str, Any], chain: Sequence[tuple[str, Mapping[str, Any]]], name: str, min_n: int) -> dict[str, Any]:
    """The entry of `name`: its own AUC when measured, else the nearest measured parent's AUC with `measured_at`."""
    out = {"n_real": own["n_real"], "n_fake": own["n_fake"], "auc": own["auc"], "measured_at": None, "note": None,
           "support": "unsupported", "support_n_real": None, "support_n_fake": None}
    if own["auc"] is not None:
        out.update(measured_at=name, support="direct", support_n_real=own["n_real"], support_n_fake=own["n_fake"])
        return out
    for parent, entry in chain:
        if entry["auc"] is not None:
            out.update(auc=entry["auc"], measured_at=parent,
                       support="pooled" if min(own["n_real"], own["n_fake"]) > 0 else "unsupported",
                       support_n_real=entry["n_real"], support_n_fake=entry["n_fake"],
                       note=f"merged with parent slice {parent} ({entry['n_real']} reals, {entry['n_fake']} fakes): "
                            f"fewer than {min_n} items of a class in {name}")
            return out
    out["note"] = f"fewer than {min_n} items of a class in {name} and in every parent slice"
    return out


def leaf_slices(ids: Sequence[str], s: np.ndarray, y: np.ndarray, *, min_n: int = MIN_CLASS_N) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for k, slice_id in enumerate(ids):
        for name in (slice_id, *parents(slice_id)):
            groups[name].append(k)
    measured = {name: _measure(s[idx], y[idx], min_n) for name, idx in groups.items()}
    return {leaf: _resolve(measured[leaf], [(p, measured[p]) for p in parents(leaf)], leaf, min_n)
            for leaf in sorted(set(ids))}


def image_leaf_slices(ev: Scored, *, min_n: int = MIN_CLASS_N) -> dict[str, dict[str, Any]]:
    ids = [abstain.image_slice(i.width, i.height, i.jpeg_q, i.face_px) for i in ev.items]
    recorded = getattr(ev, "slice_ids", None)
    if recorded:
        ids = [sid or fallback for sid, fallback in zip(recorded, ids, strict=True)]
    return leaf_slices(ids, np.asarray(ev.s, dtype=np.float64), np.asarray([i.y for i in ev.items], dtype=np.int64), min_n=min_n)


def video_leaf_slices(ev: Scored, *, min_n: int = MIN_CLASS_N) -> dict[str, dict[str, Any]]:
    """H-9 (EA-02 for video): runtime ids `video/<height>/<duration>/<largest face>` (`abstain.video_slice`), same rule."""
    ids = [abstain.video_slice(i.height, getattr(i, "duration_s", None), i.face_px) for i in ev.items]
    recorded = getattr(ev, "slice_ids", None)
    if recorded:
        ids = [sid or fallback for sid, fallback in zip(recorded, ids, strict=True)]
    return leaf_slices(ids, np.asarray(ev.s, dtype=np.float64), np.asarray([i.y for i in ev.items], dtype=np.int64), min_n=min_n)


def audio_leaf_slices(ev: Scored, *, min_n: int = MIN_CLASS_N) -> dict[str, dict[str, Any]]:
    """Runtime ids `audio/<duration>` (`abstain.audio_slice`), same rule."""
    ids = [abstain.audio_slice(getattr(i, "duration_s", None)) for i in ev.items]
    recorded = getattr(ev, "slice_ids", None)
    if recorded:
        ids = [sid or fallback for sid, fallback in zip(recorded, ids, strict=True)]
    return leaf_slices(ids, np.asarray(ev.s, dtype=np.float64), np.asarray([i.y for i in ev.items], dtype=np.int64), min_n=min_n)


@dataclass(frozen=True)
class Condition:
    id: str
    label: str
    keep: Callable[[SliceItem], bool]
    parent: str | None  # the nearest parent condition; `image` is every evaluated item


def _megapixels(item: SliceItem) -> float:
    return (item.width or 0) * (item.height or 0) / 1e6


CLEAN_CONDITIONS = (
    Condition("image", "every clean test image", lambda i: True, None),
    Condition("jpeg_q>=90", "JPEG q ≥ 90", lambda i: i.jpeg_q is not None and i.jpeg_q >= 90, "image"),
    Condition("jpeg_q>=90_1mp_noface", "JPEG q ≥ 90, ≥ 1 MP, no face",
              lambda i: i.jpeg_q is not None and i.jpeg_q >= 90 and _megapixels(i) >= 1.0 and not i.face_px, "jpeg_q>=90"),
    Condition("jpeg_q70-89", "JPEG q 70–89", lambda i: i.jpeg_q is not None and 70 <= i.jpeg_q < 90, "image"),
    Condition("lossless", "PNG / lossless (no JPEG quantisation tables)", lambda i: i.jpeg_q is None, "image"),
    Condition("face", "face present", lambda i: bool(i.face_px), "image"),
)
ROBUSTNESS_CONDITIONS = (("social", "social recompressed (social_ig, social_x)", SOCIAL_TRANSFORMS),
                         ("screenshot", "screenshot (screenshot, screenshot_jpeg)", SCREENSHOT_TRANSFORMS))
FIELD_CONDITIONS = ("jpeg_q>=90_1mp_noface", "jpeg_q70-89", "lossless", "face", "social", "screenshot")


def _pooled(evs: Iterable[Scored]) -> tuple[np.ndarray, np.ndarray]:
    evs = [e for e in evs if len(e.items)]
    if not evs:
        return np.zeros(0), np.zeros(0, dtype=np.int64)
    return (np.concatenate([np.asarray(e.s, dtype=np.float64) for e in evs]),
            np.concatenate([np.asarray([i.y for i in e.items], dtype=np.int64) for e in evs]))


def field_conditions(ev: Scored, perturbed: Mapping[str, Scored] | None = None, *,
                     min_n: int = MIN_CLASS_N) -> dict[str, dict[str, Any]]:
    """Every field condition with its measurement or merge. Robustness conditions pool the scored copies of their
    transforms; their parent is every scored robustness transform, then every clean test image."""
    s, y = np.asarray(ev.s, dtype=np.float64), np.asarray([i.y for i in ev.items], dtype=np.int64)
    perturbed = perturbed or {}
    measured: dict[str, dict[str, Any]] = {}
    parent_of: dict[str, str | None] = {"robustness": "image"}
    labels: dict[str, str] = {}
    for c in CLEAN_CONDITIONS:
        idx = [k for k, item in enumerate(ev.items) if c.keep(item)]
        measured[c.id], parent_of[c.id], labels[c.id] = _measure(s[idx], y[idx], min_n), c.parent, c.label
    measured["robustness"] = _measure(*_pooled(perturbed.values()), min_n)
    transforms: dict[str, list[str]] = {}
    for cid, label, members in ROBUSTNESS_CONDITIONS:
        transforms[cid] = [t for t in members if t in perturbed]
        measured[cid] = _measure(*_pooled(perturbed[t] for t in transforms[cid]), min_n)
        parent_of[cid], labels[cid] = "robustness", label

    def chain(cid: str) -> list[tuple[str, Mapping[str, Any]]]:
        out, parent = [], parent_of[cid]
        while parent is not None:
            out.append((parent, measured[parent]))
            parent = parent_of[parent]
        return out

    return {cid: {"label": labels[cid], **({"transforms": transforms[cid]} if cid in transforms else {}),
                  **_resolve(measured[cid], chain(cid), cid, min_n)} for cid in FIELD_CONDITIONS}


def envelope_fraction(items: Iterable[Mapping[str, Any]], slices: Mapping[str, Mapping[str, Any]], *,
                      kind: str = "image", min_auc: float = abstain.MIN_SLICE_AUC) -> dict[str, Any]:
    """Field items of `kind` with a recorded slice id (`result.slice_id`): how many lie inside the slice map."""
    rows = [it for it in items if it.get("category") == kind]
    with_id = [(it.get("result") or {}).get("slice_id") for it in rows]
    known = [sid for sid in with_id if sid]
    inside = [sid for sid in known if abstain.envelope(sid, slices, min_auc=min_auc).within]
    by_slice: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "inside": 0})
    for sid in known:
        by_slice[sid]["n"] += 1
        by_slice[sid]["inside"] += int(sid in inside)
    return {"kind": kind, "n": len(rows), "with_slice_id": len(known), "inside": len(inside),
            "fraction": len(inside) / len(known) if known else None, "by_slice": dict(sorted(by_slice.items()))}
