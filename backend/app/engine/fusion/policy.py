"""Fit policy for image size and face context (D-45; ML audit P0-1, recipe steps 1 and 3).

v2.1 image stackers learned "large image means real": additive `min_side` terms and a bias that together separate the
bench classes without any detector input. The D-45 policy for image stackers:

- **No additive size or face terms** (`features.SIZE_FACE_TERMS`). `has_face` stays an interaction group and a gate.
- **Resolution balance.** Reals are weighted to the fakes' min-side distribution in the stacker fit and in Platt
  (`stacker.resolution_weights`); τ and the conformal thresholds stay on raw `cal` reals.
- **One Platt intercept.** Per-context slopes a_k are shrunk as before; every context uses the global b.
- **Neutral size-driven missing.** A detector whose `not_applicable` is size-driven (`detector_meta[...]["missing"] ==
  "size"`, set by `calibrate.load_records` from the score cache reasons) has its missing weight fixed at 0.

J-5 checked the other modalities with gate 2b:

- **Video** (`VIDEO_2B`): the H-8 policy refit on the J-4 scope gave a context-only AUC of 0.8160 on `cal` (limit 0.55), so
  video stackers carry no additive size or face terms either and use one effective intercept. Resolution balance and neutral
  size-driven missing stay off: they are defined on image min-side buckets and size reasons, and the video fit has 214 items.
- **Audio** keeps `LEGACY`: its context-only AUC is 0.5000 (no context term varies on fit).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class SizePolicy:
    additive_size_face_terms: bool
    resolution_balance: bool
    global_intercept: bool
    neutral_size_missing: bool

    def as_json(self) -> dict[str, bool]:
        return dataclasses.asdict(self)


LEGACY = SizePolicy(additive_size_face_terms=True, resolution_balance=False, global_intercept=False, neutral_size_missing=False)
D45 = SizePolicy(additive_size_face_terms=False, resolution_balance=True, global_intercept=True, neutral_size_missing=True)
VIDEO_2B = SizePolicy(additive_size_face_terms=False, resolution_balance=False, global_intercept=True, neutral_size_missing=False)


def default_policy(modality: str) -> SizePolicy:
    return {"image": D45, "video": VIDEO_2B}.get(modality, LEGACY)
