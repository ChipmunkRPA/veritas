"""User-facing "not validated" disclosures (L-3; PRD Q6; `not_validated.yaml`, `docs/limitations.md`).

`not_validated.yaml` says each unvalidated family, stratum, transform or lane is disclosed "in the report". This module holds the
plain-words disclosure of every entry and the rule for which results it affects; the analysis API (`not_validated`), the UI summary
and the PDF all read `for_result`, so the screens and the reports say the same thing. Every figure is copied from
`docs/limitations.md` (vb-1.1); `tests/test_l3_findings_copy.py` checks that each yaml entry has a disclosure.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

Result = Mapping[str, Any]


def _kind(*kinds: str) -> Callable[[Result], bool]:
    return lambda r: r.get("kind") in kinds


def _has_face(r: Result) -> bool:
    profile = r.get("condition_profile") or {}
    return r.get("kind") == "image" and bool(profile.get("min_face_px") or ("/face" in str(profile.get("slice_id", ""))
                                                                          and "/noface" not in str(profile.get("slice_id", ""))))


def _ran(*detectors: str) -> Callable[[Result], bool]:
    return lambda r: any(s.get("detector_id") in detectors and s.get("status", "ok") == "ok" for s in r.get("signals") or [])


@dataclass(frozen=True)
class Disclosure:
    id: str  # the `not_validated.yaml` entry id
    applies: Callable[[Result], bool]
    label: str  # a short name for one-line lists ("Not validated for this result: …")
    text: str


DISCLOSURES: tuple[Disclosure, ...] = (
    Disclosure("image.families", _kind("image"), "generator families",
               "Performance varied when the earlier benchmark left a generator family out of training. The check flagged "
               "58 % of generated images on average and 25 % from the weakest tested family."),
    Disclosure("image.real_strata", _kind("image"), "kinds of genuine photograph",
               "In an earlier benchmark, 3.6 % of genuine camera originals and 11.7 % of genuine social-platform images were "
               "flagged. A separate field test flagged 5 of 52 genuine camera originals."),
    Disclosure("image.calibration", _kind("image"), "calibration",
               "Calibration in an earlier test did not meet its accuracy criterion. Error for this particular upload is unknown."),
    Disclosure("image.transforms", _kind("image"), "re-saved and edited copies",
               "Re-saved and edited copies are not validated. Detection falls by 13 to 28 points after JPEG or WebP re-saving or "
               "halving the size, and false alarms rise by up to 23 points with added noise or a screenshot."),
    Disclosure("image.faces", _has_face, "AI-generated faces",
               "An earlier image benchmark flagged 58 of 92 generated faces and 95 of 185 face-swap stills. Results for this "
               "image depend on the type of face change and the condition of the file."),
    Disclosure("image.localisers", _ran("loc.mesorch", "loc.radar", "loc.fused"), "region maps",
               "Region maps are not validated and carry no weight in the decision. An earlier evaluation of one local editing "
               "check marked regions on 36 % of genuine photographs; this is not an error rate for every check or image. "
               "A marked region is not evidence of an edit on its own."),
    Disclosure("image.splice_regions", _ran("loc.splice"), "local editing regions",
               "The PNG region check looks for local recompression differences. It can flag ordinary processing, "
               "and small or weak edits can go unmarked. Region measurements carry no weight in the whole-image estimate."),
    Disclosure("image.native_regional", _ran("loc.native_regional"), "sensitive regional screening",
               "The sensitive regional check finds editing cues and can also highlight untouched images or ordinary edits. "
               "AI authorship is inconclusive from this check alone. Its uncalibrated responses and regional highlights "
               "carry no weight in the overall authorship confidence."),
    Disclosure("image.unadmitted_detectors", _ran("img.spai", "img.univfd"), "supplementary image checks",
               "Supplementary image checks can show individual scores and carry no weight in the decision."),
    Disclosure("video.synthesis", _kind("video"), "fully generated video",
               "Detection varied by generator in an earlier video benchmark. In separate benchmark comparisons, one generator "
               "had 4 of 19 generated clips flagged and the genuine set had 4 of 162 clips flagged. These counts describe "
               "those test sets. Performance on this upload is unknown."),
    # N-17 (browser QA P1-5 c): the entry's checks are the lip-sync checks and the face-swap check of a whole track. The synthesis
    # frame checks and the per-frame face-swap check are fitted, by their one score for the whole video.
    Disclosure("video.lanes", _kind("video"), "the lip-sync and face-track lanes, and the segments of every lane",
               # N-19q (N-20b's read of the reports, finding 6): the sentence was read as said of the face-swap detectors on the
               # tracked face, whose row the breakdown shows in the fused model; it names the two apart
               "The lip-sync checks and the face-swap check that scores a whole face track at once are not in the fitted model: "
               "their lanes are shown for inspection only and carry no weight in the decision. The face-swap detectors on the "
               "tracked face, which score its frames, are in it. The segments flagged on any lane are not "
               "validated either: the decision uses each fitted check's one score for the whole video, as listed where the estimate "
               "is explained."),
    Disclosure("audio.synthetic_voice", _kind("audio"), "cloned voices",
               "An earlier voice benchmark flagged only 1 of 89 in-the-wild cloned voice recordings. A low score provides "
               "limited evidence about whether a voice was cloned."),
)


# D-68: every figure in the table above was measured on the calibrated fusion model. A result of the uncalibrated fallback quotes
# none of them: nothing was measured for the fallback, and that is its one disclosure. Not a `not_validated.yaml` entry.
FALLBACK = Disclosure("fallback.fusion", lambda r: (r.get("decision") or {}).get("fusion_mode") == "fallback",
                      "the uncalibrated fallback",
                      "Nothing in this result is validated: it comes from the uncalibrated fallback with hand-set weights, for "
                      "which no detection rate, false-alarm rate or calibration error has been measured.")


# N-19n (design review D3-02): under a provenance record that decides, the result is the record's, and what is not validated is
# the statistical checks beside it ("Nothing in this result is validated" stood under "A signed record inside the file,
# validated as untampered, …"). Read from the decision, so a stored result says it too: a record decides where the decision
# names it, or where H3 drives (a result stored before the decision named its record)
FALLBACK_BESIDE_RECORD = ("The statistical checks of this result are not validated: they are combined by the uncalibrated fallback "
                          "with hand-set weights, for which no detection rate, false-alarm rate or calibration error has been "
                          "measured.")
H3 = "H3_provenance_integrity"


def record_decides(result: Result) -> bool:
    decision = result.get("decision") or {}
    return bool(decision.get("provenance_record")) or decision.get("driving_hypothesis") == H3


def for_result(result: Result) -> list[dict[str, str]]:
    """The disclosures that affect one published result (kind, condition profile, signals), in table order.

    The measured figures are those of the fusion that produced the result (D-68): a fallback result gets the one fallback
    disclosure, on the screen (the analysis API) and in the reports (since report template set r21)."""
    from app.detectors.audio.calibration import NOTE, SHA256

    if (result.get("decision") or {}).get("fuser_sha256") == SHA256:
        return [{"id": "audio.local_screening", "label": "local speech calibration", "text": NOTE}]
    if FALLBACK.applies(result):
        text = FALLBACK_BESIDE_RECORD if record_decides(result) else FALLBACK.text
        return [{"id": FALLBACK.id, "label": FALLBACK.label, "text": text}]
    return [{"id": d.id, "label": d.label, "text": d.text} for d in DISCLOSURES if d.applies(result)]


def ids() -> Sequence[str]:
    return [d.id for d in DISCLOSURES]
