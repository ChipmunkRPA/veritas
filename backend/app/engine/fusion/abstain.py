"""Validation slices and abstention (PRD EA-02; ARCHITECTURE §4.4 "Abstention"; D-02).

- **Slice ids** bucket the condition profile the way the fuser's context does (ML §4.2), so a slice is a set of inputs the
  benchmark can measure:
  - image: `image/<jpeg>/<shorter side>/<largest face>`, e.g. `image/q70-89/side512-1024/face120-256`;
  - video: `video/<height>/<duration>/<largest face>`;
  - document: `document/<length>`;
  - audio: `audio/<duration>` (A-1).
- **Envelope.** Slice metrics come from the loaded artefact (`metrics.test.slices`, written by `vbench-report`). An input
  is eligible with finite AUC ≥ 0.70 and direct or pooled support. A leaf missing either class is unsupported.
  Pooled evidence remains usable when both classes occur in the leaf, and is explicitly identified as broader evidence.
  Legacy records without support metadata retain their AUC policy and are labelled legacy, never direct.
- **Size floor (L-1).** An input smaller than the smallest size at which the bench's test split holds examples of both
  classes (`validated_sizes.json`) is outside the envelope, whatever the AUC of its (pooled) slice.
- **Content the bench does not hold (D-65; N-18).** An image is outside the envelope when it is screenshot-like or
  graphic-like (the screenshot rule below, or a share of flat 8 × 8 blocks above the bench's limit) or greyscale
  (`content_kinds`, `content_kinds.json`). The condition stage measures; `content_outside` gives the reason.
- **Cap.** Outside the envelope, `*_strong`, `ai_supported` and `no_ai_indicators` become `inconclusive`, and the reason
  names the limiting condition. Documentary overrides (trusted C2PA declaration, trusted capture) are not pixel
  statistics and are not capped.
- **Two conditions on a verdict (D-64; N-18).** A strong code needs two checks that count (`strong_needs_two`): with
  one, `*_strong` becomes the code of its side without `_strong`. A check counts when its row in the published view is
  not zero (`checks_entered`; N-19f): a check with a score that the model gates off for the input, or gives no weight on
  the side of its score, does not. An uncalibrated fallback result needs a check of the content
  (`content_check_scored`): with metadata and provenance records only it is `not_assessable`. Both only weaken.
  The runtime and `bench/report.py` call the functions of this module.
- **Not assessable** when no applicable detector produced a usable score, or when every check ran or was not applicable
  and none that scored counts (`NO_WEIGHT_REASON`; N-19f), and no documentary override applies.
- **Limiting conditions** describe what degrades the checks (low resolution, heavy compression, small faces,
  screenshot-like files, short clips or texts, silent sound tracks). They are descriptive; the cap rule is the slice rule.
"""
from __future__ import annotations

import functools
import json
import math
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core import effect, rounding
from app.core.canonical import FLOAT_DP
from app.engine.detector import ON_THIS_SERVER, of_the_server

MIN_SLICE_AUC = 0.70
VALIDATED_SIZES = Path(__file__).with_name("validated_sizes.json")
MIN_SUPPORT_CLASS_N = 100
CAPPED_CODES = frozenset({"ai_supported_strong", "ai_supported", "no_ai_indicators", "no_ai_indicators_strong"})
MODEL_UNUSED = "no calibrated fusion model was used for this analysis"  # N-19k: `runtime.FALLBACK_ENVELOPE_UNUSED`
NOT_ASSESSABLE_VERBAL = "Veritas could not assess this file reliably"  # UX §7.1 basis `insufficient`
# Screenshot rule: a lossless file without EXIF whose size is a common display resolution (either orientation).
SCREEN_SIZES = frozenset({
    (1280, 720), (1366, 768), (1440, 900), (1536, 864), (1600, 900), (1680, 1050), (1920, 1080), (1920, 1200), (2560, 1440),
    (2560, 1600), (2880, 1800), (3024, 1964), (3456, 2234), (3840, 2160), (750, 1334), (828, 1792), (1080, 1920),
    (1080, 2340), (1080, 2400), (1125, 2436), (1170, 2532), (1179, 2556), (1242, 2688), (1284, 2778), (1290, 2796),
    (1440, 3200)})
LOSSLESS_FORMATS = frozenset({"PNG", "BMP", "TIFF"})
# D-65: the conditions under which an image is content the bench does not hold, by the names the bench report counts
FLAT_BLOCKS, SCREENSHOT_RULE, GREYSCALE = "flat_blocks", "screenshot_rule", "greyscale"
CONTENT_CONDITIONS = (FLAT_BLOCKS, SCREENSHOT_RULE, GREYSCALE)
# D-64: a strong code needs this many checks that scored
MIN_CHECKS_FOR_STRONG = 2
WEAKER_CODE = {"ai_supported_strong": "ai_supported", "no_ai_indicators_strong": "no_ai_indicators"}
WEAKER_BAND = {"very_likely_ai": "likely_ai", "very_likely_authentic": "likely_authentic"}
ONE_CHECK_SENTENCE = "One check alone does not give the strongest result, so the result is one step weaker than the estimate's own."
# D-64: families that read records about the file, not its pixels, frames, sound or text
RECORD_FAMILIES = frozenset({"provenance", "codec"})
NO_CONTENT_CHECK_REASON = ("No check of the {content} produced a score: only metadata or provenance records were read, and "
                           "metadata is never evidence of authenticity.")
CONTENT_NAMES = {"image": "pixels", "video": "frames or the sound track", "audio": "sound", "document": "text"}


def jpeg_bucket(q: int | None) -> str:
    if q is None:
        return "lossless"
    return "q>=90" if q >= 90 else ("q70-89" if q >= 70 else "q<70")


def side_bucket(min_side: int | None) -> str:
    if min_side is None:
        return "side-unknown"
    return "side<512" if min_side < 512 else ("side512-1024" if min_side <= 1024 else "side>1024")


def face_bucket(face_px: int | None) -> str:
    if not face_px:
        return "noface"
    return "face<120" if face_px < 120 else ("face120-256" if face_px < 256 else "face>=256")


def image_slice(width: int | None, height: int | None, jpeg_q: int | None, face_px: int | None) -> str:
    min_side = min(width, height) if width and height else None
    return f"image/{jpeg_bucket(jpeg_q)}/{side_bucket(min_side)}/{face_bucket(face_px)}"


def video_slice(height: int | None, duration_s: float | None, face_px: int | None) -> str:
    res = "res-unknown" if not height else ("res<480" if height < 480 else ("res480-719" if height < 720 else "res>=720"))
    dur = "dur-unknown" if duration_s is None else ("dur<4s" if duration_s < 4 else ("dur4-60s" if duration_s <= 60 else "dur>60s"))
    return f"video/{res}/{dur}/{face_bucket(face_px)}"


def document_slice(words: int | None) -> str:
    length = "words-unknown" if words is None else ("words<150" if words < 150 else ("words150-600" if words <= 600 else "words>600"))
    return f"document/{length}"


def screenshot_likelihood(fmt: str | None, width: int | None, height: int | None, has_exif: bool) -> float | None:
    """Rule-based indicator, not a calibrated probability: 1.0 when the file matches the screenshot rule, else 0.0."""
    if not (fmt and width and height):
        return None
    matches = fmt.upper() in LOSSLESS_FORMATS and not has_exif and (width, height) in SCREEN_SIZES | {(h, w) for w, h in SCREEN_SIZES}
    return 1.0 if matches else 0.0


def image_conditions(width: int | None, height: int | None, jpeg_q: int | None, face_px: int | None,
                     screenshot: float | None) -> list[str]:
    out = []
    if width and height and min(width, height) < 512:
        out.append(f"low resolution (shorter side {min(width, height)} px)")
    if jpeg_q is not None and jpeg_q < 70:
        out.append(f"heavy JPEG compression (estimated quality {jpeg_q})")
    if face_px and face_px < 120:
        out.append(f"small faces (largest {face_px} px)")
    if screenshot:
        out.append("screenshot-like file (lossless, screen resolution, no camera metadata)")
    return out


def video_conditions(height: int | None, duration_s: float | None, face_px: int | None, audio_status: str | None, *,
                     audio_decoded_s: float | None = None) -> list[str]:
    out = []
    if height and height < 480:
        out.append(f"low resolution ({height} px high)")
    if duration_s is not None and duration_s < 4:
        out.append(f"short clip ({duration_s:.1f} s)")
    if face_px and face_px < 120:
        out.append(f"small faces (largest {face_px} px)")
    if audio_status == "silent":
        out.append("sound track mostly silent")
    if audio_status in {"analysed", "silent", "short"} and audio_decoded_s is not None and duration_s is not None \
            and duration_s > audio_decoded_s + 0.5:
        out.append(f"audio decoding covered the first {audio_decoded_s:.0f} s; video duration is {duration_s:.0f} s")
    return out


def document_conditions(words: int | None) -> list[str]:
    return [f"short text ({words} words)"] if words is not None and words < 150 else []


def audio_slice(duration_s: float | None) -> str:
    dur = "dur-unknown" if duration_s is None else ("dur<4s" if duration_s < 4 else ("dur4-60s" if duration_s <= 60 else "dur>60s"))
    return f"audio/{dur}"


def audio_conditions(duration_s: float | None, shares: Mapping[str, float], *, codec: str | None, lossless: bool,
                     analysed_s: float | None = None) -> list[str]:
    out = []
    if duration_s is not None and duration_s < 4:
        out.append(f"short recording ({duration_s:.1f} s)")
    if codec and not lossless:
        out.append(f"lossy compression ({codec})")
    if shares.get("speech", 0.0) < 0.5:
        out.append(f"little speech ({round(100 * shares.get('speech', 0.0))} % of the recording)")
    for label in ("music", "noise"):
        if shares.get(label, 0.0) >= 0.2:
            out.append(f"{label} in {round(100 * shares[label])} % of the recording")
    if analysed_s is not None and duration_s is not None and duration_s > analysed_s + 0.5:
        out.append(f"only the first {analysed_s:.0f} s of {duration_s:.0f} s were analysed")
    return out


@dataclass(frozen=True)
class Envelope:
    slice_id: str
    within: bool
    auc: float | None
    reason: str  # the limiting condition when outside; empty when inside
    support: str = "unsupported"  # direct, pooled, unsupported, or legacy (scope not recorded)
    measured_at: str | None = None


def envelope(slice_id: str, slices: Mapping[str, Mapping[str, Any]] | None, *, bench_version: str | None = None,
             min_auc: float = MIN_SLICE_AUC) -> Envelope:
    """The shared serving/benchmark envelope policy; pooled AUC is not direct leaf validation.

    Existing records with ``measured_at`` retain their pooling policy, except an explicitly empty class now abstains.
    New maps carry measurement counts, which must meet the existing 100/class rule. Older AUC-only maps remain readable
    with ``legacy`` support; no missing count is invented. This support check is not a statistical coverage guarantee.
    """
    metrics = (slices or {}).get(slice_id) or {}
    measured_at = metrics.get("measured_at")
    support = "pooled" if measured_at and measured_at != slice_id else "direct" if measured_at == slice_id else "legacy"
    auc = metrics.get("auc")
    if auc is None:
        return Envelope(slice_id, False, None, f"no validation metrics exist for inputs like this (slice {slice_id})")
    auc, min_auc = float(auc), max(MIN_SLICE_AUC, min_auc)
    if not math.isfinite(auc) or not 0 <= auc <= 1:
        return Envelope(slice_id, False, None, f"invalid validation AUC for slice {slice_id}")
    leaf_counts = (metrics.get("n_real"), metrics.get("n_fake"))
    if any(n is not None and n <= 0 for n in leaf_counts):
        return Envelope(slice_id, False, auc, f"this slice lacks validation examples of one class (slice {slice_id}); "
                        "broader evidence cannot establish performance here", "unsupported", measured_at)
    support_counts = (metrics.get("support_n_real"), metrics.get("support_n_fake"))
    if support == "direct" and support_counts == (None, None):
        support_counts = leaf_counts
    if support_counts != (None, None) and any(n is None or n < MIN_SUPPORT_CLASS_N for n in support_counts):
        return Envelope(slice_id, False, auc, f"too few validation examples at {measured_at or slice_id}; "
                        f"at least {MIN_SUPPORT_CLASS_N} of each class are required", "unsupported", measured_at)
    if auc < min_auc:
        where = f"{bench_version}, " if bench_version else ""
        return Envelope(slice_id, False, auc, f"the measured AUC for inputs like this is {auc:.2f}, below "
                        f"{min_auc:.2f} ({where}slice {slice_id})", support, measured_at)
    return Envelope(slice_id, True, auc, "", support, measured_at)


def cap_for_slice(code: str, slice_id: str, slices: Mapping[str, Mapping[str, Any]] | None, *,
                  bench_version: str | None = None, min_auc: float = MIN_SLICE_AUC,
                  below_floor: str | None = None) -> tuple[str, Envelope]:
    """Shared final statistical decision cap; probabilities and conformal sets remain unchanged. `below_floor`: the
    `size_floor` reason, which puts the input outside the envelope."""
    env = envelope(slice_id, slices, bench_version=bench_version, min_auc=min_auc)
    if below_floor and env.within:
        env = Envelope(slice_id, False, env.auc, below_floor, "unsupported", env.measured_at)
    return cap(code, env.within), env


@functools.cache
def validated_sizes() -> Mapping[str, Any]:
    """`validated_sizes.json` by bench manifest sha256 (written by `scripts/v2/validated_sizes.py`)."""
    return json.loads(VALIDATED_SIZES.read_text(encoding="utf-8"))["by_manifest_sha256"]


def size_floor(kind: str, profile: Mapping[str, Any], bench: Mapping[str, Any] | None) -> str | None:
    """L-1: the reason when an input is smaller than the smallest size at which the artefact's bench holds validation
    examples of both classes (image: shorter side; video: frame height), else None. Pooled slice evidence does not reach
    below that size. An artefact whose bench has no entry (fixtures) has no floor."""
    entry = validated_sizes().get(str((bench or {}).get("manifest_sha256")))
    floor = ((entry or {}).get("modalities") or {}).get(kind)
    if not floor:
        return None
    width, height = profile.get("width"), profile.get("height")
    size = (min(width, height) if width and height else None) if kind == "image" else height
    if size is None or size >= floor["floor"]:
        return None
    what = "shorter side" if kind == "image" else "frame height"
    return (f"the {what} is {size} px, below {floor['floor']} px, the smallest size at which the validation set "
            f"({entry['bench']}) holds examples of both classes")


def percent(share: float) -> str:
    return f"{round(share * 100)} %"


def content_conditions(kind: str, profile: Mapping[str, Any], context: Mapping[str, Any] | None,
                       bench: Mapping[str, Any] | None) -> dict[str, str]:
    """D-65 (N-18): the conditions under which an image is of a kind the artefact's bench does not hold, each with its
    reason, in the order they are said: a share of flat 8 × 8 blocks above the bench's limit, else a file that matches
    the screenshot rule; and a greyscale image. `context` is the condition stage's (`flat_block_share`, `greyscale`), or
    the bench report's measurement of the scored bytes. A bench without an entry in `content_kinds.json` (fixtures) has
    no such rule, as it has no size floor; a result whose stage did not measure is not capped by a measurement."""
    from app.engine.fusion import content_kinds

    entry = content_kinds.limits().get(str((bench or {}).get("manifest_sha256")))
    if kind != "image" or not entry:
        return {}
    name, context, reasons = entry["bench"], context or {}, {}
    share, limit = context.get("flat_block_share"), float(entry["flat_blocks"]["limit"])
    if share is not None and share > limit:
        # N-19k (design re-review D2-12): what the image looks like first, then the measurement behind it. The page and the
        # reports cut the reason out of the decision's sentence by a pattern and print it as it is composed here
        reasons[FLAT_BLOCKS] = (
            "it looks like a screenshot of text or an interface, or a flat graphic: "
            f"{percent(share)} of the image's 8 × 8 pixel blocks are flat (one value in each colour channel), above "
            f"{percent(limit)}, which 99.5 % of the images the checks were fitted and calibrated on ({name}) stay below")
    elif (profile.get("screenshot_likelihood") or 0) >= 0.5:
        reasons[SCREENSHOT_RULE] = ("the file matches the screenshot rule (lossless, a screen resolution, no camera metadata), "
                                    f"and screenshots were not measured on the validation set ({name})")
    if context.get("greyscale"):
        grey, total = entry["greyscale"]["n"], entry["n_images"]
        reasons[GREYSCALE] = (f"the image is greyscale, and {grey} of the {total:,} images the checks were fitted and "
                              f"calibrated on ({name}) are")
    return reasons


def content_outside(kind: str, profile: Mapping[str, Any], context: Mapping[str, Any] | None,
                    bench: Mapping[str, Any] | None) -> str | None:
    """The reasons of `content_conditions`, joined; None when the image is of a kind the bench holds."""
    return "; ".join(content_conditions(kind, profile, context, bench).values()) or None


def outside_reason(kind: str, profile: Mapping[str, Any], context: Mapping[str, Any] | None,
                   bench: Mapping[str, Any] | None) -> str | None:
    """Why an input is outside what the artefact's bench validated, whatever its slice says: below the size floor (L-1),
    or content the bench does not hold (D-65). The reasons that apply, joined; None when none does."""
    reasons = [r for r in (size_floor(kind, profile, bench), content_outside(kind, profile, context, bench)) if r]
    return "; ".join(reasons) or None


# N-19f (code review R4-03): the condition stage of an image or a video did not complete, so nothing says whether the input
# is below the size floor or of a content the bench does not hold. The fallback caps such a result as the size floor does
CONDITIONS_NOT_MEASURED = "the conditions of the input could not be measured, so it was not compared with what was validated"


def cap(code: str, within: bool) -> str:
    return "inconclusive" if not within and code in CAPPED_CODES else code


# N-19h (E-20, 8): "outside what was validated", for a result that publishes a number, read from the result itself. It
# is the key of N-10b (the reasons a one-page report gives) and of what stands beside a record: the statistical number is
# printed beside no record there. A calibrated result says it in its condition profile. The fallback has no validated
# envelope at all (`runtime.fallback_profile`: the flag is false for every fallback result), so its results say it in
# their reason: a statistical one in the reason of its cap, one that a provenance record decides in the sentence that
# `stages/common.fallback_fusion` adds to the record's reason (the look that caps a statistical result, made for it too)
OUTSIDE_RECORD_SENTENCE = "This input is outside what was validated: {reason}."
_CAP_REASON = re.compile(r"^Capped at inconclusive: (.+?)(?:; conditions: .*?)?\. Before the cap")
_OUTSIDE_RECORD = re.compile(r" This input is outside what was validated: (.+)\.$")


def cap_reason(decision: Mapping[str, Any]) -> str | None:
    """The validation reason of a decision that the envelope rule or the fallback's look capped ("Capped at inconclusive:
    <reason>[; conditions: …]. Before the cap …"), or None."""
    m = _CAP_REASON.match(str(decision.get("reason") or ""))
    return m.group(1) if m else None


def record_outside_reason(decision: Mapping[str, Any]) -> str | None:
    """Why the input of a fallback result that a provenance record decides is outside what was validated, or None."""
    if decision.get("fusion_mode") != "fallback":
        return None
    m = _OUTSIDE_RECORD.search(str(decision.get("reason") or ""))
    return m.group(1) if m else None


def outside_validated(decision: Mapping[str, Any] | None, profile: Mapping[str, Any] | None) -> bool:
    """Whether the input of a result that publishes a number is outside what was validated (answer.ts `outsideValidated`).
    A result without a number (not assessable; a check did not run and no record decides) is not asked: it has other
    words."""
    if not decision or decision.get("code") == "not_assessable":
        return False
    if decision.get("probability_ai_calibrated") is None and decision.get("probability_ai_uncalibrated") is None:
        return False
    if cap_reason(decision) is not None or record_outside_reason(decision) is not None:
        return True
    return decision.get("fusion_mode") == "artefact" and (profile or {}).get("within_validated_envelope") is False


def outside_why(decision: Mapping[str, Any] | None, profile: Mapping[str, Any] | None) -> str | None:
    """N-19n (browser QA, the rest of Q2-14): why the input of a calibrated result with a number is outside what was
    validated, also where the cap changed no code (the result is inconclusive by its own estimate, so its reason names no
    cap): the reason of the cap, else the envelope's own reason, which the runtime puts last among the limiting
    conditions of an input outside the envelope. None for any other result."""
    if not decision or decision.get("fusion_mode") != "artefact" or not outside_validated(decision, profile):
        return None
    capped = cap_reason(decision)
    if capped is not None:
        return capped
    conditions = [str(c) for c in (profile or {}).get("limiting_conditions") or []]
    return conditions[-1] if conditions else None


# N-19f (code review R4-05, E-18): every check ran or was not applicable, checks scored, and no row of the published view is
# not zero. The reason of the decision; the reports and the page know the case by it (`view.no_weight`, answer.ts `noWeight`)
NO_WEIGHT_REASON = "For an input like this one the fusion model gives the checks that scored no weight."


def row_counts(row: Any, outside: Collection[str] = ()) -> bool:
    """N-19i (R5-01): whether a row of the published view is one of the checks a decision counts. One answer for "k of n
    detectors", D-64, "no check entered", D-76 and what the runtime publishes with the row (`Contribution.counted`), which
    the number line, the count "in the estimate" and the sentence on how many checks entered the model read."""
    return (row.kind == "detector" and row.z is not None and row.detector not in outside
            and effect.entered(round(row.logit, FLOAT_DP)))


def checks_entered(rows: Iterable[Any], outside: Collection[str] = ()) -> int:
    """D-64 as clarified on 2026-10-04 (N-19f): the checks that count in a decision. `rows`: the rows of the published view
    (`Fuser.neutral_view`, D-63). A check counts when it scored and its row as published (the canonical six decimals) is not
    zero by the rule the number line uses (`effect.entered`; `reports/view.scored_checks`), also on the side where its
    score has no weight and the row is the step from its neutral point. `outside`: detectors that are not counted (H3).
    Not the fuser's own decomposition, whose rows are zero elsewhere."""
    return sum(1 for r in rows if row_counts(r, outside))


def strong_needs_two(code: str, n_scored: int) -> str:
    """D-64: with fewer than two checks that count (`checks_entered`), a strong code becomes the code of its side without
    `_strong`. Every other code is returned as it is."""
    return WEAKER_CODE.get(code, code) if n_scored < MIN_CHECKS_FOR_STRONG else code


# D-76 (N-19i; browser QA re-review Q2-01, P1): a decisive code needs a check that supports it. A calibrated result is
# decisive when the conformal set of its estimate holds one answer, and nothing asked that a check supports that answer: a
# camera original read "Indications of AI generation or manipulation", 68 %, on a start of 53 and three rows that each
# "had little effect". The side of each decisive code, and the sentence that ends the reason of a result the rule changed
# (the page and the reports know such a result by it: `reports/view.unsupported`, answer.ts `unsupported`)
SIDE_OF_CODE = {"ai_supported_strong": 1, "ai_supported": 1, "no_ai_indicators": -1, "no_ai_indicators_strong": -1}
# the sentence names the answer of the estimate (the result itself is inconclusive, so nothing else in it says which side
# the estimate was on), and ends in words that are the same for both sides
NO_SUPPORT_ANSWER = {1: "AI generation or manipulation", -1: "No AI generation or manipulation"}
NO_SUPPORT_REASON = ("is the answer of the estimate, and no check gives at least weak support for it: the estimate rests on the "
                     "model's starting point for inputs like this one, so the result is inconclusive.")


def no_support_reason(code: str) -> str:
    """The sentence that ends the reason of a result whose decisive `code` no check supports."""
    return f"{NO_SUPPORT_ANSWER[SIDE_OF_CODE[code]]} {NO_SUPPORT_REASON}"


def unsupported_side(decision: Mapping[str, Any] | None) -> str | None:
    """D-76: "for" or "against" (AI generation or manipulation), the answer of the estimate of a result that is inconclusive
    because no check supports that answer; None for any other result. Read from the reason, as a result without weight
    is known by its reason (N-19f).
    N-19m (D-76 as amended): in either form, no single check or the checks together; a result that D-77 took is not
    D-76's (`rule_taken`)."""
    taken = rule_taken(decision)
    return taken[1] if taken is not None and taken[0] != RULE_SCALE else None


def supported(code: str, rows: Iterable[Any], outside: Collection[str] = ()) -> bool:
    """D-76: whether a check supports the code. `rows`: the rows of the published view (`Fuser.neutral_view`, D-63). A check
    supports a decisive code when it counts for D-64 (`checks_entered`: it scored, it is outside `outside`, the detectors of
    H3) and its row as published (the canonical six decimals) is at least ln 2 toward the side of the code (`effect.LN2`,
    the page's "weak support"). One check, not the sum of several; the start and a "missing" term are not checks. True
    for a code that is not decisive: the rule asks nothing of it."""
    side = SIDE_OF_CODE.get(code)
    if side is None:
        return True
    return any(row_counts(r, outside) and side * round(r.logit, FLOAT_DP) >= effect.LN2 for r in rows)


# N-19m, third part (D-77; D-76 as amended on 2026-10-05; browser QA Q3-01, P0, and Q3-10, P1). Two conditions of a decisive
# code, asked where D-76 is asked and in this order. (1) D-77: the estimate as published, in whole points, stands in the
# part of the scale of the code's answer (a clean headline stood at "49 % estimated likelihood": in the image model's
# low-quality context the AI answer leaves the 5 % set below 0.496). (2) D-76: the checks that count support the code, one
# of them (`supported`) and all of them together: their published rows sum to at least ln 2 toward its side (a clean
# headline stood on a start of 13 with rows of +2.36 and −1.29). The names of the conditions, as the stage's output and
# the decision's `statistical_alone` give them (`rule`), and the sentence that ends the reason of a result each one took
RULE_SCALE, RULE_NO_CHECK, RULE_TOGETHER_LESS, RULE_TOGETHER_AGAINST = "scale", "no_check", "together_less", "together_against"
SCALE_CLEAN_MAX, SCALE_AI_MIN = 34, 65  # the outer segments of the scale (`core.evidence.band_for`), in printed points
OFF_SCALE_REASON = ("is the answer of the calibrated levels for inputs like this one, but the estimate stands in the middle part "
                    "of the scale, so the result is inconclusive.")
TOGETHER_LESS_REASON = ("is the answer of the estimate, and a check gives at least weak support for it, but the checks together "
                        "give it less than weak support, so the result is inconclusive.")
TOGETHER_AGAINST_REASON = ("is the answer of the estimate, and a check gives at least weak support for it, but the checks "
                           "together point the other way, so the result is inconclusive.")
RULE_REASONS = {RULE_SCALE: OFF_SCALE_REASON, RULE_NO_CHECK: NO_SUPPORT_REASON, RULE_TOGETHER_LESS: TOGETHER_LESS_REASON,
                RULE_TOGETHER_AGAINST: TOGETHER_AGAINST_REASON}
# N-19n (supervisor check 29): under a configured prior the estimate that D-77 reads need not stand in the middle part. The
# code is taken from the calibrated levels before the prior's shift, and the estimate is published after it: at a prior of
# 0.1 a picture with a face to which the levels give the AI code at 70 is published at 21, and at 0.9 a small picture to
# which they give the clean code at 30 is published at 79. The reason of such a result has an ending of its own (it said
# "the middle part of the scale" over a printed 21). It names no side: the reason names the answer of the levels first
OFF_SCALE_OTHER_REASON = ("is the answer of the calibrated levels for inputs like this one, but the estimate stands in the other "
                          "answer's part of the scale, so the result is inconclusive.")
# every ending by which a result says which condition took its code (`rule_taken`; answer.ts `RULE_REASONS`): D-77 has two
RULE_ENDINGS = ((RULE_SCALE, OFF_SCALE_REASON), (RULE_SCALE, OFF_SCALE_OTHER_REASON), (RULE_NO_CHECK, NO_SUPPORT_REASON),
                (RULE_TOGETHER_LESS, TOGETHER_LESS_REASON), (RULE_TOGETHER_AGAINST, TOGETHER_AGAINST_REASON))
# N-19n: the caveat of a configured prior (`VERITAS_PRIOR_AI`), which the runtime writes with the share between these
# words. A result that a condition took says of its estimate that it is adjusted for that share (`reports/view.py`
# `prior_share`, answer.ts `priorShare`); the caveat is known by its fixed words, not by its number
PRIOR_CAVEAT = ("Probabilities assume that ", " of inputs like this are AI-generated or manipulated (configured prior); without it "
                "they assume 50:50.")


def prior_caveat(prior_ai: float) -> str:
    return f"{PRIOR_CAVEAT[0]}{prior_ai:.0%}{PRIOR_CAVEAT[1]}"


def published_points(estimate: float) -> int:
    """D-77: the estimate as a result prints it, in whole points. `estimate`: the statistical estimate as the runtime
    holds it (after a configured prior, clamped). It is published at the canonical six decimals and printed by
    `reports.formatting.numeral` and the page's `numeral`, a half upwards; the engine does not import `reports`, so the
    points are taken here (a test holds the two equal). Asked of the unrounded value, 0.3449997 would keep a clean code
    over a result that is published as 0.345 and prints 35."""
    return rounding.half_up(round(estimate, FLOAT_DP) * 100)


def part_of_scale(estimate: float) -> str | None:
    """N-19n: the part of the scale in which the estimate as published prints: "against" for the clean answer's part (34
    or less), "for" for the AI answer's (65 or more), None for the middle part (35 to 64)."""
    points = published_points(estimate)
    return "against" if points <= SCALE_CLEAN_MAX else "for" if points >= SCALE_AI_MIN else None


def in_part_of_scale(code: str, estimate: float) -> bool:
    """D-77: whether the estimate as published stands in the part of the scale of the code's answer: 34 or less for a
    code of no indications, 65 or more for an AI code. True for a code that is not decisive: the rule asks nothing of it."""
    side = SIDE_OF_CODE.get(code)
    if side is None:
        return True
    points = published_points(estimate)
    return points >= SCALE_AI_MIN if side > 0 else points <= SCALE_CLEAN_MAX


def together(code: str, rows: Iterable[Any], outside: Collection[str] = ()) -> float:
    """D-76, the second condition: the sum of the published rows (the canonical six decimals) of the checks that count,
    toward the side of the decisive `code`."""
    return SIDE_OF_CODE[code] * math.fsum(round(r.logit, FLOAT_DP) for r in rows if row_counts(r, outside))


def rule_for(code: str, estimate: float, rows: Iterable[Any], outside: Collection[str] = ()) -> str | None:
    """The condition that takes a decisive `code`, in the order of the decisions; None where both are met, and for a code
    that is not decisive. D-77 first (`RULE_SCALE`), then D-76: no single check (`RULE_NO_CHECK`), else the checks
    together: less than ln 2 toward the code or a sum of zero (`RULE_TOGETHER_LESS`), a sum toward the other answer
    (`RULE_TOGETHER_AGAINST`). One function for the runtime and the benchmark report, so that a mark is a result's reason."""
    if code not in SIDE_OF_CODE:
        return None
    rows = list(rows)
    if not in_part_of_scale(code, estimate):
        return RULE_SCALE
    if not supported(code, rows, outside):
        return RULE_NO_CHECK
    total = together(code, rows, outside)
    return None if total >= effect.LN2 else RULE_TOGETHER_LESS if total >= 0 else RULE_TOGETHER_AGAINST


def rule_reason(code: str, rule: str, estimate: float | None = None) -> str:
    """The sentence that ends the reason of a result whose decisive `code` the condition `rule` took. `estimate`: the
    statistical estimate as the runtime holds it. N-19n: where D-77 took the code and the estimate as published stands
    in the part of the scale of the other answer (a configured prior), the ending says so."""
    ending = RULE_REASONS[rule]
    if rule == RULE_SCALE and estimate is not None and part_of_scale(estimate) is not None:
        ending = OFF_SCALE_OTHER_REASON
    return f"{NO_SUPPORT_ANSWER[SIDE_OF_CODE[code]]} {ending}"


def rule_taken(decision: Mapping[str, Any] | None) -> tuple[str, str] | None:
    """(the condition, the side) of a result that is inconclusive because a condition took the decisive code of its
    estimate, read from the end of its reason (answer.ts `ruleTaken`); the side is "for" or "against" AI generation or
    manipulation, the answer that the result does not give. None for any other result."""
    if not decision or decision.get("code") != "inconclusive" or decision.get("fusion_mode") != "artefact":
        return None
    # the sentence stands after another one: matched from its first word, since "No AI generation …" ends as "AI generation …" does
    reason = str(decision.get("reason") or "")
    return next(((rule, "for" if side > 0 else "against") for rule, ending in RULE_ENDINGS
                 for side, answer in NO_SUPPORT_ANSWER.items() if reason.endswith(f". {answer} {ending}")), None)


def content_check_scored(entries: Sequence[Mapping[str, Any]]) -> bool:
    """D-64: whether a check of the pixels, frames, sound or text scored. `entries`: `{"family", "scored"}` of each check
    that entered the result. Metadata, provenance and container records (`RECORD_FAMILIES`) are not checks of the
    content; a check without a family counts as one, so that a result is never withdrawn for a missing label."""
    return any(e.get("scored") and e.get("family") not in RECORD_FAMILIES for e in entries)


def no_content_check_reason(kind: str) -> str:
    return NO_CONTENT_CHECK_REASON.format(content=CONTENT_NAMES.get(kind, "content"))


def not_assessable_reason(entries: Sequence[Mapping[str, Any]]) -> str:
    missing = [e for e in entries if e.get("status") != "ok"]
    if not entries:
        return "No check that tests for AI generation or manipulation ran on this file."
    if not missing:
        # L-3: never "too few checks" when all of them ran; N-17 (D-68): and no "fused estimate" where none was formed
        return "Every check ran, but none that tests for AI generation or manipulation produced a score that can be used."
    # N-14a: a check skipped for the server's memory or for weights that are not installed is said as the server's
    names = "; ".join(f"{e.get('name') or e.get('detector_id')} "
                      f"({ON_THIS_SERVER if of_the_server(e.get('status'), e.get('reason')) else e.get('status')}: "
                      f"{e.get('reason') or 'no reason given'})" for e in missing[:6])
    return f"Too few checks could run: {names}."
