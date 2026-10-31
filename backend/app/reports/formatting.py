"""Deterministic formatting for report templates (UX §6.1 dates and hashes, §7.4 numbers). No locale, no wall clock."""
from __future__ import annotations

import re
from datetime import UTC, datetime

from markupsafe import Markup, escape

from app.core import effect
from app.core.rounding import half_up

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _datetime(value: datetime | str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    return (dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)).astimezone(UTC)


def utc(value: datetime | str | None) -> str:
    """`13 Sep 2026 14:02 UTC`."""
    if value is None:
        return "not recorded"
    dt = _datetime(value)
    return f"{dt.day} {MONTHS[dt.month - 1]} {dt.year} {dt.hour:02d}:{dt.minute:02d} UTC"


def utc_date(value: datetime | str | None) -> str:
    """`13 Sep 2026`."""
    if value is None:
        return "not recorded"
    dt = _datetime(value)
    return f"{dt.day} {MONTHS[dt.month - 1]} {dt.year}"


def utc_seconds(value: datetime | str | None) -> str:
    """`13 Sep 2026 14:02:07 UTC` (custody log)."""
    if value is None:
        return "not recorded"
    dt = _datetime(value)
    return f"{dt.day} {MONTHS[dt.month - 1]} {dt.year} {dt.hour:02d}:{dt.minute:02d}:{dt.second:02d} UTC"


def mmss(seconds: float | None) -> str:
    """`01:02.5` media time."""
    if seconds is None:
        return "–"
    s = max(0.0, float(seconds))
    return f"{int(s // 60):02d}:{s % 60:04.1f}"


def short_hash(sha256: str | None) -> str:
    """First and last 8 hex characters (UX §6.1 running footer)."""
    return "not recorded" if not sha256 else f"{sha256[:8]}…{sha256[-8:]}"


def size(n: int | None) -> str:
    """The size of a file, in bytes in every profile. N-19c (design review, P2 "PDFs"): the summary said "945.4 KB" and the cover
    of the full report "968 054 bytes"."""
    return "size not recorded" if n is None else f"{grouped(n)} bytes"


def grouped(n: int) -> str:
    """`2 481 223` with narrow no-break spaces."""
    return f"{n:,}".replace(",", " ")


def numeral(p: float | None) -> str:
    """The one display formatter of a probability out of 100 (UX §5.3, §7.4; DR P0-2): whole numbers clamped to 1–99, with `<1`
    and `>99` as text. The estimate, the waterfall, the summary and the sign-off all use it. A half goes up, as on the page
    (`rounding.half_up`; answer.ts `numeral`)."""
    if p is None:
        return "–"
    v = p * 100
    return "<1" if v < 1 else ">99" if v > 99 else str(half_up(v))


def span(low: float | None, high: float | None) -> str | None:
    """A plausible range as whole numbers out of 100 clamped to 1–99: `97–99`."""
    if low is None or high is None:
        return None
    return f"{min(99, max(1, half_up(low * 100)))}–{min(99, max(1, half_up(high * 100)))}"


ACQUISITION_WORD = {  # frontend/src/features/cases/custody.ts ACQUISITION_WORD
    "not_stated": "acquisition not stated", "custodian_upload": "upload from custodian", "url_download": "downloaded from URL",
    "screen_capture": "screen capture", "email_forward": "forwarded by email", "quick_upload": "quick upload", "other": "other acquisition",
}
ACTION_WORD = {  # custody.ts WORD: custody actions in words
    "intake": "received", "original_viewed": "original viewed", "original_downloaded": "original downloaded", "hash_verified": "hash verified",
    "hash_mismatch": "hash mismatch", "analysis_started": "analysis started", "analysis_finished": "analysed", "rerun": "re-analysed",
    "manifest_recorded": "manifest recorded", "annotation_created": "annotation added", "annotation_edited": "annotation edited",
    "annotation_deleted": "annotation deleted", "note_added": "note added", "status_changed": "status changed",
    "conclusion_recorded": "conclusion recorded", "signoff": "signed off", "report_generated": "report generated",
    "package_exported": "exported", "redaction": "redacted", "retention_purge": "purged", "deletion_requested": "deletion requested",
    "permission_changed": "members changed", "fuser_loaded": "fusion model loaded", "case_created": "case opened",
    "case_updated": "case details changed", "login": "signed in", "login_failed": "sign-in failed", "policy_changed": "policy changed",
    "webhook_configured": "webhook configured", "fuser_review_recorded": "fusion model review recorded",
    "fuser_review_withdrawn": "fusion model review withdrawn", "signing_key_adopted": "signing key file written",
}
AUTOMATED_REVIEW = "pending human review"  # N-19d: the effective state only (it read "automated fit — pending human review")
IMPLICIT_PREFIX = "UNFILED-"  # app.cases.intake: the implicit case of a quick analysis


def action_word(action: str, details: dict | None = None) -> str:
    if action == "signoff" and (details or {}).get("kind") == "peer_review":
        return "peer review signed"
    return ACTION_WORD.get(action, action.replace("_", " "))


def _automated(value: str) -> bool:
    from app.engine.fusion.artefact import LOOP_REVIEW_FLAG

    return value == LOOP_REVIEW_FLAG


def _on(value: str | None) -> str:
    """` on 1 Oct 2026` for the date of a review record, nothing without one (a result published before N-19d holds none)."""
    try:
        return f" on {utc_date(value)}" if value else ""
    except ValueError:
        return f" on {value}"


def reviewer(value: str | None, date: str | None = None) -> str:
    """The "Reviewed by" cell of a fitted artefact (D-17, D-57; DR P0-3; N-19d): the effective state only. The reviewer of the
    review record with its date, "pending human review" while no record names the artefact (the artefact's own field then
    holds the build loop's flag, which is never printed), or not reviewed."""
    return "not reviewed" if not value else AUTOMATED_REVIEW if _automated(value) else value + _on(date)


def review_phrase(value: str | None, date: str | None = None) -> str:
    """The same in prose: `reviewed by <name> on <date>`, `pending human review`, or `not reviewed`."""
    return "not reviewed" if not value else AUTOMATED_REVIEW if _automated(value) else f"reviewed by {value}{_on(date)}"


def threshold_word(kind: str | None) -> str:
    """An error-rate threshold kind in words: `conformal_0.05` → `false-alarm target 5 %` (DR P0-3)."""
    if kind and kind.startswith("conformal_"):
        try:
            return f"false-alarm target {float(kind.split('_', 1)[1]) * 100:g} %"
        except ValueError:
            pass
    return words(kind)


def case_reference(reference: str | None) -> str:
    """A case reference as printed: the implicit case of a quick analysis is "Unfiled" (DR P2)."""
    return "Unfiled" if not reference or reference.startswith(IMPLICIT_PREFIX) else reference


def words(value: str | None) -> str:
    """An enum as words ("likely_ai" → "likely AI")."""
    return "" if value is None else re.sub(r"\bai\b", "AI", value.replace("_", " "))


def symbols(text: object) -> object:
    """σ, ≈, ≤ and ≥ in detector and system text as words: the report fonts carry no Greek or maths glyphs, and a host fallback
    font breaks the type system (DR P2 fonts; N-19c: "≤ 4 s" in the audio reports was set in Menlo). User data is printed as
    given."""
    if not isinstance(text, str):
        return text
    text = re.sub(r"≤\s*", "up to ", text)
    text = re.sub(r"≥\s*", "at least ", text)
    return text.replace("≈", "about").replace("σ", "sd")


def lower_first(text: str) -> str:
    """Lower-case the first letter for use after a colon ("Overlay: overlay unavailable …" → "Overlay: overlay …")."""
    return text[:1].lower() + text[1:] if text else text


# N-19d (design review DS-19): a validation slice in words. Its id is the build's key ("image/q>=90/side>1024/face<120",
# `app/engine/fusion/abstain.py`): the result document and the attached manifest keep it, a report does not print it.
_SLICE_FIXED = {"lossless": "lossless file", "noface": "no face", "face": "with a face", "jpeg": "JPEG", "unknown": "no validation slice"}
_SLICE_NOUN = {"q": ("JPEG quality ", ""), "side": ("shorter side ", " px"), "face": ("face ", " px"), "res": ("", " px high"),
               "dur": ("duration ", " s"), "words": ("", " words")}
_SLICE_UNKNOWN = {"side": "size unknown", "res": "resolution unknown", "dur": "duration unknown", "words": "length unknown"}
_SLICE_BOUND = re.compile(r"^(q|side|face|res|dur|words)(>=|<=|<|>)(\d+)s?$")
_SLICE_RANGE = re.compile(r"^(q|side|face|res|dur|words)(\d+)-(\d+)s?$")
CONDITION_WORDS = {"slice_id": "validation slice", "jpeg_quality_est": "estimated JPEG quality", "min_face_px": "smallest face (px)",
                   "median_face_px": "median face (px)", "duration_s": "duration (s)", "within_validated_envelope": "validated envelope",
                   "width": "width (px)", "height": "height (px)", "screenshot_likelihood": "screenshot rule (0 or 1)"}


def slice_words(slice_id: str | None) -> str:
    """`image/q>=90/side>1024/face<120` → "JPEG quality 90 or more, shorter side over 1,024 px, face under 120 px". The first
    part (the kind of media) is left out; a part outside the build's grammar keeps its own words."""
    if not slice_id:
        return "not recorded"
    kind, *parts = str(slice_id).split("/")
    out = []
    for part in parts:
        bound, span = _SLICE_BOUND.match(part), _SLICE_RANGE.match(part)
        if bound:
            (before, unit), n = _SLICE_NOUN[bound.group(1)], f"{int(bound.group(3)):,}"
            phrase = {">=": f"{n}{unit} or more", "<=": f"{n}{unit} or less", "<": f"under {n}{unit}", ">": f"over {n}{unit}"}[bound.group(2)]
            out.append(before + phrase)
        elif span:
            before, unit = _SLICE_NOUN[span.group(1)]
            out.append(f"{before}{int(span.group(2)):,} to {int(span.group(3)):,}{unit}")
        elif part.endswith("-unknown") and part.removesuffix("-unknown") in _SLICE_UNKNOWN:
            out.append(_SLICE_UNKNOWN[part.removesuffix("-unknown")])
        else:
            out.append(_SLICE_FIXED.get(part, part.replace("_", " ")))
    return ", ".join(out) or kind


_SLICE_IN_TEXT = re.compile(r"(\(slice |\bslice )?\b((?:image|video|audio|document)/(?:[\w<>=.-]+/)*"
                            r"(?:(?:q|side|face|res|dur|words)(?:[<>]=?\d|\d+-\d|-unknown)|noface|lossless)[\w<>=/-]*)")


def slices_in_text(text: str) -> str:
    """N-19d: a slice id inside a sentence the engine stored (a decision's reason, a limiting condition: "… (slice
    audio/dur4-60s)"), in words. "(slice <id>)" reads "(<words>)", "slice <id>" reads "slice “<words>”", a bare id its words."""
    def said(m: re.Match[str]) -> str:
        words, lead = slice_words(m.group(2)), m.group(1) or ""
        return f"({words}" if lead.startswith("(") else f"slice “{words}”" if lead else words
    return _SLICE_IN_TEXT.sub(said, text)


_FALLBACK_REASON = re.compile(r"(no validation slice applies|weights each check by hand-set weights) \(uncalibrated fallback\)")
# the note of the one assessment a result without a fusion model holds (`schemas/legacy.V1_NOTE`)
_V1_NOTE = re.compile(r"\bv1: hypotheses not separated\b")
QUESTIONS_TOGETHER = "Not assessed separately: without a fusion model one score answers the questions together"
# N-19k: the coverage finding of a result stored before tpl-2.14's later parts names its error-rate context by the key
_CONTEXT_IN_TEXT = re.compile(r"\b(" + "|".join(map(re.escape, effect.CONTEXT_WORDS)) + r") inputs\b")


# N-19q (N-20b's read of the reports, finding 9): the fallback's two stored caveats (`schemas.legacy.FALLBACK_CAVEAT`, the v0.3
# fusion's caveat) as a report prints them: L-01 and L-03 of a document read "probability is not a frequency estimate" and
# "Probabilities are statistical estimates …" beside "Score 16 / 100, not calibrated". The page's copy: `sliceWords.ts`
# N-19q (finding 10): the note of a question that the fusion model has no fitted part for, as a result stores it
# (`runtime.assessments`), in the report's words (no "fuser artefact", no "sub-fuser")
NO_PART_WORDS = (("this fuser artefact has no fitted H1 sub-fuser", "the fusion model has no part fitted for this question"),
                 ("this fuser artefact has no fitted H2 sub-fuser", "the fusion model has no part fitted for this question"))
# N-19q (finding 6): the title of the earlier method's per-frame series under a fusion model (`stages.video.SERIES_TITLE_OUTSIDE`),
# cited by the face-swap check's finding, whose row is in the estimate: "(not in the estimate)" was read as said of that check
EARLIER_SERIES_WORDS = (("Per-frame scores of the earlier method (not in the estimate)",
                         "Per-frame values of the earlier method (the estimate does not read them)"),)
FALLBACK_SCORE_WORDS = (
    ("Uncalibrated fallback: probability is not a frequency estimate", "Uncalibrated fallback: the score is not a frequency estimate"),
    ("Probabilities are statistical estimates from open-source detectors: evidence", "Scores from open-source detectors are evidence"),
)


def stored_words(text: str) -> str:
    """N-19d: a sentence the engine stored, as a report prints it: a slice id in words (`slices_in_text`); the limiting
    condition of a result without a fusion model ("no calibrated fusion model is loaded, so no validation slice applies")
    and the note of a check against its estimate (template sets before tpl-2.12) without the build's name for that path in
    brackets after them; the note of such a result's assessment in words; and the error-rate context of a coverage finding
    ("video T2V inputs", "portrait face inputs") in the words of `context_words`."""
    said = _V1_NOTE.sub(QUESTIONS_TOGETHER, _FALLBACK_REASON.sub(r"\1", slices_in_text(text)))
    # N-19q (finding 9; N-19n's rule): the caveats of the fallback call its number "the score", never a probability
    for old, new in (*FALLBACK_SCORE_WORDS, *NO_PART_WORDS, *EARLIER_SERIES_WORDS):
        said = said.replace(old, new)
    # N-19n: the caveat of a failed step that a result stored with the stage's id in backticks, in that step's words
    return effect.step_in_words(_CONTEXT_IN_TEXT.sub(lambda m: f"{effect.CONTEXT_WORDS[m.group(1)]} inputs", said))


_PERCENT = re.compile(r"(?<=\d)%(?![0-9A-Za-z])")


def percent_spaced(text: str) -> str:
    """N-19d (design review, "57%" against "70 %"): one percent format. The page and the reports write "83 %"; the sentences
    the engine stores write "83%" ("estimate 83%", "assume that 50% of inputs"). A percent sign that stands directly after
    a digit gets the space; one followed by a letter or a digit (an encoded address) is left."""
    return _PERCENT.sub(" %", text)


def holds_percent(text: str) -> bool:
    return bool(_PERCENT.search(text))


# N-19d (design review DS-19): the values of the analysis record in words. The result and the attached manifest keep the
# recorded value ("mps", "commercial+allowlist", "tpl-2.11", "pts-plan-3", "legacy-fuser"); a report prints the words. The
# page's copy is `frontend/src/lib/recordWords.ts`; both read the `record` vectors of `slice_words_vectors.json`.
_DEVICE = {"mps": "Apple GPU", "cuda": "NVIDIA GPU", "cpu": "CPU"}
_PRECISION = {"fp16": "16-bit floats", "float16": "16-bit floats", "fp32": "32-bit floats", "float32": "32-bit floats"}
_POLICY = {"commercial": "commercial licences only", "research": "research use, every licence class", "custom": "a custom licence policy",
           "allowlist": "an allow list", "denylist": "a deny list"}
_VERSIONED = re.compile(r"^(?:tpl|pts-plan)-(\d+(?:\.\d+)*)$")
FUSION_WORDS = {"fallback": "no fusion model (hand-set weights, not calibrated)", "artefact": "calibrated fusion model"}
NO_FUSION_MODEL = "hand-set weights (no fusion model)"
_METHOD = {"legacy-heuristic": "not calibrated", "platt": "Platt scaling", "isotonic": "isotonic regression"}


def device_words(device: object, precision: object = None) -> str | None:
    """`mps`, `fp16` → "Apple GPU · 16-bit floats"; `cuda:1` → "NVIDIA GPU 1"; a value outside the list keeps its words."""
    parts = []
    if device:
        head, _, index = str(device).partition(":")
        parts.append(f"{_DEVICE[head]} {index}".strip() if head in _DEVICE else str(device))
    if precision:
        parts.append(_PRECISION.get(str(precision), str(precision)))
    return " · ".join(parts) or None


def policy_words(label: object) -> str | None:
    """`commercial+allowlist` → "commercial licences only, with an allow list" (as a model card says it)."""
    if not label:
        return None
    head, *rest = (_POLICY.get(part, part.replace("_", " ").replace("-", " ").strip()) for part in str(label).split("+"))
    return head + (", with " + " and ".join(rest) if rest else "")


def version_words(value: object) -> str | None:
    """The sentence templates of an analysis (`tpl-2.11`) and the sampling plan of a video (`pts-plan-3`) by their version:
    "version 2.11", "version 3". Another value is printed as recorded."""
    if not value:
        return None
    match = _VERSIONED.match(str(value))
    return f"version {match.group(1)}" if match else str(value)


def fusion_words(mode: object) -> str | None:
    return FUSION_WORDS.get(str(mode), str(mode)) if mode else None


def calibration_subject(subject: object) -> str:
    """The subject of a calibration record: the name of a fusion model or a check, and for the record of a result without a
    fusion model (`legacy-fuser`, `v1-fuse@…`) what it is."""
    text = str(subject or "")
    return NO_FUSION_MODEL if text == "legacy-fuser" or text.startswith("v1-fuse") else text or "not recorded"


def calibration_method(method: object) -> str:
    return _METHOD.get(str(method), str(method)) if method else "not recorded"


def context_words(context: str) -> str:
    """An error-rate context as a report says it (`effect.CONTEXT_WORDS`): the key "video T2V" (the test videos generated from
    text, the context of a video without a face) reads "text-to-video", "portrait face" reads "portrait"; a key outside the
    table is words already."""
    return effect.context_words(context)


_UNNAMED_TAG = re.compile(r"^(?:(?P<group>[A-Za-z][A-Za-z0-9]*):)?(?:(?P<dec>\d{1,5})|0x(?P<hex>[0-9a-fA-F]{4}))$")


def tag_words(tag: object) -> str:
    """N-19k (the PDF words of both reviews: the EXIF tag "41733"): a metadata tag that the reader has no name for is recorded
    by its number ("41733", "ExifIFD:0xa305"). A table of tags prints it as an unnamed tag with that number, in decimal:
    "unnamed tag 41733". A named tag is printed as recorded. The page's copy is `tagWords` of `provenance.ts`."""
    match = _UNNAMED_TAG.match(str(tag))
    if match is None:
        return str(tag)
    number = int(match["dec"]) if match["dec"] else int(match["hex"], 16)
    return (f"{match['group']}: " if match["group"] else "") + f"unnamed tag {number}"


# N-19k (browser QA "PDF rendering" and its note 16; design re-review, "p(synthetic)=0.2 / 00"): a cell of a narrow column
# may break a long value anywhere, and it broke numbers, dates and tag names with it ("111 / 8", "2026-09-1 / 2",
# "Claim_Generator_InfoNa / me"). A date (also in the form of EXIF, "2026:08:21"), a time of day or a timecode, and a number
# with its sign and its percent sign
_UNBROKEN = re.compile(r"(?<!\d)\d{4}([-:])\d{2}\1\d{2}(?!\d)"
                       r"|(?<![\d:.])\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?![\d:])"
                       r"|(?<![\w.,:/-])[+\-−]?\d+(?:[.,]\d+)*(?:\s?%)?(?!\w)")
UNBROKEN_MAX, NAME_MAX = 14, 32


def unbroken(value: object, name: bool = False) -> Markup:
    """The text of a table cell with what must stay on one line set in a span that does not wrap (`.nb`): each date, time
    and number of at most 14 characters, which fits the narrowest column that prints values; with `name`, the whole value
    where it is one token of at most 32 characters (a tag name, a field name). Anything longer (a hash, a path) breaks as
    before. Everything is escaped as the template would escape it."""
    text = "" if value is None else str(value)
    if name and 2 <= len(text) <= NAME_MAX and not any(ch.isspace() for ch in text):
        return Markup('<span class="nb">{}</span>').format(text)
    out, at = [], 0
    for match in _UNBROKEN.finditer(text):
        if not 2 <= len(match.group(0)) <= UNBROKEN_MAX:
            continue
        out.append(escape(text[at:match.start()]))
        out.append(Markup('<span class="nb">{}</span>').format(match.group(0)))
        at = match.end()
    out.append(escape(text[at:]))
    return Markup("").join(out)


WORD_CELL_CHARS, WORD_CELL_COLUMNS = 18, 6


def word_cell(value: object, columns: int = 1) -> Markup:
    """N-19c: the class of a table cell whose words are short enough to stay whole (`td.words`): every word of at most 18
    characters, in a table of at most six columns, so that six such cells still fit the page. A longer token (a hash, a path,
    a value without spaces) keeps the rule that breaks it anywhere."""
    text = str(value)
    whole = bool(text.strip()) and columns <= WORD_CELL_COLUMNS and all(len(word) <= WORD_CELL_CHARS for word in text.split())
    return Markup(' class="words"') if whole else Markup("")


FILTERS = {"word_cell": word_cell, "unbroken": unbroken, "lower_first": lower_first, "utc": utc, "utc_date": utc_date, "utc_seconds": utc_seconds, "mmss": mmss, "short_hash": short_hash, "size": size,
           "grouped": grouped, "numeral": numeral, "words": words, "symbols": symbols}
