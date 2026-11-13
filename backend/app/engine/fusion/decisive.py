"""Stage 0 provenance overrides (D-15, D-33; ARCHITECTURE §4.4 Stage 0; ML §4.7) and the bound on other provenance states.

- **Overrides**, in precedence order:
  1. `decisive_ai`: a C2PA manifest that validates against the trust list and declares generative AI. s ≥ 0.97.
  2. `trusted_capture`: a trusted capture manifest covering the current bytes. s ≤ 0.10.
  Unsigned generator metadata, untrusted declarations and public-message watermarks are advisory only.
  Their historical `self_declared_ai` classification remains readable, but never changes a fresh estimate.
  Signed documentary evidence outranks unsigned markers, so a trusted capture manifest wins over a self-declared marker;
  the conflict is stated in a caveat.
- **The statistical estimate** before the override is always kept and shown (invariant 4, D-33).
- **Every other provenance state** (absent or valid-untrusted credentials, invalid credentials, EXIF) may move s by at most
  `settings.h3_influence_bound` (default |Δs| ≤ 0.05, DC-06). `bound_logit` enforces that against the same input with
  those detectors' contributions removed.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from app.engine.fusion.calibration import clamp_probability

# The self-declared value is retained only to interpret historical records.
OVERRIDE_PROBABILITY = {"decisive_ai": 0.97, "self_declared_ai": 0.85, "trusted_capture": 0.10}
PRECEDENCE = ("decisive_ai", "trusted_capture", "self_declared_ai")
ACTIVE_PRECEDENCE = ("decisive_ai", "trusted_capture")
ADVISORY_CAVEAT = ("An unverified origin marker was detected. It does not authenticate AI generation, "
                   "does not change the content estimate and cannot stop further checks.")
MARKER_DETECTORS = frozenset({"img.meta.generator", "img.watermark.dwtdct", "prov.c2pa", "prov.c2pa.video", "prov.c2pa.audio", "prov.c2pa.pdf"})
H3_INFLUENCE_BOUND = 0.05
CONFLICT_LOGIT = 3.0  # ML §4.7: a statistical estimate beyond |η'| > 3 in the opposite direction adds a caveat
SELF_DECLARED_CAVEAT = ("Self-declared: the file's metadata names an AI generator. Metadata can be added or edited by anyone, "
                        "so it can be added to a genuine image.")
SELF_DECLARED_WATERMARK_CAVEAT = ("Self-declared: the image carries a public AI-generator watermark message. The message is public, "
                                  "so anyone can embed it in a genuine image.")
WATERMARK_DETECTORS = frozenset({"img.watermark.dwtdct"})
STATISTICAL_ESTIMATE_CAVEAT = "Statistical estimate before the provenance override: {:.0%}."
# D-68: on the uncalibrated fallback the number before the override is a score, printed as one
FALLBACK_STATISTICAL_CAVEAT = "Statistical score before the provenance override: {:.0f} / 100, not calibrated."
CONFLICT_CAVEAT = ("The detectors' statistical estimate points strongly the other way; review the provenance record and the detector "
                   "results together.")
# D-49: while a check did not run no statistical estimate is published, so these caveats are dropped (`runtime.fuse_with_artefact`)
STATISTICAL_CAVEATS = (STATISTICAL_ESTIMATE_CAVEAT.split(":", 1)[0], FALLBACK_STATISTICAL_CAVEAT.split(":", 1)[0], CONFLICT_CAVEAT)
# N-19h (E-20, 8): outside what was validated a statistical number is not shown where it stands alone (D-65, D-68), and it
# is printed beside no record either. The caveat says that it is not shown, and why; the decision's field keeps the number
STATISTICAL_WITHHELD_CAVEAT = ("The statistical estimate before the provenance override is not shown: this input is outside what "
                               "was validated.")
FALLBACK_STATISTICAL_WITHHELD_CAVEAT = ("The statistical score before the provenance override is not shown: this input is outside "
                                        "what was validated.")
_NUMBER_CAVEATS = (STATISTICAL_CAVEATS[0] + ":", STATISTICAL_CAVEATS[1] + ":")


def withhold_statistical(caveats: Iterable[str], *, fallback: bool = False) -> list[str]:
    """`caveats` with the caveat that prints the statistical number before the override replaced, in its place, by the one
    that says that it is not shown (a caveat keeps its line: an examiner acknowledges limitations by their number). The
    form follows the caveat that is replaced: an estimate of a fusion model, or a score of the fallback (also for a
    result stored before N-7, whose fallback caveat spoke of an estimate: `fallback`)."""
    out = []
    for c in caveats:
        if str(c).startswith(_NUMBER_CAVEATS[1]) or (fallback and str(c).startswith(_NUMBER_CAVEATS[0])):
            out.append(FALLBACK_STATISTICAL_WITHHELD_CAVEAT)
        elif str(c).startswith(_NUMBER_CAVEATS[0]):
            out.append(STATISTICAL_WITHHELD_CAVEAT)
        else:
            out.append(c)
    return out
# the fallback's reasons for a documentary override (image and video); the calibrated path words its own (`runtime`)
FALLBACK_DECLARED_AI_REASON = ("A C2PA manifest that validates against the C2PA Trust List declares generative AI. This documentary "
                               "evidence decides the result.")
FALLBACK_TRUSTED_CAPTURE_REASON = ("A trusted C2PA capture manifest covers the current bytes, so the signed capture state decides the "
                                   f"result and caps the uncalibrated score at {OVERRIDE_PROBABILITY['trusted_capture'] * 100:.0f} / 100.")


def fallback_floor_reason(source: str, probability: float, marker: str | None = None) -> str:
    """The self-declared floor on the fallback: `source` is what the file carries; the floor is a score, not a likelihood.
    N-19h: `marker`, the kind of a marker that is Content Credentials, whose words say who can sign such credentials (or,
    for a signer on the trust list, nothing of the kind); every other marker keeps "it can be added or edited by anyone"."""
    floor = f"{source}. This self-declared evidence floors the uncalibrated score at {probability * 100:.0f} / 100"
    if marker in (CREDENTIALS_LISTED, CREDENTIALS_UNLISTED_DECLARED, CREDENTIALS_UNLISTED_NAMED):
        anyone = marker_anyone(marker)
        return floor + (f"; {anyone}." if anyone else ".")
    return f"{floor}; it can be added or edited by anyone."


# N-19h (E-21): a result says which record its decision applied (`record`), and of which kind a marker is. A marker is
# unsigned metadata that names a generator, a public watermark message, or Content Credentials that are no trusted
# declaration, in three forms: a generator named under a signer on the trust list (no AI source type declared); under a
# signer that is not on it, a declaration of AI, or a generator named
CREDENTIAL_DETECTORS = ("prov.c2pa", "prov.c2pa.video", "prov.c2pa.audio", "prov.c2pa.pdf")
METADATA, WATERMARK = "metadata", "watermark"
CREDENTIALS_LISTED, CREDENTIALS_UNLISTED_DECLARED, CREDENTIALS_UNLISTED_NAMED = (
    "credentials_listed_signer", "credentials_unlisted_signer_declared", "credentials_unlisted_signer_named")
# N-19n (Q4-01; D-15 as amended on 2026-10-05): unsigned metadata whose marker is the IPTC digital source type names no
# generator. It is a kind of its own, worded by what the file holds: a declaration of AI generation, or of a composite with
# AI-generated parts. The metadata check hands the kind on in its metrics (`names_generator`, `source_type_trained`,
# `source_type_composite`); a result stored before holds none of them and keeps the kind of a named generator
METADATA_DECLARED, METADATA_COMPOSITE = "metadata_declared", "metadata_declared_composite"
# N-19n (D3-07, 9; D-15 as amended): unsigned metadata is worded by what the file holds. A declaration of AI generation
# without a named generator is one by the source type or by the phrase "AI generated" in a field that software writes
# (`phrase_declared`); generation parameters that match no fingerprint and name no tool are a kind of their own
# (`sd_params` without `names_generator`). A generator that the metadata names, or whose form it has, keeps its words
METADATA_PARAMETERS = "metadata_parameters"
MARKER_KINDS = (METADATA, WATERMARK, CREDENTIALS_LISTED, CREDENTIALS_UNLISTED_DECLARED, CREDENTIALS_UNLISTED_NAMED,
                METADATA_DECLARED, METADATA_COMPOSITE, METADATA_PARAMETERS)
UNSIGNED_METADATA = (METADATA, METADATA_DECLARED, METADATA_COMPOSITE, METADATA_PARAMETERS)


def marker_kind(entries: Iterable[dict]) -> str | None:
    """The kind of the self-declared marker among `entries` (detector entries or signals: `detector_id`, `metrics`,
    `provenance_override`); None without one. A watermark message alone is a watermark; with another marker beside it the
    words are the other marker's, the first in the order of the entries (as `watermark_only` words them today)."""
    markers = [e for e in entries if e.get("provenance_override") == "self_declared_ai"]
    if not markers:
        return None
    first = next((e for e in markers if e.get("detector_id") not in WATERMARK_DETECTORS), None)
    if first is None:
        return WATERMARK
    metrics = first.get("metrics") or {}
    if first.get("detector_id") not in CREDENTIAL_DETECTORS:
        if metrics.get("names_generator") == 0.0:  # (a result stored before holds no such metric and names a generator)
            if metrics.get("source_type_trained") or metrics.get("phrase_declared"):
                return METADATA_DECLARED
            if metrics.get("source_type_composite"):
                return METADATA_COMPOSITE
            if metrics.get("sd_params"):
                return METADATA_PARAMETERS
        return METADATA
    if metrics.get("trusted"):
        return CREDENTIALS_LISTED
    return CREDENTIALS_UNLISTED_DECLARED if metrics.get("ai_declared") else CREDENTIALS_UNLISTED_NAMED


# N-19h (E-21): the words of a marker, by its kind. Each names what the file carries, and who can make such a marker where
# anyone can: nothing says "metadata can be added or edited by anyone" of Content Credentials, and a signer on the trust list
# is not said to be anyone. `one` is the kind of file the words stand in ("image", "video"): no "genuine image" under a video
_ONE = {"image": "image", "video": "video", "audio": "recording", "document": "document"}
_CARRIES = {
    METADATA: "The file's metadata names an AI generator",
    WATERMARK: "The {one} carries a public AI-generator watermark message",
    CREDENTIALS_LISTED: "The file's Content Credentials name an AI generator and declare no AI generation",
    CREDENTIALS_UNLISTED_DECLARED: "The file's Content Credentials declare AI generation, and their signer is not on the trust list",
    CREDENTIALS_UNLISTED_NAMED: "The file's Content Credentials name an AI generator, and their signer is not on the trust list",
    METADATA_DECLARED: "The file's metadata declares AI generation",
    METADATA_COMPOSITE: "The file's metadata declares a composite with AI-generated parts",
    METADATA_PARAMETERS: "The file's metadata holds AI generation parameters",
}
_ANYONE = {METADATA: "metadata can be added or edited by anyone", WATERMARK: "the message is public, so anyone can embed it",
           CREDENTIALS_LISTED: None, CREDENTIALS_UNLISTED_DECLARED: "anyone can sign such credentials",
           CREDENTIALS_UNLISTED_NAMED: "anyone can sign such credentials",
           METADATA_DECLARED: "metadata can be added or edited by anyone", METADATA_COMPOSITE: "metadata can be added or edited by anyone",
           METADATA_PARAMETERS: "metadata can be added or edited by anyone"}
_CAVEAT_TAIL = {
    METADATA: "Metadata can be added or edited by anyone, so it can be added to a genuine {one}.",
    WATERMARK: "The message is public, so anyone can embed it in a genuine {one}.",
    CREDENTIALS_LISTED: "A generator named without a declaration of AI generation counts as self-declared.",
    CREDENTIALS_UNLISTED_DECLARED: "Anyone can sign such credentials, so they can be attached to a genuine {one}.",
    CREDENTIALS_UNLISTED_NAMED: "Anyone can sign such credentials, so they can be attached to a genuine {one}.",
    METADATA_DECLARED: "Metadata can be added or edited by anyone, so it can be added to a genuine {one}.",
    METADATA_COMPOSITE: "Metadata can be added or edited by anyone, so it can be added to a genuine {one}.",
    METADATA_PARAMETERS: "Metadata can be added or edited by anyone, so it can be added to a genuine {one}.",
}
_VERBAL = {
    METADATA: "The file's metadata names an AI generator",
    WATERMARK: "The {one} carries a public AI-generator watermark",
    CREDENTIALS_LISTED: "The file's Content Credentials name an AI generator",
    CREDENTIALS_UNLISTED_DECLARED: "The file's Content Credentials declare AI generation; their signer is not on the trust list",
    CREDENTIALS_UNLISTED_NAMED: "The file's Content Credentials name an AI generator; their signer is not on the trust list",
    METADATA_DECLARED: "The file's metadata declares AI generation",
    METADATA_COMPOSITE: "The file's metadata declares a composite with AI-generated parts",
    METADATA_PARAMETERS: "The file's metadata holds AI generation parameters",
}


def one_of(kind: str) -> str:
    return _ONE.get(kind, "file")


def marker_carries(marker: str | None, kind: str = "image") -> str:
    """What the file carries, as a sentence without its full stop: "The file's metadata names an AI generator"."""
    return _CARRIES[marker or METADATA].format(one=one_of(kind))


def marker_anyone(marker: str | None) -> str | None:
    """Who can make such a marker, as a clause; None for credentials under a signer on the trust list."""
    return _ANYONE[marker or METADATA]


def marker_verbal(marker: str | None, kind: str = "image") -> str:
    """The decision's own headline for a marker that decides (`Decision.verbal`)."""
    return _VERBAL[marker or METADATA].format(one=one_of(kind))


def marker_caveat(marker: str | None, kind: str = "image") -> str:
    """The caveat of a self-declared marker: what the file carries and why that is no more than self-declared."""
    carries = marker_carries(marker, kind)
    return f"Self-declared: {carries[0].lower()}{carries[1:]}. {_CAVEAT_TAIL[marker or METADATA].format(one=one_of(kind))}"


def floor_reason(marker: str | None, kind: str, floor: str) -> str:
    """The reason of a calibrated result that a marker's floor decides. `floor`: the floor as printed ("85%")."""
    anyone = marker_anyone(marker)
    return f"{marker_carries(marker, kind)}. This self-declared evidence floors the estimate at {floor}" + (f"; {anyone}." if anyone else ".")


def record(kind: str | None, entries: Iterable[dict], signal_ids: Iterable[str]) -> dict | None:
    """`Decision.provenance_record` of a decision that applied the override `kind`: the kind, the kind of marker for a
    self-declared one, and the signals of the record (every override entry of that kind). None without an override."""
    if kind is None:
        return None
    return {"kind": kind, "marker": marker_kind(entries) if kind == "self_declared_ai" else None, "signal_ids": list(signal_ids)}


def watermark_only(entries: Iterable[dict]) -> bool:
    """True when every self-declared override comes from a public watermark decoder, so the texts name the watermark."""
    declared = [e.get("detector_id") for e in entries if e.get("provenance_override") == "self_declared_ai"]
    return bool(declared) and all(d in WATERMARK_DETECTORS for d in declared)
OVERRIDE_LABELS = {
    "decisive_ai": "Content Credentials declare AI generation (validated against the trust list)",
    "trusted_capture": "Trusted capture credentials cover these bytes",
    "self_declared_ai": "Generator named in unsigned metadata or a public watermark (self-declared)",
}
# N-19l (code review R6-04, 1): the row of a self-declared record names the kind of that record. One label stood for all
# three kinds: under "The file's Content Credentials name an AI generator" the breakdown said "Generator named in unsigned
# metadata or a public watermark (self-declared)". The label above is what a result published before holds
# (contributions.ts `MARKER_LABELS`)
MARKER_LABELS = {
    METADATA: "Generator named in unsigned metadata (self-declared)",
    WATERMARK: "Public AI-generator watermark message (self-declared)",
    CREDENTIALS_LISTED: "Content Credentials name an AI generator without declaring AI generation (self-declared)",
    CREDENTIALS_UNLISTED_DECLARED: "Content Credentials declare AI generation; their signer is not on the trust list (self-declared)",
    CREDENTIALS_UNLISTED_NAMED: "Content Credentials name an AI generator; their signer is not on the trust list (self-declared)",
    METADATA_DECLARED: "AI generation declared in unsigned metadata (self-declared)",
    METADATA_COMPOSITE: "A composite with AI-generated parts declared in unsigned metadata (self-declared)",
    METADATA_PARAMETERS: "AI generation parameters in unsigned metadata (self-declared)",
}


def override_label(kind: str, marker: str | None = None) -> str:
    """The label of the row of the record that a decision applied: by the kind of the record, and for a self-declared one
    by what it is (`marker_kind`). Without a known kind of marker, the label of all self-declared records."""
    if kind == "self_declared_ai" and marker in MARKER_LABELS:
        return MARKER_LABELS[marker]
    return OVERRIDE_LABELS[kind]


def printed_label(label: str, marker: str | None) -> str:
    """A row's label as it is printed: the label that a result published before N-19l holds for every self-declared record,
    by the kind of the record that decides it (the reports' and the page's reading of a stored row)."""
    return MARKER_LABELS[marker] if label == OVERRIDE_LABELS["self_declared_ai"] and marker in MARKER_LABELS else label


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def resolve(kinds: Iterable[str | None], *, historical: bool = False) -> str | None:
    present = {k for k in kinds if k}
    unknown = present - set(PRECEDENCE)
    if unknown:
        raise ValueError(f"unknown provenance overrides {sorted(unknown)}")
    return next((k for k in (PRECEDENCE if historical else ACTIVE_PRECEDENCE) if k in present), None)


def floor_or_cap(s: float, kind: str) -> float:
    if kind == "self_declared_ai":
        return s
    target = OVERRIDE_PROBABILITY[kind]
    return min(s, target) if kind == "trusted_capture" else max(s, target)


@dataclass(frozen=True)
class OverrideOutcome:
    kind: str | None
    probability: float  # s after Stage 0
    statistical: float  # s before Stage 0 (always kept)
    caveats: tuple[str, ...] = ()
    not_applied: tuple[str, ...] = ()


# N-19m (D3-01; D-15): the decision's own headline for a trusted capture that decides, on both fusion paths (it was the
# sentence of the decision code, "No indications of AI generation or manipulation found")
CAPTURE_VERBAL = "The file's Content Credentials record a capture of these bytes"
SELF_DECLARED_WATERMARK_VERBAL = "The image carries a public AI-generator watermark"  # UX §4.3 metadata row, applied to I-5


def apply(s_statistical: float, kinds: Iterable[str | None], *, watermark: bool = False, marker: str | None = None,
          kind_of_file: str = "image") -> OverrideOutcome:
    """`watermark`: every self-declared override comes from a public watermark decoder (`watermark_only`). N-19h: `marker`,
    the kind of the marker (`marker_kind`), and `kind_of_file`, the kind of file, word the self-declared caveat; without
    `marker` the caveat is the one of unsigned metadata or, with `watermark`, of a watermark message, of an image."""
    present = {k for k in kinds if k}
    kind = resolve(present)
    if kind is None:
        return OverrideOutcome(None, s_statistical, s_statistical,
                               (ADVISORY_CAVEAT,) if "self_declared_ai" in present else (),
                               ("self_declared_ai",) if "self_declared_ai" in present else ())
    s = floor_or_cap(s_statistical, kind)
    # N-19h (E-20, 7): the caveat prints the number that the field beside it holds, clamped (it read "100%" beside 0.98)
    caveats = [STATISTICAL_ESTIMATE_CAVEAT.format(clamp_probability(s_statistical))]
    if "self_declared_ai" in present:
        caveats.append(marker_caveat(marker or (WATERMARK if watermark else METADATA), kind_of_file))
    others = tuple(k for k in PRECEDENCE if k in present and k != kind)
    if others:
        # N-19k (D2-06): a record by its words, never the value of its kind ("trusted_capture", "decisive_ai")
        # N-19n (D3-07): what was set aside and what takes precedence, without a reason: "because signed evidence outranks
        # unsigned markers" does not hold for what it lists (a capture is signed; a marker under a declaration points the same way)
        applied = RECORD_WORDS[kind]
        caveats.append(f"Provenance evidence that was not applied: {', '.join(RECORD_WORDS[k] for k in others)}. "
                       f"{applied[0].upper()}{applied[1:]} takes precedence.")
    eta = _logit(min(max(s_statistical, 1e-12), 1 - 1e-12))
    if (kind == "trusted_capture" and eta > CONFLICT_LOGIT) or (kind != "trusted_capture" and eta < -CONFLICT_LOGIT):
        caveats.append(CONFLICT_CAVEAT)
    return OverrideOutcome(kind, s, s_statistical, tuple(caveats), others)


# N-19k (D2-06): the kinds of provenance record as a caveat names them
RECORD_WORDS = {"decisive_ai": "a trusted declaration", "trusted_capture": "a trusted capture record",
                "self_declared_ai": "a self-declared marker"}


def bound_logit(logit_total: float, h3_logit: float, bound: float = H3_INFLUENCE_BOUND) -> float:
    """logit s with the influence of non-override provenance contributions limited to |Δs| ≤ bound.

    `h3_logit` is the sum of those detectors' contributions (their detector and missing rows)."""
    if not (math.isfinite(logit_total) and math.isfinite(h3_logit)):
        raise ValueError("logits must be finite")
    s, s0 = _sigmoid(logit_total), _sigmoid(logit_total - h3_logit)
    if s > s0 + bound:
        return _logit(s0 + bound)
    if s < s0 - bound:
        return _logit(s0 - bound)
    return logit_total
