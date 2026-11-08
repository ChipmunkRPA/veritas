"""View model of the report templates (UX §6.1, §6.2, §7.1–7.4; D-10, D-16, D-25; PRD RP-01 (1), RP-03).

Pure functions of a `ReportSnapshot`: every sentence is a fixed template filled with snapshot values, and the headline, guidance,
natural frequencies, key findings and next step follow the answer module (`frontend/src/features/analysis/answer.ts`), so paper
says what the screen says (P8).

- **Summary** (D-10, UX §6.2): identity, assessment with the scale, what this means / does not mean, in testing (icon arrays
  only when the counts come from ≥ 100 test items, else "k of n"), key findings with the counter-evidence finding, next step,
  review, closing line.
- **Newsroom** (D-10, RP-03 minimum): the summary layout with the filename, the case reference, source fields and metadata
  withheld: metadata and credential findings keep their strength but not their details, and the PDF metadata names no file.
"""
from __future__ import annotations

import math
import re
from typing import Any, NamedTuple

from app.cases.review import PEER_APPROVES
from app.core import effect
from app.core.rounding import half_up
from app.engine.customer_assessment import text_reference_rate
from app.engine.detector import ON_THIS_SERVER, SERVER_REASONS
from app.engine.fusion import abstain, decisive, disclosures
from app.engine.fusion.runtime import D49_NOTES, MODEL_UNUSED
from app.reports import figures, formatting
from app.reports.heatmaps import THUMBNAIL_NOTE
from app.reports.snapshot import ReportSnapshot, SnapshotReview
from app.schemas.legacy import uncalibrated_score

H3 = "H3_provenance_integrity"
CODE_BAND = {"ai_supported_strong": "very_likely_ai", "ai_supported": "likely_ai", "inconclusive": "uncertain",
             "no_ai_indicators": "likely_authentic", "no_ai_indicators_strong": "very_likely_authentic"}
BAND_STEP = {"very_likely_authentic": 1, "likely_authentic": 2, "uncertain": 3, "likely_ai": 4, "very_likely_ai": 5}
HEADLINE = {  # UX §7.1 L1 headlines, verbatim (answer.ts HEADLINE)
    "very_likely_ai": "Strong indications of AI generation or manipulation",
    "likely_ai": "Indications of AI generation or manipulation",
    "uncertain": "Inconclusive: the evidence does not support a firm assessment",
    "likely_authentic": "No significant indications of AI generation or manipulation",
    "very_likely_authentic": "No indications of AI generation or manipulation found",
    "documentary_declared": "The file’s Content Credentials declare it was created with AI",
    "documentary_metadata": "The file’s metadata names an AI generator",
    "documentary_capture": "The file’s Content Credentials record a capture of these bytes",  # N-19m (D3-01; D-15)
    "insufficient": "Chipmunk Verify could not assess this file reliably",
}
MEDIA = {"image": ("image", "images"), "video": ("video", "videos"), "document": ("document", "documents"), "audio": ("recording", "recordings")}
CONFOUNDERS = {"image": "heavy compression or re-saving, or beauty filters and HDR processing",
               "video": "heavy compression or re-saving, or dubbing and poor audio sync",
               "audio": "heavy compression or re-saving", "document": "templated or grammar-tool-edited writing"}
PROFILE_TITLES = {"summary": "Executive summary", "standard": "Report", "forensic": "Forensic report", "newsroom": "Newsroom explainer"}
UNREVIEWED = "Machine assessment — not reviewed by an examiner"  # D-25 review status line
SIGNED_STATES = frozenset({"reviewed", "approved", "disputed"})  # the examiner has signed (D-27, D-56)
SIGNOFF_DECISION = {"concur": "concurs", "concur_with_changes": "concurs with changes", "non_concur": "does not concur"}
NOT_KEY = frozenset({"coverage", "F9_distribution_conditions"})
SIDE = {"ai_supported_strong": "for", "ai_supported": "for", "no_ai_indicators": "against", "no_ai_indicators_strong": "against"}
BARS = {"none": 0, "weak": 1, "moderate": 2, "strong": 3}
STRENGTH_WORDS = {"none": "little effect", "weak": "weak support", "moderate": "moderate support", "strong": "strong support"}
WITHHELD = {"F1_provenance": "Content Credentials", "F2_self_declared_metadata": "Self-declared metadata",
            "F8_metadata_consistency": "Metadata consistency"}  # newsroom: metadata and credential details
WITHHELD_REASON = "Details withheld in this version: they come from the file’s metadata."
SUMMARY_KEY_FINDINGS = 3
AUDIO_MISS = "Cloned voices are often missed: in testing, only 1 of 89 cloned voices recorded in the wild was flagged."
ICON_ARRAY_MIN_N = 100
CLOSING = "Probabilistic evidence from automated checks: an estimate subject to uncertainty and error, not a finding of fact. This summary does not list every check, method, error rate or limitation."
NO_ESTIMATE_OUTSIDE = "No estimate is shown: this input is outside what was validated."  # AnswerModule.tsx, the capped result
# N-17 (D-68; design review DS-10): a result that is not assessable does not speak of "the estimate" or "the probability". Its
# one sentence on the matter starts with these words (answer.ts `NO_ESTIMATE_FORMED`); the closing line of the one-page reports
# has its own form for such a result.
NO_ESTIMATE_FORMED = "No estimate was formed"
CLOSING_NOT_ASSESSABLE = ("The automated checks could not assess this file. That is not a finding either way. This summary does not "
                          "list every check, method or limitation.")

def no_model(profile: dict[str, Any] | None, loaded: str = "was loaded") -> str:
    """N-19k (Q2-14; methods.ts `noModel`): how a fallback result names its path: "No calibrated fusion model was loaded"
    (`loaded`: the tense of the sentence), and where one is installed and this analysis did not use it (the result's
    limiting condition says so: `runtime.FALLBACK_ENVELOPE_UNUSED`), that none was used."""
    unused = any(str(c).startswith(MODEL_UNUSED) for c in (profile or {}).get("limiting_conditions") or [])
    return "No calibrated fusion model was used for this analysis" if unused else f"No calibrated fusion model {loaded}"


def not_assessable(decision: dict[str, Any] | None) -> bool:
    return bool(decision) and decision.get("code") == "not_assessable"  # type: ignore[union-attr]


speaks_of_estimate, without_estimate = effect.speaks_of_estimate, effect.without_estimate


# N-19n (design review D3-04): the foot of the one-page reports of a result that a provenance record decides. Such a result is
# no "estimate subject to uncertainty and error"
CLOSING_RECORD = ("A provenance record of the file decides this result. The automated checks beside it are probabilistic evidence, "
                  "subject to uncertainty and error; neither is a finding of fact. This summary does not list every check, method, "
                  "error rate or limitation.")


# N-19q (N-20b's read of the reports, finding 9; N-19n's rule): the foot of a one-page report of the fallback says "a score"
CLOSING_SCORE = CLOSING.replace("an estimate subject to", "a score subject to", 1)


def closing(decision: dict[str, Any] | None) -> str:
    if not_assessable(decision):
        return CLOSING_NOT_ASSESSABLE
    if record_of(decision) is not None:
        return CLOSING_RECORD
    return CLOSING_SCORE if (decision or {}).get("fusion_mode") == "fallback" else CLOSING
# what the summaries of a capped result say in place of its findings (N-10b): the one-page summaries, and the summary
# section of a report that has a findings chapter
CAPPED_WHY = "They are not given as reasons here, because no test shows how well they work on inputs like this one."
CAPPED_CHECKS = f"The checks that ran are in the standard and forensic reports. {CAPPED_WHY}"
CAPPED_CHECKS_BELOW = f"The checks that ran are listed under Key findings. {CAPPED_WHY}"
CAP_CAVEAT = "Not validated for inputs like this: "  # the caveat of a capped result (`runtime.fuse_with_artefact`, `common.fallback_fusion`)
_DIRECTION = re.compile(r"support (for|against)\b")
_CAP_REASON = re.compile(r"^Capped at inconclusive: (.+?)(?:; conditions: .*?)?\. Before the cap")


def media(kind: str) -> tuple[str, str]:
    return MEDIA.get(kind, ("file", "files"))


def finding_direction(finding: dict[str, Any]) -> str:
    if finding.get("direction"):
        return str(finding["direction"])
    m = _DIRECTION.search(str(finding.get("title", "")))
    return m.group(1) if m else "neutral"


def is_documentary(finding: dict[str, Any]) -> bool:
    return finding.get("strength") == "decisive" or str(finding.get("title", "")).startswith("Documentary:")


# --- what a finding of the provenance families is (N-19n; design review D3-07; record.ts `markOf`) --------------------------
# The finding of a self-declared marker is a record's, not a likelihood's: its mark is an open diamond where a meter
# would stand (the filled one is for a record that was validated, bars for what was measured), in the muted ink where the
# decision did not apply it, and its strength word is "self-declared", or "not applied". What a finding is, is read from
# what a result stores, which holds no variant: so the marker's finding of a stored result loses its bars too.
SELF_DECLARED_WORD, NOT_APPLIED_WORD = "self-declared", "not applied"
# N-19n (D3-07, 8): Content Credentials that do not validate are no self-declared marker and no check. Their word is the
# product's own for the state (`c2pa.STATE_LABELS`), their mark the open diamond in the muted ink, and what the result
# means names them under every headline, on both fusion paths
INVALID_WORD = "invalid"
INVALID_MEANS = ("The file’s Content Credentials do not validate ({id}). A certificate, signer-trust, signature or "
                 "content-binding check failed. This alone does not establish tampering, AI generation or originality.")


def first_check(finding: dict[str, Any]) -> str:
    """The check of the finding's first evidence id (a signal's id begins with its detector's: "prov.c2pa:global")."""
    first = str((finding.get("evidence_ids") or [""])[0])
    return first.removeprefix("detector:").split(":", 1)[0]  # ("detector:prov.c2pa": a finding of a check without a signal)


def is_marker(finding: dict[str, Any]) -> bool:
    """The finding of a self-declared marker: of self-declared metadata with strength strong (a mention's has none); or of
    provenance, strong and "for" and no documentary record, where the check in the first evidence id is a watermark
    decoder (the other such finding is that of Content Credentials that do not validate)."""
    if is_documentary(finding) or finding.get("strength") != "strong":
        return False
    if finding.get("family") == "F2_self_declared_metadata":
        return True
    return (finding.get("family") == "F1_provenance" and finding_direction(finding) == "for"
            and first_check(finding) in decisive.WATERMARK_DETECTORS)


def is_invalid_credentials(finding: dict[str, Any]) -> bool:
    """The finding of Content Credentials that do not validate: of provenance, strong and "for" and no documentary record,
    where the check in the first evidence id reads Content Credentials (the other such finding is a watermark message's)."""
    return (not is_documentary(finding) and finding.get("family") == "F1_provenance" and finding.get("strength") == "strong"
            and finding_direction(finding) == "for" and first_check(finding) in decisive.CREDENTIAL_DETECTORS)


def record_word(finding: dict[str, Any], decision: dict[str, Any] | None) -> str | None:
    """The strength word of a finding that is a record without a ratio: a marker's ("self-declared", "not applied"), or
    "invalid" for credentials that do not validate; None for any other finding."""
    return INVALID_WORD if is_invalid_credentials(finding) else marker_word(finding, decision)


def invalid_sentence(findings: list[dict[str, Any]]) -> str | None:
    """What the result means says of credentials that do not validate (answer.ts `invalidSentence`); None without them."""
    invalid = next((f for f in sorted(findings, key=_rank_order) if is_invalid_credentials(f)), None)
    return None if invalid is None else INVALID_MEANS.format(id=invalid.get("id", ""))


def marker_word(finding: dict[str, Any], decision: dict[str, Any] | None) -> str | None:
    """The strength word of a marker's finding: "self-declared", or "not applied" where the decision did not apply it (a
    record outranks it, or the statistical checks alone decide); None for any other finding."""
    if not is_marker(finding):
        return None
    return SELF_DECLARED_WORD if of_record(finding, decision) else NOT_APPLIED_WORD


def finding_mark(finding: dict[str, Any], decision: dict[str, Any] | None) -> str | None:
    """The mark in the meter's place (`figures.marks`): `documentary`, `documentary_muted` (a documentary record that the
    decision did not apply: a capture under a declaration), `self_declared`, `not_applied`; None for a finding with a meter."""
    if is_documentary(finding):
        ids = (record_of(decision) or {}).get("signal_ids") or []
        applied = not ids or any(e in ids for e in finding.get("evidence_ids") or [])
        return "documentary" if applied else "documentary_muted"
    word = record_word(finding, decision)
    return None if word is None else {SELF_DECLARED_WORD: "self_declared", NOT_APPLIED_WORD: "not_applied", INVALID_WORD: "invalid"}[word]


# N-19o (browser QA Q5-01): the metadata check's outcome of a hit in free text (a caption, a description, a comment) is a
# mention, no marker: it sets no number, so its ledger row points neither way, has no P(AI) and no dot in the figure, and
# says what the finding says (`synthesis._mention`), not the check's label for the pattern. Mirror: `contributions.ts`
MENTION_DIRECTION = "neither way"


def is_mention(signal: dict[str, Any]) -> bool:
    """A signal of the metadata check that holds a hit in free text and no marker (`provenance_override` is unset)."""
    return (signal.get("detector_id") == "img.meta.generator" and signal.get("provenance_override") is None
            and float((signal.get("metrics") or {}).get("free_text_hits") or 0) > 0)


def signal_observed(signal: dict[str, Any]) -> str:
    """What a ledger row prints as observed: a mention in the finding's words, else the check's observation."""
    if is_mention(signal):
        from app.engine.synthesis.engine import _mention

        return _mention(signal)
    return str(signal.get("observed") or "")


def own_finding(signal_id: str, findings: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The finding synthesis built from this signal: one per signal, citing the signal's own id first (EA-07)."""
    return next((f for f in findings if f.get("family") not in NOT_KEY and (f.get("evidence_ids") or [None])[0] == signal_id), None)


def finding_strength(finding: dict[str, Any]) -> str:
    """The finding's strength after family caps and corroboration: `none` … `strong`, or `documentary`."""
    return "documentary" if is_documentary(finding) else str(finding.get("strength", "none"))


_OVERRIDE_NOTE = "provenance override: "  # the note of the H3 assessment of a calibrated result (`runtime`)
# the fixed sentences with which a stored decision names the record it applied, on either fusion path (`runtime`, `decisive`)
_STORED_REASONS = (("decisive_ai", ("A C2PA manifest that validates against the trust list declares generative AI.",
                                    decisive.FALLBACK_DECLARED_AI_REASON)),
                   ("trusted_capture", ("A trusted C2PA capture manifest covers the current bytes,",)),
                   ("self_declared_ai", ("The file's metadata names an AI generator.", "The image carries a public AI-generator watermark message.",
                                         "The file's Content Credentials name an AI generator without a declaration")))


def stored_record(decision: dict[str, Any], assessments: list[dict[str, Any]] | None = None,
                  signals: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """N-19h (E-21), for a result stored before the decision named its record: where H3 drives, the override of the note of
    the H3 assessment where the result has one (a calibrated result), else the record the decision's reason names (the
    fallback, which has no such assessment: a fixed sentence). None where another hypothesis drives, whatever the note says,
    and where neither names a record.

    N-19l (code review R6-01): a stored result holds what the runtime took when it named the record, on every signal its
    `provenance_override`, its detector and its metrics. Where H3 drives and a stored signal holds an override, the record
    is worked out from the signals by the functions the runtime calls: the override that wins by the runtime's precedence
    (`decisive.resolve`), and `decisive.record` over the signals of that kind, which gives the kind of marker and the
    record's signals. The result is then read as a fresh one (which findings are the record's, the words of the marker,
    the checks that scored). The driving hypothesis is asked first: a marker beside a result that the checks decide is no
    record that decides. Without `signals`, or where none holds an override, the rule above stays."""
    if decision.get("driving_hypothesis") != H3:
        return None
    overrides = [s for s in signals or [] if s.get("provenance_override") in decisive.PRECEDENCE]
    if overrides:
        applied_kind = decisive.resolve((s["provenance_override"] for s in overrides), historical=True)
        applied = [s for s in overrides if s["provenance_override"] == applied_kind]
        return decisive.record(applied_kind, applied, [str(s.get("id")) for s in applied])
    note = next((str(a.get("note") or "") for a in assessments or [] if a.get("hypothesis") == H3), "")
    reason = str(decision.get("reason") or "")
    kind = note.removeprefix(_OVERRIDE_NOTE) if note.startswith(_OVERRIDE_NOTE) else next(
        (k for k, starts in _STORED_REASONS if reason.startswith(starts)), None)
    if kind not in decisive.PRECEDENCE:
        return None
    marker = None
    if kind == "self_declared_ai":
        watermark = decision.get("verbal") == decisive.SELF_DECLARED_WATERMARK_VERBAL or reason.startswith("The image carries a public")
        marker = decisive.WATERMARK if watermark else decisive.METADATA
    return {"kind": kind, "marker": marker, "signal_ids": []}


def record_of(decision: dict[str, Any] | None, assessments: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """The provenance record that decides a result (answer.ts `recordOf`): what the decision says (`provenance_record`), or
    for a result stored before it said so, `stored_record`. None when no record decides. A decision that names no record
    while H3 drives is a stored one: since N-19h a decision that H3 drives always names its record, and the stored field
    is read as empty by the schema, so its absence says nothing."""
    if not decision:
        return None
    return decision.get("provenance_record") or stored_record(decision, assessments)


_STORED_STATISTICAL = re.compile(r"^Statistical (?:score before the provenance override: (\d+) / 100, not calibrated"
                                 r"|estimate before the provenance override: (\d+)\s?%)\.")


def stored_statistical(decision: dict[str, Any], caveats: list[str], signals: list[dict[str, Any]]) -> float | None:
    """N-19h (E-20, 1; record.ts `storedStatistical`): a result of the image fallback stored before the stage published the
    statistical score beside a record holds that score only in its caveat, in whole points ("Statistical score before the
    provenance override: 30 / 100, not calibrated."; before N-7 "Statistical estimate before the provenance override:
    30%."). It is read from there where a check of the content scored, so that the result is not taken for one that a
    record decides alone. Without such a check the caveat holds the empty "50 / 100", which is no score. None for any
    other result."""
    if (decision.get("fusion_mode") != "fallback" or decision.get("statistical_estimate_before_provenance") is not None
            or record_of(decision) is None):
        return None
    from app.engine.stages.common import fallback_checks

    if not abstain.content_check_scored(fallback_checks(signals)):
        return None
    found = next((m for c in caveats if (m := _STORED_STATISTICAL.match(str(c)))), None)
    return None if found is None else int(found.group(1) or found.group(2)) / 100


def decision_of(result: dict[str, Any]) -> dict[str, Any]:
    """The decision of a result as the reports read it: with the record that decides, also for a result stored before the
    decision named it, and with the statistical score that a stored result of the image fallback holds only in its caveat
    (`stored_statistical`). The stored result is not changed (its hash is the stored one)."""
    decision: dict[str, Any] = result.get("decision") or {"code": "not_assessable"}
    if not decision.get("provenance_record"):
        decision = {**decision, "provenance_record": stored_record(decision, list(result.get("assessments") or []),
                                                                   list(result.get("signals") or []))}
    statistical = stored_statistical(decision, list(result.get("caveats") or []), list(result.get("signals") or []))
    return decision if statistical is None else {**decision, "statistical_estimate_before_provenance": statistical}


BASIS_OF_RECORD = {"decisive_ai": "documentary_declared", "self_declared_ai": "documentary_metadata",
                   "trusted_capture": "documentary_capture"}
# N-19m: a basis under which the statistical checks are read as a statistical result is read (a check outside the estimate
# that flagged the file is named): the checks' own, and a capture's, whose decision code is the clean one
READ_AS_STATISTICAL = ("inferential", "documentary_capture")


def basis_of(decision: dict[str, Any], findings: list[dict[str, Any]]) -> str:
    """The basis of a result (answer.ts `basisOf`). N-19h (E-21): from the record the decision applied, not from the findings:
    a trusted declaration has the declaration's basis and a marker the marker's. N-19m (design review D3-01; D-15): a
    trusted capture that decides has a basis of its own too, also when the file carries a marker that was not applied (it
    had none, so its result was headed by its decision code and read as a clean statistical one). A result stored before
    N-19h whose record neither its assessments nor its reason name keeps the reading by its findings."""
    if decision.get("code") == "not_assessable":
        return "insufficient"
    record = record_of(decision)
    if record is not None:
        return BASIS_OF_RECORD.get(str(record.get("kind")), "inferential")
    if decision.get("driving_hypothesis") == H3:  # a stored result that names its record nowhere
        if any(f.get("strength") == "decisive" and finding_direction(f) == "for" for f in findings):
            return "documentary_declared"
        if any(f.get("family") == "F2_self_declared_metadata" and finding_direction(f) == "for" for f in findings):
            return "documentary_metadata"
    return "inferential"


def step_of(decision: dict[str, Any], basis: str) -> int:
    """Verdict step of the headline colour and the scale; 0 = insufficient."""
    if basis == "insufficient":
        return 0
    if basis == "documentary_declared":
        return 5
    if basis == "documentary_metadata":
        return 4
    if basis == "documentary_capture":  # N-19m (D-15): the first segment, in which the record's 10 stands, on both fusion paths
        return 1
    return BAND_STEP[CODE_BAND[decision["code"]]]


def record_set_number(decision: dict[str, Any] | None) -> bool:
    """N-19h (E-20; record.ts `recordSetNumber`): the record that decides set the number the result publishes. Read from
    the result's numbers, not from the words of its reason: a record decides, a number is published, and it differs from
    the statistical one beside it or none stands beside it. A marker or a capture that did not move the estimate did not."""
    if not decision or record_of(decision) is None:
        return False
    published, statistical = estimate_of(decision), decision.get("statistical_estimate_before_provenance")
    return published is not None and (statistical is None or abs(float(statistical) - float(published)) > 1e-9)


def record_alone(decision: dict[str, Any] | None) -> bool:
    """N-19h (E-19; record.ts `recordAlone`): a record decides alone: it decides the result, and the result holds no
    statistical number beside it (a check did not run; no check that scored has weight for the input; on the fallback no
    check of the content scored). Nothing then says that a statistical estimate is reported, that checks entered a model
    or that a model's estimate was adjusted. Read from what the result holds, so a stored result follows the same rule."""
    return (bool(decision) and record_of(decision) is not None and estimate_of(decision) is not None  # type: ignore[arg-type]
            and decision.get("statistical_estimate_before_provenance") is None)  # type: ignore[union-attr]


# the notes of the H1 assessment by which a calibrated result says why it holds no statistical estimate (`runtime.D49_NOTES`)
_ALONE_WHY = ((D49_NOTES[1], "a check that this analysis depends on did not run"),
              (D49_NOTES[2], "for an input like this one the fusion model gives the checks that scored no weight"),
              (D49_NOTES[0], "no check produced a score that can be used"))


def alone_why(decision: dict[str, Any], assessments: list[dict[str, Any]] | None, kind: str) -> str | None:
    """Why no statistical number stands beside a record that decides alone, as a clause (contributions.ts `aloneWhy`): on
    the fallback no check of the content scored (D-64); a calibrated result says it in the note of its H1 assessment.
    None where the result does not say."""
    if decision.get("fusion_mode") == "fallback":
        return f"no check of the {abstain.CONTENT_NAMES.get(kind, 'content')} produced a score"
    note = next((str(a.get("note") or "") for a in assessments or [] if a.get("hypothesis") == "H1_whole_content_synthesis"), "")
    return next((words for prefix, words in _ALONE_WHY if note.startswith(prefix)), None)


def statistical_sentence(decision: dict[str, Any], profile: dict[str, Any] | None = None) -> str:
    """What the statistical checks alone give beside a record's value, as a sentence (answer.ts `statisticalSentence`): an
    estimate of a fusion model, a score of the fallback, that they gave none, or (N-19h, E-20, 8) that theirs is not shown
    because the input is outside what was validated."""
    statistical = decision.get("statistical_estimate_before_provenance")
    if statistical is None:
        return "The statistical checks alone gave no estimate."
    if statistical_withheld(decision, profile):
        number = "score" if decision.get("fusion_mode") == "fallback" else "estimate"
        return f"The {number} of the statistical checks alone is not shown: this input is outside what was validated."
    other_way = alone_other_way(decision)
    if decision.get("fusion_mode") == "fallback":
        if other_way:  # N-19n (D4-02): "The statistical checks alone point the other way: their score is 90 / 100, not calibrated."
            return f"{ALONE_OTHER_WAY}: their {uncalibrated_score(statistical).replace('score ', 'score is ', 1)}."
        return f"The statistical checks alone give a {uncalibrated_score(statistical)}."
    if alone_inconclusive(decision):  # N-19m (R7-01, R7-02): never the bare estimate where the checks alone do not decide
        return f"The statistical checks alone do not decide: their estimate is {alone_number(decision)} out of 100."
    if alone_one_check(decision):  # N-19n: nor where one check alone counts and decides; with the way it points (D4-02)
        scored = "Only one statistical check scored, and it points the other way" if other_way else "Only one statistical check scored"
        return f"{scored}: its estimate is {numeral(statistical)} out of 100."
    if other_way:  # N-19n (D4-02): where the checks alone decide the other answer than the record's, the sentence says so
        return f"{ALONE_OTHER_WAY}: their estimate is {numeral(statistical)} out of 100."
    return f"The statistical checks alone estimate {numeral(statistical)} out of 100."


# N-19n (D4-02; D-68 as amended on 2026-10-05; record.ts `ALONE_OTHER_WAY`, `aloneOtherWay`): the statistical checks alone
# decide the other answer than the record's. Their number beside the record says so ("98 out of 100, they point the
# other way"), and the sentence of the number in the full reports and on the page.
ALONE_OTHER_WAY = "The statistical checks alone point the other way"
OTHER_WAY, OTHER_WAY_ONE = "they point the other way", "it points the other way"
_AI_CODES, _CLEAN_CODES = ("ai_supported", "ai_supported_strong"), ("no_ai_indicators", "no_ai_indicators_strong")


def alone_other_way(decision: dict[str, Any] | None) -> bool:
    """The statistical checks alone decide the other answer than the record that decides: an AI code under a trusted
    capture, a clean code under a marker or a declaration (`Decision.statistical_alone`). The fallback publishes no such
    code: its score's own ranges are read (`core.evidence.band_for`), 65 or more under a capture and under 35 under a
    marker or a declaration. False without a record, on the fallback without a statistical score, and for a result
    stored before N-19l on the fusion model, which does not say what the checks alone decided."""
    kind = (record_of(decision) or {}).get("kind")
    if kind is None:
        return False
    capture = kind == "trusted_capture"
    if decision.get("fusion_mode") == "fallback":  # type: ignore[union-attr]
        p = decision.get("statistical_estimate_before_provenance")  # type: ignore[union-attr]
        # N-19o (code review R9-04): on the printed whole points, as D-77 reads them (a score of 0.645 prints 65)
        return p is not None and (half_up(float(p) * 100) >= 65 if capture else half_up(float(p) * 100) <= 34)
    alone = ((decision or {}).get("statistical_alone") or {}).get("code")
    return alone in (_AI_CODES if capture else _CLEAN_CODES)


# N-19q (N-20b's read of the reports, finding 5): under contribution rows the count is the decision's, the checks whose rows
# it counts (D-64); it read "only one check scored" beside "0 of 5 scored checks flagged this file" (five checks scored, one
# counts in the estimate). Without rows (the fallback) the words of what scored stay. answer.ts, resultNumber.ts
ONE_CHECK = "only one check counts in the estimate"
NO_CHECK_COUNTS = "no check counts in the estimate"


def few_checks(decision: dict[str, Any], scored: int) -> str:
    """The reason that no agreement is reported, and that the number is printed as text, for fewer than two checks."""
    if has_contribution_rows(decision):
        return ONE_CHECK if scored == 1 else NO_CHECK_COUNTS
    return "only one check scored" if scored == 1 else "no check scored"


def alone_one_check(decision: dict[str, Any] | None) -> bool:
    """N-19n (the code reviewer's read of N-19m's draft; resultNumber.ts `aloneOneCheck`): beside a record, the statistical
    checks alone decide and one check counts. The same picture without the record reads "Estimate 98–98: only one check
    scored"; beside a capture it read "statistical checks alone: 98 out of 100" and "The statistical checks alone estimate
    98 out of 100.", and N-19m took the fact "Agreement — only one check scored" away there. False where they alone do not
    decide (those words stand), on the fallback, and for a result stored before N-19l, which does not say what they decide."""
    alone = ((decision or {}).get("statistical_alone") or {}).get("code")
    return bool(alone) and alone != "inconclusive" and has_contribution_rows(decision) and len(counted_rows(decision)) == 1


def alone_inconclusive(decision: dict[str, Any] | None) -> bool:
    """N-19m (code review R7-01; D-76 beside a record; record.ts `aloneInconclusive`): the result says that the statistical
    checks alone are inconclusive, by the band or by a decision rule (`Decision.statistical_alone`). Their number beside
    the record that decides is then printed with the words that they alone do not decide, whether the record set the
    result's number or not. False for a result stored before N-19l, which does not say: its number stands as it did."""
    return ((decision or {}).get("statistical_alone") or {}).get("code") == "inconclusive"


def alone_number(decision: dict[str, Any]) -> str:
    """The statistical number where the checks alone do not decide: the range where the result holds one (the record did
    not set the number, so the range is the statistical one: CAPUNS), else the estimate (record.ts `aloneNumber`)."""
    span = None if record_set_number(decision) else interval_text(decision)
    return span or numeral(float(decision["statistical_estimate_before_provenance"]))


# N-19h (E-20, 2): a capture's result is the signed record's, not what the combined evidence indicates
CAPTURE_MEANS = "A signed capture record that covers the file’s current bytes decides this result."
# N-19m (D3-01; D-15): what such a result does not mean is said of the record, not of detectors ("Detectors miss some AI
# content." under a result that no detector decides), and its next step is the record's (answer.ts, the same names)
CAPTURE_NOT = ("The record states what the capture device declared. It does not show what was in front of the device: a screen "
               "or a print can be captured. A signing key can be misused.")
CAPTURE_STEP = "Check who signed the capture record, then confirm where the file came from."
CAPTURE_STEP_AGAINST = "The statistical checks alone point the other way: weigh the record’s signer against them."
CAPTURE_BASIS = "Documentary · Content Credentials (capture record)"  # the fact "Basis" of the page; the one-page line begins with it


def marker_of(decision: dict[str, Any] | None) -> str:
    """The kind of the marker that decides a result (`decisive.MARKER_KINDS`; record.ts `markerOf`); unsigned metadata where
    the result does not say (a result stored before N-19h, whose words were those of metadata)."""
    record = record_of(decision)
    return str((record or {}).get("marker") or decisive.METADATA)


def _typographic(text: str) -> str:
    return text.replace("'", "’")


# N-19h (E-21): what the kinds of marker add to "what the file carries", as a sentence of their own
_MARKER_SECOND = {decisive.CREDENTIALS_LISTED: "Their signer is on the trust list."}
_MARKER_NOT = {
    decisive.METADATA: "the same fields can be added to any file",
    decisive.WATERMARK: "the same message can be embedded in any {one}",
    decisive.CREDENTIALS_LISTED: "the credentials name a generator and do not declare that it made the {one}",
    decisive.CREDENTIALS_UNLISTED_DECLARED: "credentials from a signer that is not on the trust list can be attached to any file",
    decisive.CREDENTIALS_UNLISTED_NAMED: "credentials from a signer that is not on the trust list can be attached to any file",
    decisive.METADATA_DECLARED: "the same fields can be added to any file",
    decisive.METADATA_COMPOSITE: "the same fields can be added to any file",
    decisive.METADATA_PARAMETERS: "the same fields can be added to any file",
}


def marker_headline(marker: str, kind: str = "image") -> str:
    """The headline of a result that a marker decides, in the words of the marker's kind (answer.ts `markerHeadline`)."""
    return _typographic(decisive.marker_verbal(marker, kind))


def marker_guidance(marker: str, kind: str) -> tuple[str, str]:
    """("What this means", "What this does not mean") of a result that a marker decides (answer.ts `markerGuidance`): what the
    file carries and who can make such a marker; nothing says of Content Credentials that metadata can be edited, and
    nothing says of a signer on the trust list that anyone can sign."""
    one, anyone = media(kind)[0], decisive.marker_anyone(marker)
    second = _MARKER_SECOND.get(marker) or f"{anyone[0].upper()}{anyone[1:]}."  # type: ignore[index]
    means = _typographic(f"{decisive.marker_carries(marker, kind)}. {second}")
    return means, (f"It is not a finding that the {one} is AI-generated: {_MARKER_NOT[marker].format(one=one)}. "
                   "The statistical checks are still reported below.")


def headline(decision: dict[str, Any], basis: str) -> str:
    if basis == "documentary_metadata":  # N-19h: the words of the marker's kind
        return marker_headline(marker_of(decision))
    return HEADLINE[basis] if basis != "inferential" else HEADLINE[CODE_BAND[decision["code"]]]


def estimate_of(decision: dict[str, Any]) -> float | None:
    p = decision.get("probability_ai_calibrated")
    return p if p is not None else decision.get("probability_ai_uncalibrated")


def numeral(p: float) -> str:
    """Whole numbers clamped to 1–99, with `<1` and `>99` as text (UX §5.3): `formatting.numeral`, the one display formatter."""
    return formatting.numeral(p)


def interval_text(decision: dict[str, Any]) -> str | None:
    """The plausible range as printed, clamped like the estimate (DR P0-2); `interval` keeps the numbers for the scale figure."""
    ci = decision.get("ci90")
    return formatting.span(ci[0], ci[1]) if ci and len(ci) == 2 else None


def interval(decision: dict[str, Any]) -> tuple[int, int] | None:
    ci = decision.get("ci90")
    return (half_up(ci[0] * 100), half_up(ci[1] * 100)) if ci and len(ci) == 2 else None


def cap_reason(decision: dict[str, Any]) -> str | None:
    """L-1 (answer.ts `capReason`): the validation reason of a decision the envelope rule or the fallback's size floor capped
    ("Capped at inconclusive: <reason>[; conditions: …]. Before the cap …"), or None."""
    m = _CAP_REASON.match(str(decision.get("reason") or ""))
    return m.group(1) if m else None


# N-19n (D3-09, Q3-02; answer.ts `outsideInputSentence`): wherever a result says that its input is outside what was validated,
# the summary says why, from the condition of the input. A result that is inconclusive by its own estimate names no cap in
# its reason, and its summary gave no reason for "outside what was validated" (a screenshot and a flat graphic on the
# packaged models); where the cap changed the code the sentence stands in the place of "The result is capped at
# inconclusive because …", so that the reason is said once
OUTSIDE_WHY = "This input is outside what was validated: {why}."
CAPPED_THEREFORE = "The result is therefore capped at inconclusive."
# the reason of a share of flat blocks says what the image looks like, then the measurement behind it (`abstain`): the
# summary gives the first; the measurement stands with the input's conditions and in the caveat of a capped result
_FLAT_MEASURE = re.compile(r"(it looks like a screenshot of text or an interface, or a flat graphic): .*? stay below")


def outside_input_sentence(decision: dict[str, Any] | None, profile: dict[str, Any] | None) -> str | None:
    """Why the input of a result that no record decides is outside what was validated, as the summary says it; None for
    any other result. With a fusion model the reason is the cap's, else the envelope's own among the input's conditions
    (`abstain.outside_why`); on the fallback it is the reason of the cap."""
    if not decision or record_of(decision) is not None:
        return None
    capped = cap_reason(decision)
    why = abstain.outside_why(decision, profile) or capped
    if not why:
        return None
    said = OUTSIDE_WHY.format(why=_FLAT_MEASURE.sub(r"\1", why))
    return f"{said} {CAPPED_THEREFORE}" if capped is not None else said


def outside_validated(decision: dict[str, Any] | None, profile: dict[str, Any] | None) -> bool:
    """The key of N-10b (N-19h): the input of a result with a number is outside what was validated. It is what the test for
    a capped number knows without the number, so that it also holds where a record set the number or decides without one
    of its own: a calibrated result outside what was validated, or the cap's reason of a fallback result. The fallback
    has no validated envelope at all (`runtime.fallback_profile`: the flag is false for every fallback result), so the
    profile's flag alone is no key. A result without a number (not assessable; a check did not run) is not asked."""
    return abstain.outside_validated(decision, profile)


def statistical_withheld(decision: dict[str, Any] | None, profile: dict[str, Any] | None) -> bool:
    """N-19h (E-20, 8; answer.ts `statisticalWithheld`): the statistical number that stands beside a record is not shown,
    because the input is outside what was validated (D-65, D-68: it is not shown where it stands alone either). The
    decision's field keeps it. False where there is none to withhold."""
    return bool(decision) and decision.get("statistical_estimate_before_provenance") is not None and outside_validated(decision, profile)  # type: ignore[union-attr]


STATISTICAL_WITHHELD = "not shown (outside what was validated)"


def shown_caveats(result: dict[str, Any], decision: dict[str, Any]) -> list[str]:
    """The caveats of a result as they are printed. A result stored before N-19h holds, outside what was validated, the
    caveat with the statistical number before the override; it is printed in its place as the caveat that says that the
    number is not shown (the same rule, from what the result stores). Every other caveat is printed as it is."""
    caveats = [str(c) for c in result.get("caveats") or []]
    if record_of(decision) is None:
        return caveats
    if outside_validated(decision, result.get("condition_profile")):
        return decisive.withhold_statistical(caveats, fallback=decision.get("fusion_mode") == "fallback")
    # N-19m (R7-02): where the statistical checks alone do not decide, the caveat does not give their estimate bare
    # ("Statistical estimate before the provenance override: 90%."); the stored caveat stays as it is
    if alone_inconclusive(decision) and decision.get("statistical_estimate_before_provenance") is not None:
        return [alone_caveat(decision) if c.startswith(decisive.STATISTICAL_CAVEATS[0] + ":") else c for c in caveats]
    # N-19n (D4-04): where the checks alone decide the caveat is shown in one form too, with the unit of the number beside
    # it ("Statistical estimate before the provenance override: 98%." stood beside "Statistical checks alone: 98 out of
    # 100"); on the fallback, which never says what they decided, by its score. A result of the fusion model stored before
    # N-19l does not say either, and keeps its caveat as stored (`test_n19m_beside.py`)
    if decision.get("statistical_estimate_before_provenance") is not None and (
            decision.get("fusion_mode") == "fallback" or decision.get("statistical_alone") is not None):
        return [without_record_caveat(decision) if c.startswith(_NUMBER_CAVEATS) else c for c in caveats]
    return caveats


_NUMBER_CAVEATS = (decisive.STATISTICAL_CAVEATS[0] + ":", decisive.STATISTICAL_CAVEATS[1] + ":")


def without_record_caveat(decision: dict[str, Any]) -> str:
    """N-19n (D4-04; record.ts `withoutRecordCaveat`): the caveat with the statistical number as it is shown where the checks
    alone decide: "Statistical estimate without the provenance record: 98 out of 100."; on the fallback "Statistical score
    without the provenance record: 90 / 100, not calibrated.". The stored caveat stays as it is."""
    p = float(decision["statistical_estimate_before_provenance"])
    if decision.get("fusion_mode") == "fallback":
        return f"Statistical score without the provenance record: {uncalibrated_score(p).removeprefix('score ')}."
    return f"Statistical estimate without the provenance record: {numeral(p)} out of 100."


def alone_caveat(decision: dict[str, Any]) -> str:
    """The caveat with the statistical number as it is shown where the checks alone do not decide (record.ts `aloneCaveat`)."""
    return f"Statistical estimate without the provenance record: {alone_number(decision)} out of 100; the statistical checks alone do not decide."


def counted_rows(decision: dict[str, Any] | None) -> list[dict[str, Any]]:
    """N-19i (R5-01; answer.ts `countedRows`): the rows of the checks that the decision counts, which is one count wherever
    a count is taken: "k of n detectors" in the reason, the number line and the agreement, the count "in the estimate" and
    the sentence on how many checks entered the model. The runtime says it with the row (`counted`): not a row of zero,
    and not the row of a check that declares H3 (the container check of a video: its row was counted here and not by the
    runtime, so a video read "1 of 4 detectors" under a large numeral). A result published before N-19i has no such word:
    its rows count when they are not zero."""
    return [r for r in (decision or {}).get("contributions") or []
            if r.get("kind") == "detector" and r.get("counted") is not False and abs(float(r["logit"])) > 1e-9]


def not_counted_rows(decision: dict[str, Any] | None) -> tuple[int, int]:
    """N-19i: the rows of checks that are published and not counted: (rows of zero, rows of checks that declare H3)."""
    rows = [r for r in (decision or {}).get("contributions") or [] if r.get("kind") == "detector"]
    zero = sum(1 for r in rows if abs(float(r["logit"])) <= 1e-9)
    return zero, sum(1 for r in rows if r.get("counted") is False and abs(float(r["logit"])) > 1e-9)


def not_counted_lines(decision: dict[str, Any] | None) -> list[str]:
    """N-19i (contributions.ts `notCountedLines`): what the sentence on the checks that entered the model says of a row that
    is published and not counted."""
    zero, record = not_counted_rows(decision)
    out = []
    if zero:
        out.append("One more row is shown and not counted: it is zero, so its check did not move the estimate." if zero == 1 else
                   f"{zero} more rows are shown and not counted: they are zero, so their checks did not move the estimate.")
    if record:
        out.append("One more row is shown and not counted: its check tests the file’s provenance and integrity, and is not one of "
                   "the statistical checks." if record == 1 else
                   f"{record} more rows are shown and not counted: their checks test the file’s provenance and integrity, and are "
                   "not statistical checks.")
    return out


def scored_checks(decision: dict[str, Any], findings: list[dict[str, Any]]) -> int:
    """Checks that scored in this decision (answer.ts `scoredChecks`): the rows the decision counts (`counted_rows`), else the
    findings with at least weak support. N-19h (E-19, 4): a finding of the record that decides is not a check that scored
    (a marker's finding is not documentary, and was counted: "only one check scored" where none did)."""
    if has_contribution_rows(decision):
        return len(counted_rows(decision))
    return sum(1 for f in findings if f.get("family") not in NOT_KEY and not is_documentary(f) and f.get("strength") != "none"
               and not of_record(f, decision))


class ResultNumber(NamedTuple):
    """The number a result shows (D-68; N-7): `frontend/src/features/analysis/resultNumber.ts`, rule for rule, so a score that
    is not a calibrated likelihood never reads as one on paper and not on the screen.

    - `calibrated`: a decisive calibrated result in which at least two checks scored: the estimate and the scale marker.
    - `text`: a calibrated result that is inconclusive, or in which fewer than two checks scored: "Estimate 53–78: inconclusive".
      N-19l (R6-05; D-68 as amended): also the statistical number under a record that decides and did not set it, where
      the statistical checks alone are inconclusive, by the band or by D-76: "Estimate of the statistical checks alone
      83–96: they alone do not decide". The result is the record's and is not inconclusive, so the words are their own.
    - `uncalibrated`: the fallback: "score n / 100, not calibrated", no scale marker.
    - `capped`: outside what was validated: no number, no range and no scale marker; the reason is printed instead.
    - `record` (N-19h, E-20): a number that the provenance record set (a marker's floor, a capture's cap): said in words
      with what set it, never as an estimated likelihood, a score of the fallback or a marker on the scale. The statistical
      checks' own number stands beside it (`statistical_alone`).
    - `none`: no estimate, a result that is not assessable, or one a validated declaration decides."""
    kind: str
    estimate: float | None = None
    span: str | None = None  # the plausible range as printed
    text: str | None = None


def result_number(decision: dict[str, Any] | None, findings: list[dict[str, Any]] | None = None,
                  profile: dict[str, Any] | None = None) -> ResultNumber:
    known = findings is not None or has_contribution_rows(decision or {})  # as `agreement`: without either, nothing says what scored
    findings = list(findings or [])
    if not decision or decision.get("code") == "not_assessable" or basis_of(decision, findings) == "documentary_declared":
        return ResultNumber("none")
    p = estimate_of(decision)
    if p is None:
        return ResultNumber("none")
    if record_set_number(decision):  # N-19h (E-20, 1): the record's value, on the fallback as on the calibrated path
        return ResultNumber("record", p, None, record_phrase(decision))
    if outside_validated(decision, profile):
        return ResultNumber("capped")
    if decision.get("fusion_mode") == "fallback":
        return ResultNumber("uncalibrated", p, None, uncalibrated_score(p))
    span = interval_text(decision)
    if alone_undecided(decision):  # N-19l (R6-05): as the result of the statistical checks alone would show it, in words
        return ResultNumber("text", p, span, f"{ALONE_ESTIMATE} {span or numeral(p)}: {ALONE_UNDECIDED}")
    scored = scored_checks(decision, findings) if known else 2
    why = "inconclusive" if decision.get("code") == "inconclusive" else few_checks(decision, scored) if scored < 2 else None
    if why is not None:
        # N-19q (N-20b's read of the reports, finding 11): a range is named as the plausible range, not as the estimate ("Estimate
        # 24–85: inconclusive" stood beside "49 out of 100"); without a range the estimate itself is printed
        return ResultNumber("text", p, span, f"{RANGE_WORD} {span}: {why}" if span else f"Estimate {numeral(p)}: {why}")
    return ResultNumber("calibrated", p, span)


RANGE_WORD = "Plausible range"  # resultNumber.ts `RANGE_WORD`


# N-19l (R6-05; D-68 as amended): the statistical number under a record that decides and did not set it, where the
# statistical checks alone do not decide (resultNumber.ts `ALONE_ESTIMATE`, `ALONE_UNDECIDED`)
ALONE_ESTIMATE = "Estimate of the statistical checks alone"
ALONE_UNDECIDED = "they alone do not decide"


def alone_undecided(decision: dict[str, Any] | None) -> bool:
    """N-19l (R6-05; record.ts `aloneUndecided`): a record decides, the number the result publishes is the statistical one,
    and the statistical checks alone are inconclusive, by the band or by D-76 (`Decision.statistical_alone`: the result
    says what they alone decided). The number is then no large numeral and no marker on the scale. A result stored
    before the decision said so keeps the form it had."""
    if not decision or record_of(decision) is None or record_set_number(decision) or estimate_of(decision) is None:
        return False
    return (decision.get("statistical_alone") or {}).get("code") == "inconclusive"


# N-19h (E-20, 1): what set a number that a record set (record.ts `RECORD_SET`)
RECORD_SET = {"self_declared_ai": "the floor of self-declared evidence", "trusted_capture": "the cap of a trusted capture record"}


def _record_set(decision: dict[str, Any]) -> str:
    return RECORD_SET.get(str((record_of(decision) or {}).get("kind")), "the value of the provenance record")


def record_phrase(decision: dict[str, Any]) -> str:
    """A number that a record set, beside a headline (record.ts `recordPhrase`): "set by the record: 85 out of 100, the
    floor of self-declared evidence". Not an estimated likelihood and not a score of the fallback."""
    return f"set by the record: {numeral(estimate_of(decision))} out of 100, {_record_set(decision)}"  # type: ignore[arg-type]


def record_sentence(decision: dict[str, Any]) -> str:
    """The same as a sentence under the headline of the full reports."""
    return f"The record sets the number: {numeral(estimate_of(decision))} out of 100, {_record_set(decision)}."  # type: ignore[arg-type]


def number_phrase(n: ResultNumber, own: bool = False) -> str | None:
    """The number in a sentence beside a headline (the sign-off block's machine assessment; resultNumber.ts `numberPhrase`),
    or None when the result shows none. `own` (N-19n, D4-03): a record decides and did not set the number, so the
    calibrated numeral is the statistical checks' own, and the phrase says whose it is."""
    if n.kind == "calibrated":
        if own:
            return f"{OWN_LIKELIHOOD}: {numeral(n.estimate)} %"  # type: ignore[arg-type]
        return f"estimated likelihood {numeral(n.estimate)} %"  # type: ignore[arg-type]
    if n.kind == "capped":
        return "no estimate shown: outside what was validated"
    return n.text


# N-19n (D4-03; resultNumber.ts `OWN_LIKELIHOOD`): under a record that did not set the number, the calibrated numeral shown is
# the statistical checks' own, and its caption and lines say so ("2 % estimated likelihood" stood under a capture's headline)
OWN_LIKELIHOOD = "estimated likelihood, statistical checks alone"


def likelihood_words(decision: dict[str, Any] | None) -> str:
    """The words after a calibrated numeral in a line: whose estimate it is where a record decides beside it."""
    return OWN_LIKELIHOOD if record_of(decision) is not None else "estimated likelihood"


def statistical_alone(decision: dict[str, Any], profile: dict[str, Any] | None = None) -> str:
    """The statistical checks beside a validated declaration or beside a number that a record set (resultNumber.ts
    `statisticalAlone`): their estimate, the fallback's score said as one, that there is none, or (N-19h, E-20, 8) that
    it is not shown because the input is outside what was validated."""
    p = decision.get("statistical_estimate_before_provenance")
    if p is None:
        return "no estimate"
    if statistical_withheld(decision, profile):
        return STATISTICAL_WITHHELD
    other_way = alone_other_way(decision)  # N-19n (D4-02): "…, they point the other way"
    if decision.get("fusion_mode") == "fallback":
        return f"{uncalibrated_score(p)}; {OTHER_WAY_ONE}" if other_way else uncalibrated_score(p)
    # N-19m (R7-01; D3-05): the number with its unit, as the record's value beside it has one ("85 out of 100" stood over
    # "13"), and with what the checks alone decided where they do not decide ("statistical checks alone: 90" stood bare)
    if alone_inconclusive(decision):
        return f"{alone_number(decision)} out of 100, {ALONE_UNDECIDED}"
    if alone_one_check(decision):  # N-19n: "98 out of 100, only one check scored" (it stood bare); "…; it points the other way"
        return f"{numeral(p)} out of 100, {ONE_CHECK}" + (f"; {OTHER_WAY_ONE}" if other_way else "")
    return f"{numeral(p)} out of 100" + (f", {OTHER_WAY}" if other_way else "")


def agreement_beside(n: ResultNumber, agree: str) -> str:
    """The agreement beside a number that already says why it is not reported ("Estimate 93–98: only one check scored"): the
    reason is said once."""
    why = agree.removeprefix("not reported (").removesuffix(")")
    return "not reported" if n.kind == "text" and why != agree and str(n.text).endswith(f": {why}") else agree


def scale(decision: dict[str, Any], basis: str, step: int, n: ResultNumber) -> Any:
    """The scale figure of a result (AnswerModule.tsx): the marker for a `calibrated` number only; the range without a marker
    for a `text` number; the verdict's segment filled where there is neither (so for a number that a record set, which has
    no place on the scale); no figure beside a validated declaration."""
    if basis == "documentary_declared":
        return None
    span = interval(decision)
    low, high = span if span else (None, None)
    if n.kind == "calibrated":
        return figures.scale_svg(step, float(n.estimate) * 100, low, high)  # type: ignore[arg-type]
    if n.kind == "none":
        return figures.scale_svg(step, None, None, None)
    phrase = number_phrase(n) or "No estimate"
    in_range = n.kind == "text"
    return figures.scale_svg(step, None, low if in_range else None, high if in_range else None, band=True,
                             label=phrase[:1].upper() + phrase[1:])


# D-63 "Headline rule in words" (browser QA P1-2): under a calibrated result the scale says what its marker is and what the
# headline follows, in place of the three labels that name its segments (AnswerModule.tsx `scaleNote`)
# N-19m (D-77): and the condition that the headline of one answer has on the scale itself
# N-19n (the design reviewer's read of the sentences of N-19m's third part): the note read "…, not the segment, and a headline
# of one answer also needs the marker in its part of the scale.", which reads for a moment as a list of what the headline
# does not follow, and "its part" has three possible owners
SCALE_NOTE_MARKER = ("The marker is the estimate. The headline follows the calibrated levels for inputs like this one, not the "
                     "segment; a headline of one answer also needs the marker in that answer’s part of the scale.")
SCALE_NOTE = ("The headline follows the calibrated levels for inputs like this one, not the segments of this scale; a headline of "
              "one answer also needs the estimate in that answer’s part of the scale.")
# N-19n (code review R8-02): the notes as they were before D-77, without the condition. They stand under a stored result
# whose headline of one answer has its marker outside that answer's part of the scale, a result made before D-77: a clean
# headline at 42 read "… also needs the marker in its part of the scale." under a marker that is not there
SCALE_NOTE_MARKER_BEFORE = "The marker is the estimate; the headline follows the calibrated levels for inputs like this one, not the segment."
SCALE_NOTE_BEFORE = "The headline follows the calibrated levels for inputs like this one, not the segments of this scale."
# N-19i (D-76): under a result that is inconclusive because no check supports the answer of its estimate, the headline does
# not follow the levels: the estimate reached one
# N-19q (finding 1): the rule reads each check's part in the estimate, and the note says so (it read "no check supports that
# answer" beside a check that taken alone gives weak support)
SCALE_NOTE_UNSUPPORTED = ("The estimate reaches a calibrated level for inputs like this one, but no check's part in it gives that "
                          "answer weak support, so the headline is inconclusive.")
# N-19m (D-76 as amended): a check supports the answer, and the checks together do not
SCALE_NOTE_TOGETHER = ("The estimate reaches a calibrated level for inputs like this one, but the checks together do not support "
                       "that answer, so the headline is inconclusive.")
# N-19m (D-77): the estimate stands in the middle part of the scale. The note names the answer of the levels and does not
# quote the headline that the result does not have
SCALE_NOTE_OFF_SCALE = ("The estimate stands in the middle part of the scale, so the headline is inconclusive, although the "
                        "calibrated levels for inputs like this one would have given {answer}.")
OFF_SCALE_ANSWER = {"for": "the AI answer", "against": "the clean answer"}
# N-19n (supervisor check 29): under a configured prior the estimate of a result that D-77 took can stand in the part of
# the other answer (published at 21 where the levels give the AI answer: the note said "the middle part of the scale"). The
# note names the part in which the estimate prints. The part's owner and its bound, by the side of that part
ANSWERS_PART = {"for": "the AI answer’s part of the scale", "against": "the clean answer’s part of the scale"}
PART_BOUND = {"for": "65 or more", "against": "34 or less"}
SCALE_NOTE_OTHER_PART = ("The estimate stands in {part}, although the calibrated levels for inputs like this one would have given "
                         "{answer}, so the headline is inconclusive.")


def scale_note(decision: dict[str, Any], n: ResultNumber) -> str | None:
    """The scale's caption for a result of a calibrated fusion model; None on the fallback, whose headline does follow the
    segments, and where no estimate was formed. N-19h (E-20, 4): None in a result that a provenance record decides, whose
    headline is the record's and does not follow the calibrated levels. N-19n (D3-10): None under a result that shows no
    estimate because its input is outside what was validated: there are no calibrated levels for such an input."""
    if decision.get("fusion_mode") != "artefact" or n.kind in ("none", "capped") or record_of(decision) is not None:
        return None
    if (side := off_scale(decision)) is not None:  # N-19m (D-77)
        if (part := off_scale_part(decision)) is not None:  # N-19n: in the other answer's part, under a configured prior
            return SCALE_NOTE_OTHER_PART.format(part=ANSWERS_PART[part], answer=OFF_SCALE_ANSWER[side])
        return SCALE_NOTE_OFF_SCALE.format(answer=OFF_SCALE_ANSWER[side])
    if unsupported(decision) is not None:  # N-19i (D-76); N-19m: in the form of the condition that took the code
        return SCALE_NOTE_TOGETHER if rule_taken(decision) in TOGETHER_RULES else SCALE_NOTE_UNSUPPORTED
    estimate = estimate_of(decision)
    if estimate is not None and not abstain.in_part_of_scale(str(decision.get("code")), float(estimate)):  # made before D-77 (R8-02)
        return SCALE_NOTE_MARKER_BEFORE if n.kind == "calibrated" else SCALE_NOTE_BEFORE
    return SCALE_NOTE_MARKER if n.kind == "calibrated" else SCALE_NOTE


def agreement(decision: dict[str, Any], coverage: dict[str, Any] | None, findings: list[dict[str, Any]] | None = None, *,
              failed: bool = False) -> str:
    """The backend's confidence (answer.ts `agreement`, L-3): "not reported" without an estimate or with fewer than two scored
    checks; capped at medium when a check failed or timed out or entered checks point against the headline, and at low when
    documentary evidence does. `failed` (N-19n, Q4-01): the step that writes the findings failed. The agreement is worked
    out from the findings, so it is not reported: without them it read "no check scored" on the fallback, and with a
    fusion model it lost its cap for a check that points the other way."""
    known = findings is not None or has_contribution_rows(decision)  # without findings or rows the evidence gates cannot be read
    findings = list(findings or [])
    rows = has_contribution_rows(decision)
    scored = scored_checks(decision, findings)
    if decision.get("code") == "not_assessable":
        return "not reported (not assessed)"  # N-17 (D-68)
    if estimate_of(decision) is None:
        return "not reported (no estimate)"
    if record_alone(decision):  # N-19h (E-19, 4): no statistical estimate to agree on; never "no check scored" of a check that scored
        return f"not reported (no statistical {'score' if decision.get('fusion_mode') == 'fallback' else 'estimate'})"
    if failed:
        return effect.AGREEMENT_NOT_WRITTEN
    if known and scored < 2:
        return f"not reported ({few_checks(decision, scored)})"
    by_status = (coverage or {}).get("by_status", {})
    level = {"high": 3, "medium": 2}.get(str(decision.get("confidence")), 1)
    side = SIDE.get(str(decision.get("code")))
    opposite = "against" if side == "for" else "for"
    counts = (lambda f: entered_decision(f, decision)) if rows else (lambda f: f.get("family") not in NOT_KEY and f.get("strength") != "none")  # noqa: E731
    against = [f for f in findings if side and counts(f) and decision_direction(f, decision) == opposite]
    if (by_status.get("failed", 0) + by_status.get("timeout", 0)) > 0 or against:
        level = min(level, 2)
    if any(is_documentary(f) for f in against):
        level = 1
    return {3: "high", 2: "medium", 1: "low"}[level]


# --- checks that could not run for a reason of the server (N-14a; answer.ts `serverSentence`, `nextStep`) ---------------

SERVER_FEWER = "This result rests on fewer checks than the profile holds."  # `engine.detector.SERVER_FEWER`: the caveat ends with it
OBTAIN_ANOTHER = " Obtain the original file or another source."


def not_run_on_server(coverage: dict[str, Any] | None) -> dict[str, int]:
    """Checks skipped for the server's memory budget or for model files that are not installed, by reason; empty when none."""
    found = (coverage or {}).get("not_run_on_server") or {}
    return {reason: int(found[reason]) for reason in SERVER_REASONS if int(found.get(reason) or 0) > 0}


def server_count(coverage: dict[str, Any] | None) -> str | None:
    """"3 of the 4 checks could not run on this server"; None when every check that did not run has a reason of the file."""
    n = sum(not_run_on_server(coverage).values())
    total = int((coverage or {}).get("detectors_total", 0))
    if not n:
        return None
    if n >= total:
        return "The one check of this analysis could not run on this server" if n == 1 else f"None of the {n} checks could run on this server"
    return f"{n} of the {total} checks could not run on this server"


def server_sentence(coverage: dict[str, Any] | None) -> str | None:
    """The count with the server's reason: its memory budget, model files that are not installed, or both."""
    by_reason, head = not_run_on_server(coverage), server_count(coverage)
    if head is None:
        return None
    memory, files = by_reason.get("memory budget", 0), by_reason.get("model not installed", 0)
    if memory and files:
        return f"{head}: its memory budget is too small for {memory}, and the model files of {files} are not installed."
    if memory:
        return f"{head}: its memory budget is too small for {'it' if memory == 1 else 'them'}."
    return f"{head}: {'its' if files == 1 else 'their'} model files are not installed."


def server_step(coverage: dict[str, Any] | None) -> str | None:
    n = sum(not_run_on_server(coverage).values())
    if not n:
        return None
    return f"Ask the administrator of this server to make {'this check' if n == 1 else 'these checks'} able to run, then analyse the file again."


def no_weight(decision: dict[str, Any] | None) -> bool:
    """N-19f (code review R4-05; answer.ts `noWeight`): a result that is not assessable because the fusion model gives the
    checks that scored no weight for the input. The checks scored and the fusion was reached, so no surface says that
    nothing ran, scored or was reached."""
    return bool(decision) and decision.get("code") == "not_assessable" and decision.get("reason") == abstain.NO_WEIGHT_REASON


# N-19f: the one-page reports have no line for checks outside the estimate; under such a result they say where those are
NO_WEIGHT_CHECKS = ("The checks that scored carry no weight for an input like this one, so none of their findings is a reason "
                    "here. The standard and forensic reports list them.")


def no_estimate_line(coverage: dict[str, Any] | None, decision: dict[str, Any] | None = None) -> str:
    """L-3 (answer.ts `noEstimateLine`): what happened, never "too few checks" beside "4 of 4 checks"."""
    ok = int((coverage or {}).get("by_status", {}).get("ok", 0))
    total = int((coverage or {}).get("detectors_total", 0))
    noun = "check" if total == 1 else "checks"
    if no_weight(decision):  # N-19f: checks scored; the model gives them no weight here
        return (f"{NO_ESTIMATE_FORMED}: {ok} of {total} {noun} ran, and for an input like this one the fusion model gives those "
                "that scored no weight.")
    server = server_count(coverage)
    if server is not None:  # N-14a: the server, not the file
        return f"{NO_ESTIMATE_FORMED}: {server[0].lower()}{server[1:]}."
    if total == 0:  # N-17 (browser QA P1-4): never "none of the 0 checks"
        return f"{NO_ESTIMATE_FORMED}: no check could be applied to this file."
    if ok == 0:
        return f"{NO_ESTIMATE_FORMED}: none of the {total} {noun} produced a result."
    return f"{NO_ESTIMATE_FORMED}: {ok} of {total} {noun} ran, but none produced a score that can be used."


def checks(coverage: dict[str, Any] | None) -> str:
    ok = int((coverage or {}).get("by_status", {}).get("ok", 0))
    return f"{ok} {'check' if ok == 1 else 'checks'}"


def checks_counted(coverage: dict[str, Any] | None, decision: dict[str, Any] | None, outside: bool = False) -> str:
    """N-19d (browser QA "Copy": "Inferential · 14 checks" beside "Three checks entered the fusion model"): the count beside
    a result says what it counts. Under a fusion model with published rows: "14 checks ran, 3 in the estimate", where the
    second number is the one of "… entered the fusion model" (N-19i: the rows the decision counts, `counted_rows`; it was
    every row of a check, a row of zero and the row of a check that declares H3 among them). Otherwise "14 checks ran".
    The page's `checksCounted`. `outside` (N-19n, D3-09): the input is outside what was validated, where no estimate of the
    checks is shown: "5 checks ran, 2 combined" (it read "2 in the estimate" beside "No estimate is shown")."""
    ran = f"{checks(coverage)} ran"
    rows = (decision or {}).get("contributions") or []
    if (decision or {}).get("fusion_mode") != "artefact" or not rows:
        return ran
    entered = len(counted_rows(decision))
    return f"{ran}, {entered or 'none'} {'combined' if outside else 'in the estimate'}"


NO_ESTIMATE_NOT_RUN = "No estimate: a check that this analysis depends on did not run."  # D-49
NOT_RUN_MEANS = "A check that this analysis depends on did not run, so no estimate is published."


def no_estimate_capped(decision: dict[str, Any]) -> bool:
    """D-49: an artefact decision other than not assessable that publishes no estimate, because a check it depends on did not run."""
    return decision.get("fusion_mode") == "artefact" and decision.get("code") != "not_assessable" and estimate_of(decision) is None


# N-19n (the browser QA's lead 3): the reason of a D-49 result names each check that did not run with why
# (`runtime.fuse_with_artefact`: "… did not run: A (could not run on this server: memory budget); B (failed: …). No estimate …")
_NOT_RUN_REASON = re.compile(r"^Capped at inconclusive: a check that this analysis depends on did not run: (.+)\. No estimate is published "
                             r"while a check did not run\.$", re.DOTALL)
# what follows the server's sentence under "What this means" of such a result, in place of a second sentence on a check that
# did not run
SERVER_NO_ESTIMATE = "No estimate is published without every check that this analysis depends on."


def not_run_of_server(decision: dict[str, Any] | None, coverage: dict[str, Any] | None) -> bool:
    """N-19n (answer.ts `notRunOfServer`): a result without an estimate because a check did not run (D-49), where the
    checks that did not run are those that could not run on this server (its memory budget; model files that are not
    installed). The one-page report of an image and the top block of its page then say so as the document's do: in the
    line under the headline and in one sentence under "What this means", not in three."""
    if not decision or not no_estimate_capped(decision) or server_count(coverage) is None:
        return False
    named = _NOT_RUN_REASON.match(str(decision.get("reason") or ""))
    return named is not None and all(ON_THIS_SERVER in part for part in named.group(1).split("; "))


def estimate_line(decision: dict[str, Any], basis: str, coverage: dict[str, Any] | None,
                  findings: list[dict[str, Any]] | None = None, profile: dict[str, Any] | None = None, *, failed: bool = False) -> str:
    """The line under the headline of the summary and newsroom reports. The number follows `result_number` (D-68).
    `failed`: the step that writes the findings failed (`agreement`)."""
    n = result_number(decision, findings, profile)
    agree = f"agreement {agreement_beside(n, agreement(decision, coverage, findings, failed=failed))}"
    ran = checks_counted(coverage, decision, outside_validated(decision, profile))
    if basis == "insufficient":
        return no_estimate_line(coverage, decision)
    if basis == "documentary_declared":
        return f"Documentary · Content Credentials · statistical checks alone: {statistical_alone(decision, profile)}"
    if basis == "documentary_capture":
        return capture_line(decision, n, profile)
    if no_estimate_capped(decision):
        # N-19n: where the server is what was short, the line says so, as the document's does
        return no_estimate_line(coverage, decision) if not_run_of_server(decision, coverage) else NO_ESTIMATE_NOT_RUN
    if n.kind == "none":
        return f"No estimate was produced · {ran}"
    if n.kind == "capped":
        return NO_ESTIMATE_OUTSIDE
    if n.kind == "record":  # N-19h (E-20, 1): the record's value in words, with the statistical checks' own beside it
        return f"{str(n.text)[0].upper()}{str(n.text)[1:]} · statistical checks alone: {statistical_alone(decision, profile)} · {agree} · {ran}"
    if n.kind == "uncalibrated":
        return f"{n.text}: uncalibrated fallback with hand-set weights · {agree} · {ran}"
    if n.kind == "text":
        return f"{n.text} · {agree} · {ran}"
    # L-7 (QA P2 "17 estimated likelihood"): the numeral carries its unit, as the screen does
    return f"{numeral(n.estimate)} % {likelihood_words(decision)}" + (f" · plausible range {n.span} %" if n.span else "") + f" · {agree} · {ran}"  # type: ignore[arg-type]


def capture_line(decision: dict[str, Any], n: ResultNumber, profile: dict[str, Any] | None = None) -> str:
    """N-19m (D3-01; D-15): the line under the headline of a result that a trusted capture decides. It begins with the
    basis, as under a declaration, and holds neither an agreement nor a count of checks: the result is the record's, and
    it read "… · agreement medium · 6 checks ran, 2 in the estimate" as a statistical one does. The number stays as D-68
    gives it: the record's value in words where the record set it, with the statistical checks' own beside it, and the
    statistical one as the checks alone would show it where it did not."""
    if n.kind == "record":
        number = f"{n.text} · statistical checks alone: {statistical_alone(decision, profile)}"
    elif n.kind == "capped":
        number = _lower_first(NO_ESTIMATE_OUTSIDE.removesuffix("."))
    elif n.kind == "uncalibrated":
        number = f"{n.text}: uncalibrated fallback with hand-set weights"
    elif n.kind == "text":
        number = _lower_first(str(n.text))
    elif n.kind == "calibrated":
        number = f"{numeral(n.estimate)} % {likelihood_words(decision)}" + (f" · plausible range {n.span} %" if n.span else "")  # type: ignore[arg-type]
    else:
        number = "no estimate was produced"
    return f"{CAPTURE_BASIS} · {number}"


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def error_rate_contexts(kind: str, profile: dict[str, Any] | None) -> list[str]:
    """ML §5.1 error-rate contexts that apply to the input, most specific first (answer.ts `errorRateContexts`)."""
    slice_id = str((profile or {}).get("slice_id", ""))
    face = "/face" in slice_id and "/noface" not in slice_id
    if kind == "image":
        out = []
        if (profile or {}).get("screenshot_likelihood"):
            out.append("screenshot")
        if "/face<120" in slice_id:
            out.append("small face (< 120 px)")
        elif face:
            out.append("portrait face")
        if "/side<512" in slice_id:
            out.append("low-res (< 512)")
        if "/q<70" in slice_id:
            out.append("social-recompressed")
        return out
    if kind == "video":
        return ["video talking head" if face else "video T2V"]
    lossy = any(str(c).startswith("lossy compression") for c in (profile or {}).get("limiting_conditions") or [])
    return ["audio lossy"] if kind == "audio" and lossy else []  # L-3: a FLAC or WAV recording is not lossy


def frequency(rate: float, n: int) -> str:
    """Natural frequency (UX §7.4): whole numbers per 100, "fewer than 1 in 100", or "k of n" below 100 test items."""
    if n < ICON_ARRAY_MIN_N:
        return f"{half_up(rate * n)} of {n}"
    v = rate * 100
    return "fewer than 1 in 100" if 0 < v < 0.5 else f"{half_up(v)} in 100"


def in_testing(kind: str, decision: dict[str, Any], profile: dict[str, Any] | None,
               calibration: list[dict[str, Any]]) -> dict[str, Any]:
    """UX §7.4 T1 + T2 (headline toward AI or inconclusive) or T3 (headline against) at the fuser's conformal 5 % threshold for
    the first matching context; icon arrays only when every count shown comes from ≥ 100 test items. Never another context's
    numbers."""
    one, many = media(kind)
    if line := text_reference_rate(kind, profile, calibration):
        return {"line": line, "scope": "Held-out writing from the local reference evaluation.", "arrays": []}
    rows = [r for c in calibration if str(c.get("subject", "")).startswith("fuser-") for r in c.get("error_rates") or []]
    contexts = error_rate_contexts(kind, profile)
    for context in contexts:
        row = next((r for r in rows if r.get("context") == context and r.get("threshold_kind") == "conformal_0.05"), None)
        if row is None:
            continue
        scope = f"{row['bench_version']}, {formatting.context_words(context)} inputs"
        tpr, fpr, n_pos, n_neg = float(row["tpr"]), float(row["fpr"]), int(row["n_pos"]), int(row["n_neg"])
        if decision.get("code") in ("no_ai_indicators", "no_ai_indicators_strong"):
            caption = f"{frequency(1 - tpr, n_pos)} AI-generated or manipulated {many} scored below the flagging threshold"
            arrays = [(half_up((1 - tpr) * 100), caption)] if n_pos >= ICON_ARRAY_MIN_N else []
            line = f"Detectors miss some: in testing ({scope}), {caption}."
        else:
            false_alarms = f"{frequency(fpr, n_neg)} genuine {many} scored above the flagging threshold"
            detections = f"{frequency(tpr, n_pos)} AI-generated or manipulated {many} did"
            arrays = ([(half_up(fpr * 100), false_alarms), (half_up(tpr * 100), detections)]
                      if min(n_pos, n_neg) >= ICON_ARRAY_MIN_N else [])
            line = f"In testing ({scope}), {false_alarms}, and {detections}."
        return {"line": line, "scope": f"Test set: {scope}.",
                "arrays": [{"svg": figures.icon_array_svg(k), "caption": text} for k, text in arrays]}
    line = (f"Known error rates are not established: no evaluation is on record for {formatting.context_words(contexts[0])} inputs." if contexts
            else f"Known error rates are not established: no evaluation is on record for this kind of {one}.")
    return {"line": line, "scope": None, "arrays": []}


def guidance(kind: str, decision: dict[str, Any], basis: str, coverage: dict[str, Any] | None = None,
             findings: list[dict[str, Any]] | None = None, signals: list[dict[str, Any]] | None = None,
             profile: dict[str, Any] | None = None, *, caveats: list[Any] | None = None) -> tuple[list[str], list[str]]:
    """UX §7.1 "What this means / What this does not mean" (answer.ts `guidance` without the natural-frequency line, which the
    report prints under "In testing"). N-14a: a result in which checks could not run on this server says so in "What this
    means" (the insufficient variant says it first among its reasons, `insufficient_reasons`), and is not told to obtain
    another file where the server is what was short. N-18 (browser QA P1-1): under a clean headline a check outside the
    estimate that flagged the file is named (`flagged_outside_sentence`). `caveats` (N-19n): the result's caveats; where
    they hold the caveat of a configured prior, the sentence of a condition that took the code is followed by the sentence
    on the prior (`PRIOR_ADJUSTED`)."""
    means, does_not_mean = _guidance(kind, decision, basis, profile)
    share = prior_share(caveats)
    if (beside := unsupported_beside_record(decision, profile)) is not None:  # N-19l (R6-05): after the record's own
        means = [*means, _with_prior(beside, share, start=True), *_readings_beside(decision, findings or [])]
    if basis == "inferential":  # N-19m (D-77): one reason, D-77's or D-76's
        if (why := unsupported_ai_sentence(decision)) is not None:  # a headline published before N-19i: no condition took it
            means = [*means, why]
        elif (why := unsupported_sentence(decision)) is not None:
            means = [*means, _with_prior(why, share, start=True), *_readings_own(decision, findings or [])]
        elif (why := off_scale_sentence(decision)) is not None:
            means = [*means, _with_prior(why, share, start=False)]
    # N-19o (browser QA Q5-02): wherever the number of a result under a configured prior is printed, one sentence gives the
    # share, once: D-76's and D-77's sentences above state it already
    if share is not None and not any(PRIOR_SAID in m for m in means) and (whose := prior_number_printed(decision, findings or [], profile)):
        means = [*means, (PRIOR_SHARE_CHECKS if whose == "beside" else PRIOR_SHARE).format(share=share)]
    if basis in READ_AS_STATISTICAL and (flagged := flagged_outside_sentence(kind, decision, findings or [], signals)) is not None:
        means = [*means, flagged]
    if (unapplied := unapplied_sentence(kind, decision, findings or [], signals)) is not None:  # N-19l (R6-03, 1): on both fusion paths
        means = [*means, unapplied]
    # N-19n (D3-07, 8): credentials that do not validate, under every headline and on both fusion paths (the sentence of a
    # check outside the estimate stood under a clean headline of a calibrated result only, and nothing anywhere else)
    if not not_assessable(decision) and (invalid := invalid_sentence(findings or [])) is not None:
        means = [*means, invalid]
    server = server_sentence(coverage)
    if server is None or basis == "insufficient":
        return means, does_not_mean
    if not_run_of_server(decision, coverage):  # N-19n: one sentence, with the server's reason; it said three times that a check did not run
        means = [f"{server} {SERVER_NO_ESTIMATE}", *(s for s in means if s != NOT_RUN_MEANS)]
        return means, [text.removesuffix(OBTAIN_ANOTHER) if text.endswith(OBTAIN_ANOTHER) else text for text in does_not_mean]
    return ([*means, f"{server} {SERVER_FEWER}"],
            [text.removesuffix(OBTAIN_ANOTHER) if text.endswith(OBTAIN_ANOTHER) else text for text in does_not_mean])


def _readings_own(decision: dict[str, Any], findings: list[dict[str, Any]]) -> list[str]:
    """N-19q (finding 1): after the sentence of D-76 in its first form, the sentence on the other readings, if any."""
    side = unsupported(decision)
    if side is None or rule_taken(decision) in TOGETHER_RULES:
        return []
    said = other_readings_sentence(decision, findings, side)
    return [said] if said else []


def _readings_beside(decision: dict[str, Any], findings: list[dict[str, Any]]) -> list[str]:
    """N-19q (finding 1): the same beside a record, of the statistical checks' estimate."""
    alone = decision.get("statistical_alone") or {}
    if alone.get("unsupported") not in UNSUPPORTED_ANSWER or alone.get("rule") in TOGETHER_RULES:
        return []
    said = other_readings_sentence(decision, findings, str(alone["unsupported"]), "that estimate")
    return [said] if said else []


# N-18 (browser QA P1-1): what a flagged check outside the estimate looked at, by the family of its finding
FLAGGED_SUBJECT = {"A1_voice": "the sound track", "V1_face_track": "a face track", "V2_lip_sync": "the lip movement",
                   "V3_synthetic_video": "the frames", "V4_codec": "the encoding", "F3_whole_image_generation": "the image",
                   "F4_localised_manipulation": "a region of the image", "F7_duplication": "a region of the image",
                   "F10_face": "a face", "T1_ai_text": "the text"}
FLAGGED_STRENGTHS = ("moderate", "strong")
FLAGGED_RAW_SCORE = 10 / 11  # odds of 10 to 1, the lower edge of "moderate" for a score that is not calibrated (`strength.py`)


def _flat_raw_score(finding: dict[str, Any], signals: dict[str, dict[str, Any]]) -> float | None:
    """The score of a check whose stand-alone fit in the model says nothing at any score (a likelihood ratio of exactly 1:
    the served video model fitted the voice check on eight clips, with slope 0) while the score itself is at least 10 to
    1 for manipulation. None for any other check."""
    for sid in finding.get("evidence_ids") or []:
        signal = signals.get(sid)
        if signal is None or signal.get("raw_kind") != "probability" or signal.get("raw_score") is None or signal.get("log_lr") is None:
            continue
        if abs(float(signal["log_lr"])) <= 1e-9 and float(signal["raw_score"]) >= FLAGGED_RAW_SCORE:
            return float(signal["raw_score"])
    return None


def flagged_outside(decision: dict[str, Any], findings: list[dict[str, Any]],
                    signals: list[dict[str, Any]] | None = None) -> list[tuple[dict[str, Any], float | None]]:
    """The checks outside the estimate that flagged the file, under a clean headline of a calibrated result (answer.ts
    `flaggedOutside`): a finding with at least moderate stand-alone support for manipulation, or, where the model's
    stand-alone fit of the check says nothing, a score of the check itself of at least 10 to 1 (returned beside the
    finding). Empty for any other headline: there the estimate's own checks speak."""
    if SIDE.get(str(decision.get("code"))) != "against" or not has_contribution_rows(decision):
        return []
    by_id = {str(s.get("id")): s for s in signals or []}
    out: list[tuple[dict[str, Any], float | None]] = []
    for f in sorted(findings, key=_rank_order):
        if f.get("family") in NOT_KEY or entered_decision(f, decision):
            continue
        # N-19l (R6-03, 1): a provenance record is no check, and it flags no part of the file (`unapplied_sentence`); N-19o
        # (R9-02): known by its family as well, as `provenance_aside` knows it, so that it holds without the signals
        if f.get("family") in PROVENANCE_FINDINGS or any((by_id.get(str(e)) or {}).get("provenance_override") for e in f.get("evidence_ids") or []):
            continue
        if is_invalid_credentials(f):
            continue  # N-19n (D3-07, 8): nor are credentials that do not validate (`invalid_sentence`)
        if finding_direction(f) == "for" and f.get("strength") in FLAGGED_STRENGTHS:
            out.append((f, None))
        elif f.get("strength") == "none" and (raw := _flat_raw_score(f, by_id)) is not None:
            out.append((f, raw))
    return out


# N-19l (R6-03, 1): what a self-declared marker that the decision did not apply is, by its kind (`decisive.MARKER_KINDS`)
UNAPPLIED_MARKER = {
    decisive.METADATA: "the file’s metadata names an AI generator",
    decisive.WATERMARK: "the {one} carries a public AI-generator watermark message",
    decisive.CREDENTIALS_LISTED: "the file’s Content Credentials name an AI generator",
    decisive.CREDENTIALS_UNLISTED_DECLARED: "the file’s Content Credentials declare AI generation, from a signer that is not on the trust list",
    decisive.CREDENTIALS_UNLISTED_NAMED: "the file’s Content Credentials name an AI generator, from a signer that is not on the trust list",
    decisive.METADATA_DECLARED: "the file’s metadata declares AI generation",
    decisive.METADATA_COMPOSITE: "the file’s metadata declares a composite with AI-generated parts",
    decisive.METADATA_PARAMETERS: "the file’s metadata holds AI generation parameters",
}


def unapplied_markers(decision: dict[str, Any], findings: list[dict[str, Any]],
                      signals: list[dict[str, Any]] | None) -> list[tuple[str, str]]:
    """N-19l (R6-03, 1; answer.ts `unappliedMarkers`): under a trusted capture that decides, the self-declared markers that
    the file also carries: (the kind of the marker, the id of its finding), in the order of the findings. They point the
    other way than the capture's headline, and the decision did not apply them. Empty under any other result: under a
    declaration a marker points the way of the headline, and nothing more is said of it."""
    if (record_of(decision) or {}).get("kind") != "trusted_capture":
        return []
    by_id = {str(s.get("id")): s for s in signals or []}
    out = []
    for f in sorted(findings, key=_rank_order):
        mine = [by_id[str(e)] for e in f.get("evidence_ids") or [] if (by_id.get(str(e)) or {}).get("provenance_override") == "self_declared_ai"]
        if mine and (marker := decisive.marker_kind(mine)) is not None:
            out.append((marker, str(f.get("id", ""))))
    return out


def unapplied_sentence(kind: str, decision: dict[str, Any], findings: list[dict[str, Any]],
                       signals: list[dict[str, Any]] | None = None) -> str | None:
    """N-19l (R6-03, 1): what the result means says of every record that the decision did not apply and that points the
    other way than the headline what it is, in the words of its kind, and that the signed capture record outranks it. It
    read N-18's sentence of a check without weight ("Part of the file was flagged by a check that carries no weight in this
    estimate (F4 …)"); on the fallback nothing was said. One sentence for one such record and for several (answer.ts
    `unappliedSentence`)."""
    markers = unapplied_markers(decision, findings, signals)
    if not markers:
        return None
    one, _ = media(kind)
    said = [f"{UNAPPLIED_MARKER[marker].format(one=one)} ({fid})" for marker, fid in markers]
    listed = said[0] if len(said) == 1 else f"{', '.join(said[:-1])} and {said[-1]}"
    it, them = ("that marker", "it") if len(said) == 1 else ("those markers", "them")
    return (f"Beside the capture record, {listed}. The decision did not apply {it}: the signed capture record outranks {them}, "
            "and a self-declared marker can be added to any file.")


def flagged_subject(kind: str, finding: dict[str, Any]) -> str:
    if kind == "audio" and finding.get("family") == "A1_voice":
        return "the recording"
    return FLAGGED_SUBJECT.get(str(finding.get("family")), "part of the file")


def flagged_outside_sentence(kind: str, decision: dict[str, Any], findings: list[dict[str, Any]],
                             signals: list[dict[str, Any]] | None = None) -> str | None:
    """N-18 (browser QA P1-1): camera footage with a synthetic voice read "No significant indications … 2 %" while the voice
    check, which carries no weight in the video model, scored 0.91. A clean headline names such a check.

    N-19k (browser QA Q2-12): the sentence carries both halves, the score and why it is not counted, the reason taken
    from the record of that check for this input (`effect.outside_why`; the figures stand in its finding): "A check of the
    sound track (F4) scored 0.91 and is not counted in this estimate: in testing it was at chance on inputs like this
    one. The headline does not take it into account. To assess the voice, examine the sound track as audio." It read "The
    sound track was flagged by a check that carries no weight in this estimate (…)", beside "0 of 5 scored checks flagged
    this file" and the check's own finding, "provides no information about manipulation": an examiner could not tell
    whether 0.91 was a warning or noise. One sentence for each such check (answer.ts `flaggedOutsideSentence`)."""
    flagged = flagged_outside(decision, findings, signals)
    if not flagged:
        return None
    by_id = {str(s.get("id")): s for s in signals or []}
    said = []
    for f, raw in flagged:
        record = next((by_id[e]["outside_estimate"] for e in f.get("evidence_ids") or [] if (by_id.get(e) or {}).get("outside_estimate")), None)
        why = effect.outside_why(record, f.get("benchmark_note"), figures=False)
        basis = f"gives {f.get('strength')} support for manipulation on its own" if raw is None else f"scored {raw:.2f}"
        said.append(f"A check of {flagged_subject(kind, f)} ({f.get('id', '')}) {basis} and is not counted in this estimate: {why}.")
    said.append("The headline does not take it into account." if len(flagged) == 1 else "The headline does not take them into account.")
    if kind == "video" and any(f.get("family") == "A1_voice" for f, _ in flagged):
        said.append(effect.VOICE_STEP)
    return " ".join(said)


def _counted(n: int, one: str, many: str) -> str:
    return one if n == 1 else f"{n} {many}"


def _joined(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} and {parts[-1]}"


def _start_and_terms(decision: dict[str, Any], sign: int) -> list[str]:
    """The starting point, and the fitted terms of checks that could not score on the side `sign` (+1 toward AI, −1
    against), counted: what an estimate stands on before any check that scored."""
    parts = [f"the starting point of {start_where(decision)} out of 100 for inputs like this one"]
    missing = sum(1 for r in decision.get("contributions") or [] if r["kind"] == "missing" and sign * float(r["logit"]) > 0)
    if missing:
        parts.append(_counted(missing, "the fitted term of a check that could not score", "fitted terms of checks that could not score"))
    return parts


def _comes_from(decision: dict[str, Any], sign: int) -> str:
    """Where an estimate that no check supports comes from, on the side `sign` (+1 toward AI, −1 against): the starting
    point, the fitted terms of checks that could not score, the checks that pushed the other way and the checks that each
    had little effect. N-19n (D4-02 … D4-05, point 3): the checks of little effect are counted as the breakdown counts
    them, every counted row under ln 2 in either direction ("one check that had little effect" stood over "None pushed the
    estimate up, none pushed it down, two had little effect."), and a counted row of ln 2 or more against the answer is
    named as a check that pushed the other way (it was not counted at all)."""
    parts = _start_and_terms(decision, sign)
    rows = [float(r["logit"]) for r in counted_rows(decision)]  # N-19i: the checks that count
    against = sum(1 for logit in rows if sign * logit <= -effect.LN2)
    small = sum(1 for logit in rows if abs(logit) < effect.LN2)
    if against:
        parts.append(_counted(against, "one check that pushed the other way", "checks that pushed the other way"))
    if small:
        parts.append(_counted(small, "one check that had little effect", "checks that each had little effect"))
    return _joined(parts)


def unsupported_ai_sentence(decision: dict[str, Any]) -> str | None:
    """D-63 (N-17; browser QA P0-1): an AI headline under which no check gives at least weak support says where the estimate
    comes from: the starting point, the fitted term of a check that could not score, and checks that each had little effect
    (answer.ts `unsupportedAiSentence`). None for any other result, and for a result published before N-17. Since N-19i
    (D-76) no new result has such a headline; the sentence stays for the results published earlier."""
    if not measured_from_neutral(decision) or SIDE.get(str(decision.get("code"))) != "for" or estimate_of(decision) is None:
        return None
    if any(float(r["logit"]) >= effect.LN2 for r in decision.get("contributions") or [] if r["kind"] == "detector"):
        return None
    return f"No check gave at least weak support for it on its own. The estimate comes from {_comes_from(decision, 1)}."


def unsupported(decision: dict[str, Any] | None) -> str | None:
    """N-19i (D-76; answer.ts `unsupported`): the side of the estimate ("for", "against") of a result that is inconclusive
    because no check supports the answer of its estimate; None for any other result. N-19m (D-76 as amended): or because
    the checks together do not support it; a result that D-77 took is not one of these (`off_scale`)."""
    return abstain.unsupported_side(decision)


TOGETHER_RULES = (abstain.RULE_TOGETHER_LESS, abstain.RULE_TOGETHER_AGAINST)


def rule_taken(decision: dict[str, Any] | None) -> str | None:
    """N-19m (answer.ts `ruleTaken`): the condition that took the decisive code of an inconclusive result's estimate
    (`abstain.RULE_*`), read from the end of its reason; None for any other result."""
    taken = abstain.rule_taken(decision)
    return taken[0] if taken is not None else None


def off_scale(decision: dict[str, Any] | None) -> str | None:
    """N-19m (D-77; answer.ts `offScale`): the side ("for", "against") of the answer that the calibrated levels would have
    given, in a result that is inconclusive because its estimate stands in the middle part of the scale; None for any
    other result."""
    taken = abstain.rule_taken(decision)
    return taken[1] if taken is not None and taken[0] == abstain.RULE_SCALE else None


def off_scale_part(decision: dict[str, Any] | None) -> str | None:
    """N-19n (supervisor check 29; answer.ts `offScalePart`): in a result that D-77 took, the part of the scale of the
    other answer in which its published estimate prints ("for": the AI answer's part, 65 or more; "against": the clean
    answer's, 34 or less); None where it prints in the middle part, and for any other result. Read from where the
    estimate prints, in the rule's whole points, and not from the end of the reason: a result that the build of N-19m's
    third part stored with the ending of the middle part reads right too."""
    side, estimate = off_scale(decision), estimate_of(decision) if decision else None
    if side is None or estimate is None:
        return None
    part = abstain.part_of_scale(float(estimate))
    return part if part != side else None


# N-19m (D-77): what the levels would have read, and the bound that a headline of that answer has. Neither quotes the
# headline that the result does not have
OFF_SCALE_READ = {"for": ("indications", "65 or more"), "against": ("no significant indications", "34 or less")}


def off_scale_sentence(decision: dict[str, Any]) -> str | None:
    """N-19m (D-77; answer.ts `offScaleSentence`): what such a result says where "What this means" stands: its estimate,
    that it stands in the middle part of the scale, what the calibrated levels would have read, and the bound. None for
    any other result."""
    side = off_scale(decision)
    if side is None or estimate_of(decision) is None:
        return None
    read, bound = OFF_SCALE_READ[side]
    if (part := off_scale_part(decision)) is not None:  # N-19n: it said "the middle part of the scale (35 to 64)" of a printed 21
        return (f"The estimate itself, {numeral(estimate_of(decision))} out of 100, stands in {ANSWERS_PART[part]} ({PART_BOUND[part]}). "
                f"The calibrated levels for inputs like this one would have read {read}; a headline of that answer also needs an "
                f"estimate of {bound}.")
    return (f"The estimate itself, {numeral(estimate_of(decision))} out of 100, stands in the middle part of the scale (35 to 64), "
            f"where Chipmunk Verify gives no answer. The calibrated levels for inputs like this one would have read {read}; a headline "
            f"of that answer also needs an estimate of {bound}.")


# N-19n (supervisor check 29): none of these sentences said why the levels and the estimate part ways under a configured
# prior (`VERITAS_PRIOR_AI`): the estimate uses the prior, the levels do not; and the start that the sentences of D-76 name
# holds the shift too (a start of 13 reads 2 at a prior of 0.1). A result that D-77 or either condition of D-76 took, and
# that holds the caveat of a configured prior, adds one sentence after its own, with the caveat's share: the one-page
# reports print no caveat. The first where D-77 took the code, the second where D-76 did, alone or beside a record
PRIOR_ADJUSTED = ("The estimate is adjusted for the share of AI-generated or manipulated inputs that this server is set to assume, "
                  "{share}; the calibrated levels are not.")
PRIOR_ADJUSTED_START = ("The estimate and its starting point are adjusted for the share of AI-generated or manipulated inputs that "
                        "this server is set to assume, {share}; the calibrated levels are not.")


# N-19o (Q5-02): the first sentence of `PRIOR_ADJUSTED` without its clause on the calibrated levels, said beside the number
# of a result that no condition took; `PRIOR_SAID` knows a sentence that states the prior already
PRIOR_SHARE = PRIOR_ADJUSTED.split(";")[0] + "."
# N-19p (browser QA Q6-02): beside a record, where the number that the prior shifts is the statistical checks'
PRIOR_SHARE_CHECKS = PRIOR_SHARE.replace("The estimate is", "The statistical checks' estimate is", 1)
PRIOR_SAID = "this server is set to assume"


def prior_number_printed(decision: dict[str, Any], findings: list[dict[str, Any]], profile: dict[str, Any] | None) -> str | None:
    """Which number that a configured prior shifts is printed for this result: `own` (an estimate, a score, the words of an
    inconclusive estimate), `beside` (the statistical checks' beside a record, where it is shown: N-19h, E-20), or None."""
    if result_number(decision, findings, profile).kind not in ("none", "capped", "record"):
        return "own"
    shown = decision.get("statistical_estimate_before_provenance") is not None and not statistical_withheld(decision, profile)
    return "beside" if shown else None


def prior_share(caveats: list[Any] | None) -> str | None:
    """The share of a configured prior as a result's caveat prints it ("10 %"; answer.ts `priorShare`); None for a result
    without that caveat. The caveat is known by its fixed words (`abstain.PRIOR_CAVEAT`), not by its number: a prior of
    0.004 prints "0 %", and one of 0.5 writes the caveat and shifts nothing."""
    head, tail = abstain.PRIOR_CAVEAT
    for caveat in caveats or []:
        text = str(caveat)
        if text.startswith(head) and text.endswith(tail) and len(text) > len(head) + len(tail):
            return f"{text[len(head):-len(tail)].removesuffix('%').strip()} %"
    return None


def _with_prior(sentence: str, share: str | None, *, start: bool) -> str:
    return sentence if share is None else f"{sentence} {(PRIOR_ADJUSTED_START if start else PRIOR_ADJUSTED).format(share=share)}"


def _rests_on(decision: dict[str, Any], sign: int) -> str:
    """N-19m (D-76, the second condition): the sentence on what the estimate comes from counts only checks of little
    effect, and here checks moved it; the estimate is said to rest on the start. N-19n (supervisor check 29): and on the
    fitted terms of the checks that could not score, on the side `sign` of the answer, as `_comes_from` words them: on the
    video model a result holds such terms beside a start of 69 (two frame checks scoring, the other three not applicable),
    and the sentence named the start alone."""
    return _joined(_start_and_terms(decision, sign))


UNSUPPORTED_ANSWER = {"for": "toward AI generation or manipulation", "against": "away from AI generation or manipulation"}


def unsupported_sentence(decision: dict[str, Any]) -> str | None:
    """N-19i (D-76; answer.ts `unsupportedSentence`): what such a result says where "What this means" stands: which way its
    estimate points, that no check supports that answer, and where the estimate comes from (N-17's words, for both
    sides). None for any other result."""
    side = unsupported(decision)
    if side is None or estimate_of(decision) is None or not has_contribution_rows(decision):
        return None
    rule = rule_taken(decision)  # N-19m: the second condition of D-76 has two forms of its own
    sign = 1 if side == "for" else -1
    if rule == abstain.RULE_TOGETHER_LESS:
        return (f"The estimate alone points {UNSUPPORTED_ANSWER[side]}, but taken together the checks gave less than weak support "
                f"for that, so the result is inconclusive: what one check adds, another takes away. The estimate rests on "
                f"{_rests_on(decision, sign)}.")
    if rule == abstain.RULE_TOGETHER_AGAINST:
        return (f"The estimate alone points {UNSUPPORTED_ANSWER[side]}, but taken together the checks point the other way, so "
                f"the result is inconclusive. The estimate rests on {_rests_on(decision, sign)}.")
    return (f"The estimate alone points {UNSUPPORTED_ANSWER[side]}, but {NO_PART} toward that answer, so the result is "
            f"inconclusive. The estimate comes from {_comes_from(decision, sign)}.")


# N-19q (N-20b's read of the reports, finding 1): D-76 reads each check's row in the estimate (D-63), and the sentence said "no
# check gave at least weak support for that on its own" beside the finding "Taken alone, the score is weak support for
# manipulation (likelihood ratio about 2.0)" and a check outside the estimate at moderate support: a reader took "no check"
# literally. The sentence names what the rule reads (a factor of 2 is ln 2, the edge of weak support)
NO_PART = "no check's part in this estimate reached weak support (a factor of 2)"


def other_readings(decision: dict[str, Any], findings: list[dict[str, Any]], side: str) -> tuple[int, int]:
    """N-19q (finding 1; answer.ts `otherReadings`): the checks of a result that D-76 took whose reading other than their part
    in the estimate gives the answer `side` at least weak support: (taken alone: a check that entered, by the stand-alone
    words its finding stores; outside the estimate: a check that did not enter, by its finding's own words). A check that
    declares H3 is not counted by the decision and is left out; a result stored before N-19q stores no stand-alone words."""
    alone = sum(1 for f in findings if f.get("family") not in NOT_KEY and f.get("family") not in PROVENANCE_FINDINGS
                and f.get("hypothesis") != H3 and entered_decision(f, decision)
                and f.get("standalone_direction") == side and f.get("standalone_strength") not in (None, "none"))
    outside = sum(1 for f in context_findings(findings, decision) if f.get("hypothesis") != H3 and finding_direction(f) == side)
    return alone, outside


def other_readings_sentence(decision: dict[str, Any], findings: list[dict[str, Any]], side: str, within: str = "this estimate") -> str | None:
    """N-19q (finding 1; answer.ts `otherReadingsSentence`): beside the sentence of D-76 in its first form, one sentence where a
    check taken alone or a check outside the estimate gives the answer weak support or more: such a reading does not decide
    the result, and why (its part in the estimate is small, or it carries no weight). None where no reading does."""
    alone, outside = other_readings(decision, findings, side)
    scores = "one check's score gives" if alone == 1 else f"the scores of {alone} checks give"
    if alone and outside:
        checks = "does one check" if outside == 1 else f"do {outside} checks"
        return (f"Taken alone, {scores} that answer weak support or more, and so {checks} outside {within}, but neither reading "
                f"decides the result here: a check's part in {within} is small, and a check outside it carries no weight.")
    if alone == 1:
        return (f"Taken alone, {scores} that answer weak support or more, but its part in {within} is small, so that reading does "
                "not decide the result here.")
    if alone:
        return (f"Taken alone, {scores} that answer weak support or more, but their parts in {within} are small, so those "
                "readings do not decide the result here.")
    if outside == 1:
        return (f"One check outside {within} gives that answer weak support or more, but it carries no weight, so that reading "
                "does not decide the result here.")
    if outside:
        return (f"{outside} checks outside {within} give that answer weak support or more, but they carry no weight, so those "
                "readings do not decide the result here.")
    return None


# N-19n (the design reviewer's read of the sentences of N-19m's third part): beside a record the three sentences of D-76 open
# alike. The first form opened "The statistical checks alone point away from AI generation or manipulation, but no check gave
# at least weak support for that on its own", which says of the checks that they point and that none of them does. Under a
# capture the sentence follows the sentence of the number ("The statistical checks alone do not decide: their estimate is
# …"), so it begins with the pronoun, and the sentence after it with "It": two sentences in a row began "The statistical
# checks alone …". (the opening, the subject of the sentence after it): answer.ts `BESIDE_OPENS`
BESIDE_OPENS = ("The estimate of the statistical checks alone points", "Their estimate")
BESIDE_OPENS_AFTER_NUMBER = ("Their estimate points", "It")
# what follows "…, but " in each form; the form of less than weak support keeps its last words beside a record too: the
# only place that says how it can stand a few lines above the key finding "Weak support against AI generation"
BESIDE_HOW = {abstain.RULE_TOGETHER_LESS: "taken together the checks gave less than weak support for that: what one check adds, another takes away",
              abstain.RULE_TOGETHER_AGAINST: "taken together the checks point the other way"}
BESIDE_NO_CHECK = "no check's part in it reached weak support (a factor of 2) toward that answer"  # N-19q (finding 1)


def unsupported_beside_record(decision: dict[str, Any], profile: dict[str, Any] | None = None) -> str | None:
    """N-19l (R6-05; D-76 in its form beside a record; answer.ts `unsupportedBesideRecord`): where a record decides and
    D-76 took the decisive code of the statistical estimate away: which way the statistical checks alone point, that no
    check supports that on its own, and where their estimate comes from. It does not say that the result is
    inconclusive: the result is the record's. N-19m (R7-01): said wherever their number is shown, whether the record
    set the result's number or not (it was said only where the record did not set it, which left the estimate bare
    beside a capture's cap, a declaration and a marker's floor: the common case). None for any other result, where no
    statistical number stands beside the record, and outside what was validated, where it is not shown."""
    side = (decision.get("statistical_alone") or {}).get("unsupported")
    if side not in UNSUPPORTED_ANSWER or record_of(decision) is None or decision.get("statistical_estimate_before_provenance") is None:
        return None
    if estimate_of(decision) is None or not has_contribution_rows(decision) or outside_validated(decision, profile):
        return None
    # N-19m (code review, the read of the third part): the sentence follows the condition. Under the second one a check
    # does support the answer ("no check gave at least weak support" stood above "Weak support against AI generation").
    # A result stored before the third part names no condition and reads as the first form
    rule = (decision.get("statistical_alone") or {}).get("rule")
    sign = 1 if side == "for" else -1
    after_number = (record_of(decision) or {}).get("kind") == "trusted_capture"  # `_guidance`: after `statistical_sentence`
    opens, then = BESIDE_OPENS_AFTER_NUMBER if after_number else BESIDE_OPENS
    if rule in TOGETHER_RULES:
        return f"{opens} {UNSUPPORTED_ANSWER[side]}, but {BESIDE_HOW[rule]}. {then} rests on {_rests_on(decision, sign)}."
    return f"{opens} {UNSUPPORTED_ANSWER[side]}, but {BESIDE_NO_CHECK}. {then} comes from {_comes_from(decision, sign)}."


def _guidance(kind: str, decision: dict[str, Any], basis: str, profile: dict[str, Any] | None = None) -> tuple[list[str], list[str]]:
    one, _ = media(kind)
    confounders = CONFOUNDERS.get(kind, CONFOUNDERS["image"])
    if basis == "insufficient":
        return ["Chipmunk Verify could not assess this file, for these reasons:"], ["It is not a finding either way."]  # L-3, N-17; answer.ts
    if basis == "documentary_declared":
        return (["A signed record inside the file, validated as untampered, states that the signing software generated or edited it."],
                ["Credentials record what the signing software declared. They don’t describe the scene or who directed it."])
    if basis == "documentary_metadata":  # N-19h: in the words of the marker's kind
        means, does_not_mean = marker_guidance(marker_of(decision), kind)
        return [means], [does_not_mean]
    if no_estimate_capped(decision):  # D-49
        return ([NOT_RUN_MEANS],
                [f"It is not a finding of manipulation, and not a finding that the {one} is unaltered. Obtain the original file or another source."])
    band = CODE_BAND[decision["code"]]
    if (record_of(decision) or {}).get("kind") == "trusted_capture":  # N-19h (E-20, 2): the record decides, and what stands beside it
        miss = [AUDIO_MISS] if kind == "audio" and decision.get("fuser_sha256") != "912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf" else []
        return [f"{CAPTURE_MEANS} {statistical_sentence(decision, profile)}"], [CAPTURE_NOT, *miss]  # N-19m (D3-01)
    if band == "very_likely_ai":
        return (["The combined evidence strongly points toward AI generation or manipulation."],
                [f"It does not prove the {one} was made or edited with AI, say who made it, or show exactly what changed. "
                 f"{confounders[0].upper()}{confounders[1:]} can produce similar patterns."])
    if band == "likely_ai":
        return (["The combined evidence points toward AI generation or manipulation."],
                [f"It is not a finding that the {one} is AI-generated. Similar patterns can come from {confounders}. Treat it as a reason to examine further."])
    if band == "uncertain":
        # N-7: the summary and newsroom reports print why a result was capped, as the screen does. N-19n: and why its
        # input is outside what was validated where the cap changed no code, in one sentence for both
        outside = outside_input_sentence(decision, profile)
        levels = levels_sentence(decision)  # N-19q (finding 12)
        return (["The combined evidence does not support a conclusive assessment.", *([outside] if outside else []),
                 *([levels] if levels else [])],
                [f"It is not a finding of manipulation, and not a finding that the {one} is unaltered. Obtain the original file or another source."])
    miss = [AUDIO_MISS] if kind == "audio" and decision.get("fuser_sha256") != "912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf" else []  # L-3: `docs/limitations.md`, "Synthetic voice"
    if band == "likely_authentic":
        return (["The combined evidence does not show significant indications of AI generation or manipulation."],
                [f"It does not confirm the {one} is authentic or unedited.", *miss])
    return (["The combined evidence does not indicate AI generation or manipulation."],
            [f"It does not confirm the {one} is authentic or unedited. Detectors miss some AI content.", *miss])


# N-19q (N-20b's read of the reports, finding 12): an inconclusive result of the calibrated levels themselves (no condition of
# D-76 or D-77 took a code, no cap) said only "The combined evidence does not support a conclusive assessment."; the note under
# the scale gave the reason. The conformal set of the moderate level, as the decision's reason records it (`runtime._SET_TEXT`)
LEVELS_SETS = {"{AI, non-AI}": "both answers", "∅": "neither answer"}


def levels_sentence(decision: dict[str, Any] | None) -> str | None:
    """N-19q (finding 12; answer.ts `levelsSentence`): why the calibrated levels give no answer, read from the decision's
    reason; None for any other result."""
    d = decision or {}
    if d.get("code") != "inconclusive" or d.get("fusion_mode") != "artefact" or rule_taken(d) is not None or cap_reason(d) is not None:
        return None
    reason = str(d.get("reason") or "")
    which = next((words for text, words in LEVELS_SETS.items() if f" {text}." in reason), None)
    if which is None:
        return None
    return f"At the calibrated levels for inputs like this one, the estimate fits {which}, so the result is inconclusive."


def insufficient_reasons(decision: dict[str, Any], profile: dict[str, Any] | None, coverage: dict[str, Any] | None = None) -> list[str]:
    """The reasons under "Chipmunk Verify could not assess this file": the server's first (N-14a), then the decision's and the input's.

    N-19k (browser QA lead 3): the line under the headline already counts the checks that could not run on this server, so
    where the decision's reason names them ("2 checks could not run on this server: A (memory budget); B …") the count is
    not given a third time: the one-page report of a document read the fact in three sentences. A result whose reason does
    not name them keeps the server's sentence, which holds the cause."""
    reason = decision.get("reason")
    server = None if ON_THIS_SERVER in str(reason or "") else server_sentence(coverage)
    return list(dict.fromkeys(r for r in [server, reason, *((profile or {}).get("limiting_conditions") or [])] if r))


def next_step(decision: dict[str, Any], basis: str, coverage: dict[str, Any] | None = None, *, failed: bool = False) -> str:
    """PRD RP-01 (1), J5 (answer.ts `nextStep`). N-14a: where the step was to obtain another file and checks could not run on
    this server, the step is the administrator's; a result with an estimate keeps its step and adds that one. `failed`
    (N-19n, Q4-01): the step that writes the findings failed, and the step is to run the analysis again: there are no
    key findings to examine."""
    if failed:
        return effect.run_again()
    server = server_step(coverage)
    if server is None or basis in ("documentary_declared", "documentary_metadata"):
        return _next_step(decision, basis)
    if basis == "insufficient" or no_estimate_capped(decision) or decision.get("code") == "inconclusive":
        return server
    return f"{_next_step(decision, basis)} For a result with every check: {server[0].lower()}{server[1:]}"


def _next_step(decision: dict[str, Any], basis: str) -> str:
    if basis == "insufficient":
        return "Obtain the original file or a higher-quality copy, then analyse it again."
    if basis == "documentary_declared":
        return "Check who signed the credentials, then ask the source for the original file."
    if basis == "documentary_metadata":
        return "Ask the source for the original file and how it was made."
    if basis == "documentary_capture":  # N-19m (D3-01): the record's step; where the checks alone give an AI code, it says so
        # N-19n (D4-02): on the fallback too, by its score's own ranges (`alone_other_way`)
        return CAPTURE_STEP + (f" {CAPTURE_STEP_AGAINST}" if alone_other_way(decision) else "")
    if decision.get("code") in ("ai_supported_strong", "ai_supported"):
        return "Examine the key findings below, and look for the original source before relying on this file."
    if decision.get("code") == "inconclusive":
        return "Look for the original file or another source, then analyse it again."
    return "If this file supports a consequential decision, confirm where it came from: detectors miss some AI content."


def _rank_order(f: dict[str, Any]) -> tuple[int, str]:
    return int(f.get("rank", 0)), str(f.get("id", ""))


def row_label(row: dict[str, Any], decision: dict[str, Any] | None) -> str:
    """N-19l (R6-04, 1; contributions.ts `rowLabel`): the label of an override row as it is printed. A result published
    before holds one label for every self-declared record; it is printed by the kind of the record that decides."""
    return decisive.printed_label(str(row.get("label") or ""), (record_of(decision) or {}).get("marker"))


def has_contribution_rows(decision: dict[str, Any]) -> bool:
    """L-3 (answer.ts `hasContributionRows`): an artefact decision whose contribution rows say which checks entered it."""
    return decision.get("fusion_mode") == "artefact" and bool(decision.get("contributions"))


def of_record(finding: dict[str, Any], decision: dict[str, Any] | None) -> bool:
    """N-19h (E-21; record.ts `ofRecord`): a finding of the record that decides the result: one of the record's signals is
    among its evidence. A stored result names no signals; there the finding of a marker that decides is the one of the
    family of self-declared metadata that points toward AI (a declaration's and a capture's findings are documentary)."""
    record = record_of(decision)
    if record is None:
        return False
    ids = record.get("signal_ids") or []
    if ids:
        return any(e in ids for e in finding.get("evidence_ids") or [])
    return (record.get("kind") == "self_declared_ai" and finding.get("family") == "F2_self_declared_metadata"
            and finding_direction(finding) == "for")


def entered_decision(finding: dict[str, Any], decision: dict[str, Any] | None = None) -> bool:
    """A non-zero contribution after gates, or documentary (a Stage 0 override); weight-0 checks did not enter. N-19h: where
    a record decides, every finding of that record entered the decision, with or without an override row (`decision`)."""
    c = finding.get("contribution_logit")
    return is_documentary(finding) or (c is not None and abs(float(c)) > 1e-9) or of_record(finding, decision)


def little_of(decision: dict[str, Any] | None) -> float:
    """The size below which a contribution "had little effect" (`app.core.effect`; vocabulary.ts `littleOf`): ln 2 (D-63), and
    0.05 for a result published before N-17, which keeps its labels."""
    return effect.LEGACY_LITTLE if decision is not None and start_kind(decision) == "context_prior" else effect.LN2


def measured_from_neutral(decision: dict[str, Any] | None) -> bool:
    """D-63: an artefact decision whose rows are measured from each check's neutral point (a `neutral_start` row)."""
    return decision is not None and has_contribution_rows(decision) and start_kind(decision) == "neutral_start"


def decision_direction(finding: dict[str, Any], decision: dict[str, Any] | None = None) -> str:
    """The finding's direction in this decision: by its contribution (neutral when it had little effect), else its own."""
    c = finding.get("contribution_logit")
    if is_documentary(finding) or c is None:
        return finding_direction(finding)
    return effect.direction(float(c), little_of(decision))


# N-19k (Q2-14; D-64): the families of findings that read a record of the file (its metadata, its container, its compression
# history, its Content Credentials) and not its content
RECORD_FAMILIES = frozenset({"F1_provenance", "F2_self_declared_metadata", "F5_compression_history", "F6_capture_pipeline",
                             "F8_metadata_consistency", "V4_codec"})
SET_ASIDE_WORDS = "not counted"
SET_ASIDE_TITLE = "A record of the file, not evidence of authenticity"
SET_ASIDE_NOTE = ("No check of the content gave a score that can be used, so this finding is shown for inspection only: metadata "
                  "is never evidence of authenticity.")


def set_aside(finding: dict[str, Any], decision: dict[str, Any] | None) -> bool:
    """N-19k (Q2-14; D-64): under a result that is not assessable, the finding of a record of the file that points away
    from AI generation (a camera's EXIF block). It is listed with what it read; it is not headed "support against AI
    generation or manipulation", and it is no key finding (`key_findings`)."""
    return (not_assessable(decision) and not is_documentary(finding) and finding.get("family") in RECORD_FAMILIES
            and finding.get("strength") != "none" and finding_direction(finding) == "against")


# N-19n (D3-07, 3; answer.ts `provenanceAside`): a finding of the provenance families (Content Credentials, self-declared
# metadata, a watermark message) that did not enter the decision is no check outside the estimate: a record is no check, and
# nothing says of it that its score is not counted. It is listed after the checks outside the estimate, under words of its
# own, each row with its diamond and no meter. It is known by its family, as the page knows it, and not by the signals
PROVENANCE_FINDINGS = frozenset({"F1_provenance", "F2_self_declared_metadata"})
ASIDE_TITLE = "Provenance findings that did not decide this result"
ASIDE_NOTE = "These records of the file did not decide this result; each says why. They are included as context."


def provenance_aside(findings: list[dict[str, Any]], decision: dict[str, Any]) -> list[dict[str, Any]]:
    """The provenance findings that did not decide this result, in rank order. With contribution rows: every finding of the
    provenance families with at least weak strength that did not enter the decision ("No Content Credentials" has none).
    Without rows (the fallback, where every finding with a strength is a key finding): the finding of a record that the
    decision did not apply: a marker that a record outranks or that the score left unapplied, and credentials that do not
    validate (strong); a provenance check that the fallback weighs stays among the key findings."""
    rows = has_contribution_rows(decision) or no_weight(decision)
    return sorted((f for f in findings if f.get("family") in PROVENANCE_FINDINGS and not entered_decision(f, decision)
                   and (f.get("strength") != "none" if rows else f.get("strength") == "strong")), key=_rank_order)


def key_findings(findings: list[dict[str, Any]], decision: dict[str, Any], limit: int = SUMMARY_KEY_FINDINGS) -> list[tuple[dict[str, Any], bool]]:
    """UX §5.4, §6.2 (answer.ts `keyFindings`): the strongest finding against the headline is always listed, last, as
    counter-evidence (§7.3 rule 8). L-3: with contribution rows, only findings that entered the decision, and the counter-evidence
    is the largest contribution against the headline; otherwise the rank order with at least weak support either way.
    Returns (finding, is counter-evidence)."""
    side = SIDE.get(str(decision.get("code")))
    opposite = "against" if side == "for" else "for"
    if no_weight(decision):  # N-19f: every finding of a check is outside the estimate (`context_findings`); none is a reason
        return []
    if not_assessable(decision):
        # N-19k (Q2-14; D-64): no check of the content scored, so a finding that points away from AI generation is a
        # record's (a metadata block), which is never evidence of authenticity: it is not a key finding of such a result
        findings = [f for f in findings if finding_direction(f) != "against"]
    # N-19l (R6-01, 4): the findings of the record that decides stand first, with the ids they were stored with. A fresh
    # result ranks them first (N-19h); a result stored before ranked a marker's and a capture's finding behind the checks
    # that entered, and over three such checks the record's finding stood in no list
    def order(f: dict[str, Any]) -> tuple:
        return (not of_record(f, decision), *_rank_order(f))

    if has_contribution_rows(decision):
        ranked = sorted((f for f in findings if f.get("family") not in NOT_KEY and entered_decision(f, decision)), key=order)
        against = [f for f in ranked if decision_direction(f, decision) == opposite and not of_record(f, decision)] if side else []
        counter = max(against, key=lambda f: abs(float(f.get("contribution_logit") or 0.0)), default=None)
    else:
        # N-19n (D3-07, 3): the finding of a record that was not applied is no key finding (FT, KD: it stood last, as
        # counter-evidence); it stands in the list of provenance findings that did not decide, which follows them
        aside = {id(f) for f in provenance_aside(findings, decision)}
        ranked = sorted((f for f in findings if f.get("family") not in NOT_KEY and id(f) not in aside
                         and (f.get("strength") != "none" or of_record(f, decision))), key=order)
        counter = next((f for f in ranked if finding_direction(f) == opposite and not of_record(f, decision)), None) if side else None
    rest = [f for f in ranked if f is not counter][:limit - 1 if counter is not None else limit]
    return [(f, False) for f in rest] + ([(counter, True)] if counter is not None else [])


def context_findings(findings: list[dict[str, Any]], decision: dict[str, Any]) -> list[dict[str, Any]]:
    """L-3: findings with at least weak support that did not enter an artefact decision (weight 0): context, never reasons.
    N-19f: under a result that is not assessable because the model gives the checks that scored no weight, all of them."""
    if not has_contribution_rows(decision) and not no_weight(decision):
        return []
    # N-19n (D3-07, 3): a record of the file is no check outside the estimate (`provenance_aside`)
    return sorted((f for f in findings if f.get("family") not in NOT_KEY and f.get("family") not in PROVENANCE_FINDINGS
                   and f.get("strength") != "none" and not entered_decision(f, decision)), key=_rank_order)


START_KINDS = ("neutral_start", "context_prior")  # the starting row: D-63 since N-17; `context_prior` in results published earlier


def start_kind(decision: dict[str, Any]) -> str | None:
    """Which starting row an artefact decision holds (contributions.ts `startKind`): `neutral_start` (D-63: every scoring check
    at its neutral point), `context_prior` (a result published before N-17: every check at 50), or None without rows."""
    kinds = {r["kind"] for r in decision.get("contributions") or []}
    return next((k for k in START_KINDS if k in kinds), None)


def start_where(decision: dict[str, Any]) -> str:
    """The starting row out of 100, in the words of the page (`startWhere`)."""
    logit = math.fsum(float(r["logit"]) for r in decision.get("contributions") or [] if r["kind"] in START_KINDS)
    p = 1.0 / (1.0 + math.exp(-logit))
    return "above 99" if p >= 0.995 else "below 1" if p < 0.005 else f"{half_up(p * 100)}"


def starting_point_note(decision: dict[str, Any]) -> str | None:
    """D-63, for the one-page reports (the page's Plain view: `startingPointNote`): where the estimate starts and that the
    start is not evidence about the file. None for a result without a `neutral_start` row, or without an estimate to explain."""
    if not has_contribution_rows(decision) or start_kind(decision) != "neutral_start" or estimate_of(decision) is None:
        return None
    return (f"The estimate starts at {start_where(decision)} out of 100 with every check at its neutral point, the score at which "
            "a check alone favours neither answer. That start comes from fitting the model; it is not evidence about this file.")


def not_counted(finding: dict[str, Any], decision: dict[str, Any] | None) -> bool:
    """N-19o (D5-04; answer.ts `notCounted`): the finding of a check whose row the decision publishes and does not count
    (`counted` false: a check that declares H3, the container check of a video)."""
    ids = set(finding.get("evidence_ids") or [])
    return any(r.get("kind") == "detector" and r.get("counted") is False and effect.entered(r.get("logit")) and r.get("signal_id") in ids
               for r in (decision or {}).get("contributions") or [])


NOT_COUNTED = "not counted"  # contributions.ts `NOT_COUNTED`


def row_not_counted(signal_id: str | None, decision: dict[str, Any] | None) -> bool:
    """N-19p (design review D6-05, code review R10-05; contributions.ts `rowNotCounted`): the signal of a check whose row the
    decision publishes and does not count (`counted` false: a row of zero, the row of a check that declares H3). Its row
    reads "not counted", with no bar and no "−0.00", in the breakdown and in the ledgers (it read "little effect | little
    effect | −0.00" under "One more row is shown and not counted")."""
    return any(r.get("kind") == "detector" and r.get("counted") is False and r.get("signal_id") == signal_id
               for r in (decision or {}).get("contributions") or [])


def log_odds_size(c: float) -> str:
    """The size of a contribution as the line prints it: never "0.00" (N-19o, D5-04)."""
    return "less than 0.01" if abs(c) < 0.005 else f"{abs(c):.2f}"


def contribution_line(finding: dict[str, Any], decision: dict[str, Any]) -> str | None:
    c = finding.get("contribution_logit")
    if not has_contribution_rows(decision) or is_documentary(finding) or c is None or not_counted(finding, decision):
        return None
    moved = f"it {'raised' if float(c) > 0 else 'lowered'} the estimate by {log_odds_size(float(c))} log-odds"
    if start_kind(decision) != "context_prior":  # D-63: the word is the finding's own; this line gives the number behind it
        return f"In this decision {moved} from its neutral point." if effect.entered(float(c)) else None
    if abs(float(c)) < effect.LEGACY_LITTLE:
        return "In this decision it had little effect on the estimate."
    return f"In this decision {moved} from the model’s starting point."


def strength_words(finding: dict[str, Any], decision: dict[str, Any]) -> str:
    """The reason-row label: documentary, the contribution's effect when the finding entered an artefact decision, else the
    finding's own strength and direction."""
    if is_documentary(finding):
        return "documentary"
    if (word := record_word(finding, decision)) is not None:  # N-19n (D3-07): a record that anyone can write has no strength
        return word
    if not_counted(finding, decision):  # N-19o (D5-04): it read "little effect" beside a row that is not counted
        return SET_ASIDE_WORDS
    c = finding.get("contribution_logit")
    if measured_from_neutral(decision) and effect.entered(c):  # D-63: the one vocabulary, from the contribution
        return effect.words(float(c))
    if has_contribution_rows(decision) and c is not None:
        return "little effect" if abs(float(c)) < effect.LEGACY_LITTLE else ("raised the estimate" if float(c) > 0 else "lowered the estimate")
    strength, direction = str(finding.get("strength", "none")), finding_direction(finding)
    words = STRENGTH_WORDS.get(strength, "little effect")
    return f"{words} {direction}" if strength != "none" and direction in ("for", "against") else words


def reason_of(finding: dict[str, Any], counter: bool) -> str:
    sentences = finding.get("sentences") or []
    first = sentences[0]["text"] if sentences else str(finding.get("statement", ""))
    return f"{first} {finding['dissent_note']}" if counter and finding.get("dissent_note") else first


LEDGER_CLAUSE = "; what it measured is listed in the evidence ledger."
FULL_REPORT_CLAUSE = "; the full report lists what it measured."
_NO_SCORE = re.compile(r"^(.+?) produced a result without a score$")


def page_reason(text: str) -> str:
    """N-19d (browser QA "Copy"): the reason of a finding on a one-page profile. Such a page has no evidence ledger, so the
    generated sentence of a check's score points at the full report; a check without a score is said as the full reports say
    it (`prose.system_prose`). Any other sentence is printed as it is."""
    if not text.endswith(LEDGER_CLAUSE):
        return text
    return _NO_SCORE.sub(r"\1: no score was produced", text[:-len(LEDGER_CLAUSE)]) + FULL_REPORT_CLAUSE


EXAMINER_STATUS = {"accepted": "Accepted", "disputed": "Disputed", "needs_check": "Needs check", "not_relevant": "Not relevant"}


def finding_statuses(snapshot: ReportSnapshot) -> dict[str, str]:
    """The examiner's status of each finding, in words (RW-01: a status is an annotation, and the latest one stands). D-69
    (N-19; browser QA P1-6): it is printed wherever the finding is listed, in all four profiles: a finding marked "Disputed"
    was listed as "F2 Strong support for AI generation or manipulation" with no mark in the summary and newsroom PDFs."""
    out: dict[str, str] = {}
    for a in sorted(snapshot.annotations, key=lambda a: (a.edited_at or a.created_at, a.id)):
        if a.examiner_status and a.target_type == "finding" and a.target_id:
            out[a.target_id] = EXAMINER_STATUS.get(a.examiner_status, a.examiner_status)
    return out


def record_findings(findings: list[dict[str, Any]], decision: dict[str, Any]) -> list[tuple[dict[str, Any], bool]]:
    """N-19h (E-21): the findings of the record that decides, in rank order, as `key_findings` returns them. In a result
    outside what was validated they are the reasons that are given, beside the note that the findings of the checks are
    withheld (N-10b: those were not validated for the input; the record is no check)."""
    return [(f, False) for f in sorted((f for f in findings if f.get("family") not in NOT_KEY and of_record(f, decision)), key=_rank_order)]


def withheld_title(finding: dict[str, Any], signals: dict[str, dict[str, Any]]) -> str | None:
    """N-19l (R6-04, 3): the title under which the newsroom report prints a finding whose details it withholds, or None
    for a finding that is printed whole. What that report withholds is what a file's metadata and credentials say.

    A finding of a provenance record (one of its signals holds an override, applied by the decision or not) is titled by
    the kind of that record: "Content Credentials" for credentials of every kind, "Self-declared metadata" for a marker in
    the metadata. The title was the family's, and the families do not follow the kinds: the finding of Content Credentials
    that name a generator is of the family of self-declared metadata, and that of a public watermark message of the family
    of provenance ("Content Credentials" under "The image carries a public AI-generator watermark"). The finding of a
    public watermark message is not withheld: the message is a public constant of generator code and holds nothing of a
    person. A withheld finding that is no record's keeps the title of its family ("Metadata consistency")."""
    family = finding.get("family")
    if family not in WITHHELD:
        return None
    mine = [signals[i] for i in finding.get("evidence_ids") or [] if i in signals and signals[i].get("provenance_override")]
    if not mine:
        return WITHHELD[family]
    detectors = {str(s.get("detector_id")) for s in mine}
    if detectors <= decisive.WATERMARK_DETECTORS:
        return None
    return "Content Credentials" if detectors & set(decisive.CREDENTIAL_DETECTORS) else "Self-declared metadata"


def finding_rows(snapshot: ReportSnapshot, decision: dict[str, Any], images: dict[str, str], redacted: bool,
                 thumbnail_details: dict[str, dict[str, str]] | None = None, *, record_only: bool = False) -> list[dict[str, Any]]:
    """`record_only`: only the findings of the record that decides (a result outside what was validated)."""
    rows = []
    withheld_at = ""
    statuses = finding_statuses(snapshot)
    listed = list(snapshot.result.get("findings") or [])
    signals = {str(s.get("id")): s for s in snapshot.result.get("signals") or []}
    for f, counter in (record_findings(listed, decision) if record_only else key_findings(listed, decision)):
        strength = str(f.get("strength", "none"))
        documentary = is_documentary(f)
        words = strength_words(f, decision)
        title = withheld_title(f, signals) if redacted else None
        withheld = title is not None
        if withheld:  # N-19c: the sentence once, at the first finding it applies to (the newsroom page printed it three times)
            withheld_at = withheld_at or str(f.get("id", ""))
        detail = {} if withheld else (thumbnail_details or {}).get(str(f.get("id")), {})
        rows.append({"id": f.get("id", ""), "counter": counter, "documentary": documentary, "bars": BARS.get(strength, 0), "words": words,
                     "mark": finding_mark(f, decision),  # N-19n (D3-07): a record's mark stands in the meter's place
                     "status": statuses.get(str(f.get("id"))),
                     "title": title if withheld else f.get("title", ""),
                     "reason": (WITHHELD_REASON if withheld_at == str(f.get("id", "")) else f"Details withheld, as for {withheld_at}.") if withheld
                     else page_reason(reason_of(f, counter)),
                     "thumbnail": None if withheld else images.get(str(f.get("id"))),
                     "thumbnail_caption": detail.get("caption"), "thumbnail_alt": detail.get("alt", "Evidence map")})
    return rows


class ReviewStatus(NamedTuple):
    """What a frozen report says about its review (D-25, D-56)."""
    issued: bool  # the snapshot is issuable: "issued", an id without DRAFT, "Reviewed by …"
    signed: bool  # the examiner has signed, so the examiner's opinion is printed; a signed report that is not issued is a draft
    line: str  # the review status line of every page
    # N-19j (browser QA Q2-02; design D2-10): why a signed report is not issued, for the running header of every page
    # ("second review outstanding", "second review: does not concur", "a review record does not verify"); None otherwise
    held: str | None = None
    refusal: str | None = None  # the sentence of a second review that does not concur, beside the examiner's opinion


def review_status(snapshot: ReportSnapshot) -> ReviewStatus:
    return review_status_of(snapshot.review)


def review_status_of(review: SnapshotReview) -> ReviewStatus:
    """The review status line follows the frozen `issuable` flag (D-56), not the state alone: a report signed while a required
    second review is outstanding, or whose peer reviewer does not concur, is not issued and says why. The peer decision is
    printed with the peer's name. A review record whose signature did not verify when the snapshot was frozen is printed as such
    and is never issued (N-3)."""
    examiner = next((s for s in review.signoffs if s.kind == "examiner"), None)
    if review.state not in SIGNED_STATES or examiner is None:
        return ReviewStatus(False, False, UNREVIEWED)
    peer = next((s for s in review.signoffs if s.kind == "peer_review"), None)
    conclusion_invalid = review.conclusion is not None and review.conclusion.signature_valid is False
    invalid = conclusion_invalid or any(s.signature_valid is False for s in review.signoffs)
    issued = review.issuable and (peer is None or peer.decision in PEER_APPROVES) and not invalid

    def unverified(record: Any) -> str:
        return " (the record of this sign-off does not verify)" if record.signature_valid is False else ""

    line = f"{'Reviewed' if issued else 'Signed'} by {examiner.principal_name}, {formatting.utc_date(examiner.signed_at)}{unverified(examiner)}"
    held = refusal = None
    if peer is not None:
        # N-19j (Q2-03): the reviewer's decision with its object
        line += (f" · peer review by {peer.principal_name}, {formatting.utc_date(peer.signed_at)}: "
                 f"{decision_words('peer_review', peer.decision)}{unverified(peer)}")
        if peer.decision not in PEER_APPROVES:
            held = SECOND_REVIEW_REFUSED
            refusal = REFUSAL_SENTENCE.format(name=peer.principal_name)
    elif review.requires_second_review or (review.requires_second_review is None and not issued and not invalid):
        line += "; second review outstanding"  # snapshots frozen before N-3 hold no policy: a signed draft without a peer was one
        held = SECOND_REVIEW_OUTSTANDING
    if conclusion_invalid:
        line += " · the record of the conclusion does not verify"
    if not issued and held is None:
        held = RECORD_NOT_VERIFIED if invalid else NOT_ISSUED
    return ReviewStatus(issued, True, line, None if issued else held, refusal)


# N-19j (browser QA Q2-03): a decision is printed with its object, in the wording of the dialog and the record of the page
# (review.ts `DECISION_WORD`). "Examiner sign-off · does not concur" beside "Peer review · concurs" read as a dispute where
# the examiner departed from the machine assessment and the reviewer agreed with the examiner
SIGNOFF_OBJECT = {"examiner": "the machine assessment", "peer_review": "the examiner’s opinion"}
NO_COMPARISON = "opinion recorded, no comparison with the machine assessment"


def decision_words(kind: str, decision: str) -> str:
    """"concurs with the machine assessment", "concurs with the examiner’s opinion, with changes", "does not concur with …"."""
    what = SIGNOFF_OBJECT.get(kind)
    if what is None or decision not in SIGNOFF_DECISION:
        return SIGNOFF_DECISION.get(decision, decision)
    return (f"does not concur with {what}" if decision == "non_concur"
            else f"concurs with {what}" + (", with changes" if decision == "concur_with_changes" else ""))


def review_reference(decision: dict[str, Any] | None, conclusion: Any = None) -> str | None:
    """N-19n (design review D4-01, P1: the rest of D3-01; review.ts `reviewReference`): what the examiner's layer names
    where it compares, in a result that a trusted capture decides: "the step of the capture record". The page and the
    reports head such a result by the record's sentence, and the layer said "the machine decision" and "the machine
    assessment" of a decision code that no surface prints there ("Examiner's opinion: No indications of AI generation or
    manipulation found, agrees with the machine assessment" directly under the capture's headline, over "statistical
    checks alone: 98 out of 100"). None: the words of before, which stand under a declaration (the words of the scale
    point the way of the record), under a marker and without a record.

    `conclusion`: a recorded one. Its steps are the record's where it says so (`steps_from`), and where the code's step is
    the record's own (a fusion model: a conclusion recorded before reads in the new words too). One recorded before on the
    fallback, whose steps were counted from the code's step one further in, keeps the words of before with them."""
    from app.cases import review as review_rules

    kind = (record_of(decision) or {}).get("kind")
    if kind == "decisive_ai":
        # N-19o (design review D5-01): on the fallback the code of a declaration's result is the fourth step and the steps
        # are counted from the fifth; the layer called both "the machine's". With a fusion model the code is the fifth step
        code = (decision or {}).get("code")
        if code == review_rules.RECORD_STEP[review_rules.STEPS_FROM_DECLARATION]:
            return None
        if conclusion is not None and not review_rules.counts_from_declaration(conclusion, code):
            return None
        return review_rules.STEP_NAMES[review_rules.STEPS_FROM_DECLARATION]
    if kind != "trusted_capture":
        return None
    if conclusion is not None and not review_rules.counts_from_capture(conclusion):
        return None
    return review_rules.STEP_NAMES[review_rules.STEPS_FROM_CAPTURE]


def _steps(n: int) -> str:
    return f"{n} step{'s' if n != 1 else ''}"


def signoff_decision(signoff: Any, conclusion: Any, reference: str | None = None) -> str:
    """The decision of a sign-off in words, with its object (N-19j). Where the machine's result is not assessable, or there
    is no machine decision, the examiner's sign-off has no object: it records the opinion without a comparison, whatever
    decision the record stores. N-19 (browser QA P1-7): a record made before the rule can pair an examiner's "concur with
    changes" with a conclusion that departs from the machine assessment; it prints the departure, not "concurs" (the page
    printed "Departs from the machine assessment by 1 step" and "DECISION: concurs with changes"). N-19j: and the mirror,
    a record that does not concur under a conclusion equal to the machine's. `reference` (N-19n, D4-01): the step that the
    layer names under a capture (`review_reference`); the signed decision itself keeps its words: it is what was signed,
    and under a capture the machine assessment is the record's result."""
    from app.cases import review as review_rules

    if signoff.kind == "examiner" and conclusion is not None:
        departs = review_rules.departs_of(conclusion)  # N-19n (D4-01): from the stored steps
        if departs is None:
            return NO_COMPARISON
        if departs and signoff.decision in review_rules.CONCURRING:
            steps = abs(conclusion.deviation_steps or 0)
            if reference is not None:
                return f"{_steps(steps)} from {reference}" if steps else f"not on {reference}"
            return "departs from the machine assessment" + (f" by {_steps(steps)}" if steps else "")
        if not departs and signoff.decision == "non_concur":
            return ("signed as not concurring; the conclusion " + (f"is on {reference}" if reference is not None
                                                                   else "agrees with the machine assessment"))
    return decision_words(signoff.kind, signoff.decision)


# N-19 (browser QA P1-8; code re-verification CR-18): a custodian's stated hash that differs from the file received is said
# in every profile. The standard and forensic profiles print it in the custody section (N-3) and among the limitations;
# the two one-page profiles carry this line. The stated hash itself is a source field and is not printed in it.
CUSTODIAN_MISMATCH_LINE = ("The hash the custodian stated at intake differs from the file received. The difference is recorded, "
                           "not resolved: this report describes the file that was received.")
CUSTODIAN_MISMATCH_LIMITATION = ("The hash the custodian stated at intake differs from the file received. This report describes the "
                                 "file that was received; whether it is the file the custodian meant is not established.")


def custodian_mismatch(snapshot: ReportSnapshot) -> bool:
    return snapshot.media.custodian_hash_match == "mismatch"


# D-69 (N-19; design review DS-03, DS-05): what an unsigned report may claim
REDACTED_CENTRE = "Redacted copy"  # never "For publication": redaction clears nothing
DRAFT_MARK = "Draft · not reviewed"
DRAFT_MARK_NEWSROOM = "Draft · not reviewed · not for publication"


# N-19j (browser QA Q2-02; design D2-10): a signed report that is not issued said nothing in its running header ("Case … ·
# Ex. B · Confidential"), where an unsigned draft has "… · Draft · not reviewed": a revision that the second reviewer
# refused read as agreed. The header of every report that is not issuable says why, and keeps the word "Draft"
SECOND_REVIEW_OUTSTANDING = "second review outstanding"
SECOND_REVIEW_REFUSED = "second review: does not concur"
RECORD_NOT_VERIFIED = "a review record does not verify"
NOT_ISSUED = "not issued"
REFUSAL_SENTENCE = "The second reviewer, {name}, does not concur with this opinion. This revision is not issued."


def draft_mark(profile: str, status: ReviewStatus) -> str | None:
    """What the header of a report carries while it is not issued: "Draft · not reviewed" until an examiner sign-off exists,
    then "Draft · " and why (`ReviewStatus.held`). None for an issued report."""
    if status.issued:
        return None
    if status.signed:
        mark = f"Draft · {status.held or NOT_ISSUED}"
        return f"{mark} · not for publication" if profile == "newsroom" else mark
    return DRAFT_MARK_NEWSROOM if profile == "newsroom" else DRAFT_MARK


def header_centre(centre: str, mark: str | None) -> str:
    return centre if mark is None else f"{centre} · {mark}"


def case_centre(case: Any) -> str:
    """The centre of the page header of a report that names its case: the case, the exhibit and the classification. N-19c
    (design review DS-18): one text for every profile; the one-page profiles left the classification out."""
    if case is None:
        return "No case"
    reference = formatting.case_reference(case.reference)
    return (reference if reference == "Unfiled" else f"Case {reference}") + (f" · Ex. {case.exhibit_label}" if case.exhibit_label else "") \
        + (f" · {case.classification.capitalize()}" if case.classification else "")


# N-19q (N-20b's read of the reports, finding 5): an opinion on the machine's step whose sign-off concurs with changes says
# so where it says that it agrees ("Examiner’s opinion: Inconclusive, agrees with the machine assessment" and "Concurs with
# the machine assessment." stood beside the sign-off "concurs with the machine assessment, with changes")
WITH_CHANGES = ", with changes"


def with_changes(snapshot: ReportSnapshot) -> bool:
    """The examiner's sign-off of the record concurs with changes."""
    examiner = next((s for s in snapshot.review.signoffs if s.kind == "examiner"), None)
    return examiner is not None and examiner.decision == "concur_with_changes"


def opinion_position(position: str) -> str:
    """A scale position in the words of the opinion line: the headline without its explanation ("Inconclusive", not
    "Inconclusive: the evidence does not support a firm assessment"); review.ts `opinionPosition`."""
    return "No opinion on the scale" if position == "no_opinion" else HEADLINE[CODE_BAND.get(position, "uncertain")].split(":")[0]


def examiner_opinion(snapshot: ReportSnapshot, signed: bool) -> dict[str, Any] | None:
    """The examiner's opinion of a signed record, as it stands directly under the machine headline at its size, in all four
    profiles (N-19, D-69; browser QA P1-6; design review DS-06): the position, whether it departs from or agrees with the
    machine assessment and by how many steps, and the justification in full (the justification of the departure when one is
    recorded, else the rationale). It was one small line at the foot of the page, the first sentence of the rationale, under
    a headline it did not share. None until the examiner has signed: a draft conclusion is not an opinion of record."""
    from app.cases import review as review_rules

    c = snapshot.review.conclusion
    if not signed or c is None:
        return None
    steps = abs(c.deviation_steps or 0)
    departs = review_rules.departs_of(c)  # None: nothing to compare with (N-19j, D2-12); N-19n: from the stored steps
    # N-19n (D4-01): under a capture the line names the step of the capture record. "No opinion" has its line with the
    # negation: no opinion has no position to be away from
    reference = review_reference(decision_of(snapshot.result), c)
    if departs is None:
        relation = "no comparison with the machine assessment is recorded"
    elif reference is not None:
        relation = f"on {reference}" if not departs else f"{_steps(steps)} from {reference}" if steps else f"not on {reference}"
    else:
        relation = ("departs from the machine assessment" + (f" by {_steps(steps)}" if steps else "") if departs
                    else "agrees with the machine assessment")
    if departs is False and with_changes(snapshot):  # N-19q (finding 5): as the examiner's sign-off says
        relation += WITH_CHANGES
    position = opinion_position(c.scale_position)
    # N-19j (Q2-02): a refused revision says so wherever the examiner's opinion stands at headline size (it printed
    # "Examiner's opinion: …, agrees with the machine assessment" on the cover, the refusal only in the Review row)
    # N-19p (design review D6-02): "Justification of the departure" only where the conclusion departs; a justification typed
    # on the step (an API client can send one) is printed under "Justification", after the rationale, which stays
    justified = bool(c.deviation_justification) and departs is True
    also = ({"label": JUSTIFICATION_LABEL, "text": c.deviation_justification}
            if c.deviation_justification and not justified else None)
    return {"line": f"Examiner’s opinion: {position}, {relation}", "position": position, "relation": relation, "departs": departs,
            "refusal": review_status_of(snapshot.review).refusal,
            "label": DEPARTURE_LABEL if justified else "Rationale",
            "justification": c.deviation_justification if justified else c.rationale, "also": also}


DEPARTURE_LABEL, JUSTIFICATION_LABEL = "Justification of the departure", "Justification"  # review.ts, word for word


def not_validated_line(items: list[dict[str, str]]) -> str:
    """The one line of the summary and newsroom reports (the plain view's sentence): the labels of the calibrated model's
    disclosures, or the one sentence of a fallback result, for which no figure was measured (D-68)."""
    if not items:
        return ""
    if [d["id"] for d in items] == [disclosures.FALLBACK.id]:
        return items[0]["text"]
    return f"Not validated for this result: {'; '.join(d['label'] for d in items)}. The full report gives the measured figures."


def media_line(snapshot: ReportSnapshot) -> str:
    profile = snapshot.result.get("condition_profile") or {}
    parts = [media(snapshot.analysis.kind)[0].capitalize()]
    if profile.get("width") and profile.get("height"):
        parts.append(f"{profile['width']}×{profile['height']}")
    if profile.get("duration_s"):
        parts.append(f"{float(profile['duration_s']):.1f} s")
    parts += [snapshot.media.content_type, formatting.size(snapshot.media.size_bytes)]
    if snapshot.media.received_at is not None:
        # N-19: the system's time is not "received". N-19k (Q2-14): nor the day the picture was taken ("taken in 4 Oct 2026")
        parts.append(f"taken in by Chipmunk Verify on {formatting.utc_date(snapshot.media.received_at)}")
    parts.append(f"analysed {formatting.utc_date(snapshot.analysis.finished_at)}")
    return " · ".join(parts)


def page_view(snapshot: ReportSnapshot, images: dict[str, str] | None = None, *,
              thumbnail_details: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    result, report = snapshot.result, snapshot.report
    decision = decision_of(result)  # N-19h: with the record that decides, also for a stored result
    findings: list[dict[str, Any]] = list(result.get("findings") or [])
    profile: dict[str, Any] | None = result.get("condition_profile")
    kind = snapshot.analysis.kind
    coverage = snapshot.analysis.coverage.model_dump() if snapshot.analysis.coverage is not None else None
    redacted = report.profile == "newsroom" or report.redacted
    basis = basis_of(decision, findings)
    step = step_of(decision, basis)
    status = review_status(snapshot)
    centre = REDACTED_CENTRE if redacted else case_centre(snapshot.case)
    title = PROFILE_TITLES[report.profile]
    means, does_not_mean = guidance(kind, decision, basis, coverage, findings, list(result.get("signals") or []), profile,
                                    caveats=result.get("caveats"))
    number = result_number(decision, findings, profile)
    # N-19h: the key of N-10b is the input, not the number: a result that a record decides outside what was validated has
    # the record's number (a marker, a capture) or none of its own (a declaration)
    outside = outside_validated(decision, profile)
    unvalidated = disclosures.for_result({**result, "kind": kind}) if decision.get("code") != "not_assessable" else []
    # N-19n (Q4-01): the step that writes the findings failed. The page of a report says so where the findings would stand
    failed = effect.findings_failed(result.get("caveats"))
    return {
        "not_validated": not_validated_line(unvalidated),
        "title": f"Chipmunk Verify {title.lower()} {report.display_id}",
        "subject": f"{title} · {media(kind)[0]}" if redacted else f"{centre} · {snapshot.media.filename}",
        "keywords": f"Chipmunk Verify, {report.profile}, {report.display_id}" + ("" if redacted else f", analysis {snapshot.analysis.id}"),
        "header_left": f"Chipmunk Verify · {title}",
        "header_centre": header_centre(centre, draft_mark(report.profile, status)),
        # D-69: the newsroom profile repeats its mark once under the title
        "draft_note": draft_mark(report.profile, status) if report.profile == "newsroom" else None,
        "custodian_note": CUSTODIAN_MISMATCH_LINE if custodian_mismatch(snapshot) else None,
        # N-18b: a step of the run that failed, in the result's own sentence (the full reports list it with the limitations)
        "step_notes": effect.failed_steps(result.get("caveats")),
        "issued": status.issued,
        "status_line": status.line,
        "redacted": redacted,
        "heading": f"Examined {media(kind)[0]}" if redacted else snapshot.media.filename,
        "media_line": media_line(snapshot),
        "analysis_line": f"Analysis {snapshot.analysis.id} · revision {snapshot.analysis.revision} · report {report.display_id} ({report.id})",
        "step": step,
        "headline": headline(decision, basis),
        "estimate_line": estimate_line(decision, basis, coverage, findings, profile, failed=failed),
        "scale": scale(decision, basis, step, number),
        "scale_note": scale_note(decision, number),
        "means": means,
        "reasons": insufficient_reasons(decision, profile, coverage) if basis == "insufficient" else [],
        "does_not_mean": does_not_mean,
        "testing": in_testing(kind, decision, profile, list(result.get("calibration") or [])),
        # N-10b (code re-verification R2-08): a capped result prints no key findings here. Their strengths were measured on
        # inputs unlike this one ("Strong support against AI generation" under a result that shows no estimate); the reason
        # for the result is its cap, said above. The standard and forensic reports keep the findings
        # N-19h (E-21): where a record decides such a result, its finding is the reason that is given, beside the note
        "findings": finding_rows(snapshot, decision, images or {}, redacted, thumbnail_details, record_only=outside),
        "findings_beside": CAPPED_CHECKS if outside else None,
        "findings_note": (effect.findings_not_written("full") if failed else CAPPED_CHECKS if outside
                          else NO_WEIGHT_CHECKS if no_weight(decision) else "No finding gave at least weak support either way."),
        "thumbnail_note": THUMBNAIL_NOTE,
        # N-19d: the examined picture, reduced (`engine.source_plate`); absent for other media and on a copy that withholds
        "exhibit_image": None if redacted else (images or {}).get("source"),
        # D-63: where the estimate starts, said beside the findings that moved it (never under a result that shows no number)
        "start_note": None if outside or failed else starting_point_note(decision),
        "seal": figures.seal_svg(),
        "marks": figures.marks(),
        "next_step": next_step(decision, basis, coverage, failed=failed),
        "examiner": examiner_opinion(snapshot, status.signed),
        "closing": closing(decision),
        "withheld_note": ("Withheld in this version: the filename, the case reference, source fields and metadata details."
                          if redacted else None),
    }
