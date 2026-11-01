"""Client wording of model-card strings (BUILD_SPEC L-7; design review P1-3).

The manifest's card strings are the build's own record: they cite ticket and decision ids, repository paths and raw enums. A
client page prints the wording this module derives and keeps the record verbatim under "Engineering record".

- `client_text` removes the references a sentence carries (bracketed citations, trailing paths) and returns the sentence only
  when nothing of the build's vocabulary is left (`lint.internal_terms` and the admission shorthand below); otherwise it
  returns None and the string stays in the engineering record. Nothing is reworded and no number is changed.
- Enums (threats, lanes, hypotheses, licence classes, device states) map to words.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from app.reports import lint

THREAT_WORDS = {
    "full_synthesis": "fully generated images", "face_swap": "face swaps", "partial_edit": "local edits and inpainting",
    "gan_face": "generated portraits", "t2v": "fully generated video", "voice": "synthetic or cloned voices",
    "temporal": "frame-to-frame inconsistency", "lip_sync": "lip-sync manipulation", "ai_text": "AI-written text",
}
HYPOTHESIS_WORDS = {
    "H1_whole_content_synthesis": "the whole file is AI-generated",
    "H2_localised_manipulation": "part of the file was manipulated",
    "H3_provenance_integrity": "the file's provenance record",
}
LICENCE_CLASS_WORDS = {"commercial": "commercial use permitted", "non_commercial": "non-commercial only", "unknown": "licence not established"}
DEVICE_STATE_WORDS = {"ok": "supported", "fallback_cpu": "runs on the CPU instead", "unsupported": "not supported"}

_PAREN = re.compile(r"(\s*)\(([^()]*)\)")
_SEGMENT = re.compile(r"(;\s+|,\s+)")
# a reference outside brackets: ", docs/…" or "; record docs/…" up to the next clause
_TRAILING_PATH = re.compile(r"[,;]\s*(?:(?:see|record|records|threshold record)\s+)?(?:docs|backend|frontend|scripts|tests)/[\w./<>-]+")
_SPACE_BEFORE = re.compile(r"\s+([,;:.])")
# words that only introduce a reference ("upstream align_face", "v0.3 face_min_size", "see docs/…")
_FILLER = re.compile(r"\.\w+\b|\b(?:upstream|see|per|record|records|in|and|the|of|v\d[\d.]*)\b|[^\w]+")
_DATE = re.compile(r"\d{4}-\d{2}(?:-\d{2})?")
# the admission shorthand of the build, which the lint leaves to the reports that define it: criteria A1–A6, leave-group-out,
# the threshold symbol, out-of-fold, "fuser", and the codes of the benchmark's strata (R1 … R6 and CFE; N-19k, the PDF words
# of both reviews: the false alarms of a check in an image report ended "CFE 10 of 30 in both")
_SHORTHAND = re.compile(r"\bA[1-6][ab]?(?=\s+(?:AUC|pass|fail|fails|not established|and A))|\bLGO\b|τ_|\bout-of-fold\b|\bfusers?\b"
                        r"|\bfuser-\w|\bR[1-6]\b|\bCFE\b")


class _Unclean(Exception):
    """A reference is woven into words the sentence needs: the string stays an engineering note."""


def words(value: str) -> str:
    """A raw enum or a hyphenated key as words: `face_swap` → "face swap", `learned-whole-image` → "learned whole image"."""
    return value.replace("_", " ").replace("-", " ").strip()


def _residue(text: str, quoted: list[str]) -> str:
    """What a clause says beyond its references and the words that introduce them; empty for a pure reference."""
    for value in quoted:
        text = text.replace(value, " content ")  # a published identifier is content, not a reference
    for term in sorted(set(lint.internal_terms(text)), key=len, reverse=True):
        text = text.replace(term, " ")
    return _FILLER.sub("", text)


def _segment(text: str, quoted: list[str]) -> str | None:
    """A clause as the client reads it: itself when clean, None when it is only a reference, the content of "reference: content"."""
    if not lint.internal_terms(text, quoted):
        return text
    head, colon, content = text.partition(": ")
    if colon and not lint.internal_terms(content, quoted) and not _residue(head, quoted):
        return content
    if _residue(text, quoted):
        raise _Unclean(text)
    return None


def _strip_references(text: str, quoted: list[str]) -> str:
    def clean(match: re.Match[str]) -> str:
        parts = _SEGMENT.split(match.group(2))
        kept: list[str] = []  # clause, separator, clause, …
        dropped = False
        for i in range(0, len(parts), 2):
            segment = _segment(parts[i], quoted)
            if segment is None:
                dropped = True
            elif dropped and " " not in segment.strip() and not _DATE.fullmatch(segment.strip()):
                continue  # "(I-1, D13)", "(docs/…, lessons)": the code or section that follows a reference goes with it
            else:
                kept += ([parts[i - 1]] if kept else []) + [segment]
        body = "".join(kept).strip()
        return f"{match.group(1)}({body})" if body else ""

    previous = None
    while previous != text:  # innermost brackets first, then the brackets that held them
        previous = text
        text = _PAREN.sub(clean, text)
    text = _TRAILING_PATH.sub("", text)
    return _SPACE_BEFORE.sub(r"\1", " ".join(text.split())).strip()


def client_text(text: str | None, quoted: Iterable[str] = ()) -> str | None:
    """`text` without its references, or None when it still speaks the build's vocabulary. `quoted`: identifiers the page prints
    as identifiers (detector and model ids, upstream repository names), exempt as in `lint.internal_terms`."""
    if not text or not text.strip():
        return None
    quoted = sorted({q for q in quoted if q and q.strip()}, key=len, reverse=True)
    try:
        cleaned = _strip_references(text, quoted)
    except _Unclean:
        return None
    if not cleaned or lint.internal_terms(cleaned, quoted) or _SHORTHAND.search(cleaned):
        return None
    return cleaned


def split(texts: Iterable[str], quoted: Iterable[str] = ()) -> tuple[list[str], list[str]]:
    """(client wording, engineering notes): every string lands in exactly one list, the notes verbatim."""
    quoted = list(quoted)
    client: list[str] = []
    notes: list[str] = []
    for text in texts:
        cleaned = client_text(text, quoted)
        if cleaned is None:
            notes.append(text)
        elif cleaned not in client:
            client.append(cleaned)
    return client, notes


_CLAUSE = re.compile(r"(?<=[.;])\s+")


def lead(text: str | None, quoted: Iterable[str] = ()) -> tuple[str | None, bool]:
    """For a one-value field: (client wording, complete). When the whole value does not pass, the leading sentences and
    clauses that do; `complete` is False then, and the page says the rest is in the engineering record."""
    quoted = list(quoted)
    whole = client_text(text, quoted)
    if whole is not None or not text:
        return whole, whole is not None
    kept: list[str] = []
    for clause in _CLAUSE.split(text.strip()):
        cleaned = client_text(clause, quoted)
        if cleaned is None:
            break
        kept.append(cleaned)
    return (" ".join(kept).rstrip(";") or None), False


_REASONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^listed in deny_detectors$"), "on the deny list of this instance"),
    (re.compile(r"^not default-enabled and not in allow_detectors$"), "off by default and not on the allow list of this instance"),
    (re.compile(r"^no v2 adapter yet \(runs in the v1 pipeline\)$"), "runs in the earlier pipeline only"),
)
_DENIED = re.compile(r"^model (?P<model>\S+) has licence class (?P<cls>\w+), denied by the (?P<policy>\S+) licence policy$")
_INCOMPLETE = re.compile(r"^model card incomplete \(missing: (?P<fields>[^)]*)\)$")
_POLICY_PARTS = {"commercial": "commercial licences only", "research": "research use, every licence class", "custom": "a custom licence policy",
                 "allowlist": "an allow list", "denylist": "a deny list"}


OWN_SOURCE = "this repository:"
PART_OF_VERITAS = "Part of Chipmunk Verify (no model download)"
_REPOSITORY_PATH = re.compile(r"(?:docs|backend|frontend|scripts|tests)/[\w./<>-]+")


def source_words(source: str | None) -> str | None:
    """N-19d (browser QA "Model cards": the Source row printed repository paths on about 20 cards, and "(the D-37 heuristic)"
    on one): where a check comes from, in client words. A check that is part of Veritas has no download: its record names the
    files of this repository that hold it, with remarks in the build's shorthand. The row then reads "Part of Veritas (no
    model download)", followed by what the check builds on outside the repository (ExifTool, jpeglib, a vendored codec). A
    download address is printed as it is recorded."""
    if not source:
        return None
    if not source.startswith(OWN_SOURCE):
        return source
    rest = source[len(OWN_SOURCE):]
    while _PAREN.search(rest):  # remarks in brackets name functions, methods and decisions
        rest = _PAREN.sub("", rest)
    outside = [part.strip() for part in rest.split(";") if part.strip() and not _REPOSITORY_PATH.search(part)]
    return "; ".join([PART_OF_VERITAS, *outside])


def policy_words(label: str) -> str:
    """`commercial+allowlist` → "commercial licences only, with an allow list"."""
    head, *rest = label.split("+")
    base = _POLICY_PARTS.get(head, words(head))
    return base + (", with " + " and ".join(_POLICY_PARTS.get(r, words(r)) for r in rest) if rest else "")


def reason_words(reason: str) -> str:
    """A registry enablement reason in client words (the registry's own strings name settings keys and enums)."""
    for pattern, replacement in _REASONS:
        if pattern.match(reason):
            return replacement
    if m := _DENIED.match(reason):
        return (f"model {m['model']} is licensed {LICENCE_CLASS_WORDS.get(m['cls'], words(m['cls']))}, which the licence policy in "
                f"force ({policy_words(m['policy'])}) does not permit")
    if m := _INCOMPLETE.match(reason):
        return f"model card incomplete (missing: {', '.join(words(f) for f in m['fields'].split(', '))})"
    return reason


def record_title(title: str, name: str, quoted: Iterable[str] = ()) -> str:
    """An admission record's title for the page; a title that cites the build falls back to the detector's name."""
    return client_text(title, quoted) or f"Admission record: {name}"


def client_modes(modes: Iterable[str], quoted: Iterable[str] = ()) -> list[str]:
    """The false-positive modes of a card as a result publishes them: client wording only (L-7). A mode that cites the build's
    records stays on the model card's engineering record."""
    return split(modes, quoted)[0]


# N-19n (design review D3-08): a card holds the measured false-alarm row of a check in the benchmark's shorthand ("stand-alone
# at its τ_0.05, vb-1.1 fit ∪ cal R1 ∪ R2 reals: 4.8 % (50/1,035; Wilson 3.7–6.3 %), concentrated on …"), which the copy
# for a result drops; the detector block of a full report then pointed to "the engineering record of the model card", which
# only an administrator can open. The row in the reader's words, every figure taken from the card's own mode; `models.yaml`
# is not changed. A mode that still cannot be printed says so, and points nowhere
MODE_NOT_PRINTED = "recorded on the model card in the benchmark's own terms; not printed here."
_MEASURED = re.compile(
    r"^stand-alone at its τ_0\.05(?P<of>, vb-1\.1 fit ∪ cal R1 ∪ R2 reals| on the R1 ∪ R2 fit ∪ cal reals it scored): "
    r"(?P<rate>\d+(?:\.\d+)?) % \((?P<k>[\d,]+)/(?P<n>[\d,]+); Wilson (?P<low>\d+(?:\.\d+)?)–(?P<high>\d+(?:\.\d+)?) %\)"
    r"(?P<tail>.*?)(?: \(H-3, [^()]*\))?$")
_MEASURED_TAILS = (
    (re.compile(r"^$"), ""),
    (re.compile(r"^, concentrated on Commons smartphone photographs \((?P<k>[\d,]+)/(?P<n>[\d,]+)\)$"),
     ", {k} of them among its {n} smartphone photographs"),
    (re.compile(r"^; (?P<r1>\d+(?:\.\d+)?) % \((?P<k1>[\d,]+)/(?P<n1>[\d,]+)\) at ≥ 8 MP against (?P<r2>\d+(?:\.\d+)?) % "
                r"\((?P<k2>[\d,]+)/(?P<n2>[\d,]+)\) below$"),
     ": {k1} of the {n1} of 8 megapixels or more ({r1} %), and {k2} of the {n2} smaller ones ({r2} %)"),
    (re.compile(r"^, including masked faces the face stage did not find, hence the has_face gate$"),
     ", among them masked faces that the face stage did not find"),
)


def measured_sentence(mode: str) -> str | None:
    """The measured false-alarm row of a card's mode in the reader's words, or None for a mode that is no such row, or
    whose form this copy does not know (it is then not printed: nothing is guessed from a row)."""
    row = _MEASURED.match(mode.strip())
    if row is None:
        return None
    tail = next((words_.format(**found.groupdict()) for form, words_ in _MEASURED_TAILS if (found := form.match(row["tail"])) is not None), None)
    if tail is None:
        return None
    flagged = (f"{row['k']} of the benchmark's {row['n']} genuine photographs" if row["of"].startswith(",")
               else f"{row['k']} of the {row['n']} genuine photographs of the benchmark that it scored")
    return (f"Alone, at a threshold set to flag about 5 % of genuine images, this check flagged {flagged} ({row['rate']} %; 95 % interval "
            f"{row['low']} to {row['high']} %){tail}.")


def block_modes(modes: Iterable[str], quoted: Iterable[str] = ()) -> tuple[list[str], list[str]]:
    """N-19n: the false-positive modes of a card for the detector block of a full report: (the modes in client wording, as
    `client_modes` gives them; the sentences of the modes it drops: the measured row in the reader's words, or that the
    mode is not printed here). Nothing points to a record that the report's reader cannot open."""
    quoted = list(quoted)
    kept, sentences = [], []
    for mode in modes:
        client = client_modes([mode], quoted)
        if client:
            kept += client
        else:
            sentences.append(measured_sentence(mode) or MODE_NOT_PRINTED)
    return kept, list(dict.fromkeys(sentences))


_FIGURE = re.compile(r"\d|%")
CAUSE_MAX_WORDS = 14


def benign_causes(modes: Iterable[str], limit: int = 2) -> list[str]:
    """The modes a finding can name after "benign causes such as": short causes in client words ("heavy compression"), not
    measurement sentences. A sentence with figures inside those quotation marks read as a garbled citation (QA P2)."""
    causes = [m for m in client_modes(modes)
              if not _FIGURE.search(m) and not m.lower().startswith(("not established", "rates not established")) and len(m.split()) <= CAUSE_MAX_WORDS]
    return causes[:limit]
