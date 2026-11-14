"""AI-written-text analysis: extraction → sentences → classifiers (windowed) → Binoculars (token-level)
→ stylometry → per-sentence fusion with reasons → document verdict with evidence.

Two halves, so the v1 path and the v2 document pipeline share one fusion:
- `score_document` runs the detectors (v1 loads them through the shared registry) → `DocumentScores`;
- `assemble_document` is the v0.3 per-sentence fusion, smoothing, document signals and verdict over those scores.
The v2 legacy fuser (`engine/stages/document.py`) rebuilds `DocumentScores` from detector runs and calls the same
`assemble_document`.
"""
from __future__ import annotations

import io
import json
import math
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

# OO API: no pyplot global state, safe in worker threads
from PIL import Image

from app.core import deadline
from app.core.config import settings
from app.core.evidence import Signal
from app.core.logging import get_logger
from app.detectors.common.fusion import fuse, fuse_probability, prob_to_signal_fields
from app.detectors.text import extract as extract_module
from app.detectors.text import language_scope, stylometry
from app.detectors.text import segment as segment_module
from app.detectors.text.binoculars import Binoculars, borrow_binoculars, document_score, sentence_scores
from app.detectors.text.classifier import borrow_primary, borrow_secondary
from app.detectors.text.extract import Document, from_string
from app.detectors.text.segment import Sentence, segment, sentences

# Folio palette for figures (matches frontend/design/direction-b-stripe-notion.md)
INK, MUTED, RULE, PAPER = "#1B1A17", "#625D54", "#E4E0D7", "#FCFBF9"
SCALE = ["#0B6B54", "#2A7460", "#6E685D", "#AE4617", "#9B2915"]  # authentic → AI


def _style(ax, fig=None) -> None:
    if fig is not None:
        fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.grid(axis="y", color=RULE, linewidth=0.6)
    ax.set_axisbelow(True)


log = get_logger(__name__)
AI_T, HUMAN_T = 0.65, 0.40
MIN_WORDS = 15
PRIMARY_CHUNK_MAX_LEN = 768


def windows(sents: list[Sentence], k: int) -> list[str]:
    half = k // 2
    return [" ".join(s.text for s in sents[max(0, i - half) : i + half + 1]) for i in range(len(sents))]


def chunks_of(sents: list[Sentence], target_words: int = 350) -> list[tuple[str, int, int]]:
    out, buf, words, s0 = [], [], 0, 0
    for s in sents:
        if not buf:
            s0 = s.idx
        buf.append(s.text)
        words += s.n_words
        if words >= target_words:
            out.append((" ".join(buf), s0, s.idx))
            buf, words = [], 0
    if buf:
        out.append((" ".join(buf), s0, sents[-1].idx))
    return out


def chunk_words(chunks: list[tuple[str, int, int]], sents: list[Sentence]) -> list[int]:
    return [sum(s.n_words for s in sents[a : b + 1]) for _, a, b in chunks]


def chunk_average(p_chunks: list[float], words: list[int]) -> float:
    return float(np.average(p_chunks, weights=np.maximum(words, 1)))


def sentence_plot(sent_p: list[float], art_dir: Path | None) -> list[str]:
    if art_dir is None or not sent_p:
        return []
    fig = Figure(figsize=(8, 2.6), dpi=110)
    ax = fig.subplots()
    x = np.arange(len(sent_p))
    _style(ax, fig)
    colors = [SCALE[4] if p >= AI_T else SCALE[3] if p >= HUMAN_T else SCALE[1] for p in sent_p]
    ax.bar(x, sent_p, color=colors, width=0.85, linewidth=0)
    ax.axhline(AI_T, color=MUTED, ls=":", lw=0.8)
    ax.axhline(HUMAN_T, color=MUTED, ls=":", lw=0.8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("sentence (document order)")
    ax.set_ylabel("p(AI)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    Image.open(buf).convert("RGB").save(art_dir / "sentence_scores.png")
    return ["sentence_scores.png"]


def check_document_size(doc: Document) -> None:
    """Refuse a partial examination; the configured bound applies to extracted text."""
    if doc.meta.get("truncated"):
        raise ValueError(
            "An earlier extraction retained only part of this document. "
            "Split the original into smaller documents and analyse each separately; "
            "the partial text cannot support a new document result."
        )
    if len(doc.text) > settings.max_document_chars:
        raise ValueError(
            f"Document contains {len(doc.text):,} extracted characters; the limit is {settings.max_document_chars:,}. "
            "Split it into smaller documents and analyse each separately. No text was analysed."
        )


def check_document_length(doc: Document, sents: list[Sentence]) -> int:
    words = sum(s.n_words for s in sents)
    if words < MIN_WORDS or not sents:
        raise ValueError(f"Document contains too little text to analyse (fewer than {MIN_WORDS} words). " + doc.meta.get("warning", ""))
    return words


PRIMARY_REFERENCE = "DeBERTa-v3-large fine-tuned on human vs LLM text incl. 2025-26 model output; 0.5 decision boundary"
# QA P1-10: the base checkpoint must not be described as the fine-tune
BASE_REFERENCE = "DeBERTa-v3-large base classifier, not fine-tuned on 2025-26 model output; 0.5 decision boundary"


@dataclass
class DocumentScores:
    """Detector outputs the legacy document fusion reads. `None` means the detector did not run."""

    n_sentences: int
    p_win: list[float | None] | None = None  # primary classifier, per sentence window
    p_chunks: list[float] = field(default_factory=list)
    p_doc_primary: float | None = None
    primary_source: str | None = None
    primary_fine_tuned: bool = True  # False: the base checkpoint scored the document (the fine-tune is absent or not enabled)
    p_win2: list[float | None] | None = None  # secondary classifier
    p_doc_sec: float | None = None
    secondary_source: str | None = None
    bino_doc: float | None = None
    bino_sent: list[float | None] = field(default_factory=list)
    bino_tokens: int = 0
    metrics: dict | None = None  # stylometry document metrics
    p_style_doc: float | None = None
    style_reasoning: str = ""
    flags: list[list[str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.bino_sent:
            self.bino_sent = [None] * self.n_sentences
        if not self.flags:
            self.flags = [[] for _ in range(self.n_sentences)]


def score_document(doc: Document, sents: list[Sentence]) -> tuple[DocumentScores, dict[str, float]]:
    """v1: run every document detector through the shared registry."""
    timings: dict[str, float] = {}
    scores = DocumentScores(n_sentences=len(sents))

    # --- supervised classifiers on sentence windows and on document chunks ---
    t = time.perf_counter()
    with borrow_primary() as primary:
        try:
            win = windows(sents, settings.text_window_sentences)
            scores.p_win = primary.predict(win)
            chunks = chunks_of(sents)
            words = chunk_words(chunks, sents)
            scores.p_chunks = primary.predict([c for c, _, _ in chunks], max_len=PRIMARY_CHUNK_MAX_LEN)
            scores.p_doc_primary = chunk_average(scores.p_chunks, words)
            scores.primary_source = primary.source
            scores.primary_fine_tuned = primary.source != settings.text_classifier_id
        finally:
            del primary  # A released model must not stay referenced during the next acquisition.
    with borrow_secondary() as sec:
        try:
            if sec:
                scores.p_win2 = sec.predict(win)
                scores.p_doc_sec = chunk_average(sec.predict([c for c, _, _ in chunks]), words)
                scores.secondary_source = settings.text_secondary_classifier_id
        finally:
            del sec
    timings["classifiers"] = round(time.perf_counter() - t, 3)
    deadline.check("classification")

    # --- Binoculars ---
    t = time.perf_counter()
    if settings.binoculars_enabled:
        try:
            with borrow_binoculars() as b:
                try:
                    ts = b.token_scores(doc.text)
                    scores.bino_doc, scores.bino_tokens = document_score(ts)
                    scores.bino_sent = sentence_scores(ts, sents)
                finally:
                    del b
        except Exception as e:
            log.warning("binoculars.failed", error=str(e))
    timings["binoculars"] = round(time.perf_counter() - t, 3)

    # --- stylometry ---
    t = time.perf_counter()
    scores.metrics = stylometry.document_metrics([s.text for s in sents])
    scores.p_style_doc, style_reasons = stylometry.document_style_probability(scores.metrics)
    scores.style_reasoning = "; ".join(style_reasons) + "."
    scores.flags = [stylometry.sentence_flags(s.text) for s in sents]
    timings["stylometry"] = round(time.perf_counter() - t, 3)
    return scores, timings


def assemble_document(doc: Document, sents: list[Sentence], scores: DocumentScores, art_dir: Path | None, *,
                      timings: dict[str, float] | None = None, t0: float | None = None) -> dict:
    """The v0.3 per-sentence fusion, contextual smoothing, document signals and verdict."""
    timings = dict(timings or {})
    words = sum(s.n_words for s in sents)

    # --- per-sentence fusion ---
    sentences_out = []
    sent_p: list[float] = []
    for s in sents:
        pairs: list[tuple[float, float]] = []
        reasons: list[str] = []
        sentence_scores_: dict = {}
        if s.n_words < 3:
            sentences_out.append({"idx": s.idx, "start": s.start, "end": s.end, "text": s.text, "paragraph": s.paragraph, "p_ai": None, "band": "skipped", "reasons": ["Too short to assess"], "scores": {}})
            continue
        pw = scores.p_win[s.idx] if scores.p_win is not None else None
        if pw is not None:
            sentence_scores_["classifier"] = round(pw, 4)
            pairs.append((pw, settings.text_classifier_weight))
            reasons.append(f"DeBERTa AI-text classifier: {pw:.0%} on this sentence with its neighbours" + (" (AI-like)" if pw >= 0.5 else " (human-like)"))
        pw2 = scores.p_win2[s.idx] if scores.p_win2 is not None else None
        if pw2 is not None:
            sentence_scores_["classifier_secondary"] = round(pw2, 4)
            pairs.append((pw2, settings.text_secondary_weight))
        bs = scores.bino_sent[s.idx]
        if bs is not None:
            pb = Binoculars.to_probability(bs)
            sentence_scores_["binoculars"] = round(bs, 4)
            sentence_scores_["binoculars_p"] = round(pb, 4)
            pairs.append((pb, settings.binoculars_weight))
            reasons.append(
                f"Binoculars score {bs:.3f} vs threshold {settings.binoculars_threshold} — "
                + ("the wording is more predictable to an LLM than the LLM's own expectation (machine-like)" if bs < settings.binoculars_threshold else "the wording carries more surprise than an LLM would generate (human-like)")
            )
        flags = scores.flags[s.idx]
        ps = stylometry.sentence_style_probability(flags)
        if ps is not None:
            sentence_scores_["style"] = round(ps, 4)
            pairs.append((ps, settings.stylometry_weight))
            reasons.append("Stylistic tells: " + "; ".join(flags[:4]))
        if not pairs:
            sentences_out.append({"idx": s.idx, "start": s.start, "end": s.end, "text": s.text, "paragraph": s.paragraph, "p_ai": None, "band": "skipped", "reasons": ["No detector scored this sentence"], "scores": {}})
            continue
        p = fuse_probability(pairs)
        sentences_out.append({"idx": s.idx, "start": s.start, "end": s.end, "text": s.text, "paragraph": s.paragraph, "p_ai": round(p, 4), "p_raw": round(p, 4), "band": None, "reasons": reasons, "scores": sentence_scores_, "flags": flags})

    # --- contextual smoothing: machine-written text comes in contiguous runs, so an isolated spike in an
    # otherwise human passage (formulaic or encyclopedic sentences are the usual cause) is treated as noise.
    # Median of the sentence and its two scored neighbours; runs of >= 2 flagged sentences are preserved. ---
    scored = [x for x in sentences_out if x["p_ai"] is not None]
    # document prior: sentence verdicts are not independent of the document-level detectors (which see the
    # whole text). Half-strength log-odds shift toward the document-level classifier/Binoculars estimate.
    doc_parts = ([scores.p_doc_primary] if scores.p_doc_primary is not None else []) + ([Binoculars.to_probability(scores.bino_doc)] if scores.bino_doc is not None else [])
    doc_prior = float(np.mean(doc_parts)) if doc_parts else 0.5
    prior_shift = 0.5 * math.log(max(doc_prior, 0.02) / max(1 - doc_prior, 0.02))
    raw = []
    for x in scored:
        lo = math.log(max(x["p_raw"], 0.02) / max(1 - x["p_raw"], 0.02)) + prior_shift
        raw.append(1 / (1 + math.exp(-lo)))
    for k, x in enumerate(scored):
        window = raw[max(0, k - 1) : k + 2]
        sm = float(sorted(window)[len(window) // 2]) if len(window) == 3 else raw[k]
        x["p_ai"] = round(sm, 4)
        x["band"] = "ai" if sm >= AI_T else "human" if sm < HUMAN_T else "mixed"
        raw_band = "ai" if x["p_raw"] >= AI_T else "human" if x["p_raw"] < HUMAN_T else "mixed"
        if raw_band != x["band"]:
            x["reasons"].append(f"Context adjustment: this sentence alone scored {x['p_raw']:.2f}; the document-level detectors estimate {doc_prior:.2f} for the whole text and neighbouring sentences {'agree' if abs(raw[k]-sm) < 0.05 else 'do not agree'}, so the sentence verdict was adjusted to {sm:.2f}. Machine-written passages are contiguous and consistent with the document as a whole.")
        sent_p.append(sm)
    total_chars = sum(x["end"] - x["start"] for x in scored) or 1
    ai_chars = sum(x["end"] - x["start"] for x in scored if x["band"] == "ai")
    mixed_chars = sum(x["end"] - x["start"] for x in scored if x["band"] == "mixed")
    weighted = sum((x["end"] - x["start"]) * x["p_ai"] for x in scored) / total_chars
    ai_share = ai_chars / total_chars
    mixed_share = mixed_chars / total_chars

    # --- document-level signals ---
    signals: list[Signal] = []
    if scores.p_doc_primary is not None:
        p_doc_primary = scores.p_doc_primary
        signals.append(
            Signal(
                id="classifier_document", name="DeBERTa-v3-large AI-text classifier (document chunks)", category="model",
                observed=f"p(AI)={p_doc_primary:.3f} over {len(scores.p_chunks)} chunk(s)", reference=PRIMARY_REFERENCE if scores.primary_fine_tuned else BASE_REFERENCE,
                weight=settings.text_classifier_weight, **prob_to_signal_fields(p_doc_primary),
                reasoning=("The document's token-level patterns match LLM output in the classifier's learned feature space." if p_doc_primary >= 0.5 else "The document's patterns match human-written training text."),
                details={"chunks": [round(x, 4) for x in scores.p_chunks], "model": scores.primary_source},
            )
        )
    if scores.p_doc_sec is not None:
        signals.append(
            Signal(
                id="classifier_secondary", name="RoBERTa ChatGPT detector (HC3)", category="model",
                observed=f"p(AI)={scores.p_doc_sec:.3f}", reference="ChatGPT-era baseline; weaker on newer models and paraphrased text",
                weight=settings.text_secondary_weight, **prob_to_signal_fields(scores.p_doc_sec),
                reasoning="Independent classifier from a different training corpus; used as a corroborating vote.",
                details={"model": scores.secondary_source},
            )
        )
    if scores.bino_doc is not None:
        pb = Binoculars.to_probability(scores.bino_doc)
        signals.append(
            Signal(
                id="binoculars_document", name="Binoculars zero-shot LLM detector (observer/performer perplexity ratio)", category="statistical",
                observed=f"score={scores.bino_doc:.4f} over {scores.bino_tokens} tokens", reference=f"Calibrated threshold {settings.binoculars_threshold} ({settings.binoculars_observer_id.split('/')[-1]} pair); lower = more machine-like",
                weight=settings.binoculars_weight, **prob_to_signal_fields(pb),
                reasoning=("The text is markedly more predictable to the language model than the model's own cross-entropy would imply — the statistical fingerprint of LLM sampling." if pb >= 0.5 else "The text carries more lexical surprise than LLM-sampled text would, consistent with human authorship."),
                details={"observer": settings.binoculars_observer_id, "performer": settings.binoculars_performer_id},
            )
        )
    if scores.metrics is not None and scores.p_style_doc is not None:
        metrics = scores.metrics
        signals.append(
            Signal(
                id="stylometry_document", name="Stylometric profile", category="stylometric",
                observed=f"lexicon {metrics['lexicon_weighted_per_1000']}/1000w, contrast {metrics['contrast_per_1000']}/1000w, transition openers {metrics['transition_opener_pct']}%, sentence-length CV {metrics['sentence_len_cv']}, MATTR {metrics['mattr']}",
                reference="Human prose: lexicon < 3/1000, openers < 8%, CV > 0.5", weight=settings.stylometry_weight, **prob_to_signal_fields(scores.p_style_doc),
                reasoning=scores.style_reasoning, details=metrics,
            )
        )
    if scored:
        signals.append(
            Signal(
                id="sentence_consensus", name="Sentence-level consensus", category="statistical",
                observed=f"{ai_share:.0%} of text in AI-classified sentences, {mixed_share:.0%} mixed, {1-ai_share-mixed_share:.0%} human; char-weighted mean p={weighted:.3f}",
                reference="Fully human documents: < 10% AI-classified text; fully generated: > 70%", weight=1.0, **prob_to_signal_fields(weighted),
                reasoning=f"{sum(1 for x in scored if x['band']=='ai')} of {len(scored)} sentences were independently classified as AI-written by the fused sentence detectors.",
                details={"ai_share": round(ai_share, 4), "mixed_share": round(mixed_share, 4)},
            )
        )
    letters = [ch for ch in doc.text if ch.isalpha()]
    latin_share = (sum(1 for ch in letters if ch.isascii()) / len(letters)) if letters else 1.0
    non_english = latin_share < 0.6
    if non_english:
        signals.append(
            Signal(
                id="language_gate", name="Non-Latin-script text", category="statistical",
                observed=f"{latin_share:.0%} of letters are ASCII/Latin", reference="Detectors are trained on English; other scripts are unsupported",
                ai_probability=None, weight=0.0, direction="inconclusive", strength="none",
                reasoning="The classifiers and stylometry were trained on English text; their outputs on other scripts are not meaningful. Only the zero-shot perplexity signal carries any information here.",
            )
        )
        for s_ in signals:
            if s_.id in ("classifier_document", "classifier_secondary", "stylometry_document", "sentence_consensus"):
                s_.weight = round(s_.weight * 0.25, 3)
    if words < settings.text_min_words_for_verdict * 3:
        signals.append(
            Signal(
                id="length_gate", name="Short document", category="statistical", observed=f"{words} words",
                reference="Detectors are reliable from ~120 words; below 40 words they are near chance",
                ai_probability=None, weight=0.0, direction="inconclusive", strength="none",
                reasoning="Short texts provide few statistical observations; treat the verdict as indicative.",
            )
        )

    verdict = fuse(signals, subject="document")
    if non_english:
        verdict.confidence = "low"
        verdict.caveats.insert(0, "Non-English or non-Latin-script text: the detectors are English-only; treat this result as unsupported.")
    verdict.reasoning.insert(0, f"Estimated AI-written share: {ai_share:.0%} of the text (sentences ≥ {AI_T:.0%}); a further {mixed_share:.0%} is mixed/uncertain.")
    if words < settings.text_min_words_for_verdict * 3:
        verdict.confidence = "low"
        verdict.caveats.insert(0, f"Only {words} words were available; confidence is capped at low.")
    verdict.caveats.append("AI-text detectors are known to produce false positives on formulaic, non-native, or heavily edited human prose, and false negatives on paraphrased or lightly edited machine text.")
    artifacts = sentence_plot([x["p_ai"] for x in scored], art_dir)
    if t0 is not None:
        timings["total"] = round(time.perf_counter() - t0, 3)
    return {
        "verdict": verdict.model_dump(),
        "signals": [s.model_dump() for s in signals],
        "text": doc.text,
        "paragraphs": doc.paragraphs,
        "sentences": sentences_out,
        "stats": {
            "words": words, "sentences": len(sents), "scored_sentences": len(scored),
            "ai_sentences": sum(1 for x in scored if x["band"] == "ai"), "mixed_sentences": sum(1 for x in scored if x["band"] == "mixed"),
            "human_sentences": sum(1 for x in scored if x["band"] == "human"),
            "ai_share": round(ai_share, 4), "mixed_share": round(mixed_share, 4), "weighted_ai_share": round(weighted, 4),
            "binoculars_document": round(scores.bino_doc, 4) if scores.bino_doc is not None else None,
        },
        "document": doc.meta,
        "artifacts": artifacts,
        "timings": timings,
    }


def _analyze_segmented(doc: Document, sents: list[Sentence], art_dir: Path | None, t0: float) -> dict:
    check_document_length(doc, sents)
    scope = language_scope.classify(doc.text, sents)
    if scope in language_scope.MESSAGES:
        return unassessed_language_document(doc, sents, scope)
    scores, timings = score_document(doc, sents)
    return assemble_document(doc, sents, scores, art_dir, timings=timings, t0=t0)


def unassessed_language_document(doc: Document, sents: list[Sentence], scope: str) -> dict:
    headline, reason = language_scope.MESSAGES[scope]
    result = assemble_document(doc, sents, DocumentScores(n_sentences=len(sents)), None)
    result["verdict"].update(probability_ai=None, label="Not assessable", confidence="low",
                             summary=headline, reasoning=[reason], caveats=[reason])
    result["document"] = {**doc.meta, language_scope.KEY: scope}
    result["signals"] = []
    for sentence in result["sentences"]:
        sentence["reasons"] = [reason]
    return result


def analyze_document_obj(doc: Document, art_dir: Path | None) -> dict:
    """For a caller that brings its own text (the benchmark, the tests): it is segmented in this interpreter. A request's
    document goes through `analyze_document` or `decode_document` (N-14b; S2-01)."""
    check_document_size(doc)
    t0 = time.perf_counter()
    return _analyze_segmented(doc, segment(doc), art_dir, t0)


# N-10 (SEC-04): a text file holds at most this many bytes for each character of an analysable document (UTF-8 needs up to
# four; the rest is room for markup that extraction removes). A larger `.txt` or `.md` cannot be within `max_document_chars`
# short of padding, and is refused before any pattern runs over it. HTML is not bounded here: a page is many times its text.
RAW_BYTES_PER_CHAR = 8
RAW_BOUNDED = frozenset({".txt", ".md", ".markdown"})
EXTRACT_SCRIPT = Path(extract_module.__file__)
SEGMENT_SCRIPT = Path(segment_module.__file__)


def check_raw_size(path: Path, ext: str) -> None:
    """Refuse a text file that is too large to hold an analysable document, by its size alone."""
    if ext.lower() not in RAW_BOUNDED:
        return
    size, bound = path.stat().st_size, RAW_BYTES_PER_CHAR * settings.max_document_chars
    if size > bound:
        raise ValueError(
            f"Document is {size:,} bytes; a text document within the limit of {settings.max_document_chars:,} characters "
            f"is at most {bound:,} bytes. Split it into smaller documents and analyse each separately. No text was analysed."
        )


def _not_within(limit: float) -> deadline.StepTimeout:
    return deadline.StepTimeout(f"The document could not be read within the time limit of {limit:g} s")


def extract_command(path: Path, ext: str) -> list[str]:
    """The child that reads one document: this interpreter, isolated from the environment and the working directory."""
    return [sys.executable, "-I", str(EXTRACT_SCRIPT), str(path), ext]


def extract_within(path: Path, ext: str, *, native_pdf_only: bool = False) -> Document:
    """`extract` in a child process with a hard limit (`document_decode_timeout_s`). A parser or a pattern that does not
    end holds the interpreter of whoever runs it, and a deadline that is checked between stages cannot interrupt it; a
    child is killed. An error of the reader is raised here under its own class name, a `ValueError` as a `ValueError`,
    so the reason shown to the person reads as it did in process (`errors.friendly_error`)."""
    limit = settings.document_decode_timeout_s
    command = extract_command(path, ext)
    if native_pdf_only:
        command.append("--native-pdf")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=os.name == "posix")  # noqa: S603
    try:
        stdout, stderr = process.communicate(timeout=limit)
        done = subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            with suppress(ProcessLookupError):  # The child may exit at the timeout boundary.
                os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.communicate()
        raise _not_within(limit) from None
    try:
        out = json.loads(done.stdout)
    except ValueError:
        out = None
    if not isinstance(out, dict) or done.returncode != 0:
        log.error("document.extract_child_failed", returncode=done.returncode, stderr=done.stderr[-2000:].decode("utf-8", "replace"))
        raise RuntimeError(f"The document reader ended with exit code {done.returncode}")
    if "error" in out:
        failed = out["error"]
        base = ValueError if failed.get("value_error") else RuntimeError
        raise type(str(failed.get("type") or "ExtractionError"), (base,), {})(str(failed.get("message") or ""))
    return Document(out["text"], [(int(a), int(b)) for a, b in out["paragraphs"]], dict(out["meta"]),
                    list(out.get("source_map") or []), dict(out.get("source_rasters") or {}),
                    word_review=out.get("word_review"))


def segment_command() -> list[str]:
    """The child that segments one document: this interpreter, isolated as the reader is. It imports pysbd and
    `segment.py`, and nothing of the application."""
    return [sys.executable, "-I", str(SEGMENT_SCRIPT)]


def _offsets(doc: Document, said: bytes) -> list[tuple[int, int, int]] | None:
    """The segmenter's answer as (start, end, paragraph) of each sentence, or None when it is not offsets into this text:
    each sentence lies inside its paragraph, and the paragraphs come in their order."""
    try:
        found = [(int(s), int(e), int(p)) for s, e, p in json.loads(said)]
    except (ValueError, TypeError):
        return None
    last = 0
    for s, e, p in found:
        if not last <= p < len(doc.paragraphs) or not doc.paragraphs[p][0] <= s < e <= doc.paragraphs[p][1]:
            return None
        last = p
    return found


def segment_within(doc: Document, started: float | None = None) -> list[Sentence]:
    """`segment` with the sentence boundaries found in a child process that is killed at the limit (N-14b; S2-01). The time
    of pysbd grows with the sentences times the characters of a paragraph ("a. " repeated to 120,000 characters takes more
    than 100 s), inside one call that a deadline checked between stages cannot interrupt. `document_decode_timeout_s`
    bounds reading and segmenting together: `started` is when the reading began (`time.monotonic()`), and the child gets
    what is left. A segmentation that does not end is refused as a read that does not end is. The child returns offsets;
    each `Sentence` is built here from the document's own text, so the sentences are those of `segment`."""
    limit = settings.document_decode_timeout_s
    left = limit if started is None else limit - (time.monotonic() - started)
    if left <= 0:
        raise _not_within(limit)
    asked = json.dumps(doc.paragraphs).encode() + b"\n" + doc.text.encode("utf-8", "surrogatepass")
    try:
        done = subprocess.run(segment_command(), input=asked, capture_output=True, timeout=left, check=False)  # noqa: S603
    except subprocess.TimeoutExpired:
        raise _not_within(limit) from None
    found = _offsets(doc, done.stdout) if done.returncode == 0 else None
    if found is None:
        log.error("document.segment_child_failed", returncode=done.returncode, stderr=done.stderr[-2000:].decode("utf-8", "replace"))
        raise RuntimeError(f"The document segmenter ended with exit code {done.returncode}")
    return sentences(doc, found)


def prepare_document(path: Path, ext: str) -> Document:
    check_raw_size(path, ext)
    doc = extract_within(path, ext)
    check_document_size(doc)
    return doc


def decode_document(path: Path, ext: str) -> tuple[Document, list[Sentence]]:
    """One document of a request, read and segmented: both in child processes that are killed, within
    `document_decode_timeout_s` together. The order is read, `check_document_size`, segment: a document over the size
    limit is never segmented."""
    started = time.monotonic()
    doc = prepare_document(path, ext)
    return doc, segment_within(doc, started)


def analyze_document(path: Path, ext: str, art_dir: Path) -> dict:
    started = time.monotonic()
    doc = prepare_document(path, ext)
    t0 = time.perf_counter()
    return _analyze_segmented(doc, segment_within(doc, started), art_dir, t0)


def analyze_text(text: str, art_dir: Path | None = None) -> dict:
    return analyze_document_obj(from_string(text), art_dir)
