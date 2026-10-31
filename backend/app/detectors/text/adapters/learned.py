"""Learned document detectors as v2 plugins: desklib DeBERTa-v3-large (base and the Veritas fine-tune), the HC3
RoBERTa classifier and Binoculars.

- **Classifiers** score every sentence together with its neighbours (`sentence:<i>`) and the document in ~350-word
  chunks (`chunk:<k>`). Classifiers retain every tokenizer overflow window and record content-token coverage.
  Within each input, window probabilities are weighted by content tokens; the document retains its word-weighted chunk mean.
- **Binoculars** reports raw perplexity ratios (`raw_kind="score"`, lower is more machine-like) for the document and for
  each sentence. The v0.3 logistic mapping around the calibrated threshold is applied by the legacy fuser, not here.
"""
from __future__ import annotations

from typing import Any, ClassVar

from app.core.config import settings
from app.detectors.text.adapters.common import (
    DEEP_ONLY,
    FULL_DEEP,
    document_input,
    has_sentences,
    revisions,
    text_spec,
)
from app.detectors.text.binoculars import (
    CTX,
    MIN_DOCUMENT_TOKENS,
    Binoculars,
    document_score,
    sentence_scores,
)
from app.detectors.text.classifier import (
    PRIMARY_BATCH,
    SECONDARY_BATCH,
    DesklibClassifier,
    SecondaryClassifier,
)
from app.detectors.text.pipeline import PRIMARY_CHUNK_MAX_LEN, chunk_average, chunk_words, chunks_of, windows
from app.engine.detector import (
    ActivationUse,
    Applicability,
    Detector,
    DetectorResult,
    InputUnit,
    RunContext,
    UnitScore,
)


class _SequenceClassifier(Detector):
    MODEL_ID: ClassVar[str]
    CHUNK_MAX_LEN: ClassVar[int] = 512
    BATCH: ClassVar[int] = PRIMARY_BATCH

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return (ActivationUse(self.MODEL_ID, self.BATCH, "token window",
                              f"Native tokenizer windows, at most {max(512, self.CHUNK_MAX_LEN)} tokens per input including specials",
                              ("CPU tokenizer overflow buffers are not included in a measured process peak.",)),)

    def backend(self, ctx: RunContext) -> Any:
        raise NotImplementedError

    def applicable(self, ctx: RunContext) -> Applicability:
        return has_sentences(ctx)

    def run(self, ctx: RunContext) -> DetectorResult:
        _doc, sents = document_input(ctx)
        clf = self.backend(ctx)
        p_win = [float(p) for p in clf.predict(windows(sents, settings.text_window_sentences), batch=self.BATCH, check=ctx.deadline.check)]
        window_coverage = clf.last_coverage
        ctx.deadline.check("sentence windows")
        chunks = chunks_of(sents)
        words = chunk_words(chunks, sents)
        p_chunks = [float(p) for p in clf.predict([c for c, _, _ in chunks], max_len=self.CHUNK_MAX_LEN, batch=self.BATCH, check=ctx.deadline.check)]
        chunk_coverage = clf.last_coverage
        token_count = sum(c["tokens"] for c in chunk_coverage)
        token_windows = sum(c["token_windows"] for c in chunk_coverage)
        p_doc = chunk_average(p_chunks, words)
        units = [UnitScore(f"sentence:{i}", p, "probability", quality=window_coverage[i]) for i, p in enumerate(p_win)]
        units += [UnitScore(f"chunk:{k}", p, "probability", quality={"first_sentence": float(a), "last_sentence": float(b), "words": float(w), **chunk_coverage[k]})
                  for k, (p, (_, a, b), w) in enumerate(zip(p_chunks, chunks, words, strict=True))]
        aggregate = UnitScore("global", p_doc, "probability", quality={"chunks": float(len(chunks)), "sentences": float(len(sents)),
                                                                     "tokens": float(token_count), "token_windows": float(token_windows)})
        side = "LLM-written" if p_doc >= 0.5 else "human-written"
        return DetectorResult(
            status="ok", unit_scores=units, aggregate=aggregate,
            observed=f"sigmoid output {p_doc:.3f} (word-weighted mean over {len(chunks)} chunk(s)); {len(sents)} sentence "
                     f"windows of {settings.text_window_sentences} sentences scored. Document chunks cover all {token_count} content "
                     f"tokens in {token_windows} non-overlapping token window(s)",
            reference="Binary AI-text classifier; decision boundary at 0.5",
            reasoning=f"The classifier places the document on the {side} side of its decision boundary.",
            metrics={"chunks": float(len(chunks)), "sentences": float(len(sents)), "tokens": float(token_count),
                     "token_windows": float(token_windows)}, model_revisions=revisions(ctx, self.MODEL_ID))


class DesklibDetector(_SequenceClassifier):
    MODEL_ID = "hf.desklib.ai_text_detector_v1_01"
    CHUNK_MAX_LEN = PRIMARY_CHUNK_MAX_LEN
    spec = text_spec("txt.desklib", "desklib ai-text-detector v1.01 (DeBERTa-v3-large)", InputUnit.SENTENCES,
                     model_ids=(MODEL_ID,), timeout_s=120.0, version="0.4.0")

    def backend(self, ctx: RunContext) -> DesklibClassifier:
        tokenizer, model, source = ctx.lease.get(self.MODEL_ID).obj
        return DesklibClassifier.from_parts(tokenizer, model, source, ctx.device)


class DesklibFinetuneDetector(DesklibDetector):
    MODEL_ID = "local.desklib_ft_v2"
    spec = text_spec("txt.desklib_ft_v2", "desklib-ft-v2 (Veritas fine-tune, DeBERTa-v3-large)", InputUnit.SENTENCES,
                     model_ids=(MODEL_ID,), timeout_s=120.0, version="0.4.0")


class HelloSimpleAIDetector(_SequenceClassifier):
    BATCH = SECONDARY_BATCH
    MODEL_ID = "hf.hello_simpleai.chatgpt_detector_roberta"
    spec = text_spec("txt.hf.hello_simpleai", "Hello-SimpleAI chatgpt-detector-roberta", InputUnit.SENTENCES,
                     model_ids=(MODEL_ID,), timeout_s=60.0, profiles=DEEP_ONLY, version="0.4.0")

    def backend(self, ctx: RunContext) -> SecondaryClassifier:
        tokenizer, model = ctx.lease.get(self.MODEL_ID).obj
        return SecondaryClassifier.from_parts(tokenizer, model, ctx.device)


class BinocularsDetector(Detector):
    OBSERVER, PERFORMER = "hf.qwen.qwen2_5_1_5b", "hf.qwen.qwen2_5_1_5b_instruct"
    spec = text_spec("txt.binoculars", "Binoculars zero-shot (Qwen2.5-1.5B observer / 1.5B-Instruct performer)", InputUnit.DOCUMENT,
                     model_ids=(OBSERVER, PERFORMER), timeout_s=180.0, profiles=FULL_DEEP)

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return tuple(ActivationUse(m, 1, "token sequence", f"One sequence at a time, at most {CTX} tokens including optional EOS anchor",
                                   ("Both models' vocabulary logits and loss intermediates are retained; peak remains an estimate.",))
                     for m in (self.OBSERVER, self.PERFORMER))

    def applicable(self, ctx: RunContext) -> Applicability:
        return has_sentences(ctx)

    def run(self, ctx: RunContext) -> DetectorResult:
        doc, sents = document_input(ctx)
        tokenizer, observer = ctx.lease.get(self.OBSERVER).obj
        _tokenizer, performer = ctx.lease.get(self.PERFORMER).obj
        b = Binoculars.from_parts(tokenizer, observer, performer, ctx.device)
        ts = b.token_scores(doc.text, check=ctx.deadline.check)
        score, n_tokens = document_score(ts)
        if score is None:
            return DetectorResult.not_applicable(f"fewer than {MIN_DOCUMENT_TOKENS} scored tokens")
        ctx.deadline.check("sentence spans")
        per_sentence = sentence_scores(ts, sents)
        units = [UnitScore(f"sentence:{i}", sc, "score") for i, sc in enumerate(per_sentence) if sc is not None]
        aggregate = UnitScore("global", score, "score", quality={"tokens": float(n_tokens)})
        machine_like = score < settings.binoculars_threshold
        return DetectorResult(
            status="ok", unit_scores=units, aggregate=aggregate,
            observed=f"observer log-perplexity / cross-perplexity = {score:.4f} over {n_tokens} tokens; "
                     f"{len(units)} of {len(sents)} sentences scored",
            reference=f"v0.3 threshold {settings.binoculars_threshold} for this model pair (HC3 calibration); lower is more "
                      "machine-like",
            reasoning=("The text is more predictable to the observer model than the performer's predictions imply, the "
                       "statistical fingerprint of LLM sampling." if machine_like else
                       "The text carries more surprise than LLM-sampled text, consistent with human authorship."),
            metrics={"tokens": float(n_tokens), "sentences_scored": float(len(units))},
            model_revisions=revisions(ctx, self.OBSERVER, self.PERFORMER))
