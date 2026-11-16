"""Learned document detectors as v2 plugins: desklib DeBERTa-v3-large (base and the Veritas fine-tune), the HC3
RoBERTa classifier and Binoculars.

- **Classifiers** score every sentence together with its neighbours (`sentence:<i>`) and the document in ~350-word
  chunks (`chunk:<k>`). Classifiers retain every tokenizer overflow window and record content-token coverage.
  Within each input, window probabilities are weighted by content tokens; the document retains its word-weighted chunk mean.
- **Binoculars** reports raw perplexity ratios (`raw_kind="score"`, lower is more machine-like) for the document and for
  each sentence. The v0.3 logistic mapping around the calibrated threshold is applied by the legacy fuser, not here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

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
    TokenScores,
    _losses_from_logits,
    document_score,
    sentence_score_records,
)
from app.detectors.text.classifier import (
    PRIMARY_BATCH,
    SECONDARY_BATCH,
    DesklibClassifier,
    FourClassTextClassifier,
    SecondaryClassifier,
)
from app.detectors.text.extract import text_hash
from app.detectors.text.pipeline import PRIMARY_CHUNK_MAX_LEN, chunk_average, chunk_words, chunks_of, windows
from app.engine.boundary import DetectorTimeout
from app.engine.detector import (
    ActivationUse,
    Applicability,
    Detector,
    DetectorResult,
    InputUnit,
    RunContext,
    UnitScore,
)

if TYPE_CHECKING:
    from app.engine.models import ModelManager


class _SequenceClassifier(Detector):
    MODEL_ID: ClassVar[str]
    CHUNK_MAX_LEN: ClassVar[int] = 512
    BATCH: ClassVar[int] = PRIMARY_BATCH
    SCORE_KIND: ClassVar[str] = "sigmoid output"
    REFERENCE: ClassVar[str] = "Binary AI-text classifier; decision boundary at 0.5"
    AI_SIDE: ClassVar[str] = "LLM-written"
    ISOLATED_PARAGRAPH_REVIEW: ClassVar[bool] = False

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return (ActivationUse(self.MODEL_ID, self.BATCH, "token window",
                              f"Native tokenizer windows, at most {max(512, self.CHUNK_MAX_LEN)} tokens per input including specials",
                              ("CPU tokenizer overflow buffers are not included in a measured process peak.",)),)

    def backend(self, ctx: RunContext) -> Any:
        raise NotImplementedError

    def applicable(self, ctx: RunContext) -> Applicability:
        return has_sentences(ctx)

    def run(self, ctx: RunContext) -> DetectorResult:
        doc, sents = document_input(ctx)
        clf = self.backend(ctx)
        inputs = windows(sents, settings.text_window_sentences)
        p_win = [float(p) for p in clf.predict(inputs, batch=self.BATCH, check=ctx.deadline.check)]
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
        side = self.AI_SIDE if p_doc >= 0.5 else "human-written"
        half = settings.text_window_sentences // 2
        input_windows = []
        for i, (source, score) in enumerate(zip(inputs, p_win, strict=True)):
            group = sents[max(0, i - half):i + half + 1]
            input_windows.append({"unit_ref": f"sentence:{i}", "focus": {"start": sents[i].start, "end": sents[i].end},
                                  "spans": [{"start": s.start, "end": s.end} for s in group],
                                  "input_sha256": text_hash(source), "score": score,
                                  "token_coverage": window_coverage[i]})
        evidence = {"version": "scored-context/1", "offset_unit": "unicode_codepoint",
                    "extracted_text_sha256": text_hash(doc.text), "join": "single_space_between_sentences",
                    "windows": input_windows}
        model_revisions = revisions(ctx, self.MODEL_ID)
        if self.ISOLATED_PARAGRAPH_REVIEW and (settings.text_passage_position_review or settings.text_long_passage_position_review):
            from app.detectors.text.passage_position import capture
            from app.evidence.document_passage_position import KEY

            if (position := capture(clf, doc, sents, ctx, self.spec, model_revisions)) is not None:
                evidence[KEY] = position
        if (settings.environment == "dev" and settings.text_matched_context_review and settings.text_detector_mode == "candidate"
                and settings.text_window_sentences == 3):
            from app.evidence import document_matched_context

            if document_matched_context.matches(self.spec.id, self.spec.version, model_revisions):
                record = document_matched_context.capture(doc, sents, input_windows)
                if record is not None:
                    evidence["matched_context_review"] = record
        if self.ISOLATED_PARAGRAPH_REVIEW and settings.text_isolated_paragraph_review and settings.text_window_sentences == 3:
            from app.detectors.text.paragraph_review import record

            try:
                paragraph_review = record(clf, doc, sents, input_windows, ctx.deadline)
            except (DetectorTimeout, RuntimeError):
                # Optional evidence can be unavailable without discarding the ordinary scores.
                paragraph_review = None
            if paragraph_review is not None:
                paragraphs = [item["spans"][0] for item in paragraph_review["inputs"]]
                for window in input_windows:
                    focus = window["focus"]
                    if any(span["start"] <= focus["start"] < focus["end"] <= span["end"] for span in paragraphs):
                        window["score_exact_hex"] = window["score"].hex()
                evidence["isolated_paragraph_review"] = paragraph_review
                if (settings.environment == "dev" and settings.text_paragraph_evidence_priority
                        and settings.text_detector_mode == "candidate"):
                    from app.evidence import document_matched_context, document_paragraph_priority

                    if (document_matched_context.matches(self.spec.id, self.spec.version, model_revisions)
                            and document_paragraph_priority.eligible(doc.text, doc.source_map, doc.meta.get("method"))):
                        evidence[document_paragraph_priority.KEY] = document_paragraph_priority.SCHEMA
        details = {"text_evidence": evidence}
        if self.ISOLATED_PARAGRAPH_REVIEW and settings.text_character_adjusted_review:
            from app.detectors.text.adapters import character_review
            from app.evidence import document_matched_context

            if character_review.enabled() and document_matched_context.matches(self.spec.id, self.spec.version, model_revisions):
                try:
                    rows = character_review.record_primary(clf, doc, input_windows, ctx)
                except (DetectorTimeout, RuntimeError, ValueError):
                    rows = []
                if rows:
                    details[character_review.STAGED] = rows
        if self.ISOLATED_PARAGRAPH_REVIEW and settings.text_passage_intervals_review:
            from app.detectors.text.adapters import interval_review

            if intervals := interval_review.measure(clf, doc, sents, p_win, ctx, self.spec, model_revisions):
                details[interval_review.KEY] = intervals
        return DetectorResult(
            status="ok", unit_scores=units, aggregate=aggregate,
            observed=f"{self.SCORE_KIND} {p_doc:.3f} (word-weighted mean over {len(chunks)} chunk(s)); {len(sents)} sentence "
                     f"windows of {settings.text_window_sentences} sentences scored. Document chunks cover all {token_count} content "
                     f"tokens in {token_windows} non-overlapping token window(s)",
            reference=self.REFERENCE,
            reasoning=f"The classifier places the document on the {side} side of its decision boundary.",
            metrics={"chunks": float(len(chunks)), "sentences": float(len(sents)), "tokens": float(token_count),
                     "token_windows": float(token_windows)}, model_revisions=model_revisions,
            details=details)


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


class FourClassTextDetector(_SequenceClassifier):
    MODEL_ID = "hf.wasitaigenerated.text_small"
    CHUNK_MAX_LEN = PRIMARY_CHUNK_MAX_LEN
    SCORE_KIND = "AI authorship or rewriting response"
    REFERENCE = "Combined response for generation and AI rewriting. The 0.5 reference separates raw score directions."
    AI_SIDE = "AI-written or AI-rewritten"
    ISOLATED_PARAGRAPH_REVIEW = True
    spec = text_spec("txt.four_class", "Tropa Mini text classifier", InputUnit.SENTENCES,
                     model_ids=(MODEL_ID,), timeout_s=120.0, version="1.0.0")

    def backend(self, ctx: RunContext) -> FourClassTextClassifier:
        tokenizer, model = ctx.lease.get(self.MODEL_ID).obj
        return FourClassTextClassifier(tokenizer, model, ctx.device)


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
    VOCABULARY_SIZE = 151936  # Both pinned output heads include padding beyond the tokenizer's vocabulary.
    spec = text_spec("txt.binoculars", "Binoculars zero-shot (Qwen2.5-1.5B observer / 1.5B-Instruct performer)", InputUnit.DOCUMENT,
                     model_ids=(OBSERVER, PERFORMER), timeout_s=180.0, profiles=FULL_DEEP, version="0.4.1")

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return tuple(ActivationUse(m, 1, "token sequence", f"One sequence at a time, at most {CTX} tokens including optional EOS anchor",
                                   ("Both models' vocabulary logits and loss intermediates are retained; peak remains an estimate.",))
                     for m in (self.OBSERVER, self.PERFORMER))

    def applicable(self, ctx: RunContext) -> Applicability:
        return has_sentences(ctx)

    def run(self, ctx: RunContext) -> DetectorResult:
        doc, sents = document_input(ctx)
        tokenizer, observer = ctx.lease.get(self.OBSERVER).obj
        other_tokenizer, performer = ctx.lease.get(self.PERFORMER).obj
        self._check_vocabulary(tokenizer, other_tokenizer)
        b = Binoculars.from_parts(tokenizer, observer, performer, ctx.device)
        ts = b.token_scores(doc.text, check=ctx.deadline.check)
        self._check_source_tokenization(other_tokenizer, doc.text, ts)
        return self._result(ctx, doc, sents, ts, tokenizer.eos_token_id, other_tokenizer.eos_token_id,
                            revisions(ctx, self.OBSERVER, self.PERFORMER))

    def run_after_memory_refusal(self, ctx: RunContext, models: ModelManager, *, wait_s: float) -> DetectorResult | None:
        terms = models.lease_terms((self.OBSERVER, self.PERFORMER), prefer=str(ctx.device))
        if terms.device != "cpu" or terms.dtype != "float32" or settings.environment != "dev":
            return None

        import torch

        check = ctx.deadline.check
        doc, sents = document_input(ctx)
        allowances = {model_id: models.activation_bytes_for_items({model_id: 1})
                      for model_id in (self.OBSERVER, self.PERFORMER)}

        def borrow(model_ids, reserved=0):
            remaining = ctx.deadline.remaining()
            return models.lease(model_ids, prefer="cpu", activation_bytes=reserved,
                                wait_s=wait_s if remaining is None else min(wait_s, remaining))

        with borrow((self.OBSERVER,)) as lease:
            check("paired tokenizer lease")
            tokenizer = lease.get(self.OBSERVER).obj[0]
        runtime = Binoculars.from_parts(tokenizer, None, None, "cpu")
        pins = {}
        counterpart = None

        @torch.inference_mode()
        def score_chunk(ids):
            nonlocal counterpart
            reserved = ids.numel() * self.VOCABULARY_SIZE * 4
            # The observer logits stay reserved even after its weights can be evicted.
            with borrow((), reserved):
                observed = performed = loaded = None
                try:
                    check("paired writing observer lease")
                    with borrow((self.OBSERVER,), allowances[self.OBSERVER]) as observer_lease:
                        check("paired writing observer forward")
                        loaded = observer_lease.get(self.OBSERVER)
                        pins[self.OBSERVER] = str(loaded.revision)
                        observed = loaded.obj[1](ids).logits
                        loaded = None
                        check("completed paired writing observer forward")
                    if observed.device.type != "cpu" or observed.numel() * observed.element_size() > reserved:
                        raise ValueError("Paired writing logits exceed the reserved CPU tensor")
                    check("paired writing performer lease")
                    with borrow((self.PERFORMER,), allowances[self.PERFORMER]) as performer_lease:
                        check("paired writing performer forward")
                        loaded = performer_lease.get(self.PERFORMER)
                        pins[self.PERFORMER] = str(loaded.revision)
                        if counterpart is None:
                            counterpart = loaded.obj[0]
                            self._check_vocabulary(tokenizer, counterpart)
                        performed = loaded.obj[1](ids).logits
                        loaded = None
                        check("completed paired writing performer forward")
                        losses = _losses_from_logits(observed, performed, ids)
                        check("completed paired writing loss reduction")
                        return losses
                finally:
                    observed = performed = loaded = None

        ts = runtime.token_scores(doc.text, check=check, score_chunk=score_chunk)
        if counterpart is None:
            result = DetectorResult.not_applicable(f"fewer than {MIN_DOCUMENT_TOKENS} scored tokens")
        else:
            self._check_source_tokenization(counterpart, doc.text, ts)
            result = self._result(ctx, doc, sents, ts, tokenizer.eos_token_id, counterpart.eos_token_id, pins)
        result.device, result.dtype = terms.device, terms.dtype
        return result

    @staticmethod
    def _check_vocabulary(tokenizer, other_tokenizer):
        # Cross-entropy compares the same vocabulary positions. A matching excerpt
        # alone is insufficient if the paired tokenizers map other tokens differently.
        if tokenizer.get_vocab() != other_tokenizer.get_vocab():
            raise ValueError("The paired text check requires identical vocabulary IDs")

    @staticmethod
    def _check_source_tokenization(other_tokenizer, text, ts):
        counterpart = other_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        if counterpart["input_ids"] != ts.source_token_ids or [tuple(o) for o in counterpart["offset_mapping"]] != ts.offsets:
            raise ValueError("The paired text check tokenizers disagree on the actual source input")

    def _result(self, ctx, doc, sents, ts: TokenScores, anchor, paired_eos, model_revisions):
        score, n_tokens = document_score(ts)
        if score is None:
            return DetectorResult.not_applicable(f"fewer than {MIN_DOCUMENT_TOKENS} scored tokens")
        ctx.deadline.check("sentence spans")
        records = sentence_score_records(ts, sents)
        per_sentence = [record["score"] for record in records]
        units = [UnitScore(f"sentence:{i}", sc, "score") for i, sc in enumerate(per_sentence) if sc is not None]
        aggregate = UnitScore("global", score, "score", quality={"tokens": float(n_tokens)})
        machine_like = score < settings.binoculars_threshold
        contexts = {c["id"]: c for c in ts.input_contexts}
        windows = []
        omitted = []
        for sentence, record in zip(sents, records, strict=True):
            if record["score"] is None:
                omitted.append({"unit_ref": f"sentence:{sentence.idx}", "reason": "Fewer than 12 scored tokens, including the recorded neighbour fallback."})
                continue
            if ts.evidence_limit_exceeded:
                omitted.append({"unit_ref": f"sentence:{sentence.idx}", "reason": "Exact token context recording exceeded the 262144-token evidence bound; this local score is not presented as source-mapped evidence."})
                continue
            selected = [contexts[key] for key in record["context_ids"]]
            # Keep selected public records bounded without changing inference or
            # pretending a truncated input was the context that produced a score.
            if not selected or len(selected) > 16:
                omitted.append({"unit_ref": f"sentence:{sentence.idx}", "reason": "This score spans more than 16 recorded token contexts or has no exact context; local input proof is omitted."})
                continue
            lo = min(c["spans"][0]["start"] for c in selected)
            hi = max(c["spans"][0]["end"] for c in selected)
            windows.append({**record, "unit_ref": f"sentence:{sentence.idx}",
                            "focus": {"start": sentence.start, "end": sentence.end},
                            "spans": [{"start": lo, "end": hi}], "input_sha256": text_hash(doc.text[lo:hi]),
                            "score_kind": "perplexity_ratio", "threshold": settings.binoculars_threshold,
                            "score_exact_hex": float(record["score"]).hex(),
                            "threshold_exact_hex": float(settings.binoculars_threshold).hex(),
                            "mean_negative_log_likelihood_exact_hex": float(record["mean_negative_log_likelihood"]).hex(),
                            "mean_cross_entropy_exact_hex": float(record["mean_cross_entropy"]).hex(),
                            "ai_when": "lt", "is_calibrated": False,
                            "input_semantics": "Source envelope of the recorded rolling contexts. Exact paired-model input is the committed token-ID sequence, not a re-tokenized excerpt.",
                            "shared_input_anchor_id": anchor,
                            "paired_tokenizer_default_eos_ids": [anchor, paired_eos],
                            "anchor_policy": "The first tokenizer's explicit leading token is supplied in the same input sequence to both models; their default end tokens need not agree.",
                            "token_coverage": {"tokens": record["tokens_scored"], "covered_tokens": record["tokens_scored"],
                                               "token_windows": len(selected)}})
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
            model_revisions=model_revisions,
            details={"text_evidence": {"version": "scored-context/2", "offset_unit": "unicode_codepoint",
                     "extracted_text_sha256": text_hash(doc.text), "join": "single_source_span",
                     "shared_tokenization_verified": True, "windows": windows, "contexts": ts.input_contexts,
                     "omitted_windows": omitted, "unscored_first_token": ts.unscored_first_token,
                     "document_score": {"score": score, "score_kind": "perplexity_ratio", "tokens_scored": n_tokens,
                                        "threshold": settings.binoculars_threshold, "ai_when": "lt", "is_calibrated": False}}})
