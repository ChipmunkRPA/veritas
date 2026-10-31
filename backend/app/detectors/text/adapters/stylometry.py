"""v0.3 stylometric rules as a v2 plugin (model-free, CPU).

- **Scores.** The document score is `document_style_probability` ([0.2, 0.8] by construction) and each flagged sentence
  scores `sentence_style_probability`. Both are hand-set rules, reported as uncalibrated probabilities.
- **Exhibits.** A text-spans exhibit lists the stylistic tells per flagged sentence; a table lists the LLM-favoured
  vocabulary hits. The legacy fuser reads both.
"""
from __future__ import annotations

from app.detectors.text import stylometry
from app.detectors.text.adapters.common import CPU_ONLY, document_input, has_sentences, text_spec
from app.engine.detector import Applicability, Detector, DetectorResult, InputUnit, RunContext, UnitScore

SPANS_TITLE = "Stylistic tells per sentence"
LEXICON_TITLE = "LLM-favoured vocabulary hits"


class StylometryDetector(Detector):
    spec = text_spec("txt.stylometry", "Stylometric profile (v0.3 rules)", InputUnit.SENTENCES, ("text_spans", "table"),
                     lane="cpu", devices=CPU_ONLY, timeout_s=30.0)

    def applicable(self, ctx: RunContext) -> Applicability:
        return has_sentences(ctx)

    def run(self, ctx: RunContext) -> DetectorResult:
        _doc, sents = document_input(ctx)
        metrics = stylometry.document_metrics([s.text for s in sents])
        p, reasons = stylometry.document_style_probability(metrics)
        units: list[UnitScore] = []
        spans = []
        for s in sents:
            flags = stylometry.sentence_flags(s.text)
            ps = stylometry.sentence_style_probability(flags)
            if ps is None:
                continue
            units.append(UnitScore(f"sentence:{s.idx}", ps, "probability", quality={"tells": float(len(flags))}))
            spans.append({"start": s.start, "end": s.end, "sentence_idx": s.idx, "probability": ps, "label": "stylistic tells",
                          "reasons": flags})
        exhibit_ids = []
        if spans:
            exhibit_ids.append(ctx.exhibits.add(
                "text_spans", title=SPANS_TITLE, media_ref="document", spans=spans,
                caption="Sentences with LLM-favoured vocabulary, contrast constructions, transition openers, rule-of-three lists "
                        "or repeated em-dashes. Hand-set v0.3 rules, not a trained model."))
        hits = metrics["lexicon_hits"]
        if hits:
            exhibit_ids.append(ctx.exhibits.table(["phrase", "count"], [[phrase, float(n)] for phrase, n in hits.items()],
                                                  title=LEXICON_TITLE, media_ref="document"))
        aggregate = UnitScore("global", p, "probability")
        numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, int | float) and not isinstance(v, bool)}
        return DetectorResult(
            status="ok", unit_scores=units, aggregate=aggregate,
            observed=f"lexicon {metrics['lexicon_weighted_per_1000']}/1000w, contrast {metrics['contrast_per_1000']}/1000w, "
                     f"transition openers {metrics['transition_opener_pct']}%, sentence-length CV {metrics['sentence_len_cv']}, "
                     f"MATTR {metrics['mattr']}; {len(units)} sentence(s) with tells",
            reference="Human prose: lexicon < 3/1000, openers < 8%, CV > 0.5",
            reasoning="; ".join(reasons) + ".", metrics=numeric, exhibit_ids=exhibit_ids)
