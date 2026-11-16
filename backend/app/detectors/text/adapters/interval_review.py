"""Read isolated sentences for the optional plain-text passage review."""
from __future__ import annotations

import math

from app.core.canonical import FLOAT_DP
from app.core.config import settings
from app.detectors.text import language_scope
from app.detectors.text.adapters.common import FULL_DEEP
from app.engine.boundary import DetectorTimeout
from app.evidence import document_matched_context, document_passage_intervals

KEY = document_passage_intervals.KEY
EXTRA_SECONDS = 18.0


def enabled() -> bool:
    return (settings.environment == "dev" and settings.text_passage_intervals_review
            and settings.text_detector_mode == "candidate" and settings.text_window_sentences == 3)


def measure(classifier, doc, sentences, context, ctx, spec, revisions):
    if (not enabled() or ctx.profile not in FULL_DEEP or doc.meta.get("method") != "plain"
            or ctx.outputs.document.get(language_scope.KEY) != language_scope.ENGLISH
            or not document_passage_intervals.MIN_SENTENCES <= len(sentences) <= document_passage_intervals.MAX_SENTENCES
            or not document_matched_context.matches(spec.id, spec.version, revisions)):
        return None
    geometry = [[sentence.start, sentence.end, sentence.n_words] for sentence in sentences]
    remaining = ctx.deadline.remaining()
    readings = None
    if remaining is None or remaining >= EXTRA_SECONDS + 2:
        deadline = ctx.deadline.nested(EXTRA_SECONDS)
        try:
            values = classifier.predict([sentence.text for sentence in sentences], batch=12, check=deadline.check)
            deadline.check("separate writing passage responses")
            if (len(values) == len(context) == len(sentences)
                    and all(math.isfinite(value) and 0 <= value <= 1 for value in (*values, *context))):
                # Select from exactly the values the canonical evidence record will retain.
                readings = {"context": [round(float(value), FLOAT_DP) for value in context],
                            "isolated": [round(float(value), FLOAT_DP) for value in values]}
        except (DetectorTimeout, OSError, RuntimeError, ValueError):
            # The existing context and document scores are already available.
            pass
    return document_passage_intervals.capture(doc.text, doc.source_map, geometry, readings)
