"""Findings synthesis v2 (ARCHITECTURE §4.5–4.6; IF §6.1–6.4; UX §5.4, §7; PRD EA-06, EA-07; D-16, D-26).

- `engine`: ranked findings from signals, exhibits, assessments, the decision, the condition profile and card fields.
- `strength`: likelihood-ratio strength bands, family caps, corroboration and review priority.
- `render`: the versioned Jinja templates (`templates/<family>/<scope>.j2`, template set `tpl-2.17`); one sentence per line,
  each carrying its evidence ids.
- `vocabulary`: the banned-term lint (PRD RP-02 ∪ IF §6.2 ∪ UX §7.2).

No generative model is involved: every sentence is a template filled from recorded values.
"""
from app.engine.synthesis.engine import TEMPLATE_SET, SynthesisInput, synthesise

__all__ = ["TEMPLATE_SET", "SynthesisInput", "synthesise"]
