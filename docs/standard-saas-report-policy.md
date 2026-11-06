# Standard SaaS report policy - 2026-11-05

The current customer presentation is `evidence-led-assessment/v4`, with service policy `standard-saas-evidence/v2`. It makes no public fixed-threshold or minimum-accuracy claim. Internal decision, calibration and abstention safeguards remain in place. Removing the advertised cutoff does not create a conclusion from a failed check or improve detection accuracy.

Each finding separates its observed evidence from its interpretation, links the recorded source locations or checks, and uses one of four states: Leans AI, Leans original, Inconclusive, or Context only. File history, metadata, timing and other contextual observations cannot by themselves establish AI generation or original authorship.

A finding percentage requires a matching reference calibration for that check and its assessed input. Unsupported inputs, absent calibration, conflicting directions, incomplete checks and ambiguous combinations receive no numerical finding confidence. Raw responses and contribution strengths are not confidence percentages. A global estimate must not be assigned to a crop, pixel, frame, time window, sound or quoted passage without calibration for that scope.

The overall assessment preserves the existing calibrated combination and its uncertainty gates; it does not average finding percentages. Findings identify whether they enter that combination or were assessed separately. Coverage notes explain the included scope and excluded channels. Unsupported combinations show no reliable overall conclusion. Original-leaning evidence does not certify every part of a file as original.

Every standard report and ordinary result view states:

- The cloud service provides preliminary automated screening with limited scientific reliability.
- The SaaS report is not prepared for court filing, proof of authenticity or expert testimony and must not be relied on as court evidence.
- A privately researched report, expert opinion or litigation support requires a separate signed engagement with the Chipmunk Verify expert.
- Scope, methods, fees and deliverables are agreed privately through sales@chipmunkrpa.com. This is separate from automated-analysis checkout.
- A private engagement does not guarantee court admissibility.

These are product-scope and engagement requirements, rather than a claim that a contract establishes evidential reliability. In U.S. federal proceedings, Rule 702 requires sufficient facts or data, reliable methods and reliable application to the case; contracting alone does not establish these matters. [Official Federal Rules of Evidence, Rule 702](https://www.uscourts.gov/sites/default/files/document/federal-rules-of-evidence.pdf).

The policy is versioned in the backend and included in the customer assessment in API/JSON responses. Its `finding_assessments` list carries per-finding interpretation, confidence availability, scope and inclusion status, keyed by the existing finding IDs. The tested browser mirror uses the same policy and interpretation rules. New and regenerated PDF reports use template set `veritas-report-2.0.r74`; older signed report bytes remain unchanged and are not silently rewritten. Source hashes, timestamps, source coordinates and recorded evidence remain unchanged.

The November 4 evaluation remains an 80% evaluation. Its counts are not evidence for the present customer interpretation. Before the preceding policy change, all 409 frozen production source/calibration/report files were verified and archived at `backend/data/independent-evaluation/rule80-v1/frozen-production-source.tar.gz`; the earlier shared fixtures were preserved alongside it. The original published study and per-case outcomes remain unchanged. Current-code replay correctly rejects a mismatched historical code freeze. The intermediate presentation fixtures are retained under `backend/data/policy-history/rule60/`.

Local verification covers retained internal decision boundaries, rounding, inconclusive and missing-contributor gates, independent finding calibration and scope, JSON evidence immutability, frontend/backend presentation parity, policy text across all four media and report formats, and a rendered PDF from a recorded evaluation input. A read-only presentation replay uses the 288 existing result files; it is not a new accuracy benchmark or model inference run. No model weights or calibration fits are changed by this work.
