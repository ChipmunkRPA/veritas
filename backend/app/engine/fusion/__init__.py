"""Fusion and calibration library (ML §4–§5; ARCHITECTURE §4.4).

- `features`: Stage 1 z⁺ / z⁻, missing masks, context one-hots.
- `stacker`: Stage 2 monotone logistic stacker (L-BFGS-B) and grouped-CV λ selection.
- `calibration`: Stage 3 Platt per context with shrinkage.
- `conformal`: Stage 4 class-conditional split-conformal sets and bands.
- `contributions`: exact additive contributions.
- `bootstrap`: grouped bootstrap refits and the per-item 90 % interval.
- `disagreement`: per-detector Platt, disagreement index D and d95.
- `artefact`: fitting the `veritas.fuser/1` artefact and the runtime `Fuser`.
- `registry`: the loader (newest compatible reviewed artefact, else the fallback) and the `fusers` audit row.
- `decisive`: Stage 0 provenance overrides and the bound on other provenance states.
- `abstain`: validation slices, the validated envelope and the EA-02 cap.
- `runtime`: the published decision, assessments, per-signal calibration and calibration records.
"""
