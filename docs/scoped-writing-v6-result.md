# Passage-scope combination: development result

The new scoped-consistency combination did **not** meet the improvement gates.
Keep the current passage baseline. No serving score, model, public threshold,
customer API, PDF or billing behavior changed.

All eight methods were frozen in commit `f12193bdf3ad8e9def6cb1e7f833177c0a98dff5`.
Each used the same consumed writing-v5 model responses and source-disjoint
four-role assignment: 128 connected prompt/components,32 per role,16 generators.
All eight fitted artifacts and genuine-only internal points preceded reserved
evaluation. Each reserved comparison contains 96 AI-containing constructions
and 32 human constructions. These are artificial 60-word first /100-word second
passages, not fresh articles, emails or complete documents.

| Calibration and combination | AI found /96 | False AI flags /32 | Inconclusive /128 | Source/class-balanced Brier |
|---|---:|---:|---:|---:|
| Pooled baseline |86|6|0|0.146067|
| Pooled original two-family combination |71|5|0|0.132496|
| Pooled shared-family maximum |86|6|0|0.150092|
| Pooled shared-family scoped consistency |87|7|0|0.133833|
| Length-specific baseline |60|0|0|0.137012|
| Length-specific original combination |63|3|0|0.122169|
| Length-specific shared-family maximum |60|0|0|0.154593|
| Length-specific shared-family scoped consistency |78|6|0|0.130049|

The new feature preserves information that the earlier maximum discards, but
the actual tradeoff is insufficient. In the pooled comparison it finds one
more AI-containing file and introduces one additional false flag. It also
loses one short-AI-first detection: 25/32 versus 26/32 for the pooled baseline.
Length-specific consistency recovers 18/32 short-AI-first mixtures from the
length-specific reference's 0/32, but adds six human false flags. Lower Brier
alone cannot establish improved detection at a similar false-alarm rate.
Neither scoped alternative passes the predeclared continuation gates.

All 128 reserved files were numerically assessed in every arm; an assessed
probability is not a validated statement of authorship. Rates are small-panel
development observations, not estimates of customer accuracy. Same-source
agreement can reinforce shared mistakes. The pooled reference remains a
stronger detection comparator than the failed length-specific variant.

The [full result](evaluation-panels/scoped-writing-v6-result.json),
[fit completion commitment](evaluation-panels/scoped-writing-v6-fit-complete.json)
and [replay release](evaluation-panels/scoped-writing-v6-release.json) retain all
counts, source-bound character locations, parameters, probabilities, internal
points and failures. No raw passage or customer document is republished.

Local reconstruction archived 34 dependencies from the immutable science
commit. It refitted all eight arms, compared every fitted parameter, local/file
probability, exact count and decision, and reproduced them with maximum numeric
difference 0. All eight adverse checks confirmed that changed reserved responses
cannot change fitting, and a missing required check withholds the combination
instead of producing an originality verdict. The external CI wrapper is
separate from the frozen science.

Linux fitting is **not reliably portable yet**. Of four actual runs, two
refitted all eight arms and reproduced every exact count/status/decision with
maximum numerical difference 1.1546319456101628e-14. Two failed during fitting.
The first harness captured the original child diagnostic, so its cause remains
unknown. The external wrapper was changed only to expose that diagnostic;
the immutable science, thresholds and packages were not changed. A further
run then exposed `Source-balanced monotone calibration fit did not converge`
inside the unchanged v2 scalar-head fitting method. Preserve these failures;
one success is insufficient evidence of portable training reliability.

The [actual successful receipt](evaluation-panels/scoped-writing-v6-linux-review.json)
and its recorded failed attempts distinguish numerical agreement from reliable
fitting. This is an additional reason to retain the serving baseline. A future
solver repair requires its own prospective method and comparison; do not patch
the live frozen audio/image experiments to make them pass. None of these jobs
constitutes outside human review or fresh detection validation.

Run the model-free reconstruction from a full Git checkout:

```sh
PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 python -m scripts.replay_scoped_writing_v6 /private/tmp/scoped-writing-v6-review.json
```

Use the pinned NumPy 2.2.6 /SciPy 1.18.1 versions with Python 3.12. Nothing loads a
pretrained model, raw media or voice/face embedding. No untouched source reserve
was spent to rescue this rejected development candidate.
