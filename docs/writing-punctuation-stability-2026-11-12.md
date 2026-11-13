# Punctuation stability in passage checks

A punctuation stability guard removed three ordinary passage flags and lost two of seven correctly located generated edits. Requiring the same stability when two checks agreed lost one of five known edit locations and cleared no additional ordinary cases. These results leave the passage checks unchanged. [Held responses and locations](evaluation-panels/writing-punctuation-stability-2026-11-12.json)

## Frozen comparison

The 108 files come from 31 work groups outside the scan experiment. The corpus contains 54 eLife scientific article excerpts, 26 Federal Register notices, 19 Wellcome excerpts and nine Gates excerpts. The 68 ordinary controls include published originals, genuine publisher revisions and constructed same-work substitutions. Four locally run, commercially permitted generators supplied 40 known edits. Earlier experiments used these texts for different questions. No score from the prior OCR comparison entered the study, and the model weights and points were fixed before inference. [eLife source commitments and terms](writing-elife-exact-sentence-confirmation-2026-11-12.md) · [Notice source qualification](writing-practical-notice-transfer-2026-11-11.md) · [Dated journal sources and checkpoint terms](writing-clipped-rank-transfer-2026-11-12.md)

For each file, the frozen variants substitute a spaced dash glyph with an ASCII hyphen or an em dash, convert curly quote and apostrophe glyphs to ASCII, or curl an ASCII apostrophe between letters. Word spelling, whitespace and sentence boundaries stay fixed. The app's native segmenter verified the same sentence offsets and paragraph membership before scoring. Sixty-three files contained eligible glyphs, including 26 generated and 37 ordinary files across three affected source cohorts. The 19 Wellcome files provide unchanged controls in this trial.

The current comparison retains the first six positive native contexts in source order. The guard retains an existing flag only if every complete changed view of that same context stays positive. It keeps existing flags when required responses are unavailable. There is no backfill, and the source ranges stay unchanged. A separate experimental arm repeats this rule on same-window agreement using the two existing checkpoints, including layouts outside the locally enabled second-check scope. A useful location covers at least half of the known edited non-whitespace characters with at least 30% intersection over union. [Current optional second-check scope](writing-checkpoint-agreement-2026-11-11.md#local-app)

The frozen gate requires zero useful generated files lost, ordinary false groups cleared in at least two cohorts, and sufficient changed examples across generators. Each method contributes zero whole-document confidence.

## Held results

| Outcome | Current contexts | With stability guard | Same-window agreement | Agreement with guard |
| --- | ---: | ---: | ---: | ---: |
| Useful known generated locations, out of 40 | 7 | 5 | 5 | 4 |
| Generated files flagged only at an insufficient location, out of 40 | 1 | 2 | 1 | 1 |
| Ordinary files flagged, out of 68 | 6 | 3 | 1 | 1 |
| Ordinary work groups flagged, out of 31 | 3 | 2 | 1 | 1 |
| Useful known locations on 26 files with an eligible typography change | 7 | 5 | 5 | 4 |

Of 1,564 original context occurrences, 315 had at least one changed version. The primary response changed from positive to negative in seven context occurrences. Four reused one exact context in related files. The guard lost one Granite edit in a Gates excerpt and one SmolLM2 edit in a Federal Register notice. The 27 selected secondary inputs produced no directional changes. These counts describe the tested punctuation pairs and their matched controls.

## Replay

The portable reviewer recomputes all 108 outcomes, the frozen advance gate, and every recorded directional transition. A separate local replay reopened all input texts and verified 195 single-codepoint substitutions, 7,820 exact context commitments, and every known edit overlap. The first checkpoint made 107 CPU forwards over 850 distinct complete inputs. The second made seven forwards over 27 distinct complete inputs. The published receipt omits source prose, generated text and model weights.

```sh
python3 -B -I -S backend/scripts/review_writing_punctuation_stability.py \
  docs/evaluation-panels/writing-punctuation-stability-2026-11-12.json
```

To rebuild the private inputs, use the eLife run's `plan.json`, the clipped-rank run's `cases.json` with `cohort=held`, and the notice run's `private-controls.jsonl` and `private-positives.jsonl` with `status=constructed`. The receipt binds those exact source files. Apply the frozen typography substitutions in the recorded code-point positions, then run the app's [native extraction](../backend/app/detectors/text/extract.py) and [sentence segmenter](../backend/app/detectors/text/segment.py). Form each input by joining its original neighboring sentences with single spaces. The [primary classifier](../backend/app/detectors/text/classifier.py) and [second check](../backend/app/detectors/text/adapters/agreement.py) use their serving input caps and CPU microbatches of eight and four, respectively. Every scored source position remains the original half-open Unicode range.
