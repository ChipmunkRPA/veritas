# Frozen exact-sentence check on held eLife articles

The frozen exact-sentence rule located **0 of 24** AI rewrites from 15 held eLife works. The incumbent context check located **1 of 24**. Adding the exact-sentence rule recovered no additional edits and flagged **3 of 15** ordinary work groups. The candidate missed the predeclared follow-up requirements and remains in research. [Public results and source bindings](evaluation-panels/writing-elife-exact-sentence-confirmation-2026-11-12.json)

## Source qualification

The independent sentence-only check previously located two of 14 rewrites on an archived technical-blog panel. This follow-up holds the exact checkpoint, sentence eligibility, winner selection and comparison point fixed. It measures a different publisher and two different offline generation families. The exact-sentence point comes from the earlier fitting articles. eLife responses never enter selection or calibration. [Earlier comparison and point](writing-multiblock-independent-2026-11-12.md)

The source-only preflight reopened 30 retained eLife final JATS files. Of those, 25 had an exact DOI and recorded byline, matched the archived file bytes, and carried explicit article-level CC BY 4.0 terms. Fifteen supported four to six consecutive normalized English paragraphs with an eligible interior sentence. Their publication records span 2014–2019. The selected original prose had no normalized exact 12-word overlap across its groups or with 11 earlier packets, including all works used in the technical-blog trial. Bibliography-reference nodes were omitted before normalization. Training exposure for any checkpoint remains unknown. Each source supplies an unchanged original and a matched ordinary substitution from elsewhere in the same paper. The review binds the donor to the original JATS block and native character positions. [eLife terms](https://elifesciences.org/terms) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The freeze fixes three attempts per source and generator, all seeds and prompt settings, the two pinned checkpoints, and the unchanged comparison point. The earliest completion meeting the exact one-sentence, length, copy and source-geometry rules supplies each rewrite. Admission never sees detector responses. All 90 attempts completed, with 13 admissible Granite rewrites and 11 OLMo rewrites. The 30 ordinary controls and 24 generated variants make 54 scored articles. [Granite terms](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct) · [OLMo terms](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct)

The incumbent comparison uses its first six source-ordered positive three-sentence contexts. The independent arm reads only the exact focus sentence of 12–65 lexical words, when both immediate source neighbors contain 8–80 words. A unique highest response must strictly exceed the frozen point. Useful localization requires at least 50% of the edited non-whitespace characters and at least 0.30 intersection over union. The frozen follow-up required six admitted examples per generator, complete measured target coverage, two additional source groups across both generators, and at most one additional ordinary group. The additive comparison retains existing useful locations.

## Held results

| Outcome | Current context | Independent sentence | Both |
| --- | ---: | ---: | ---: |
| Granite edits located | 1/13 | 0/13 | 1/13 |
| OLMo edits located | 0/11 | 0/11 | 0/11 |
| Generated files flagged only at a wrong location | 0/24 | 5/24 | 5/24 |
| Ordinary originals flagged | 0/15 | 3/15 | 3/15 |
| Ordinary substitutions flagged | 0/15 | 3/15 | 3/15 |
| Ordinary work groups flagged | 0/15 | 3/15 | 3/15 |

The same three ordinary work groups flagged in both their original and substitution views. Their selected sentences were unchanged. All 24 exact generated targets were eligible and measured in a complete tokenizer window. The exact-sentence check ranked seven of the 24 targets first before applying the frozen point. All seven responses fell below that point. The earlier two gains did not transfer to this publisher and generator combination. This study covers one English scientific publisher and constructed single-sentence edits. It leaves wider mixed-writing detection unqualified.

Local inference completed 105 incumbent forwards on 419 unique native contexts and 68 independent forwards on 272 exact sentences. The portable verifier reproduced all 54 held outcomes. The native replay verified 15 original JATS files, 1,083 context occurrences, and 846 exact-focus occurrences. It also independently selected each first admissible rewrite. Four intentional changes to the point, outcome totals, a case decision and a source coordinate were rejected. The sandbox denied Metal access on the first generation launch before inference. The same frozen plan completed with local GPU access.

## Replay

The [portable receipt](evaluation-panels/writing-elife-exact-sentence-confirmation-2026-11-12.json) contains source-bound scalar responses, locations, counts and the fixed decision. The portable reviewer independently recomputes each arm and verifies the unchanged comparison point. The separate native replay checks all official file bytes, independently reselects the first eligible rewrite and verifies exact input and output coordinates. No model inference is needed for either replay.

```sh
python3 -B -IS backend/scripts/review_writing_elife_exact_sentence_confirmation.py \
  docs/evaluation-panels/writing-elife-exact-sentence-confirmation-2026-11-12.json
```

Recreating model responses requires the retained private eLife preflight, publisher source files, pinned offline model snapshots, and the existing MLX generation environment. Set `run` to a new private directory, `preflight` to the bound source and donor handoff, and `publisher` to the official final JATS folder. The runner reopens the official source files before freezing and preserves each generator group separately during generation.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.writing_elife_exact_sentence_confirmation freeze "$run" --preflight "$preflight"
for family in granite olmo; do
  "$mlx_python" -B -m scripts.writing_elife_exact_sentence_confirmation generate "$run" --family "$family" || exit 1
done
for phase in plan independent primary; do
  backend/.venv/bin/python -B -m scripts.writing_elife_exact_sentence_confirmation "$phase" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.writing_elife_exact_sentence_confirmation publish "$run" --output "$run/public.json"
backend/.venv/bin/python -B -m scripts.review_writing_elife_exact_sentence_confirmation \
  "$run/public.json" --native "$run" "$publisher"
```
