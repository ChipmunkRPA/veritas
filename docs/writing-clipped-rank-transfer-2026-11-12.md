# Clipped token rank on matched journal rewrites

The current passage check located **1 of 8** generated rewrites and flagged **1 of 10** genuine work groups. Clipped token rank located **0 of 8** and flagged **3 of 10**. Adding it raised the number of flagged genuine groups to four and recovered no edits. The two previously missed PDF passages also remained unmarked. The tested alternatives stay in research. [Scalars, source commitments and replay](evaluation-panels/writing-clipped-rank-transfer-2026-11-12.json)

## Matched controls

The source qualification reopens 178 retained version pairs and 48 additional pairs drawn in fixed hash order from the earlier complete publisher catalogs. All 96 new publisher XML files were acquired. Each qualified work supplies dated v1 and v2 XML with the same byline, explicit CC BY 4.0, and exact positions for a revised sentence. The earlier and later replacement runs contain the same number of sentences, at most four, with unique identical sentences just outside the run. The comparison places the verified later sentence into its earlier three-sentence context. This keeps the displayed context identical across the earlier, genuine revised, and generated variants. The genuine revision is authentic publisher text in a controlled layout. [Earlier journal intake and rights](writing-journal-body-transfer-2026-11-11.md)

Five retained groups set the operating points. Ten separately acquired groups, four Gates and six Wellcome, supply the held controls. The normalized lexical revision extent in held groups ranges from 0.289 to 0.727. The intake excludes works and recorded bylines from the earlier paired panels and screens for exact 12-word overlap in their registered excerpts. Historical editing tools and checkpoint pretraining exposure remain unknown.

Two pinned Apache 2.0 generators made eight attempts per work, for 240 attempts. Admission keeps the earliest complete single-sentence rewrite within three lexical words of its genuine counterpart, within 0.15 of the same reference-relative edit distance, with the same numeric strings and unchanged native flanks. The generation panel yields three Granite and five OLMo held variants, below the frozen minimum of six per family. Numeric preservation is checked. Complete factual fidelity is unverified. [Granite terms](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct) · [OLMo terms](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct)

## Frozen comparison

A base language model assigns vocabulary logits to the observed tokens in their exact native three-sentence input. The center and following sentence have full causal token coverage. The first sentence lacks a predictive prefix at its first token and remains outside the causal arms. The raw log-rank score is negative mean log rank. The clipped arm caps high log ranks at the genuine selection tokens' 90th percentile before averaging. The likelihood-to-rank ratio divides total token surprisal by total log rank. Its clipped form separately caps both contributions at the selection percentiles. All score directions stay fixed before held scoring. The clipping study motivates robust reduction of unusual tokens, and the short-text calibration study motivates keeping the narrow comparison separate from longer document claims. [Robust detection under contamination](https://arxiv.org/abs/2609.29935) · [Length-conditional calibration](https://arxiv.org/abs/2611.10901) · [Pinned Apache 2.0 base checkpoint](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323)

The current check uses its existing first-six raw context responses. Each alternate arm uses a strict point above its second-highest genuine selection work maximum and can mark up to two exact sentence foci. A useful location covers at least half of the known edited non-whitespace characters at 0.30 intersection over union. The primary additive comparison requires three extra works across both generation families, at least six cases per generator, preservation of current useful locations, and at most one additional genuine work flagged.

The MIT-declared TMR model is a secondary comparator on exact sentence foci, with its own genuine selection point. An earlier test rejected this checkpoint as a replacement on an exposed mixed-text panel. Its weights and the RAID dataset both carry publisher MIT declarations. This trial uses fixed published weights. The exact training subset and source-specific terms beyond the publisher declarations remain unverified. [Earlier model comparison](writing-focus-v3-development-result.md) · [Pinned model card](https://huggingface.co/Oxidane/tmr-ai-text-detector/blob/0ceddea903015ef99cbaa040a4d8a216aed9c683/README.md) · [RAID dataset card](https://huggingface.co/datasets/liamdugan/raid)

## Held results

| Method | Useful rewrites / 8 | Genuine work flags / 10 | Higher center response than genuine revision / 8 |
| --- | ---: | ---: | ---: |
| Current context | 1 | 1 | 6 |
| Raw log rank | 0 | 3 | 2 |
| Clipped log rank | 0 | 3 | 3 |
| Raw likelihood-to-rank ratio | 1 | 4 | 4 |
| Clipped likelihood-to-rank ratio | 0 | 2 | 3 |
| TMR focus | 2 | 8 | 7 |

The raw ratio found one additional OLMo edit with three additional genuine groups flagged. TMR found two additional OLMo edits and flagged seven additional genuine groups. The clipped alternatives recovered no extra locations. All arms missed both exposed diagnostic foci taken from the actual frozen PDF extraction, including the scanned passage. Those inputs were outside selection and held estimation. [Original PDF diagnostic](writing-pdf-page-coverage-2026-11-12.md)

## Replay

The standard-library reviewer checks the frozen clipping percentiles, each alternate operating point, native overlap, paired ranks and all 484 decisions. The public receipt keeps exact source references, offsets, aggregate token reductions and hashes. Exact prose, token IDs, per-position observations and full vocabulary tensors stay local.

~~~sh
python3 -I -S backend/scripts/review_writing_clipped_rank_transfer.py \
  docs/evaluation-panels/writing-clipped-rank-transfer-2026-11-12.json
~~~

The local independent replay also reopened 30 publisher versions, checked 240 generation-attempt bindings, 44 exact case positions and 132 native contexts, then recomputed 2,719 token reductions. It independently measured rank and surprisal in one saved full vocabulary row for each cohort. The recorded workload used 44 Metal causal forwards, 33 CPU incumbent forwards and 20 CPU TMR forwards.

To rebuild the private inputs, use the committed version URLs, XML hashes, block hashes and focus offsets in the receipt with the [existing publisher parser](../backend/scripts/writing_journal_body_holdout.py). The receipt includes the generation settings, source commitments, seeds, checkpoint pins and numerical runtimes. The [existing causal runner](../backend/scripts/writing_token_curvature.py) supplies the native tokenizer-offset binding and in-memory model quantization. The current-model scores use the [existing complete-context CPU scorer](../backend/scripts/writing_paired_email_revision.py). The public replay verifies the saved decisions without any model download.
