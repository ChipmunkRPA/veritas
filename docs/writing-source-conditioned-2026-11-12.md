# Testing earlier-source likelihood on edited passages

Two source-conditioned checks found **four of eight** known generated edits missed by the current passage check. Each also flagged **three of 14** ordinary work groups. Simple lexical edit distance also found four edits at that false-alarm budget, with a different set of locations. The tested likelihood signals stay outside the app. [Held results and source commitments](evaluation-panels/writing-source-conditioned-2026-11-12.json)

## Dated controls and known edits

The intake reopened retained publisher XML and admitted 14 previously unused work groups with material body revisions. Seven came from Wellcome Open Research, six from Gates Open Research and one from F1000Research. Both dates and versions were verified, with the same recorded byline and explicit article-level CC BY 4.0. Exact unchanged sentences bracket one or two changed sentences. The held works share no admitted author or normalized 12-word overlap with the comparison's earlier source inventory. Fifteen separate previously used works supply genuine-only calibration. The three journals share a publisher platform. Earlier editing tools and checkpoint pretraining exposure remain unknown. [Prior source intake and rights](writing-clipped-rank-transfer-2026-11-12.md#matched-controls) · [Journal version and license qualification](writing-journal-body-transfer-2026-11-11.md#paired-writing-checks-on-dated-journal-body-revisions)

The held comparison includes all 14 genuine earlier and 14 genuine revised passages. Two pinned Apache 2.0 generators made eight attempts per work in the initial lane. Before any scoring, a minimal copyediting prompt supplied another eight attempts only for work-generator combinations with no qualifying output. Admission held the genuine later passage's sentence count, bounded word count and reference-relative edit distance, kept visible numeric strings, and preserved both exact source flanks. It kept the first qualifying completion. Granite qualified two cases in the initial lane and three in the second. OLMo qualified three in the first and zero in the second. The 408 attempts produced eight scored generated edits, below the frozen minimum of six per family. Exact factual equivalence beyond the checked geometry and numeric strings was unverified. [Granite terms](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct) · [OLMo terms](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct)

## Frozen comparison

For each known changed passage, a pinned local causal model reads the same target tokens three times. The first view supplies only the unchanged left and right sentences. The second adds the actual earlier passage. The third adds an unrelated earlier passage, chosen deterministically by closest length with same-journal preference from the same data split. One held F1000Research group has no same-journal donor. Mean target negative log likelihood yields two differences. The first measures the improvement when adding the real earlier passage to the flanks. The second measures the improvement over the wrong-source context. The lexical comparison uses normalized word edit distance on the same bound target. Recent conditional-response research uses a generated semantic condition and supervised features. This test uses verified earlier publisher text, fixed arithmetic and no paper code or datasets. [Conditional-response research](https://arxiv.org/html/2610.38312v1) · [Pinned Apache 2.0 causal checkpoint](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323)

Each candidate must exceed the second-highest genuine calibration work. The existing comparison keeps the first six positive native contexts. A useful mark covers at least 50% of the edited non-whitespace characters at 30% intersection over union. Each candidate adds only the verified destination span. Release requires three additional useful work groups across both generators, sufficient admissions, at most one additional ordinary flag group, and a gain over lexical distance at the same ordinary false-group budget. These are review criteria for this test. No method affects whole-document confidence.

## Held results

| Check | Known generated locations | Ordinary work groups flagged |
| --- | ---: | ---: |
| Current passage check | 0/8 | 0/14 |
| Add fixed lexical distance | 0/8 | 0/14 |
| Add likelihood improvement over no reference | 4/8 | 3/14 |
| Add likelihood improvement over wrong reference | 4/8 | 3/14 |
| Lexical-distance sweep at the same held false-group budget | 4/8 | 3/14 or fewer |

Each causal rule found one Granite and three OLMo edits from three works. Genuine revisions also became easier to predict when their actual earlier text was available. The lexical sweep is a diagnostic on the held cases and supplies no new operating point. Both causal rules exceeded the frozen false-alarm allowance, fell short of the required generator sample, and matched the simpler lexical control. The results concern known earlier drafts and the specific revision shapes in this panel.

## Replay

The public reader recalculates the genuine-only points, useful locations, ordinary-group counts and matched-budget lexical comparison from saved responses. The source ledger records the exact publisher URLs, byte digests and native positions. The separate local replay reopened 28 held XML files and verified 99 source sentences, 408 generation attempts, 153 causal inputs and 172 primary context occurrences across 51 cases. The local worker made 153 Metal causal forwards and 43 CPU classifier forwards over 171 unique native inputs. Exact prose, token IDs and generated text stay local.

```sh
python3 -B -I -S backend/scripts/review_writing_source_conditioned.py \
  docs/evaluation-panels/writing-source-conditioned-2026-11-12.json
```

To rebuild inputs privately, retrieve each exact XML witness using the receipt's DOI, URL and file digest. Recover the held source sentences from the recorded JATS blocks and ranges using the [journal source reader](../backend/scripts/writing_journal_body_holdout.py). For the earlier calibration cohorts, use the [retained clipped-rank intake](writing-clipped-rank-transfer-2026-11-12.md#replay). The receipt includes the prompt templates, model and file pins, and seed families. Join the unchanged flanks and admitted target using single spaces. Use the [existing causal token coverage helper](../backend/scripts/writing_token_curvature.py) and 4-bit affine quantization with a group size of 64. Require identical target token IDs in all three views before calculating mean negative log likelihood.
