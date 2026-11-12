# Exact independent sentence responses in longer articles

The independent sentence-only check located **2 of 14** known AI rewrites, compared with **0 of 14** for the existing context rule. Its two hits came from different works and generators. It flagged **2 of 10** ordinary source groups, compared with **5 of 10** for the existing rule. The separately frozen combined rule located **1 of 14** and missed its release criterion. [Public receipt](evaluation-panels/writing-multiblock-independent-2026-11-12.json)

## Sources and frozen comparison

This study reuses 55 known-origin, multi-paragraph English articles from the earlier Kubernetes archive trial. The 12 older works supply fitting data, and ten distinct works supply the held readout. The source text and generated edits had prior project exposure under other rules. This comparison measures a new set of independent inputs. It adds 14 same-work sentence substitutions as ordinary controls, seven per partition. Each donor is the closest-length eligible sentence from the recorded excerpt elsewhere in the same article. Exact text and segmentation are unchanged outside the target. An unchanged quote or a donor already present in the article cannot supply a control. These are constructed ordinary edits made with genuine archived prose. [Earlier source audit](writing-multiparagraph-localization-transfer-2026-11-12.md) · [Kubernetes terms](https://raw.githubusercontent.com/kubernetes/website/main/LICENSE)

The incumbent is the first six native three-sentence inputs that meet its raw review point. The new arms rank exact sentences with one preceding and one following sentence. These are the same native neighbors even when the source crosses a paragraph break. The separate classifier reads the exact focus, each neighbor separately, the full three-sentence input, and the two neighbors with the focus omitted. Omission measures how much the saved response changes when text is removed. Length, coherence and punctuation can contribute to that change. Recent research on hybrid writing tests how local context affects the scores used to propose source locations. The study uses its own source-bound measurements and the two installed pinned Apache-2.0 checkpoints. [LA-CPD](https://arxiv.org/abs/2609.33787) · [Primary checkpoint](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small) · [Independent checkpoint](https://huggingface.co/rasbt/ai-text-detector-modernbert)

The four rankers use the incumbent response, the independent isolated sentence response, the independent omission response, and a fitted combination. The combined fit also measures the isolated sentence's log-odds contrast with its stronger neighbor. It standardizes and fits on the selection works only. Every ranker selects a unique highest focus and uses the third-highest ordinary fitting-group maximum as its fixed review point. Useful localization covers at least half of the edited non-whitespace characters and at least 0.30 intersection over union. The prespecified combined-rule advancement requires at least two additional held work groups across both generators, at most one additional false ordinary group, and complete location coverage on at least 12 of 14 known edits.

## Held results

| Outcome | Current contexts | Primary top focus | Independent sentence | Omit-center response | Combined fit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen rewrites located | 0/8 | 0/8 | 1/8 | 0/8 | 1/8 |
| SmolLM2 rewrites located | 0/6 | 0/6 | 1/6 | 0/6 | 0/6 |
| Genuine originals flagged | 5/10 | 5/10 | 2/10 | 3/10 | 3/10 |
| Ordinary same-work edits flagged | 2/7 | 2/7 | 2/7 | 1/7 | 2/7 |
| Ordinary work groups flagged | 5/10 | 5/10 | 2/10 | 3/10 | 3/10 |
| Generated files flagged only at an incorrect location | 5/14 | 5/14 | 2/14 | 0/14 | 3/14 |

All 14 generated target sentences had complete eligible reads, compared with six under the earlier single-paragraph geometry. Each ranker assessed all 31 held files. None of their flags landed on the deliberately substituted ordinary sentence. Adding the independent sentence result to the incumbent would recover the two exact locations and flag one new ordinary group. The simpler arm remains in research for prospective confirmation at its unchanged point. The tested material covers one publisher and two generator families. The 12 independent-sentence misses leave mixed-document screening unfinished.

The comparison retained 500 distinct exact primary responses, made 11 CPU forwards for 42 additional primary inputs, and ran 284 CPU forwards for 1,135 independent inputs. The combined fit used 155 observations, including 13 known generated foci. Native replay verified all 69 exact texts and their source spans. The portable reviewer rejected changes to the operating point, fitted coefficient, and held result count. Native replay rejected a shifted source range. A code cleanup regenerated identical frozen input bytes, preserving the original inference receipts and all outcomes.

## Replay

The public receipt has anonymous source groups, scalar responses, source positions and overlap counts. The standalone reviewer checks the fitted coefficients against the derivative of the balanced regularized objective. It reconstructs all four points and held outcomes. Its optional native replay binds the exact source texts, donor spans, model inputs and measured overlaps to the saved runs. The evaluation retains source text and model weights in the local workspace.

```sh
python3 -B -IS backend/scripts/review_writing_multiblock_independent_transfer.py \
  docs/evaluation-panels/writing-multiblock-independent-2026-11-12.json
```

For local reconstruction with the earlier article directory and both installed, pinned model snapshots, use an empty private output directory.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze primary independent; do
  backend/.venv/bin/python -B -m scripts.writing_multiblock_independent_transfer \
    "$phase" "$parent" "$run" --primary "$primary_snapshot" --independent "$secondary_snapshot" || exit 1
done
backend/.venv/bin/python -B -m scripts.writing_multiblock_independent_transfer \
  publish "$parent" "$run" --output "$run/public.json"
python3 -B -IS backend/scripts/review_writing_multiblock_independent_transfer.py \
  "$run/public.json" --native "$parent" "$run"
```
