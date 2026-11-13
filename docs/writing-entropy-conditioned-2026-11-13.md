# Token predictability conditioned on uncertainty in mixed writing

A single-file check tested whether grouping token scores by the uncertainty of
the language model's next-token prediction helps locate one rewritten sentence
inside an otherwise unchanged article excerpt. The design draws on recent work
that evaluates local score calibration and exact token positions. It uses an
independently written entropy-binned reduction and a locally installed
Apache-2.0 base checkpoint. [Local score calibration research](https://arxiv.org/abs/2605.06294) ·
[Mixed-writing token localization research](https://arxiv.org/abs/2607.21458) ·
[Pinned checkpoint and licence](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323)

## Frozen comparison

The causal model reads each native paragraph from its start. The input must
preserve the prior study's sentence offsets, and every scored sentence needs
complete tokenizer coverage after a preceding context. For each token, measure
the actual log probability minus the expected log probability under the
model's next-token distribution. The raw check sums these residuals and scales
by their model variance. The conditioned check centers and scales each residual
using genuine-writing estimates from six entropy ranges, then sums within the
same exact sentence and scales by its token count's square root. Ranges with
fewer than 100 fit tokens use pooled genuine moments. Both arms select one
unique highest sentence per document.

Fourteen F1000Research development works supply two dated versions each. Seven
works fit the moments, and the other seven set each operating point to its
second-highest ordinary work maximum. Two disjoint published panels supply the
assessment cases. Each case has the same source hash, exact sentence positions
and edit truth as its prior ledger. All held works had prior exposure under
other methods. F1000Research controls include dated versions of the same work.
The eLife panel includes untouched prose and constructed replacements using
another genuine sentence from the same work. The two generation families in
both panels are the prior Apache-2.0 Granite and Qwen checkpoints. This study
uses scholarly excerpts and reuses known source outcomes. The causal scorer
and one rewrite generator share a model family. [Development sources](writing-long-position-transfer-2026-11-12.md) ·
[F1000Research assessment sources](writing-f1000-position-transfer-2026-11-13.md) ·
[eLife assessment sources](writing-elife-long-position-transfer-2026-11-13.md)

## Held results

| Outcome | F1000Research | eLife |
| --- | ---: | ---: |
| Known edit targets measured | 18/18 | 21/21 |
| Current context plus position check, useful locations | 4/18 | 0/21 |
| Raw token check, exact locations | 0/18 | 0/21 |
| Conditioned token check, exact locations | 0/18 | 0/21 |
| Conditioned token check, wrong locations on edited files | 0/18 | 4/21 |
| Ordinary works flagged by current context plus position | 2/9 | 2/12 |
| Ordinary works additionally flagged by raw token check | 0/9 | 1/12 |
| Ordinary works additionally flagged by conditioned token check | 0/9 | 3/12 |

The conditioned check missed all 39 target sentences even though it measured
all of them. The added eLife flags and zero new locations failed the research
advancement rule. Both arms stay outside the app. The result tests a simple
entropy stratification on retained sources. Richer conditioning and naturally
mixed human–AI writing remain untested here.

## Replay

The local Apple GPU completed 164 paragraph inputs and verified native
positions for all 918 eligible readings. An earlier process stopped during
concurrent local workloads. The final run used a bounded Metal allocation
cache and the same case roles and scoring method. The public reader replays the
points, selected sentences, exact overlaps, wrong locations and ordinary-group
counts from saved scalars. The published data omit prose, token IDs, and logits.
[Decision ledger](evaluation-panels/writing-entropy-conditioned-2026-11-13.json)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.writing_entropy_conditioned_transfer verify \
  docs/evaluation-panels/writing-entropy-conditioned-2026-11-13.json
```

For model inference, use a new scratch directory, the pinned local snapshot,
retained private case files, and the recorded MLX and tokenizer versions:

```sh
export PYTHONPATH="backend:$mlx_packages" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VERITAS_OFFLINE=true
for phase in freeze score-fit fit score-held; do
  backend/.venv/bin/python -B -m scripts.writing_entropy_conditioned_transfer "$phase" "$run" \
    --development "$development_cases" --f1000 "$f1000_cases" \
    --elife "$elife_cases" --snapshot "$snapshot" || exit 1
done
backend/.venv/bin/python -B -m scripts.writing_entropy_conditioned_transfer publish "$run" --publication "$ledger"
```
