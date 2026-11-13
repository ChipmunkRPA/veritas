# Temperature sensitivity on short, edited report passages

We tested a sentence-level adaptation of the temperature-sensitivity
detector described by Ma et al. A local causal model reads the submitted
text once. The experiment measures how much the exact target tokens'
log probabilities change when the same logits are evaluated at
temperatures 0.7 and 1.4. The score subtracts the expected token-level
change under the unscaled probabilities and divides by the mean
conditional standard deviation, following the authors' implementation.
The comparison also uses mean log probability from the same forward pass.
It uses independently written code and the existing Apache 2.0 base
checkpoint, quantized to four bits locally.
[ACL paper](https://aclanthology.org/2026.acl-long.1748/) ·
[Authors' implementation](https://github.com/Shixuan-Ma/NTS/blob/main/TeSeN/detector.py) ·
[Pinned base checkpoint and license](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/8faed761d45a263340a0528343f099c05c9a4323)

The test requires complete causal token coverage of a native sentence,
at least 12 predicted tokens, and a file of at most 512 tokens. Each
method selects its highest eligible sentence only if it strictly
exceeds its own second-highest genuine-work maximum. Forty-two earlier
and later genuine versions of 21 Wellcome and Gates articles set those
points before held inference. All 21 work groups supplied eligible
positions. The primary comparator uses the first six native
three-sentence contexts above the existing research point. Other
customer worker checks are outside this experiment. A useful location
must cover at least half of the known non-whitespace edit characters and
reach 0.30 intersection over union in one proposed range.
[Genuine editorial revision sources](writing-journal-body-transfer-2026-11-11.md) ·
[Primary checkpoint](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615)

From 222 eligible IDs, 38 were examined in fixed hash order. Twenty-four
fresh GAO works passed exact native admission, 11 lacked a shared
isolated edit and three had unsupported publisher geometry. Each work
supplied an untouched clip and two first-stage publisher polishes of
the same exact source sentence, one from Gemini 2.5 Flash and one from
GPT-5.4 Nano. Every character outside that sentence stayed identical in
the clip. Twenty-three works supplied six sentences and one supplied
five. Clips were 118–246 words. The held work IDs and normalized
12-word source sequences were screened against the earlier report
banks. The held genuine controls also include eight dated forum pairs
and six dated USWDS pairs, previously used in other project studies.
[Publisher dataset and terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9) ·
[GovReport and source terms](https://gov-report-data.github.io/#license) ·
[Forum sources](writing-forum-revision-transfer-2026-11-13.md) ·
[USWDS sources](writing-uswds-revision-transfer-2026-11-13.md)

| Method | Gemini useful locations | Nano useful locations | Unchanged GAO files flagged | Genuine forum and USWDS files flagged |
| --- | ---: | ---: | ---: | ---: |
| Existing primary contexts | 8/24 | 8/24 | 10/24 | 10/28 |
| Same-base mean log probability | 0/24 | 1/24 | 10/24 | 8/28 |
| Temperature sensitivity | 0/24 | 0/24 | 0/24 | 1/28 |
| Existing plus temperature sensitivity | 8/24 | 8/24 | 10/24 | 11/28 |

All 100 held files received measurements. Mean log probability flagged
19 edited files, with one correct location already covered by the
primary comparator. Temperature sensitivity returned zero edited
locations and one genuine flag. The new cue stays in research with zero
authorship-confidence weight. These controlled clips represent one
report genre and first-stage polishing by two providers. The experiment
does not estimate accuracy on arbitrary uploads. Recent short-text
research also finds that very short samples can have limited power at
low false-positive rates. [Short-text length study](https://arxiv.org/abs/2611.10901)

## Replay

A clean checkout can recompute the operating points and decisions from
the text-free scalar receipt. The full local rerun additionally needs
the three pinned OpAI-Bench report shards and metadata, the previously
retained typography roster, and the pinned local checkpoints. The
causal phases use the logged offline MLX runtime. Run the three
preparation/scoring stages in order to keep the held responses separate
from point selection.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.writing_temperature_transfer review \
  docs/evaluation-panels/writing-temperature-transfer-2026-11-13.json
```

```sh
export PYTHONPATH=backend:"$mlx_packages" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
backend/.venv/bin/python -B -m scripts.writing_temperature_transfer roster "$run" \
  --publisher "$publisher_cache" --historical "$typography_run/roster.json"
backend/.venv/bin/python -B -m scripts.writing_temperature_transfer score-fit "$run" --snapshot "$causal_snapshot"
PYTHONPATH=backend python3 -B -S -m scripts.writing_temperature_transfer fit "$run"
backend/.venv/bin/python -B -m scripts.writing_temperature_transfer score-held "$run" --snapshot "$causal_snapshot"
backend/.venv/bin/python -B -m scripts.writing_temperature_transfer primary "$run" --snapshot "$primary_snapshot"
PYTHONPATH=backend python3 -B -S -m scripts.writing_temperature_transfer publish "$run" --output "$receipt"
```
