# Testing shorter speech locations under louder conversation

The existing voice detector averages its time-aligned features before a
linear output layer. This study applies the same output layer to each
feature frame. The average of the resulting readings reproduced the raw
window logit within 0.00000191 on both panels. The frame readings are
contextual because the encoder uses the full three-second window.

We tested intervals from 240 to 720 ms on the native 16 kHz clock. A
range needed at least half its samples admitted as speech and support
from two overlapping three-second contexts with distinct starts.
The absolute comparison ranks by the second-highest level so each
selected range exceeds the point in at least two contexts. The other
comparison subtracts the stronger of the two 240 ms neighbors
before applying the same corroboration. This tests whether a temporary
local rise is more useful than absolute score. Recent work on
time-aligned attribution motivates testing where the detector draws
its evidence. The reconstruction here follows directly from the
existing linear output and requires no extra learned weights.
[Temporal attribution study](https://arxiv.org/abs/2606.10912)
· [Model terms](commercial-model-audit.md)

## Held results

Development used 46 existing native deliveries across 23 source
utterance groups, with ordinary hosts and isolated conversation setting
two frozen operating points. The strict point adds no false source
groups in development. The wider point allows up to two otherwise-clear
ordinary reader groups and one background room. Neither point added a
correct location on the 38 development labels. The transfer reused 120
exact deliveries held outside that calibration in reader, room, and
PCM hash. It includes 24 edited recordings from six synthesis families,
24 unchanged recordings, 24 ordinary reprocessing controls and 48
isolated backgrounds. The designated foreground is 10 dB quieter
than the added conversation. Each true range and selected range has
half-open coordinates on its source sample clock. The one-to-one
matching rule requires half of a labeled edit and at least 0.30
temporal intersection over union.
[First source panel](audio-two-second-long-review-2026-11-16.md#fresh-transfer-procedure)
· [Transfer panel](audio-six-ten-context-2026-11-16.md#fresh-comparison-of-additions)
· [Partial speech dataset terms](https://zenodo.org/records/14214149)
· [Background recording terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

| Comparison | Correct locations / 164 | Added ordinary alerts / 48 | Added isolated alerts / 48 | Edited files with candidate ranges and zero correct candidate locations |
| --- | ---: | ---: | ---: | ---: |
| Existing three-second and confirmed two-second checks | 21 | — | — | — |
| Add absolute frame level, strict | 22 | 0 | 0 | 4 |
| Add absolute frame level, wider | 22 | 0 | 0 | 5 |
| Add nearby contrast, strict | 21 | 0 | 0 | 1 |
| Add nearby contrast, wider | 21 | 4 | 0 | 2 |

The existing comparison flagged eight of the 48 ordinary controls.
It correctly located two of 67 edits shorter than 600 ms. The stricter
absolute readout added one such edit. The range **7.8475–8.2075 s**
overlaps the known **7.7400–8.2500 s** edit. It yielded one correct
candidate across 24 total ranges on edited files. The nearby-contrast
readout nominated useful intervals before thresholding, but its
selected ranges did not transfer at either frozen point. The method
stays in evaluation. Tests that can measure the edited segment separately
from surrounding speech remain the next comparison for strong noise.

## Separate segment-trained checkpoint

We also evaluated the official SAL checkpoint built with WavLM. The
publisher's complete archive matched its SHA-256, and the 571 inference
tensors were extracted and checked separately from the training state.
The first XLS-R archive matched its publisher digest and lacked its ZIP
directory. The evaluation uses the complete WavLM variant. The project
declares MIT for the checkpoint and published code. Microsoft's
backbone terms are CC BY-SA 3.0. The weights stay on the local drive.
[Published checkpoints](https://huggingface.co/MaoYC/SAL)
· [Inference source and licence](https://github.com/SentryMao/SAL/tree/b4e80d1d2fd98f3d8b9a85a47837ab3a73e12485)
· [Backbone terms](https://github.com/microsoft/UniSpeech/blob/main/LICENSE)

The separate model reads the native recording and produces a response
per 160 ms bin. We scored complete ranges of one to six bins with the
same speech requirement. The contrast variant compares the range
with the stronger of its two 320 ms flanks. Each variant gets its own
two points from the same 46 development deliveries. Across all four
frozen variants, the combined check stayed at **21/164** on transfer.
At the stricter absolute point it added no ordinary or background
alerts. The wider absolute variant added six ordinary and one isolated
alert. The stricter contrast variant flagged seven of 48 isolated
backgrounds. These results favor evaluating cleaner target speech or
noise-trained segment models before adding this check to the app.
[Segment-aware learning paper](https://arxiv.org/abs/2601.21925)
· [Development counts](evaluation-panels/audio-segment-trained-development-2026-11-17.json.gz)
· [Transfer counts](evaluation-panels/audio-segment-trained-transfer-2026-11-17.json.gz)

## Recount and compare a different detector

The public receipts provide exact source input commitments, truth and
selected native intervals. They also provide source-group maximum
control readings so the two points can be recalculated. Audio,
frame-level traces, and model assets stay local.
[Development receipt](evaluation-panels/audio-frame-logit-development-2026-11-17.json.gz)
· [Transfer receipt](evaluation-panels/audio-frame-logit-transfer-2026-11-17.json.gz)

```sh
export PYTHONPATH=backend
panels=docs/evaluation-panels
development="$panels/audio-frame-logit-development-2026-11-17.json.gz"
transfer="$panels/audio-frame-logit-transfer-2026-11-17.json.gz"
python3 -B -S -m scripts.review_audio_frame_logit_localization verify "$development"
python3 -B -S -m scripts.review_audio_frame_logit_localization verify "$transfer" "$development"
python3 -B -S -m scripts.review_audio_frame_logit_localization verify \
  "$panels/audio-segment-trained-transfer-2026-11-17.json.gz" \
  "$panels/audio-segment-trained-development-2026-11-17.json.gz"
```

Use the original freeze from the transfer panel to generate an input
manifest, reference intervals and existing predictions in the project's
independent evaluation format. The public transfer source plan can
reconstruct that freeze byte for byte. A second matcher reproduces the
existing 21 correctly localized events. Use the same files to score
predictions from a separate detector. Set `out` to a writable directory.

```sh
reference="$panels/audio-six-ten-context-fresh-complete-2026-11-16.json.gz"
python3 -B -S - "$reference" "$out/source.json" <<'PY'
import sys
from pathlib import Path
from scripts.audio_focused_context_transfer import read, save
save(Path(sys.argv[2]), read(Path(sys.argv[1]))["source_plan"])
PY
python3 -B -S -m scripts.review_audio_frame_logit_localization export \
  "$transfer" "$out/source.json" "$out/independent"
```

The original signal replay uses the two archived native PCM directories
described in the source panels. Each contains `freeze.json` and PCM
objects named by digest. Define `first`, `second` and an empty writable
`out` directory. Create the development receipt before opening the
transfer readings.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
runner=backend/.venv/bin/python
run=scripts.audio_frame_logit_localization
review=scripts.review_audio_frame_logit_localization
"$runner" -B -m "$run" prepare "$first" "$second" "$out/dev.json" --development
"$runner" -B -m "$run" score "$first" "$second" "$out/dev.json" --output "$out/dev.jsonl"
python3 -B -S -m "$review" fit "$out/dev.json" "$out/dev.jsonl" \
  "$panels/audio-two-second-long-fresh-2026-11-16.json.gz" "$out/development.json.gz"
"$runner" -B -m "$run" prepare "$second" "$first" "$out/test.json"
"$runner" -B -m "$run" score "$second" "$first" "$out/test.json" --output "$out/test.jsonl"
python3 -B -S -m "$review" replay "$out/test.json" "$out/test.jsonl" \
  "$panels/audio-six-ten-context-fresh-complete-2026-11-16.json.gz" \
  "$out/transfer.json.gz" "$out/development.json.gz"
```
