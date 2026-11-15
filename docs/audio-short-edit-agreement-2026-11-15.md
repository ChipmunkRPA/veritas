# Combining short speech-edit checks under competing speech

This study tests whether three existing candidate checks can add useful
locations to the current two-second speech review on a six-second
upload. Each check points to an interval on the original 16 kHz clock.
We keep the current interval and count what each supplement adds, both
alone and in combination. The primary and transfer sets preserve matched
genuine files so a nearby human splice can be distinguished from a useful
synthetic-edit lead. No new model assets enter the study.

The experiment addresses the distinction between temporal and overlapping
source evidence in recent mixed-authenticity work. The held comparisons
separate exact localization from a file-level flag, following the
motivation of cross-corpus partial-speech evaluation. This study uses
independent project code and existing commercially reviewed assets.
[Mixed-authenticity research](https://arxiv.org/html/2609.03620v1) ·
[Out-of-domain localization research](https://arxiv.org/abs/2507.03468) ·
[Model terms](commercial-model-audit.md)

## Comparison

The candidates use earlier speech features, learned transitions between
neighboring features, and a training objective that ranks each interval
against matched controls. All three share a frozen speech encoder.
Each has two fits, one for each of the original training generators.
The primary panel uses the fit that held out the edited generator.
Transfers can use either fit. Agreement must come from two distinct
checks within the same fit direction.
[Earlier feature test](audio-short-edit-layer-ablation-2026-11-15.md) ·
[Temporal boundary test](audio-temporal-boundary-transfer-2026-11-15.md) ·
[Matched-clock training](audio-competing-speech-ranking-2026-11-15.md)

- **Any check** keeps every proposed interval above its original point.
- **Two checks** requires both proposals to exceed their original points
  and overlap by at least half of each proposal. It keeps the shorter
  original proposal, or the earlier one if their lengths match.
- **Joint calibration** uses the same overlap rule but ranks each
  response against the maxima of the original 16 calibration groups
  for that check and fit direction. The pair uses the lower of its two
  ranks. The highest and second-highest foreground group scores set
  the tight and wider points. Both points must also reach the highest
  isolated-background score. A calibration case excludes its own group
  from ranking. Each fit direction can add its strongest pair above the
  point. Ranks are integer units and carry no interpretation as AI
  probabilities.

A correct range must cover at least half of the edit and have temporal
intersection over union of at least 0.30. A newly located edit has a
matched-control collision if any control from the same group and condition
has an independently useful interval at the edit's true clock, even if
the displayed interval differs. Isolated backgrounds with their own
source group have a separate flag count.

## Held results

The primary set has 240 decisions across the two fit directions and
165 distinct submissions. The 144 ordinary decisions include 93 distinct
PCM inputs. Reported ordinary flag counts are per decision, and the
receipt separately counts flagged source groups. Three separate transfer
sets supply 255 further recordings. The same recordings had prior
project exposure. Boundary and ranking responses for the 83 random-clock
recordings are new measurements of those retained inputs.
[Primary source plan](audio-short-edit-detector-training-2026-11-14.md) ·
[Random-clock source plan](audio-random-clock-stock-2026-11-13.md)

Each entry shows useful synthetic-edit locations followed by ordinary
flags. The recorded background speech is mixed 5 dB above the foreground
speech in every row.

| Panel | Current | Any, strict | Any, wider | Two checks, strict | Two checks, wider |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary | 4/48 · 64/144 | 6/48 · 67/144 | 8/48 · 83/144 | 5/48 · 64/144 | 5/48 · 65/144 |
| Separate meeting rooms | 0/11 · 5/33 | 0/11 · 13/33 | 2/11 · 24/33 | 0/11 · 5/33 | 0/11 · 8/33 |
| Fixed-clock stock voices | 3/16 · 4/24 | 3/16 · 13/24 | 3/16 · 19/24 | 3/16 · 6/24 | 3/16 · 7/24 |
| Random-clock stock voices | 1/16 · 3/24 | 4/16 · 13/24 | 4/16 · 18/24 | 1/16 · 3/24 | 2/16 · 7/24 |

At the wider point, accepting any check adds four matched-control-free
locations on the primary panel. It also adds flags in nine ordinary
source groups and three isolated-background decisions. The separate
meeting-room set gains two locations, of which one also flags the
matched same-speaker splice. The random-clock stock bank gains three
locations, of which two also flag same- or different-reader splices at
the edit clock. The fixed-clock stock bank gains zero locations at
either point.

Requiring two checks at the wider point adds a Piper location at
**2.500–3.000 s** inside the staged **2.045–3.005 s** edit. Its matched
genuine ranges cease to flag the edit clock under the two-check rule.
The condition still adds four ordinary flags across two new source
groups. Joint calibration yields the same useful and ordinary counts
as the two-check rule across every measured condition. All foreground
conditions had available assessments. The wider union flags none of the
11 isolated backgrounds in the three transfer banks. The dry
and 0 dB results are retained in the receipt.

The strict agreement rule adds one primary location without adding
ordinary flags there. It adds no locations on the three transfer banks.
The wider rule makes a limited gain on the random-clock clips and fails
to recover the useful ranges added by the separate candidate in the
meeting-room bank. These measurements leave the combinations in
evaluation and the existing app review in use.

## Replay

The receipts contain scalar readings, exact intervals, source hashes,
and paired outcomes. They contain no audio, voice features, or weights.
The standard-library reviewer reclocks the three checks against their
committed source rows and re-derives the separate and joint points.
Python 3.12 and 3.14 produced identical results. Two native passes
produced identical additional readings across the 83 random-clock
recordings.
[Combined receipt](evaluation-panels/audio-short-edit-agreement-2026-11-15.json.gz) ·
[New random-clock readings](evaluation-panels/audio-short-edit-agreement-random-readings-2026-11-15.json)

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/audio-short-edit-agreement-2026-11-15.json.gz
python3 -B -S -m scripts.audio_short_edit_agreement review "$receipt"
```

To remeasure the random-clock files, use the retained source bank and
the original fitted directories from the linked methods. The runner
verifies the input PCM and fitted weights before measuring the two
additional responses. Use the local speech encoder and speech-activity
assets from the previous fits. Keep the media and weights local.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_short_edit_agreement capture-random \
  "$random_media" "$boundary_fit" "$ranking_fit" "$encoder" "$speech" "$run/random.json"
python3 -B -S -m scripts.audio_short_edit_agreement review "$receipt" --random "$run/random.json"
```
