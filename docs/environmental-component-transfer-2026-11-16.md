# Synthetic background sound under speech

An estimated background view picked up six of 72 generated backgrounds
under louder speech, with one alert on 24 matched ordinary backgrounds.
The held generator families supplied no calibration examples. Combining
this view with the original mixture helped on the selection sources, but
found one of 72 generated backgrounds in the held loud-speech condition.
All four candidates remain in evaluation.

## Sources and design

The pinned EnvSDD release supplies backgrounds. Match the publisher's
source crosswalk to Clotho's individual license tables. Require a
single unambiguous CC0 grant and exclude all earlier recorded source IDs
and credited recordists. The new study freezes distinct background groups
and recordists before downloading any audio. Thirty-six groups fit the
heads, 20 set operating points, and 24 supply the held comparison. Each
group supplies its original background and matching generated versions.
Fit and selection use AudioLDM 1 and 2. The held comparison uses AudioGen,
AudioLCM, and TangoFlux. All 264 publisher WAV members and their decoded
three-second excerpts have distinct commitments. The study uses released
audio and the original CC0 sources under EnvSDD's CC BY 4.0 terms.
[Pinned EnvSDD release and terms](https://huggingface.co/datasets/EnvSDD/EnvSDD/blob/b0832726deb55484dbce869c6f09c41f7e9e5fbb/README.md)
· [Clotho source licenses](https://zenodo.org/records/4783391)

Create every mixed group with exactly the same foreground samples and a
common output gain across the four or three background origins. The
foreground is genuine at 6 dB above the environment or at equal level.
The held cases add a 32 kbps Opus replay of the louder-speech mixtures
and mixtures with verified partial edits in the matched foreground.
Four genuine foreground recordings each supply fitting and selection.
The held group uses 14 same-utterance voice pairs across six generators
from the earlier voice study. It also has 28 voice-only controls, half
genuine and half containing an annotated synthetic edit. The environmental
sources are new to the earlier project panels identified by the roster.
The prior exposure of the speech estimator to these sources is unknown.
[Verified foreground source pairs](audio-six-ten-context-2026-11-16.md#same-utterance-controls)

Measure the same seven bands and temporal statistics on the original
mixture and on the residual after subtracting an estimated aligned
speech waveform. The residual can contain speech and distortions from
the estimator. Compare four small fitted checks: the mixture, residual,
both, and both with an additional objective that compares the generated
background with its matched ordinary rendition. The small ridge heads
are project implementations. The estimator uses the existing pinned MIT
checkpoint on CPU. Both operating points for each new method come from
the same 20 selection groups and four separate voice-only controls. Each
operating point applies to all conditions without knowing the actual
mixing ratio during inference. Component-level work motivates retaining
the delivered mixture alongside the estimated residual. The two previous
isolated-fit checks are evaluated at their original operating points.
[MixedTruth paper](https://arxiv.org/html/2610.14141v1)
· [Pinned estimator](https://github.com/Xiaobin-Rong/gtcrn/tree/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73)
· [Earlier spectral and texture checks](environmental-sound-transfer-2026-11-11.md)

## Held results

Each cell shows generated backgrounds flagged out of 72 and ordinary
backgrounds flagged out of 24. The new checks use their wider frozen point.
The same source groups recur across conditions. The held scores also keep
all results at the stricter point and separate the three generator families.

| Held condition | Mixture | Residual | Both | Both with matched fit | Earlier magnitude | Earlier texture |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Isolated environment | 2 / 0 | 6 / 0 | 1 / 1 | 7 / 0 | 4 / 4 | 28 / 8 |
| Genuine speech 6 dB louder | 0 / 0 | 6 / 1 | 0 / 0 | 1 / 0 | 0 / 0 | 67 / 22 |
| Genuine speech at equal level | 0 / 0 | 6 / 0 | 1 / 1 | 1 / 1 | 2 / 0 | 62 / 20 |
| Partly generated speech 6 dB louder | 0 / 0 | 6 / 0 | 0 / 0 | 0 / 0 | 1 / 0 | 64 / 21 |
| 32 kbps Opus, genuine speech 6 dB louder | 0 / 0 | 3 / 1 | 0 / 0 | 0 / 0 | 0 / 0 | 68 / 22 |

The estimated background view's six detections under louder genuine
speech include four AudioLCM and two TangoFlux outputs. It misses all 24
AudioGen backgrounds there and finds only one AudioGen output across the
five conditions. It flags zero of the 28 held voice-only controls. The
earlier texture check flags 27 of those 28 controls, even though there is
no separately added environment. On louder genuine speech, the residual's
generated response exceeds its matched ordinary response in 33 of 72
comparisons. The fitted ranking has limited transfer even before applying
the operating points. The current acoustic findings in the app describe
what is audible in the mixed track. Background provenance is unresolved.
[Per-source commitments, frozen heads and held outcomes](evaluation-panels/environmental-component-transfer-2026-11-16.json.gz)

An independent reader verifies the 80 background groups, 184 selection
deliveries, 508 held deliveries and 220 result cells. With the local media,
it checks 264 environmental sample hashes, all 1,020 rendered hashes and
all 624 uncompressed additive mixtures at sample precision. The public
packet contains source references, positions, numeric scores and compact
fitted parameters. Raw audio and per-recording features remain local.

## Replay

The portable recount uses standard-library Python and the checked-in
numeric record. The optional media check verifies the rendered inputs
without running inference or loading a checkpoint.

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/environmental-component-transfer-2026-11-16.json.gz
python3 -B -S -m scripts.review_environmental_component_transfer "$receipt"
backend/.venv/bin/python -B -m scripts.review_environmental_component_transfer "$receipt" --native "$run"
```

To rebuild the native inputs, restore the earlier EnvSDD metadata and
archived genuine foregrounds from the [earlier replay](environmental-sound-transfer-2026-11-11.md#replay).
Restore the [matched voice controls](audio-six-ten-context-2026-11-16.md#replay)
and obtain the exact estimator and MIT license from the pinned release.
Place both in `$run/assets` and use ONNX Runtime 1.23.2 and FFmpeg with
libopus. The background source revision and file addresses are in the
frozen plan. The acquisition stage alone needs network access.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
runner=backend/.venv/bin/python
"$runner" -B -m scripts.environmental_component_inputs plan "$run" "$earlier"
"$runner" -B -m scripts.environmental_component_inputs acquire "$run" "$earlier"
"$runner" -B -m scripts.environmental_component_transfer prepare "$run" \
  --speech "$archived_speech" --matched "$matched" --earlier "$earlier"
"$runner" -B -m scripts.environmental_component_transfer fit "$run"
"$runner" -B -m scripts.environmental_component_transfer assess "$run" \
  --earlier "$earlier" --output "$run/results.json.gz"
```
