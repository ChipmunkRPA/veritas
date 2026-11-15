# Dense speech measurements for short lip-sync changes

The existing lip-timing check retains a few high-confidence speech events
and compares each with mouth aperture. This evaluation asks whether
continuous phone-class probabilities provide useful timing between
those events. It uses the same pinned Apache-2.0 speech weights and
native mouth observations. It measures timing and leaves the cause of
a timing difference unresolved.
[Published weights and license](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft/tree/ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4)
· [Timing and articulation research](https://arxiv.org/html/2610.09223v1)
· [Existing sparse measurements](video-lip-viseme-transfer-2026-11-15.md)

## Comparison

At each verified 20 ms acoustic hop, group the decoder probabilities by
visible speech class. Smooth the measurements over five hops. A ridge
fit predicts mouth aperture from the grouped probabilities and speech
support at the current hop and at ±60 ms. The fit used 1,123 paired
observations from 16 original recordings and four speakers. Each
recording has equal weight. The scoring comparison uses common 40 ms
audio sample positions and the same observed pictures for every lag.
The motion arm also compares the predicted and observed changes over
80 ms. Each method compares one constant lag with a short change
supported by a common lag before and after it.

The initial dense comparison used squared aperture errors. A second
comparison divides those errors by the same squared aperture scale as
the sparse check before applying the ambiguity margin. The initial
scores remain available as a comparison. Both points for each arm use
separate ordinary controls from four calibration speakers. The stricter
point is their largest speaker maximum. A correct range must overlap the
authored interval at 0.40 temporal intersection over union and estimate
the signed change within 80 ms. These are evaluation settings.
[Lombard GRID corpus and terms](https://zenodo.org/records/3736465)

## Held results

The main PCM panel contains two cohorts totaling 64 recordings from 16
held speakers. The separate carrier panel contains 16 recordings from
eight other speakers. It reproduces the corresponding PCM and 48 kbps
AAC waveforms at their verified sample origins. The two carrier columns
represent matched deliveries of the same edits.

| Stricter point | Main PCM useful /128 | Separate PCM useful /32 | Separate AAC useful /32 |
| --- | ---: | ---: | ---: |
| Existing sparse timing standalone | 9 | 4 | 3 |
| Dense aperture, original score scale | 2 | 1 | 1 |
| Dense aperture, matched score scale | 4 | 1 | 2 |
| Dense aperture and motion, either scale | 0 | 0 | 0 |
| Sparse plus dense aperture at the matched scale | 13 | 5 | 5 |

The combined result leaves the main panel's two ordinary flags out of
192 and raises the constant-delay flags from five to seven out of 128.
On the separate carrier panel it adds no flags to 48 ordinary and 32
constant-delay controls per carrier. The normalized dense check increased
the number of main edited deliveries with resolved comparisons from 65
to 115 for the same 128 inputs. A resolved comparison includes a completed
search with no supported change. One correctly located AAC case measured
audio about 240 ms late over **0.760–1.480 s** against a known 200 ms delay.
Its before, during and after source pictures are bound by index, native
clock, crop positions and hashes.
[Main panel receipt](evaluation-panels/video-lip-dense-posteriors-2026-11-15.json.gz)
· [Separate carrier receipt](evaluation-panels/video-lip-dense-carrier-2026-11-15.json.gz)

## Native fractional-rate transfer

The same fit and operating points were tested on 24 distinct CREMA-D
actors. All 168 soundtrack digests matched the prior native-clock
replay. The study reused the original recorded picture timestamps and
raw audio presentations. It also replayed the measured mouth values on
a virtual 25 fps schedule, selecting the nearest known picture at each
tick. That arithmetic replay preserves the assignment to the recorded
picture. It creates no new frame and has no re-encode claim.

| Native actor comparison at the stricter point | Useful shifts /48 | Ordinary flags /72 | Constant-delay flags /48 |
| --- | ---: | ---: | ---: |
| Sparse, recorded 29.97 fps clocks | 2 | 0 | 1 |
| Normalized dense aperture, recorded clocks | 0 | 7 | 3 |
| Sparse, virtual 25 fps schedule | 2 | 3 | 0 |
| Normalized dense aperture, virtual 25 fps schedule | 0 | 8 | 6 |

The rescheduled pictures recovered no new correct location. The failed
transfer persists at both tested analysis schedules, which weakens the
case that frame rate alone accounts for the gap. The dense method
remains experimental. The Grid results apply to staged short timing
shifts in the tested recordings. Separate challenges include generated
mouth shapes, natural conversation, and distinct background audio.
[Native source qualification and terms](video-lip-frame-clock-2026-11-15.md#separately-recorded-2997-fps-speech)
· [Native and virtual-clock aggregate](evaluation-panels/video-lip-dense-native-2026-11-15.json)

## Replay

The portable reviewer recalculates both calibration points, sample
placements, selected boxes and registered control outcomes from the
predicted aperture series and earlier source receipts. The scoring
replay needs standard-library Python and agreed byte for byte on
Python 3.12 and 3.14. A second native pass reproduced all 24 actor
records byte for byte. The original source media, full phone outputs,
and detailed native actor records stay local.

```sh
export PYTHONPATH=backend
panel=docs/evaluation-panels/video-lip-dense-posteriors-2026-11-15.json.gz
carrier=docs/evaluation-panels/video-lip-dense-carrier-2026-11-15.json.gz
python3 -B -S -m scripts.video_lip_dense_posteriors review "$panel"
python3 -B -S -m scripts.video_lip_dense_posteriors review-transfer "$panel" "$carrier"
```

To reproduce the acoustic and native-frame measurements, use the same
pinned [Lombard media archive](video-neural-phone-clock-2026-11-14.md#independent-replay)
and [verified native source assets](video-lip-frame-clock-2026-11-15.md#replay).

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
assets=models/research/video-lip-timing
runner=backend/.venv/bin/python
module=scripts.video_lip_dense_posteriors
"$runner" -B -m "$module" capture "$run/hops.jsonl" --checkpoint "$assets/checkpoint" --sources "$archive"
"$runner" -B -m "$module" report "$run/hops.jsonl" "$run/primary.json.gz"
"$runner" -B -m "$module" transfer "$run/primary.json.gz" "$run/carrier.json.gz" \
  --checkpoint "$assets/checkpoint" --sources "$archive"
"$runner" -B -m scripts.video_lip_dense_native capture "$run/primary.json.gz" "$audit_dir" "$publisher_media_dir" \
  "$assets/checkpoint" "$assets/face_landmarker.task" "$assets/runtime" "$run/native.json.gz"
python3 -B -S -m scripts.video_lip_dense_native review "$run/primary.json.gz" "$audit_dir" "$run/native.json.gz" \
  --clock-report "$run/clock-comparison.json"
```
