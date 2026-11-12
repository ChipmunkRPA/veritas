# Short audio repetition under competing speech

A sparse spectral landmark check added two useful copy locations over the existing repetition checks. It also added four false highlights. The candidate failed the frozen qualification and remains experimental. This experiment measures acoustic repetition. A supported pair can arise from ordinary editing or naturally repeated sound and carries no AI-origin confidence.

## Design

Select 17 earlier foreground source groups for point selection and 16 disjoint groups for transfer. Each role uses one mostly sequential and one simultaneous three-second interfering speech recording from different AMI rooms. The four rooms and 33 foreground groups had earlier project exposure. New exact inputs pair each untouched foreground with a 500 ms or 650 ms known copy from earlier in that same recording, blended over 20 ms at each boundary. Each pair is delivered clean and under −5 dB recorded interference, in PCM and 32 kb/s Opus. Each role also includes unmodified room excerpts in both deliveries. The 236 development views and 200 held views reuse source groups across conditions. [Source and rights record](audio-spatial-interference-2026-11-11.md)

Compare the existing exact and two-band waveform checks, with the optional waveform fallback when both are empty. The new method extracts local spectrogram peaks from 512-sample Hann frames spaced five milliseconds apart. It pairs two quantized peak frequencies and their frame separation into a sparse fingerprint. A useful time-offset vote requires at least two frequency bands and three time cells. Select the highest-support pair in a recording. The qualification point is strictly above the second-largest development false-control maximum across distinct foreground groups. A true location must overlap at least 150 ms of both known interiors and be within 10 ms of the authored lag. The new code uses published methods as design references and includes no external code or pretrained weights. [Constellation fingerprint method](https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf) · [Spectrogram keypoint study](https://www.mdpi.com/2076-3417/15/21/11845) · [Recent keyframe comparison](https://www.sciencedirect.com/science/article/abs/pii/S0167639326000981)

## Held results

| Delivery and interference | Existing correct /16 | Landmark correct /16 | Additional correct | Additional ordinary highlights |
| --- | ---: | ---: | ---: | ---: |
| Clean PCM | 16 | 12 | 0 | 0 |
| Clean Opus | 16 | 9 | 0 | 0 |
| Sequential PCM | 8 | 2 | 0 | 2 |
| Sequential Opus | 6 | 2 | 1 | 0 |
| Simultaneous PCM | 2 | 1 | 1 | 0 |
| Simultaneous Opus | 0 | 0 | 0 | 0 |

The native controls add two more false highlights, one PCM and one Opus, from the same room excerpt. The advancement requirement was at least eight additional true locations spanning four sources, at most two additional ordinary highlights from at most one source, and no native highlights. The locked point was 52. The severe interference still defeats both repetition methods on most of these exact copies. [Frozen aggregate](evaluation-panels/audio-landmark-repeat-transfer-2026-11-11.json)

An independent replay rebuilt all 436 input commitments, including 218 Opus round trips, reconstructed the 198 edit labels, and verified the development point and every held count. The two added locations each overlap 460 ms in both true spans, with a lag error of 5 ms. The existing checks found 48 of the 96 held copies, and their union with the candidate found 50. [Independent replay](../backend/scripts/review_audio_landmark_repeat_transfer.py)

## Reproduce

Use the exact parent stereo study and a fresh local output directory. The run decodes Opus through the application decoder and stores sample commitments and truth coordinates. Original recordings and individual coordinates remain local. [Runner](../backend/scripts/audio_landmark_repeat_transfer.py)

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for stage in prepare archived transfer summarize; do
  backend/.venv/bin/python -B -m scripts.audio_landmark_repeat_transfer "$stage" "$run" --parent "$spatial_run"
done
backend/.venv/bin/python -B -m scripts.review_audio_landmark_repeat_transfer "$run" "$spatial_run" "$run/audit.json"
```
