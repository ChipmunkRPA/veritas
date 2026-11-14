# Sound during a measured mouth-region hold

Full video analysis now checks whether the soundtrack contains sustained voice-like periodicity during an independently measured local mouth-region hold. The report shows the half-open source interval, three source frames and exact mouth crops, the waveform, and the number of 40 ms audio blocks with the acoustic pattern. The browser and evidence ZIP also include the exact decoded mono WAV excerpt. The observation carries zero weight in the overall AI confidence assessment. Ordinary dubbing, another speaker, music or other periodic sounds can also produce the pattern.

The check starts only after the qualified HD mouth-region detector has measured unusually stable pixels in the mouth region while nearby face regions change. It uses whole decoded 16 kHz samples strictly inside the exact video boundaries, with a continuous sound clock. It evaluates mean-centered, amplitude-gated autocorrelation in 40 ms blocks spaced 20 ms apart. At least six blocks are required. It reports sustained periodicity when at least two-thirds show a repeat consistent with vocal pitch. It uses no learned model or voice identity. A usable soundtrack can have no sustained periodicity, and a track that is entirely silent or otherwise unavailable shows that the soundtrack was unmeasured. [HD picture methodology and source rights](video-mouth-hd-hold-2026-11-12.md)

Recent research explicitly separates audio–picture offset from whether the mouth visibly articulates particular sounds. This check measures a narrow acoustic pattern alongside a measured hold. Broader experiments using a closed-mouth geometry estimate and general speech activity warned on unchanged scripted clips and ordinary global offsets during development. Those experiments stay outside the app. [Timing and articulation](https://arxiv.org/abs/2610.09223) · [Phonetic alignment for dubbing](https://arxiv.org/abs/2610.00825)

## Local comparison

The diagnostic reused nine held, rights-reviewed Wikitongues recordings from the earlier HD mouth-hold bank. Seven supported the paired comparison using the nominal 400 ms held-picture interval. Nine supplied natural 200 ms quiet intervals. Audio donors were selected for clear periodicity by searching within the same test partition across other recordings, so the positive results show measurement feasibility. The experiment supplies no independent estimate of general dubbing or deepfake accuracy. The noise variants add deterministic band-limited random sound at roughly −28 dBFS RMS. Recordings in the development and held groups are identified only by anonymous indices in the receipt. [Source and license ledger](evaluation-panels/video-mouth-hd-hold-2026-11-12.json)

| Condition | Sustained periodicity |
| --- | ---: |
| Original soundtrack in the 400 ms tested slot | 2/7 |
| Clean speech inserted in the 400 ms slot | 7/7 |
| Same speech inserts with strong added noise | 0/7 |
| Brief natural quiet interval | 0/9 |
| Natural quiet interval with strong added noise | 0/9 |

An isolated local checkout compared two videos with identical pictures and audio outside **3.300–3.700 s**. The quiet version had speech elsewhere in the clip and measured **0/19** periodic blocks in the held interval. The inserted-speech version measured **19/19**. Chrome displayed the waveform, playable audio, and three matching source images and crops. The PDF and JSON retained the same interval and measurement. Signed replay verified both the WAV and waveform against the original video. [Local comparison counts](evaluation-panels/video-mouth-hold-sound-2026-11-13.json)

A second checkout shifted the entire audio stream origin by **500 ms** while preserving video timestamps. The same **3.300–3.700 s** source interval selected decoded audio samples **44,800–51,200**. The waveform showed the original source clock, the JSON kept both clocks, and the PDF, Chrome waveform and signed replay verified the resulting excerpt. Older signed packages still replay their original waveform bytes. [Delayed-audio replay receipt](evaluation-panels/video-mouth-hold-source-clock-replay-2026-11-13.json)

## Timestamp breaks outside the selected sound

The app now checks continuity within the selected interval. A discontinuity
elsewhere in the soundtrack can leave this local measurement available.
Every measured sample must follow an unambiguous source and decoded clock.
A gap, overlap or ambiguous mapping inside the interval leaves the sound
unassessed and retains the picture evidence.

Three isolated paid-flow runs reused the same archived pictures and an
authored periodic tone. An actual 200 ms break in the MP4 sound timestamps
occurred before, during or after the hold at **3.300–3.700 s**. The first
and last cases each returned 6,400 playable samples, the same three
source frames and crops, and a waveform on the picture clock. A separate
FFmpeg decode matched the WAV samples. The middle case kept all three
frames and withheld the waveform and audio. Standard PDF, JSON and signed
source replay agreed in all three cases. Chrome loaded the waveform and
the 400 ms clip for the case with the earlier gap. These cases test the evidence
coverage of the workflow. The measurements carry zero AI-confidence
weight. [Worker and replay receipt](evaluation-panels/video-mouth-hold-audio-gap-2026-11-13.json)

Use the digest-pinned `diego/mouth_late.mp4` from the
[HD fixture replay](video-mouth-hd-hold-2026-11-12.md#replay). The command
creates the three MP4s and uses a local isolated worker with simulated
Stripe payment.

```sh
PYTHONPATH=backend OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  backend/.venv/bin/python -B -m scripts.video_mouth_hold_clock_app "$held_hd_clip" "$new_private_run"
```

## Loud background comparison

A separate fixed DSP probe evaluated the center 400 ms of 116 retained
speech and background deliveries. Each of 36 source files has a clean
clip and two loud mixtures with speech five decibels below the added
sound. Eight isolated excerpts from the same two noise recordings are
included. Ten clean excerpts met the existing periodicity rule by
themselves. This gives 20 matched noisy opportunities. The 100 Hz and
180 Hz high-pass views use fourth-order filters forward and backward
across the complete three-second input, then measure the same 400 ms.
[Noise source terms and earlier delivery freeze](audio-background-persistence-2026-11-13.md#holdout-and-decision-rule)

| View | Original periodicity retained | Noisy periodicity recovered | Isolated background flags |
| --- | ---: | ---: | ---: |
| Existing raw view | 10/10 | 6/20 | 0/8 |
| 100 Hz high-pass | 8/10 | 7/20 | 0/8 |
| 180 Hz high-pass | 6/10 | 8/20 | 0/8 |

The 180 Hz view gained three noisy measurements and lost one. It also
lost four clean measurements. Both filters remain experimental. These
audio-only comparisons use two noise tracks and say how the periodicity
measurement behaves when an audio segment has an observable repeat. They
give no estimate of lip-sync accuracy or who is speaking.
[Measurements and frozen criteria](evaluation-panels/video-mouth-hold-noise-probe-2026-11-13.json)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.video_mouth_hold_noise_probe \
  docs/evaluation-panels/video-mouth-hold-noise-probe-2026-11-13.json
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_mouth_hold_noise_probe \
  "$new_receipt" --source "$prepared_noise_run/wav"
```

## Sound evidence on two-second clips

The same measurement now runs on matched mouth holds in the existing
50-picture, 25 fps profile for upright 360 × 288 or 480 × 360 H.264
video. This gives the short-clip picture check a playable mono excerpt,
a source-clock waveform, and an acoustic periodicity count. If the
soundtrack cannot be measured, the report keeps the picture evidence
and gives a coverage message. [Short picture profile and held results](video-mouth-local-hold-2026-11-12.md)

In two isolated local analyses, the sound was the same authored 150 Hz tone
with a 250 ms audio stream delay. The held-mouth video located the exact
**0.520–0.840 s** interval. Frames **12, 17 and 21** gave before, during and
after views. The waveform and sound excerpt came from decoded samples
**4,320–9,440**, mapped to source ticks **8,320–13,440** at 16 kHz. All
14 measured audio blocks had periodicity. The original-picture control
with the same sound returned no mouth hold. No analysis job was queued
before simulated Stripe payment. The tone demonstrates the measurement
and can trigger it by itself. This is a workflow check, with no estimate
of speech or deepfake detection accuracy.

The JSON, evidence API, Standard PDF and signed export matched. The
audio-only verifier now accounts for the mouth-hold WAV and also rebuilds
its waveform. The original-video verifier additionally rechecks the
native picture measurements. Three older HD exports with gaps before,
during or after the selected range also passed both source replays.

```sh
export PYTHONPATH=backend
python -m app.cli verify evidence.zip --public-key trusted.pem --source-video original.mp4 --source-audio original.mp4
```
