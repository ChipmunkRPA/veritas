# Speech activity during a held mouth region

The local video trial now measures how much of a verified mouth-region
hold overlaps detected speech in the mixed soundtrack. The original
pictures, waveform, and playback share one source time range. The
measurement can help review a visible hold when rain or train noise
masks periodic speech. It cannot assign the sound to a visible person
or link the hold to AI. It contributes zero weight to overall AI confidence.

The speech check reads one continuous soundtrack context of one to six
seconds around the hold. It counts only the portion of the detected speech
that intersects the held pictures. The report shows the duration in
milliseconds and describes the speech when it covers at least 200 ms.
Digital silence records zero. Unsupported context leaves the existing
periodicity and level measurements available.
[Native mouth-hold binding and picture profiles](video-mouth-hold-sound-2026-11-13.md)
· [Pinned speech-activity asset terms](https://github.com/snakers4/silero-vad/blob/066409f3d9db45a30d21dfcbf9147826ffdd7052/LICENSE)

## Background and speech transfer

Fifteen SINE readers outside the prior mouth-hold noise study supplied
15 genuine speech excerpts and their 15 publisher-labeled edited
counterparts. Each contributes one 400 ms target window centered on its
published splice clock. The 165 measurements include both versions clean
and mixed with each background, plus matched background-only controls.
The four rain and four train excerpts and the meeting backgrounds come
from existing evaluation sources. They repeat across multiple windows.
Rain and train are mixed at 5 dB above the original's six-second RMS.
The study measures speech evidence on known audio intervals. It does not
estimate lip-sync accuracy on arbitrary clips.
[Held readers and licensing](audio-narrow-flag-transfer-2026-11-14.md#new-readers-from-the-publisher-archive)
· [Pinned environmental audio](audio-focused-window-transfer-2026-11-12.md#frozen-comparison)
· [Prior activity comparison](video-mouth-hold-background-transfer-2026-11-14.md#speech-activity-comparison)

| Delivery | Sustained periodicity | Speech activity |
| --- | ---: | ---: |
| Clean speech | 3/30 | 30/30 |
| Speech with rain | 0/30 | 25/30 |
| Speech with train | 0/30 | 15/30 |
| Speech with other voices | 0/30 | 30/30 |
| Rain or train alone | 0/30 | 0/30 |
| Other voices alone | 0/15 | 6/15 |

Other voices triggered the new check on their own. These are valid speech
observations and show why the result gives no speaker attribution. A
separate local timing comparison tried 960 ms flanks to find a stable
offset before and after a short soundtrack move. Uniform delay and
matched picture-and-sound moves served as controls. Neither the aperture
and level arm nor the arm with added signed motion located any of 30
held shifts at its fixed point. Both remain outside the timing detector.
Recent synchronization research also distinguishes shifts in timing from
articulation of particular sounds. [Timing and articulation study](https://arxiv.org/abs/2610.09223)
· [Benchmark separation of timing and content](https://github.com/Duck-Boss/AV-SyncBench)
· [Transfer receipt](evaluation-panels/video-mouth-hold-speech-transfer-2026-11-14.json)

## Local flow

Enable `VERITAS_VIDEO_MOUTH_HOLD_SPEECH_CONTEXT=true` locally. In a
new video analysis with a supported mouth hold, open the finding to see
the stills, waveform, soundtrack and linked measurements.

Three simulated $9.99 checkouts reused the same held pictures at
**0.520–0.840 s**. Speech masked by train noise and isolated competing
voices each measured **320 ms** of speech activity. The train-only clip
measured **0 ms**. The API, JSON, Standard PDF and signed replay with the
original video agreed on source samples **[8,320, 13,440)** and source
frames **12, 17 and 21**. No analysis started before simulated payment.
[Local checkout receipt](evaluation-panels/video-mouth-hold-speech-app-2026-11-14.json)

## Replay

The standard-library recount verifies the exact window clocks, pinned
source roster and counts. For a native replay use the earlier held SINE
bank, the recorded rain and train files and the existing pinned activity
weights. For the checkout replay, use the retained picture-only H.264
hold from the earlier two-second verification. The runner verifies its
byte digest. No media is included in the receipts.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
receipt=docs/evaluation-panels/video-mouth-hold-speech-transfer-2026-11-14.json
python3 -B -S -m scripts.video_mouth_hold_speech_transfer "$receipt"
backend/.venv/bin/python -B -m scripts.video_mouth_hold_speech_transfer "$new_receipt" \
  --bank "$sine_bank" --sounds "$backgrounds" --checkpoint models/weights/audio-upgrade/silero_vad.jit
backend/.venv/bin/python -B -m scripts.video_mouth_hold_speech_app \
  "$local_app_output" "$mouth_hold_picture" "$sine_bank" "$backgrounds"
```

The signed package can be checked without the source files using the
instance's public key. Exact source replay also verifies the additional
speech measurement and needs the pinned activity asset in the local
project environment.

```sh
python -m app.cli verify evidence.zip --public-key trusted.pem --source-video original.mp4 --source-audio original.mp4
```
