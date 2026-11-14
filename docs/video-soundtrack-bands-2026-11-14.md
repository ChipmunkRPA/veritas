# Soundtrack timing under rain and train noise

The local two-track video check now has an optional second pass. If the
full-band check finds no local shift, the second pass looks for matching sounds
in separate frequency bands. When it finds a range, the analysis, Standard PDF
and JSON report the signed time difference, both source intervals and which
measurement supported the finding. The browser plays the exact two excerpts.
The check compares two streams in one short video and carries zero weight in
the AI estimate.

## Method

The second pass uses three Butterworth band-pass filters at 250–1,350 Hz,
1,350–3,500 Hz and 3,500–7,000 Hz. It runs the existing 80 ms waveform search
every 20 ms in each band. A window qualifies if two bands agree within 1 ms
and both pass the signal, correlation and competing-offset checks. The weaker
of the two correlations is recorded for the qualifying window. The same rules
still apply for nearby same-time matches and for same-time support across the
clip. The first-pass result takes precedence. No model weights were added.
Frequency-selective matching and time synchronization informed the design.
[Landmark cross-correlation for synchronization](https://research.adobe.com/publication/clustering-and-synchronizing-multi-camera-video-via-landmark-cross-correlation/)
· [Frequency weighting under noise](https://www.isca-archive.org/interspeech_2021/salvati21_interspeech.pdf)

## Matched test

Seven Wikitongues recordings supplied two newly prepared four-second excerpts
each. They are different recordings from the earlier two-track test and had
been used elsewhere in the project. Edits moved the soundtrack locally by
200 ms in either direction. Moderate noise was about 7 dB below voice RMS
and loud noise about 5 dB above it across the excerpt. Controls include
untouched copies, gain and filtering, different speech, the same local shift
in both tracks, whole-track delay with background sound, different recorded
backgrounds, a repeating sound bed, and unsupported stream counts and clocks.
PCM and AAC reuse the same underlying edits and controls. Sources, credits,
terms and exact file hashes are in the receipt.
[Prior two-track test](video-soundtrack-pair-2026-11-14.md)
· [Recorded noise and terms](audio-focused-window-transfer-2026-11-12.md#frozen-comparison)
· [Per-delivery receipt](evaluation-panels/video-soundtrack-bands-2026-11-14.json)

| Outcome per format | Full-band check | With second pass |
| --- | ---: | ---: |
| Clean shifts located | 28/28 | 28/28 |
| Moderate rain or train shifts located | 13/28 | 15/28 |
| Loud rain or train shifts located | 0/28 | 2/28 |
| Total useful locations | 41/84 | 45/84 |
| Incorrect range on an edited clip | 0/84 | 1/84 |
| Controls with a flagged range | 0/252 | 0/252 |

All counts in the table hold separately for PCM and AAC. A useful location
needs at least 0.50 temporal intersection over union and the intended signed
lag within 1 ms. The four new locations came from four source recordings.
One additional loud-rain range found the right lag but covered only 44% of
the edited interval. The panel includes 56 structurally unassessed controls
per format. Both formats passed the predeclared rule for a development trial.
Most shifts in loud noise still escaped localization. Single-track video and
AI voice attribution are outside the scope of the check.

Two simulated $9.99 checkouts went through the local worker. No jobs ran
before payment confirmation. On a held AAC clip with loud rain, the app
matched stream 1 at **1.140–1.460 s** to stream 2 at **1.340–1.660 s**.
The browser contract, Standard PDF, JSON and signed source replay agreed on
the 200 ms difference, both waveforms and both clips. The different-speech
control produced no timing finding. [App receipt](evaluation-panels/video-soundtrack-bands-app-2026-11-14.json)

## Replay

Enable `VERITAS_VIDEO_SOUNDTRACK_PAIR_REVIEW` and
`VERITAS_VIDEO_SOUNDTRACK_FREQUENCY_REVIEW` in development. The second pass
is off by default. Run a new analysis and open **Voice evidence from this
video**, then **Timing between audio tracks**.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
receipt=docs/evaluation-panels/video-soundtrack-bands-2026-11-14.json
python3 -B -S -m scripts.video_soundtrack_bands_transfer review "$receipt"
backend/.venv/bin/python -B -m scripts.video_soundtrack_bands_transfer run "$run" \
  --bank "$published_recordings" --sounds "$pinned_backgrounds" --examples
backend/.venv/bin/python -B -m scripts.video_soundtrack_pair_app "$run" "$app_run" \
  --band-example "$run/aac-3-44-later_rain_loud.mp4"
```

The public replay recounts the recorded results without model weights or
media. The native run uses the seven original files listed under
`freeze.sources` and the same pinned `rain.ogg` and `train.ogg` as the
earlier comparison. The app verification runs the simulated prepaid checkout.
