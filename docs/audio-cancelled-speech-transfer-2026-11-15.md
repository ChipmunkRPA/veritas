# Inspect speech lost when stereo is mixed to mono

When opposing stereo channels cancel in the analyzed mono copy, Chipmunk
Verify already withholds its recording-wide speech estimate. The stereo
comparison now adds `(left − right) / 2` as a 16-bit WAV and a fourth
waveform for the displayed cancellation intervals. This derived clip
can make opposing sound easier to hear. The original left and right
clips and the analyzed mono clip remain available for comparison.
[Audio coverage](audio-downmix-coverage-2026-11-15.md) ·
[Video soundtrack coverage](video-soundtrack-downmix-coverage-2026-11-15.md)

The difference is an inspection aid. It can contain speech, ambient
sound and recording effects. It carries zero weight in the AI estimate.
It uses the same source timestamps and one-second excerpts as the stereo
comparison, which displays up to three selected windows. The PDF labels
the derived waveform. JSON records the audio and waveform hashes with
those same samples. Signed source replay recomputes both from the
original audio or video. Historical three-track evidence remains readable.

## Voice classifier comparison

The evaluation asks if the existing voice classifier can use speech that
the mono input loses. It compares the mono window, the original left
channel, the channel difference, and the change in the response between
the difference and mono. All inputs use the existing model's
preprocessing and the pinned speech activity model. Each model reading
requires at least 400 ms detected speech. Non-mono paths also require
measured destructive cancellation in the existing stereo check.
[Commercial asset audit](commercial-model-audit.md) ·
[Measured noise variation in voice detection](https://arxiv.org/abs/2512.13744) ·
[Noisy speech localization research](https://arxiv.org/abs/2507.16220)

The freeze contains 64 genuine calibration deliveries and 328 assessment
deliveries across 66 total source groups. Sixteen PhonemeDF calibration
groups set research points from their genuine source-group maxima. The
stricter point uses the largest maximum, and the wider point uses the
nearest-rank 90th percentile. Points were fixed before any held scoring,
separately by input view and delivery condition. The declared noise
condition comes from the evaluation design. Applying a condition-specific
point to an upload would require a validated way to identify that
condition from the recording.

The held panels use 24 PhonemeDF groups and eight stock groups with
conventional splices and two synthetic replacements each, plus 18 SINE
groups with matched genuine and edited speech. Calibration and held voice
roles stayed separate. The same known edit locations appear in newly
authored lossless stereo mixtures. A one-second excerpt centered on the
known edit supplies the opposed sound, including the matching genuine
excerpt in the controls. Conditions compare isolated and louder
competing speech inside that opposed signal. Every group uses the same
genuine mono input across its labels and both conditions. The stereo
decoder recovered the exact prepared difference on all 392 files. The
cancellation check qualified 391 of them. The staged window gives the
correct clock to the evaluation, so the result measures classification
at a known interval. Field frequency, codec transfer and discovery of
arbitrary edit locations require separate evidence.
[PhonemeDF and stock source roles](audio-short-edit-room-training-2026-11-15.md#source-roles-and-delivery-conditions) ·
[SINE source separation and terms](audio-short-edit-background-training-2026-11-15.md#fresh-source-comparison)

| Held panel | Condition | Left flags, edits · genuine | Difference flags, edits · genuine | Response change flags, edits · genuine |
| --- | --- | ---: | ---: | ---: |
| PhonemeDF | Isolated | 1/48 · 0/48 | 2/48 · 0/48 | 5/48 · 1/48 |
| PhonemeDF | Competing speech | 6/48 · 10/48 | 7/48 · 9/48 | 3/48 · 4/48 |
| Stock | Isolated | 0/16 · 0/16 | 0/16 · 0/16 | 1/16 · 0/16 |
| Stock | Competing speech | 0/16 · 0/16 | 0/16 · 0/16 | 0/16 · 0/16 |
| SINE | Isolated | 2/18 · 0/18 | 1/18 · 0/18 | 7/18 · 0/18 |
| SINE | Competing speech | 4/18 · 0/18 | 2/18 · 0/18 | 2/18 · 0/18 |

At the stricter point, mono flagged zero edits and zero controls in
every panel. The PhonemeDF difference check's seven competing-speech
flags always had a flagged or unavailable matched genuine comparison.
At the wider point, the stock difference check added one finding with
a clear matched original. The numeric release counts unassessed windows
separately. For example, seven of the 18 isolated SINE edits could not
receive a difference-versus-mono response because the mono speech gate
did not admit the clip. The gains varied across source panels and types
of background. The classification candidates remain in evaluation.
[Numeric receipt](evaluation-panels/audio-cancelled-speech-transfer-2026-11-15.json.gz)

## Local app verification and replay

Three USD 9.99 payment simulations exercised a standalone WAV and two
video deliveries with ALAC and AAC soundtracks. Each queued analysis
only after payment. The new clip exactly matched the 16-bit comparison
of the decoded channels at 1–2 s in all three files. The browser
component test checks the matching playback URL and waveform on that
source clock. API and JSON evidence matched, the PDFs labeled the
derived waveform, and all three signed source replays verified. The
speech assessments were inconclusive. Two historical signed packages
also replayed against their originals.
[App receipt](evaluation-panels/audio-stereo-difference-app-2026-11-15.json)

The public replay needs only standard Python. A separate numeric recount
matched all 48 held result rows and all control points on Python 3.12
and 3.14. A second native run reproduced all 64 calibration rows exactly.
Raw speech and hidden states stay local. Native replay needs the
retained matched room and SINE inputs, the earlier SINE case WAV and PCM
reconstruction, and the three pinned runtime files. The local evidence
fixture also verifies the new and old plot and the signed source replay.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
runner=backend/.venv/bin/python
module=scripts.audio_cancelled_speech_transfer
python3 -B -S -m "$module" review docs/evaluation-panels/audio-cancelled-speech-transfer-2026-11-15.json.gz
"$runner" -B -m "$module" freeze "$room_inputs" "$sine_frozen" "$sine_reads" "$run/input"
"$runner" -B -m "$module" score "$run/input" "$voice_weights" "$run/calibration.json" --phase calibration
"$runner" -B -m "$module" score "$run/input" "$voice_weights" "$run/held.json" \
  --phase held --calibration "$run/calibration.json"
"$runner" -B -S -m "$module" combine "$run/calibration.json" "$run/held.json" "$run/receipt.json.gz"
"$runner" -B -m pytest -q backend/tests/test_audio_stereo_context.py -k actual_audio_stage_packet
```
