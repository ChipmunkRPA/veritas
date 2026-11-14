# Check brief speech after the main audio review skips its windows

The main voice check samples four-second windows. In a six-second WAV,
both longer windows can fail their speech requirement even when a
two-second window has enough speech. The local opt-in lets the existing
two-second check run following this specific coverage skip. Its own
speech requirement, score point and two-second timing are unchanged.
It gives a source-linked lead with zero weight in the overall assessment.
Trusted Content Credentials still stop the statistical checks.

## Matched local comparison

Use the 12 matched groups from the previously bound public-source
evaluation. Each group contains a synthetic edit, an untouched recording,
a splice by the same speaker, and a splice using another genuine speaker.
Four isolated room inputs give additional controls. The synthetic inserts
come from three previously identified generator families. This study
reuses their original foreground and −5 dB competing-speech mixtures.
[Source roster and terms](audio-remote-neighbor-replication-2026-11-12.md)

In each group, take the same 1.65 s excerpt centered as close as possible
to the labeled edit. Put it at 2.200 s in a six-second file with silence
outside the excerpt. Keep the same time and audio samples across each
matched comparison. This authored placement tests the gap between the
two schedules. It gives no accuracy estimate for natural pauses or edits
elsewhere in an upload. Earlier research reuse of the sources also limits
what the comparison says about unseen speech.

| Outcome | Foreground | With competing voices |
| --- | ---: | ---: |
| Longer window available | 0/52 | 0/52 |
| Edited deliveries reaching the short check | 12/12 | 12/12 |
| Edits usefully located at the existing point | 12/12 | 1/12 |
| Ordinary files flagged | 8/36 | 3/36 |
| Isolated background or silence flagged | 0/4 | 0/4 |

Useful means that a reported window covers at least half of the labeled
edit and reaches 0.30 temporal intersection-over-union. The eight
foreground flags are ordinary splices across six groups. All three noisy
ordinary flags came from the same group as the one located synthetic
edit, including its untouched match. The uncovered foreground ordinary
file and six isolated controls are included in the denominators. The
result supports a local inspection trial and provides no basis to give
these flags an authorship score or enable the route by default.
[Full counts and source commitments](evaluation-panels/audio-short-admission-transfer-2026-11-14.json)

## Local app and replay

For a development instance in candidate audio mode, set
`VERITAS_AUDIO_SHORT_ADMISSION_REVIEW=true`. The option is limited to
existing Full short-review inputs, which are six continuous seconds of
supported native mono 16 kHz WAV, FLAC or the previously qualified MP3
profile. The measured comparison used PCM16 WAV only. The report says
that the longer windows were skipped and explains that brief ordinary
speech can produce an elevated response. The waveform marks the full
two-second window. Browser, PDF and JSON identify each checked or skipped
window and its source timestamps. The primary assessment can still be
inconclusive.

Three simulated prepaid runs checked a known noisy edit, its noisy
untouched match, and silence. Both speech files flagged 2.000–4.000 s.
Silence ran no voice inference in the short check. Each result kept a
blank overall probability and no supporting confidence contributions.
The paid worker, PDF, JSON and signed ZIP agreed. A separate verifier
recreated the per-window speech counts from the exact originals.
[App receipt](evaluation-panels/audio-short-admission-app-2026-11-14.json) ·
[Original-audio verification](audio-interval-learning-2026-11-13.md#app-coverage)

The scalar reader needs only the repository and standard-library Python.
For native replay, use the retained 52 original mixtures and stems from
the parent study, plus the pinned voice and speech-activity assets. The
native audit verifies the 104 delivered PCM hashes and source positions.
A second native scoring run reproduced the complete receipt byte for byte.
No raw audio or model weights are included with this receipt.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
receipt=docs/evaluation-panels/audio-short-admission-transfer-2026-11-14.json
python3 -B -S -m scripts.audio_short_admission_transfer review "$receipt"
backend/.venv/bin/python -B -m scripts.audio_short_admission_transfer audit "$receipt"
backend/.venv/bin/python -B -m scripts.audio_short_admission_transfer score "$new_receipt" --fixtures "$new_fixture_dir"
```
