# Withhold the voice estimate when stereo cancellation hides sound

Chipmunk Verify scores the decoded mono waveform for synthetic speech.
Two opposing stereo channels can cancel some of the sound before that
check sees it. A usable mono response can still come from the sound that
remains. Reporting it as a recording-wide estimate would extend the
conclusion to audio that the detector could not hear.

New standalone audio analyses use the existing aligned stereo measurements
to decide when to withhold the recording-wide estimate. In each complete
one-second window of the first 30 seconds, the rule requires both a strong
drop in the mono level and strongly opposing channel waveforms. The
existing measurement defines a strong drop as at least 12 dB below the
louder channel or digital silence with an audible original channel. The
correlation must be at most −0.85. The rule describes coverage and supplies
zero directional weight. It does not identify the hidden sound or explain
why the channels oppose each other.
[Original stereo measurement method](audio-paired-stereo-overlap-2026-11-11.md)

The report explains the withheld estimate. The evidence view and Standard
PDF show the affected times next to the channel comparison. The JSON
`/coverage/mono_voice_coverage_limit` entry points to the exact measured
windows. Their WAV excerpts and waveform retain the left, right and
analyzed mono signals. Existing interval scores remain available for
inspection. The guard applies to the two standalone voice-fusion paths.
The [channel-difference view](audio-cancelled-speech-transfer-2026-11-15.md)
adds derived playback to selected cancellation intervals.
The [video soundtrack version](video-soundtrack-downmix-coverage-2026-11-15.md)
uses the same stereo measurements. This rule covers
measurable stereo cancellation on supported clocks and does not measure
speech masked acoustically in the individual channels.

## Local verification

Four held PhonemeDF prompts from three readers supplied matched
three-second input groups. Each group contains a mono original, an
equal-channel stereo copy, a one-sided stereo copy, and a stereo copy
with opposing sound at 1–2 seconds. That opposing sound includes a
known replacement from a second prompt. The fixtures alternate two
synthesis systems. All four files in each group had byte-identical
decoded mono samples. The stereo measurements place the cancellation
at 1–2 seconds in all four modified files. Their measured loss ranges
from 13.393 to 23.717 dB.
[Source terms and role separation](audio-short-edit-detector-training-2026-11-14.md#source-and-fitting-roles)

| Files | Previous rule | With the coverage check |
| --- | --- | --- |
| Four with stereo cancellation | Four leaned original | Four inconclusive, without a recording-wide estimate |
| Twelve ordinary controls | Twelve leaned original | All twelve decisions and estimates matched |

The comparison ran the actual local voice model and both versions of
the coverage policy. All 32 simulated USD 9.99 checkouts started their
analyses only after payment. The API and JSON agreed on each result's
evidence. All four changed results bound the original-channel playback
and waveform bytes to the affected interval. One generated Standard PDF
matched the JSON, and its signed package was verified against the
original stereo file. The portable review checks counts, original
sample hashes and times without loading a model. The fixtures test a
coverage failure in controlled WAVs. They supply no real-world error
rate for synthetic voice, cancellation under codecs, or background noise.
[Numeric verification receipt](evaluation-panels/audio-downmix-coverage-app-2026-11-15.json)

## Replay

```sh
export PYTHONPATH=backend
module=scripts.audio_downmix_coverage_app
receipt=docs/evaluation-panels/audio-downmix-coverage-app-2026-11-15.json
python3 -B -S -m "$module" --review "$receipt"
```

For the native app replay, recreate the input directory (`$inputs`)
with the room-training freeze, and use the existing local speech assets.
Choose new temporary directories for the two runs. Keep source audio,
exports and keys in the local runs.
[Native source reconstruction](audio-short-edit-room-training-2026-11-15.md#reproduce-locally) ·
[Local voice setup](local-audio-rollout.md)

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
backend/.venv/bin/python -B -m "$module" "$inputs" "$run/before" --before
backend/.venv/bin/python -B -m "$module" "$inputs" "$run/after"
```
