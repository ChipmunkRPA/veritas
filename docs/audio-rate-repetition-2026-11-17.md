# Repeated sound at different playback speeds

Chipmunk Verify compares quieter sound at different speeds in short lossless
stereo recordings and eligible video soundtracks. Each match shows two separate
source time ranges, the relative playback speed, the aligned waveforms and the
original stereo clips. The PDF and JSON use the same source ranges and asset
identities. Loops, recurring alerts and ordinary editing can produce a match.
The finding contributes zero weight to the overall AI assessment.

## What was checked

The existing stereo alignment estimates a shared delay and level over active
windows. The new comparison reduces that shared sound, then proposes locations
from short waveform anchors at six relative speeds from approximately 1.05× to
1.25×. Two frequency bands must agree over a sustained interval of at least
180 ms on the longer occurrence. The compared ranges are disjoint and their
endpoints are mapped back to the actual source samples. The displayed waveforms
show each aligned excerpt at its recorded speed. Earlier fingerprinting research
motivated comparing across time and pitch scales. This implementation uses local
waveform correlations and introduces no new weights or third-party model code.
[Rate-aware fingerprinting research](https://biblio.ugent.be/publication/5754913)

The comparison runs alongside the same-speed check on eligible continuous
2.5–4 second 48 kHz stereo PCM16 or 16-bit FLAC. The tested speed changes shift
pitch together with speed. Pitch-preserving changes, AAC and arbitrary speeds
remain outside this check's verified scope. Set
`VERITAS_AUDIO_STEREO_RATE_REPEAT_REVIEW=false` to use only the existing
same-speed check when native stereo review is enabled.

## Local confirmation

The confirmation used 32 new DEMAND excerpts from a kitchen, open field, street
and metro. A different AMI meeting from development supplied the foreground
speech. Copies were placed before or after the source occurrence, with exact
durations recorded before mixing. A correct match requires both ranges to
overlap at least half their union with the expected ranges and the candidate to
identify the right speed ratio and faster occurrence. Each loudness uses the
same 32 source excerpts. [DEMAND and its CC BY-SA 3.0 licence](https://zenodo.org/records/1227121)
· [AMI and its CC BY 4.0 licence](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

| Confirmation outcome | Existing same-speed stereo check | With playback-speed comparison |
| --- | ---: | ---: |
| Correct pair with speech 10 dB louder | 0/32 | 24/32 |
| Correct pair with speech 18 dB louder | 0/32 | 24/32 |
| Controls with a playback-speed finding | — | 0/176 assessed |
| Controls that the new check could not assess | — | 48/224 |

The new check also returned ten incorrect pairs across the 64 staged files.
Six files received a pair without a correct match. Controls included unchanged
rooms, ordinary stereo mixes, unrelated pasted room sound, moving foreground
speech, retimed foreground speech, same-speed background copies and steady tones.
The same-speed comparison still reported 17 of the 32 deliberate same-speed
copies. The raw recordings and rendered media stay local. The published receipts
contain input commitments, ranges, statuses and counts.
[Development receipt](evaluation-panels/audio-rate-repetition-development-2026-11-17.json.gz)
· [Confirmation receipt](evaluation-panels/audio-rate-repetition-held-2026-11-17.json.gz)

## App replay

Seven USD 9.99 checkouts used simulated payment. No analysis or job was created
before paid confirmation. The same recording returned the same AI assessment
with playback-speed review enabled or disabled. WAV, FLAC and a video with a
soundtrack offset of 0.400 seconds produced the expected separate time ranges.
The report embeds both waveform images. Browser playback uses both original
sample counts. The signed packages and exported JSON reproduced the recorded
evidence against the supplied originals.
[App receipt](evaluation-panels/audio-rate-repetition-app-2026-11-17.json)

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
panel=docs/evaluation-panels/audio-rate-repetition-held-2026-11-17.json.gz
python3 -B -S -m scripts.audio_rate_repetition_panel review "$panel"
"$runner" -B -m scripts.audio_rate_repetition_panel measure "$demand" "$ami" \
  confirmation "$run/confirmation"
npm --prefix frontend run build
"$runner" -B -m scripts.audio_native_alignment_app "$run/confirmation" "$run/app" --rate --browser
```

The [confirmation roster](evaluation-panels/audio-aac-repetition-sources-2026-11-17.json)
pins the two original DEMAND channels per scene. Give the panel script that
roster as `recorded-sources.json` under `demand`, with the extracted WAVs in
their named scene directories. Use the original `TS3004a.Array1-05.wav` under
`ami`. The development sources use the [lossless stereo roster](evaluation-panels/audio-native-alignment-sources-2026-11-17.json)
and `ES2004a.Array1-05.wav`. Both commands verify the source hashes before use.
