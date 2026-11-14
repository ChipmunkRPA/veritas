# Local loss of stereo differences

New audio and video analyses can locate short periods of identical left
and right audio with differing channels before and after. The ordinary
view shows the source clock, a highlighted waveform, and left, right
and mono playback with 80 ms of context on either side. The Standard
PDF and JSON use the same source positions. A signed replay from the
original recomputes the exact intervals and verifies the exhibits.

Audio recorded or mixed intentionally as dual mono can legitimately
contain matching channels. An isolated match can come from an ordinary
mix or a mono insert. This observation carries zero weight in the
overall AI assessment. It does not attribute an edit to AI, identify
the changed channel, or establish who is speaking.
[SWGDE audio recording guidance](https://www.swgde.org/documents/published-complete-listing/08-a-001-swgde-best-practices-for-forensic-audio/)

## Method

The check uses the aligned 16 kHz stereo prefix, capped at 30 seconds,
and keeps sample-exact equality runs lasting 160 ms to 2 s. It requires
nonconstant audio in each 80 ms block in the run, including the last
80 ms, and separately measures variation of half the left-minus-right
signal in each 80 ms flank. All three parts must fit within one
continuous source clock segment. The 80 ms signal floor is −70 dBFS.
Both flanks must meet −75 dBFS on the stereo difference signal. These
are measurement floors, not probabilities. Whole-track dual mono
cannot supply two distinct adjoining flanks. The three longest
qualifying intervals are presented in source order. Exact boundaries
describe the decoded clock and can include naturally equal edge samples.
Lossy encoding or resampling can remove the exact equality.

## Matched recordings

The first comparison used a −50 dBFS floor on both measurements.
On its two held AMI meetings, it located four of 48 lossless edits
exactly, all on one meeting, and flagged none of 144 controls. The
first comparison failed its frozen target. Inspection of the two
development meetings showed that native differences were often
below −60 dBFS. The revised floors were fixed before opening the
two new held microphone pairs for measurement.
[First comparison](evaluation-panels/audio-stereo-identical-initial-2026-11-14.json)

The revised held set uses 24 different three-second excerpts of
**ES2010a** and **IS1005a**, distinct from the two development meetings
and the earlier held set. Each excerpt contributes four edited
deliveries and six controls, which keeps the comparison matched.
The two matched channels come from AMI's two tabletop microphones
Array1-01 and Array1-05. Conditions copy one channel locally or
write their average to both, keeping the source clock. The same two
edits were also encoded as AAC at 96 kbps. Controls are untouched
WAV, untouched AAC, a right-channel gain change, a whole-track mono
mix, a local mute on both channels and a nonidentical stereo crossfade.
The untouched speeches and room backgrounds are naturally recorded.
All edits are authored signal operations. No original audio is
included in the public receipts.
[AMI license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)
· [AMI capture notes](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml)

| Held delivery | Exact intervals | Useful overlaps |
| --- | ---: | ---: |
| Local channel copy, lossless | 24/24 | 24/24 |
| Local channel average, lossless | 23/24 | 24/24 |
| Local channel copy, AAC | 0/24 | 0/24 |
| Local channel average, AAC | 0/24 | 1/24 |
| Matched controls | — | 0 flags/144 |

Useful means temporal intersection over union of at least 0.50. The one
inexact lossless match includes equal samples next to the authored range.
The prior one-second channel check and exact silence check flagged none
of the lossless edits. The test covers two independent held meetings,
not 240 independent recordings. It does not measure general AI-audio
accuracy or field prevalence of ordinary local mono mixing.
[Revised measurements](evaluation-panels/audio-stereo-identical-held-2026-11-14.json)

## Local app and replay

A local simulated USD 9.99 checkout sent an edited held WAV through
the normal worker. Zero jobs ran before payment. The API, JSON, Standard
PDF and signed source replay agreed on samples **[11357, 15197)**,
or **0.710–0.950 s**. Both channel playbacks matched the original
decoded samples. A separate video integration test exercised a
soundtrack with a different clock from the picture.
[App receipt](evaluation-panels/audio-stereo-identical-app-2026-11-14.json)

The ledger lists the original microphone URLs, file digests and clocks.
Save each WAV in the applicable directory under the filename recorded
in the ledger. `review` uses only the Python standard library and
recounts all recorded outcomes. `run` uses the application's decoder
and recomputes the observations locally. The first experiment's exact
ledger and frozen code hashes are preserved.

```sh
export PYTHONPATH=backend
panel=docs/evaluation-panels/audio-stereo-identical-held-2026-11-14.json
python3 -B -S -m scripts.audio_stereo_identical_transfer review "$panel"
backend/.venv/bin/python -B -m scripts.audio_stereo_identical_transfer freeze "$new_freeze" \
  --development "$development_wavs" --held "$held_wavs"
backend/.venv/bin/python -B -m scripts.audio_stereo_identical_transfer run "$new_receipt" \
  --development "$development_wavs" --held "$held_wavs" --freeze "$new_freeze"
backend/.venv/bin/python -B -m scripts.audio_stereo_identical_app "$held_wavs" "$panel" "$new_app_run"
```
