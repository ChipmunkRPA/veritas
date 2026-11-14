# Ranking local soundtrack differences after checking the flanks

The earlier-copy soundtrack trial compares two clips that have the same picture
clock. It looks for a short loss of matching sound surrounded by matching
sound on both sides. The current rule ranks possible intervals and then checks
that the top interval has sufficiently strong flanks. A lower-ranked interval
can have more convincing flanks. This comparison tests checking those flanks
before ranking, using the same 0.84 correlation floor in two voice bands.
The second candidate uses a 0.52 floor and suppresses the proposed range if
an offset search on band-limited sound supports a local delay. The original
short timing comparison runs first in all three arms. No additional model
weights or code from the referenced papers were used. [Previous check](video-reference-sound-difference-2026-11-14.md)
· [Matched ordinary splice controls](https://arxiv.org/abs/2610.04479)
· [Separate speech and background measurements](https://arxiv.org/abs/2608.09593)

## Inputs and outcomes

Six commercially permitted Wikitongues recordings were separate from the
earlier soundtrack-difference panels. They had prior project use for a
different video benchmark. This evaluation used two four-second excerpts per
recording. The adapter produced aligned 720p25 pictures and a continuous
16 kHz soundtrack, including clock normalization before constructing the
controls and edits. Each excerpt contributed two staged speech substitutions
at the same picture positions, using words from the same or another speaker.
Ordinary controls included local timing edits, filtering, gain changes,
background-only differences, constant timing shifts, unrelated speech and
unchanged copies. Added rain, train or competing speech was about 5 dB above
the original voice RMS. PCM and 64 kbps AAC used the same cases.
[Source names, licenses and original file hashes](evaluation-panels/video-sound-difference-guards-2026-11-14.json)
· [Background recordings](audio-focused-window-transfer-2026-11-12.md#frozen-comparison)

Each condition had 24 staged substitutions and 96 controls in each format.
A useful location requires at least 0.50 temporal intersection over union.
The frozen rule requires three useful gains across two or more recordings,
preservation of all previous useful locations, at most two new control flags
per carrier, and no new flags on unrelated or constant-shift pairs.

| Carrier and condition | Current useful | Strong flanks checked first | Lower floor with delay check |
| --- | ---: | ---: | ---: |
| AAC, clean | 11/24 | 17/24 | 16/24 |
| AAC, rain | 0/24 | 7/24 | 12/24 |
| AAC, train | 0/24 | 0/24 | 13/24 |
| AAC, competing speech | 0/24 | 0/24 | 14/24 |
| PCM, clean | 11/24 | 17/24 | 16/24 |
| PCM, rain | 0/24 | 4/24 | 12/24 |
| PCM, train | 0/24 | 0/24 | 13/24 |
| PCM, competing speech | 0/24 | 0/24 | 14/24 |

Checking strong flanks first kept all previous useful locations and added 13
AAC and ten PCM locations across five source recordings. Control flags rose
from one to four of 384 in AAC and from one to five of 384 in PCM. All added
control flags involved a real local timing edit. These are useful acoustic
differences for inspection, with an unresolved cause in this check. The
measurements produce two additional wrong locations on edited AAC clips
and five on edited PCM clips. The tradeoff supports retaining the candidate
for further comparison with a stronger test for movable audio.

The lower-floor arm kept the previous useful locations and added 44 in
each carrier, including 39 noisy locations. It added 21 AAC and 19 PCM
control flags. Those flags included local filtering, gain and local delay.
The two candidates fell outside the frozen control limits for replacing the
existing selection. The stronger-flank check is now available as a
supplemental observation in the explicitly opted-in local comparison, with
zero weight in the AI estimate. The more permissive arm stays in evaluation.

## Local trial and evidence

The trial first checks for a supported timing shift and preserves an existing
soundtrack range when it has strong matches on both sides. If the strongest
range has weak flanks, the trial can display a different range that meets the
same matching requirements. These additional ranges appear as **Soundtrack
difference with unresolved cause**, with the original timestamps, both
waveforms and playable clips, and three matching picture pairs. The same
selection and exhibits can be reproduced from both source files and the JSON
with the existing comparison verifier.

Three simulated $9.99 checkouts exercised the local app. The rain-masked
replacement and rain-masked local delay both showed **1.400–2.200 s** and six
verified frames. This check could not distinguish their causes. The clean
delay was identified by the app's existing timing comparison, which took
precedence and suppressed the additional range. All three record and asset
replays matched. No jobs ran before payment.

## Replay

The public receipt binds the clock, conditions, original video digests,
measured ranges, band values and hashes of the decoded audio pairs. It
contains no recorded clips. The standard-library replay recalculates all
counts. For native replay use the six publisher files named in
`freeze.sources`, along with the pinned `rain.ogg` and `train.ogg` files.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 -B -S -m scripts.video_sound_difference_guard_transfer review \
  docs/evaluation-panels/video-sound-difference-guards-2026-11-14.json
python3 -B -S -m scripts.video_sound_difference_guard_transfer freeze "$run/freeze.json"
backend/.venv/bin/python -B -m scripts.video_sound_difference_guard_transfer run "$run/receipt.json" \
  --freeze "$run/freeze.json" --bank "$video_sources" --sounds "$background_sources"
```
