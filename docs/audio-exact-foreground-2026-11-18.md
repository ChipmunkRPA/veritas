# Short speech edit checks on the exact foreground

The two-track separator did not add any edit that cleared its matched ordinary
control. This test asks whether the same 750 ms speech detector can locate
more edits when given the exact original foreground before it was mixed with
the louder room. The comparison reuses the separator study's fixed sample
positions, two decision points, and control recordings.
[Two-track diagnostic](audio-severe-overlap-separation-2026-11-18.md)
· [Original Opus experiment](audio-recurrent-spectrum-opus-2026-11-18.md)

## Fixed comparison

Eleven source groups supply an edited utterance, a matched ordinary utterance,
and their shared room recording. The underlying clips contain 75 known edits
from six generator families. At least ten of the 11 groups have multiple
simultaneous talkers in the room capture itself. The clean-source comparison
subtracts the exact prepared interference from the archived lossless mix
and verifies that the parts sum to the original samples. Only those isolated
foreground versions of the edited and ordinary utterances receive new model
calls. The original decoded Opus measurement and isolated-room control readings
come from the previous study. The exact foreground is available in the
benchmark because the mixture was constructed with known inputs.
[Matched-source preparation](audio-six-ten-context-2026-11-16.md#same-utterance-controls)
· [Published partial-rewrite sources](https://zenodo.org/records/14214149)
· [LibriTTS terms](https://www.openslr.org/60/)
· [AMI terms and attribution](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)
· [Current model asset terms](commercial-model-audit.md)

Every file uses the 4.000 and 4.500 second crop starts. The first five fixed
groups also use 6.000 and 6.500 seconds. Both 750 ms crops in each pair must
have at least half their samples admitted as speech and individually exceed
the same existing point. Each proposal must cover at least half the known
edit and reach 0.30 temporal intersection over union. Each proposal can
match one edit. A matched-control-clear location requires that neither
the same-time matched ordinary file nor the isolated room also provide
a matching proposal at that location.

Only eight of the 75 known edits have a geometrically matching proposal
on this schedule. In each foreground arm, 15 of the 16 scheduled pairs had
both crops admitted. The other pair was withheld on one edited file and
one ordinary file. The submitted mixture admitted all 16 pairs per arm.
The 62 new one-thread CPU calls kept the same weights, speech check, and
decision points. The points were originally measured on the submitted
carrier. The trial did not fit a separate point on the clean tracks.

## Results

| Reading | Known locations out of 8 | Locations clear of matched controls | Ordinary alerts out of 11 | Background alerts out of 11 | Edited files flagged only at the wrong time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stricter point, submitted mixture | 2 | 2 | 2 | 0 | 1 |
| Stricter point, exact foreground | 3 | 3 | 0 | 0 | 2 |
| Wider point, submitted mixture | 3 | 2 | 2 | 0 | 0 |
| Wider point, exact foreground | 6 | 6 | 0 | 0 | 1 |

At the wider point, there are four additional matched-control-clear
locations across three source groups. Two locations in one group share
the same two-crop reading. All previously control-clear locations were
retained. The stricter point adds one such location. Among the eight
reachable labels, the lower log odds of the two windows was higher in
the edited recording than its matched ordinary recording in six cases
on the mixture and eight cases on the exact foreground. The median of
those paired margins was 0.41 on the mixture and 12.35 on the exact foreground.

The exact foreground ran before the Opus encode. This comparison therefore
changes both interference and codec exposure relative to the submitted
mixture. It measures the potential of the short detector when the foreground
is known, on a limited set of fixed windows. The earlier six-second
experiment used a different detector and saw no gain on its exact
foreground inputs. A deployable follow-up needs to show that a view derived
from the delivered audio can reproduce additional locations without matching
ordinary alerts or leaving incorrect ranges. No serving or confidence
change follows from this comparison.
[Earlier six-second foreground diagnostic](audio-overlap-reconstruction-2026-11-16.md#exact-foreground-comparison)

## Offline recount

The numeric capsule binds the source manifest, native readings, and the
published mixture and timing ledger by SHA-256. It keeps anonymous group
slots, window admission and decision bits, and the paired scalar margins
used for the table. It contains no speech, reconstructed samples, speaker
embeddings, identifying records, or local paths. The standard-library
recount uses those retained bits and margins. Re-running detector inference
requires the original local sources and model files.
[Numeric capsule](evaluation-panels/audio-exact-foreground-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_audio_exact_foreground \
  docs/evaluation-panels/audio-exact-foreground-2026-11-18.json.gz
```
