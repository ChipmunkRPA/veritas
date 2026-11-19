# Denser comparison with a clean earlier audio copy

The denser comparison located 12 of 64 noisy synthetic replacements, up from
two with the current twelve-second earlier-copy check. The ten additions came
from three reader groups and both synthesis families. It added four ordinary
traffic-control highlights from one host and one misplaced highlight on a clean
synthetic recording. This candidate stays in development and contributes zero
weight to overall confidence.

## What information the check uses

The user would need to supply a clean earlier recording of the same speech.
The candidate compares that copy with the submitted recording. Those two
recordings form its entire input. Both use the same continuous twelve-second,
16 kHz mono PCM16 clock.

This is a physical comparison of two supplied versions. Genuine splices and
ordinary processing can also reduce matching waveform. Authorship and
individual-speaker attribution require separate evidence. Single-upload
screening remains a separate task.

## Fixed method and source scope

The current twelve-second route combines six six-second measurements, starting
1.2 seconds apart, and reports a 720 ms range when neighboring measurements
agree. The existing six-second dense method instead measures a 960 ms range
using 120 ms frames at 40 ms steps. The candidate applies that existing dense
method to the same six starts and keeps the original neighboring-window
agreement. It retains every current range and declines a retry when the
current reference-match or whole-file frequency-balance requirements fail.
Each dense window also keeps its existing reference-match and frequency guard.
The existing weights, decision points, speech gates and source clocks stay fixed.
[Existing six-second method](audio-dense-clean-reference-2026-11-13.md)
· [Current twelve-second method](audio-clean-reference-duration-2026-11-13.md)

The 336 foreground pairs reuse the completed short-window reader study's 16
hosts from eight declared reader groups. Each host has two 960 ms synthetic
replacements from the licensed Piper and Kitten stock-voice sources, two genuine
splice controls, two local resampling roundtrips and an unchanged version. The
same target events recur in clean, traffic and competing-speech conditions.
Both noisy conditions put recorded background 10 dB above the original target
speech. Four background captures recur across the panel. The speech originated
in Opus and was independently decoded into the exact native PCM16 fixtures.
This development plan uses previously observed sources and earlier standalone
results. Prospective independent source validation remains outstanding.
[Prior source qualification and results](audio-short-window-reader-holdout-2026-11-18.md)
· [MLS source and CC BY 4.0 terms](https://www.openslr.org/94/)
· [AMI source terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)
· [Traffic source one and CC0 declaration](https://commons.wikimedia.org/w/index.php?title=File:Sonido_trafico_cercano_jardin_botanico_ucm.ogg&oldid=991122728)
· [Traffic source two and CC0 declaration](https://commons.wikimedia.org/w/index.php?title=File:Temporary_traffic_signal_at_Colonel_By_Drive.webm&oldid=1230992220)

An additional 144 wrong-reference pairs use an unrelated reader, a 320 ms
shift with zero padding, or a reference whose second half comes from another
reader. Another 144 pairs apply a whole-file volume change, a 1,900 Hz low-pass
filter or a 5 dB high-frequency shelf to an unchanged submission. Together
there are 624 pairs, 208 per condition. The comparison uses model-free
measurements throughout.

## Results

A useful location covers at least half the known replacement and reaches 0.30
temporal intersection over union. Across the two noise conditions, counts cover
64 deliveries from 32 paired target events. Genuine splices are change positives
for this physical measurement. Ordinary processing controls supply the negative
examples.

| Condition | Synthetic locations, current / candidate | Genuine splice locations, current / candidate | Ordinary processing flags, current / candidate | Wrong-reference flags, current / candidate |
| --- | ---: | ---: | ---: | ---: |
| Clean | 1 / 13 of 32 | 2 / 14 of 32 | 0 / 0 of 96 | 0 / 0 of 48 |
| Traffic, -10 dB | 2 / 6 of 32 | 2 / 6 of 32 | 4 / 8 of 96 | 0 / 0 of 48 |
| Competing speech, -10 dB | 0 / 6 of 32 | 0 / 6 of 32 | 13 / 13 of 96 | 0 / 0 of 48 |

The 96 ordinary processing negatives in each condition comprise 16 unchanged
copies, 32 local roundtrips and 48 whole-file processing controls. The four new
traffic highlights were the unchanged recording, both local roundtrips and its
volume-adjusted version from one host. The three new highlights in the original
foreground roster fell outside their matched target interval. The volume-only
control had no target label. A further clean synthetic highlight missed the
required overlap. The noisy synthetic recordings gained zero new misplaced
highlights. Existing misplaced highlights remain in both arms.

Reference admission was unchanged. Two clean, 14 traffic and 16 competing-speech
synthetic comparisons out of 32 remained unassessed for reference matching.
The candidate leaves 19 clean and 26 synthetic deliveries in each noisy
condition without a correct location. Reliable target-speaker detection under
strong background sound remains an open evaluation problem.

The frozen gate required four additional noisy synthetic locations across at
least two reader groups and both families, no more than one newly flagged
ordinary reader group per condition, no wrong-reference flags and zero new
off-target foreground highlights. The candidate met the gain, reader, family,
control-group and reference conditions. It failed the off-target requirement
with four additional highlights, comprising the clean synthetic example and
three ordinary traffic examples. The plan and gate were preserved after the
result. The observed tradeoff is worth a prospective confirmation on new
readers, independently recorded backgrounds and gain/clock perturbations.

## Independent source replay and public recount

A separately implemented reviewer uses its own measurement implementation.
It rehashed all 368 original delivered PCM
inputs, reconstructed all 624 supplied-source pairs and their controlled
transformations, and recomputed both measurement procedures. It reproduced
1,464 dense-window admission and measurement records, selected ranges,
neighboring-window clocks and every grouped result. NumPy and SciPy supply the
same pinned numerical filtering runtime. Independent implementation replay
used the same original sources. Prospective source validation remains outstanding.

The [public aggregate](evaluation-panels/audio-dense-native12-reference-2026-11-18.json)
contains grouped condition and role counts, the frozen method, privacy scope
and commitments to the closed native records. Media, individual decisions,
scores, reader identifiers, transcripts, local paths and per-case timestamps
remain private. The standard-library reviewer checks its tables and frozen gate.
Reader-group counts are native-audited aggregate inputs. Reproducing individual
source measurements uses the retained private pairs and fixed transformations.

```sh
python3 -B -S backend/scripts/review_audio_native12_dense_reference.py \
  docs/evaluation-panels/audio-dense-native12-reference-2026-11-18.json
```

This model-free method uses the previously qualified research-input terms
above. The candidate stays in development and contributes zero weight to AI
confidence.
