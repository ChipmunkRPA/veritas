# Local first-formant evidence under competing speech

Two lightweight single-upload checks searched for a short reduction in
the width, frequency, or relative position of the first vocal resonance.
Neither located a known edit at its frozen point on 178 held deliveries.
The existing short voice check located one edit in the three-generator
replication. Both formant checks stay experimental.
[Source commitments and measurements](evaluation-panels/audio-formant-local-transfer-2026-11-13.json)

## Frozen comparison

Recent research identifies first-formant statistics as interpretable
utterance-level cues that transferred across several codecs. The same study
reports significant remaining error and sensitivity to telephone bandwidth.
Here the question is narrower in time and different in content. Can the
same kind of measurement place a short edit under competing voices?
The new implementation uses only the submitted mixture, with no earlier
copy, transcript, speaker identities or pre-trained weights. The cited
work does not validate these local measurements or their scores.
[First-formant research](https://arxiv.org/html/2611.13258v1)
· [Cross-condition speech localization research](https://arxiv.org/html/2609.25197v1)

Analyze 30 ms Hann-windowed frames every 15 ms. Retain two plausible
speech resonances on periodically voiced frames using linear prediction.
The first arm tests whether the estimated first resonance narrows inside
a 600 ms range relative to **both** 360 ms flanks. The second arm uses
only the frames where order-10 and order-12 trackers agree. It takes the
median drop in log bandwidth, log center frequency, and log first-to-second
frequency ratio. Both arms require at least six supported frames per flank
and ten inside, with 45 ms guards. The strongest supported range wins.
Changing vowels, recording paths and interfering speakers can all change
estimated resonances. Agreement between the two orders addresses tracker
stability alone. Overlapping voices can affect both estimators together.

Select each point as the strict maximum on the same 38 ordinary and
isolated-background development files at dry, +5, 0 and −5 dB. The
points are **1.3287088** for bandwidth alone and **0.3935440** for the
two-order arm. Thirty-eight and 35 development controls were assessed,
respectively. The 178 held deliveries are 168 distinct native PCM files
across three retained source panels. The main panel repeats four exact
synthetic deliveries across four noise conditions. The three-generator
panel contains four Chatterbox (DuplexGen), four ElevenLabs v4 and four
Kokoro replacements, with genuine same-reader and other-reader splices,
untouched hosts and isolated AMI room controls. The additional bank uses
three separate held readers, two other AMI sessions and two mechanisms
from one generator family. All banks had previous project exposure for
other methods. No held response set a point or selected a measurement.
[Same-word edit sources and rights](audio-phoneme-short-edit-intake-2026-11-13.md)
· [Three-generator source qualification](audio-remote-neighbor-replication-2026-11-12.md)
· [Additional held readers and background sources](audio-clean-reference-transfer-2026-11-13.md)

A useful range covers at least half the known edit and reaches 0.30
temporal intersection over union. Compare the exact incumbent score and
range only on the matched inputs where the incumbent was run. A candidate
needs two new useful locations across two generators in the three-generator
bank, at most one added ordinary group, and no room-only flag. It also
needs two useful locations at each of −5 and −10 dB in the additional
bank, with at most one ordinary group per level.

## Held results

| Input | Incumbent useful | Bandwidth useful | Two-order useful | Ordinary groups flagged, incumbent / bandwidth / two-order |
| --- | ---: | ---: | ---: | ---: |
| Same-word dry | 4/4 | 0/4 | 0/4 | 0 / 0 / 0 |
| Same-word +5 dB | 3/4 | 0/4 | 0/4 | 1 / 1 / 1 |
| Same-word 0 dB | 2/4 | 0/4 | 0/4 | 0 / 1 / 1 |
| Same-word −5 dB | 0/4 | 0/4 | 0/4 | 0 / 1 / 0 |
| Three generators −5 dB | 1/12 | 0/12 | 0/12 | 3 / 0 / 1 |
| Additional bank 0 dB | — | 0/6 | 0/6 | — / 1 / 0 |
| Additional bank −5 dB | — | 0/6 | 0/6 | — / 0 / 1 |
| Additional bank −10 dB | — | 0/6 | 0/6 | — / 0 / 0 |

On the three-generator panel, both candidates' strongest ranges overlapped
one true edit each before applying a point, compared with five for the
incumbent. At a descriptive cutoff of zero observed false groups, the
bandwidth arm's range at **0.540–1.140 s** covers **0.420 s** of a
**0.320–0.960 s** Chatterbox edit. It differs from the incumbent's one
useful case, and its response is below the point selected on development.
Neither candidate's strongest range covered any of the 18 additional-bank
synthetic edits usefully, even before applying a point. The bandwidth arm
also falsely flagged isolated room audio in three same-word mixture
conditions. The two-order arm left ten of 52 three-generator deliveries
unassessed, including two synthetic replacements. Those unassessed
deliveries are not evidence of an original recording. Changing the
operating point would not fix the poor localization on the transfer bank.

## Replay

The native runner locks all 232 PCM hashes, source clocks, upstream
receipts, incumbent window hashes, and the implementation before candidate
inference. It seals development before assessment and reconstructs the
public receipt from the WAV and PCM inputs byte for byte. The public
standard-library reader verifies the selected range geometry, recorded
acoustic contrasts, points and paired false-group accounting on Python
3.11 and 3.12. No source waveforms, transcripts or model weights are
included in the public receipt. It records only anonymized source
commitments, small scalar summaries and source-relative ranges.

Rebuild the primary and additional WAV caches as described in the
[two-pitch comparison](audio-overlap-pitch-transfer-2026-11-13.md#replay).
The prior three-generator path must contain the original `pcm`, freeze,
roster and incumbent receipt. Pass it with `--held` if it differs from
the existing local default.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze fit score replay; do
  backend/.venv/bin/python -B -m scripts.audio_formant_local_transfer "$phase" \
    --run "$run" --primary-wavs "$primary_wavs" --transfer-wavs "$transfer_wavs" \
    --held "$prior_three_generator_run" || exit 1
done
python3 -B -IS backend/scripts/review_audio_formant_local_transfer.py \
  docs/evaluation-panels/audio-formant-local-transfer-2026-11-13.json
```
