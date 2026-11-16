# Compare stereo channel readings under loud competing speech

This comparison tests whether the short speech check can better locate a
local voice replacement when it reads the original channels separately
or removes part of the interference measured by the stereo difference.
The test uses the existing voice and speech-activity checkpoints.
Recent mixed-audio research measures foreground and background separately.
The filter here follows the general approach of estimating interference
from a second signal. It uses no code or weights from the cited studies.
[MixFake](https://arxiv.org/abs/2605.23201v2) · [PC-Mix](https://arxiv.org/abs/2607.10345)
· [Transfer-function interference cancellation](https://israelcohen.com/wp-content/uploads/2018/05/TASL_May2008.pdf)
· [Existing model assets and terms](commercial-model-audit.md)

## Inputs and comparisons

Twenty-four ordinary six-second stereo deliveries from three Surrey
speaker pairs set the internal comparison points. Each pair includes
untouched and genuine-splice variants, with measured impulse responses
and loud competing speech. The maximum and second-highest ordinary
speaker-pair scores set the two points separately for each rule.
Both points are fixed before scoring the held files. Three pairs give
a coarse calibration. [Original stereo benchmark](audio-two-voice-components-2026-11-15.md)

The held comparison replays 80 BBC-room deliveries from six separate
speaker pairs. They include genuine versions and synthetic replacements
from four generator families, in the quieter voice, the louder voice,
or both. A separate bank uses 32 previously qualified SINE reader pairs
and six unchanged AMI room controls. Each SINE pair is newly rendered
with two recorded microphone channels from the same meeting excerpt.
For half the pairs the embedded target is equal in both channels. For
the other half the opposite channel has lower gain and a three-sample
delay. Each mix holds the room 5 dB above the original target by mono
RMS and uses the same room recording and gain for both members.
These are new deliveries of previously used speakers and source excerpts.
[Matched SINE source selection](audio-short-edit-union-holdout-2026-11-15.md)
· [AMI capture details](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml)
· [AMI source license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

Each view uses the same five fixed two-second windows and independently
requires one second of detected speech inside the window. The baseline
uses the app's existing short-window rule on the arithmetic mono samples.
This is a comparison of inputs to that rule. The live app's short scan
requires mono or [verified identical lossless channels](audio-short-identical-stereo-2026-11-15.md).
The channel rule takes the higher admitted score from the left and
right channels at each window. The residual rule fits a complex linear
transfer from side to mid per frequency over the whole clip and removes
the estimated interference in the mid signal. The agreement rule uses
the lower of the residual and channel scores at the same window. No
source identity, room identity, known stem, or edit clock selects an
input view. The scorer reports one interval. A useful interval covers
half a known edit and at least 30% of the temporal union. The receipt
counts additional ordinary flags, missed edits, label coverage and
unavailable views for each rule. Before speech admission or scoring,
the five-window grid could fit 16 of 32 PhonemeDF edit labels, all 32
stock edit labels and 7 of 32 SINE edit labels at this overlap rule.
Changing the input alone cannot locate the other labels usefully with
this window schedule.

## Detection results

Each cell gives useful edit locations and ordinary flags in that order.
The held set includes 80 edited recordings and 70 ordinary recordings.
One isolated-room recording was unassessed under every rule.

| Rule | PhonemeDF (24 edited, 16 ordinary) | Stock (24 edited, 16 ordinary) | SINE centered (16 each) | SINE lateral (16 each) | Isolated room flags (6 ordinary) | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mono, existing point | 0 / 0 | 2 / 0 | 0 / 3 | 1 / 7 | 0 | 3 / 10 |
| Mono, recalibrated tighter point | 0 / 0 | 5 / 3 | 1 / 11 | 1 / 15 | 0 | 7 / 29 |
| Add physical-channel fallback, tighter | 0 / 0 | 4 / 1 | 0 / 3 | 1 / 7 | 0 | 5 / 11 |
| Add physical-channel fallback, wider | 1 / 0 | 6 / 2 | 0 / 3 | 1 / 11 | 0 | 8 / 16 |
| Add interference-reduced fallback, tighter | 0 / 0 | 4 / 2 | 1 / 14 | 1 / 14 | 1 | 6 / 31 |
| Add same-window agreement fallback, tighter | 0 / 0 | 4 / 2 | 1 / 10 | 1 / 14 | 0 | 6 / 26 |

The tighter physical-channel fallback kept all three baseline locations.
It added two useful stock deliveries and one ordinary flag. Both added
locations came from the same recorded scene, with the edit in the louder
voice in one file and edits in both voices in the other. The wider
fallback added five useful deliveries across four speaker pairs and six
ordinary flags. Four of those extra ordinary flags came from lateral
SINE mixes, where the wider point recovered no extra useful edits. The
strict residual and agreement fallbacks had substantial ordinary-flag
costs. The physical-channel fallback warrants a separate test on new
speakers and rooms with its points frozen. The new scoring rules stay
in evaluation. [Held and calibration receipt](evaluation-panels/audio-stereo-regression-2026-11-15.json.gz)

## Acoustic verification

The filter is fitted on the mixture alone. Using those same fixed
coefficients, the run replays each known stem separately and checks
that the outputs add back to the filtered mixture. The centered SINE
layout improved the median target-to-interference ratio by **0.40 dB**
across 16 edited clips. The lateral layout lost **1.76 dB** across 16
edited clips and reduced the target level by **2.15 dB** at the median,
before any peak-safety scaling.
The corresponding median ratio changes were **−0.23 dB** on the
PhonemeDF single-edited mixes and **+0.01 dB** on the stock mixes, with
16 edited clips each. These results measure the exact prepared stems.
When the target reaches the microphones differently, the interference
estimate can remove some of the target itself.

## Replay

The public receipt holds waveform digests, sample positions, speech
coverage and scalar audio responses. The standard Python review
recalculates the frozen points and the held decisions from those
responses. It needs neither model weights nor the audio. Python 3.12
and 3.14 produced identical results. A separate count checked the paired
gains, losses and ordinary flags. Native source replay reproduced all
174 stereo inputs and 696 analysis-view hashes byte for byte.

```sh
export PYTHONPATH=backend
module=scripts.audio_stereo_regression
receipt=docs/evaluation-panels/audio-stereo-regression-2026-11-15.json.gz
python3 -B -S -m "$module" review "$receipt"
```

For native replay, reconstruct the `input` directory from the
[two-voice stereo benchmark](audio-two-voice-components-2026-11-15.md#replay)
and the `native` directory from the
[matched SINE holdout](audio-short-edit-union-holdout-2026-11-15.md#reproduce).
The `microphones` field in the public receipt gives the official AMI
microphone 05 URLs, start samples and expected 16-bit PCM digests.
For each record, `download_background` in
`scripts.audio_sine_resynthesis_transfer` fetches the exact six-second
range into the microphone directory. Save its return records in order
as `receipt.json` in that directory. The reader verifies each byte
commitment and checks it against the timestamp of microphone 01.
Use the existing three pinned audio runtime files in `models/weights/audio-upgrade`.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
runner=backend/.venv/bin/python
args=("$stereo_inputs" "$sine_inputs" "$microphones" models/weights/audio-upgrade)
"$runner" -B -m "$module" development "${args[@]}" "$run/development.json"
"$runner" -B -m "$module" held "${args[@]}" "$run/held.json" "$run/development.json"
"$runner" -B -S -m "$module" combine "$run/development.json" "$run/held.json" "$run/receipt.json.gz"
```
