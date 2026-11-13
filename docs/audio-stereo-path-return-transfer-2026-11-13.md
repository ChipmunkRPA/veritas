# Short stereo recording-path changes under competing speech

A voice replacement can carry a different stereo recording path for a short
interval before the original path returns. This experiment measures that
temporary change on the submitted stereo channels. A conventional replacement
can produce the same change. A same-path AI replacement can leave the
measurement unchanged. The score is for locating a change in the recording
path.

## Frozen comparison

Estimate speech-band interchannel lag, level balance and normalized correlation
from 80 ms frames on the native 16 kHz clock. Retain median measurements every
100 ms. Scan for 0.7, 0.9 and 1.1 second intervals whose two surrounding
0.5-second flanks agree. The strongest response describes how much the
interior differs from both flanks. Subtract penalties for disagreements
between the flanks and unstable interior frames. Compare level alone with
the joint distance. Independently set each operating point at the maximum
development response across untouched speech, same-path ordinary
replacements, gradual moves and background-only controls. The points were
frozen before assessment. No pretrained weights were used.

The development cohort uses three readers and a measured Surrey stereo path.
The assessment uses four different readers and the measured BBC stereo paths.
The speech is from the earlier LlamaPartialSpoof room study. The source readers
and waveforms were previously exposed to other methods. This study authors
new interventions from their verified source audio. The two held donor hosts
supply four published synthetic segments across four synthesis families.
Six held host and edit positions each supply the matching ordinary
replacement. All replacement destinations last 960 ms with identical 20 ms
fades. The main speech is centered, while displaced inserts use the selected
left or right response. Gradual-move controls use the same original voice
and blend the paths over 2.96 seconds. No new synthetic voices were
generated. [Published speech and labels, CC BY 4.0](https://doi.org/10.5281/zenodo.14214149)
· [Prior exact source qualification](audio-wpe-fresh-room-transfer-2026-11-13.md)
· [Surrey measurement and terms](https://doi.org/10.15126/surreydata.00813511)
· [BBC static response terms and attribution](audio-bbc-room-assets-v303-method.md)

Each comparison uses the same peak gain and background source across edit
variants. The background comes from a role-specific AMI meeting with
nonoverlapping exact excerpts outside the prior WPE study's marked intervals.
Both meetings have prior project exposure. Background-only controls retain
the same rendering. Use dry, +5, 0 and −5 dB room-mean target-to-background
levels. An interval must cover at least half the 960 ms edit and achieve
0.30 time intersection over union. These mixes use measured room responses
and known render interventions. Real microphone switching and moving-speaker
recordings were outside the test. [AMI recordings and terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Results

| Held condition | Joint strongest range aligned before applying point | Joint localized at fixed point | Gradual-move controls flagged |
| --- | ---: | ---: | ---: |
| Room only, foreground present | 9/10 | 0/10 | 0/6 |
| +5 dB competing speech | 7/10 | 1/10 | 2/6 |
| 0 dB competing speech | 1/10 | 0/10 | 0/6 |
| −5 dB competing speech | 3/10 | 0/10 | 2/6 |

Each localization denominator contains six ordinary and four synthetic
displaced replacements. The only accepted range was 1.70–2.40 s at +5 dB,
covering part of an ordinary replacement authored at 1.45–2.41 s. Level-only
accepted no held ranges. The joint view did not flag unchanged,
same-path ordinary or isolated-background inputs. It left one room-only
gradual move unassessed. Before applying the operating point, the joint
arm's strongest range aligned with two of ten same-path replacements in
room-only foreground, compared with nine of ten displaced replacements.
That separation largely disappeared when competing speech became as loud
as the target. The occasional false flag on a gradual move further limits
the recording-path explanation at these fixed settings. The check remains
experimental and leaves customer evidence and confidence unchanged.

## Replay

The runner froze 57 development and 146 assessment deliveries, then replayed
all 203 waveform commitments and strongest ranges exactly. NumPy and SciPy
FFT paths reproduced per-cell features to within (10^{-9}) and agreed on
every reported range. The standard-library reader verifies the two points,
source-clock intervals, availability and aggregate decisions from the public
scalar receipt. The original waveforms and source identifiers remain local.
[Anonymous receipt](evaluation-panels/audio-stereo-path-return-transfer-2026-11-13.json)

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze fit score replay; do
  backend/.venv/bin/python -B -m scripts.audio_stereo_path_return_transfer "$phase" \
    --run "$run" --prefix "$publisher_prefix" --publisher "$publisher_run" \
    --directional "$directional_run" --aachen "$aachen_assets" --stress-run "$stress_run" \
    --prior-run "$wpe_fresh_run" --ami "$ami_sources" --surrey "$surrey_room" \
    --bbc "$bbc_room" || exit 1
done
python3 -B -IS backend/scripts/review_audio_stereo_path_return_transfer.py "$receipt"
```
