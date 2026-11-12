# How competing vocal events change the speech result

The production voice adapter leaned AI on all six speech clips with a known
AI insert and no added event. Adding a louder ordinary vocal event reduced that to three of
six WAVs and two of six MP3s. Two WAVs and one MP3 with known AI inserts
leaned original. Short sound-category windows missed all added vocal events
in the speech mixtures at the fixed comparison point. The proposed
sound-category safeguard withheld none of the directional voice decisions.
[Case recipes and measurements](evaluation-panels/audio-vocal-event-mixtures-2026-11-12.json)

## Frozen design

Start with six distinct four-second speech sources from the retained
call-processing panel. Each has a matching version with a known 0.80–0.96
second AI speech insert. The inserts cover two sources from each of three
generator families. All the speech sources had prior project use. The event
roster provides three generated recordings from one family and six
individually licensed ordinary recordings. The event categories follow the
publisher labels. Independent listening and additional generator families
remain pending. [Speech sources and rights](audio-remote-neighbor-replication-2026-11-12.md) ·
[Event sources and rights](audio-vocal-event-sources-2026-11-12.md)

Select each event's strongest 1.5-second section on a 100 ms step using decoded
16 kHz mono floating-point PCM. Use a 30 ms fade at each end. Overlay a
matched ordinary or generated event in the same interval of the same speech
at −5 or +5 dB speech-to-event RMS ratio. Apply a common gain per host.
Sum the quantized components exactly in 16-bit PCM without clipping. Repeat
the 69 cases in both native WAV and 48 kb/s MP3. Each carrier has 12
speech-only cases, 48 mixtures and nine isolated events. The WAV and MP3
results are paired observations of the same sources. The first preparation
attempt stopped at a quiet donor. The final source freeze, before inference,
uses float decoding with an explicit 0.0005 RMS floor and records source
levels. The quiet synthetic sniff remains a limitation of that event stratum.

The production voice adapter uses its existing speech admission, its pinned
voice calibration, and its original-leaning stability check. The separate
event-tag comparison uses one four-second view and three overlapping
two-second views with a fixed 0.50 category response. The proposed safeguard
withholds a directional voice result only when an eligible non-speech event
leads the broad view and no listed speech category appears in its top three.
This is a descriptive category rule. It supplies no independent proof of
speech absence. Neither candidate changes customer confidence.

## Results

| Known AI speech inserts | WAV leaning AI / original | MP3 leaning AI / original |
| --- | ---: | ---: |
| Clean voice, six cases | 6 / 0 | 6 / 0 |
| Louder ordinary event, six cases | 3 / 2 | 2 / 1 |
| Quieter ordinary event, six cases | 4 / 1 | 4 / 1 |
| Louder generated event, six cases | 5 / 0 | 6 / 0 |
| Quieter generated event, six cases | 6 / 0 | 5 / 0 |

Other assessed directions in the table were inconclusive. None of the 30
genuine-speech cases per carrier leaned AI. Twenty-eight WAVs and 25 MP3s
leaned original speech, with the remainder inconclusive. The production
voice adapter abstained on all nine isolated events per carrier. Both
sound-tag configurations identified one isolated cough per carrier and
flagged no speech-only controls or mixtures at the chosen point.

The placement rule gives only the center two-second window enough overlap to
pass a 0.50 temporal IoU criterion. The four-second view has a geometric
maximum of 0.375. The isolated cough selected the center window in both
carriers. These data support a comparison of category detection and changes
in the voice response. They leave variable boundary detection untested.

Ordinary event files had prior MP3 encoding while the generated events came
from float WAV. Common output encoding preserves that source-history
difference. Laughter was assigned to clone and expressive voice groups,
while breath-like events were assigned to stock voice groups. This limits
comparisons across the strata. The task tested vocal events only. Environmental
ambience, music, and general audio-generation detection need their own panels.

## Replay

The public JSON contains the component commitments, source positions, and
scalar model responses. It contains no recordings or model weights.

```sh
python3 -B -IS backend/scripts/audio_vocal_event_mixtures.py review \
  docs/evaluation-panels/audio-vocal-event-mixtures-2026-11-12.json
```

Native reproduction uses the retained event roster, speech stems, and pinned
local audio checkpoints. From the repository root, set `run` to a new working
directory and use the exact local dependency environment.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 VERITAS_AUDIO_DETECTOR_MODE=candidate
backend/.venv/bin/python -B -m scripts.audio_vocal_event_mixtures prepare "$run"
backend/.venv/bin/python -B -m scripts.audio_vocal_event_mixtures score "$run"
```

A separate implementation reopened all nine event sources and 12 speech
stems. It regenerated all 69 exact component pairs and WAV sums, and verified
the file hashes, decoded PCM and clocks of all 138 deliveries. The worst
absolute mixture-level error after quantization was 0.000044 dB.

## Can the shorter voice check catch the wrong-original results?

A separate diagnostic reran the existing two-second voice check on every
original-leaning case from the pilot. The frozen WAV and MP3 operating
points came from the earlier six-second reviews. This time the three exact
windows cover a four-second file. The rule would withhold an original
tendency if any admitted short window strictly exceeded its carrier's
point. A useful location had to cover at least half the known edit and
reach 0.30 temporal IoU.

All 174 windows in 58 files were scored. None exceeded its point. The check
would have withheld **0 of 5 wrong-original results** and **0 of 53
ordinary original-leaning results**. These are paired carrier results from
the consumed pilot, with the exact source bytes and clocks rechecked.
[Diagnostic ledger](evaluation-panels/audio-vocal-event-voice-conflict-2026-11-12.json)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.audio_vocal_event_voice_conflict review \
  docs/evaluation-panels/audio-vocal-event-voice-conflict-2026-11-12.json
```
