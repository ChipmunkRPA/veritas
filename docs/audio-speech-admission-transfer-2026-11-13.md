# Speech-window coverage in mixed audio

New audio and video analyses retain the decision to admit each candidate
voice window. The evidence view, PDF and JSON show how many windows were
scored, had too little detected speech, or fit the speech requirement but
fell outside the window limit. The PDF lists up to eight skipped windows.
The JSON keeps every measured window with sample counts and source times.
Detected speech describes the mixed track and cannot identify a speaker.

In three simulated prepaid Full analyses, a WAV and an MP4 with speech
followed by noise each recorded one scored window and three below the
speech minimum. A noise-only WAV recorded two below the minimum and zero
voice scores. The API, PDF, JSON and signed bundles agreed. Both WAVs and
the video soundtrack replayed against the supplied originals. These
fixtures check the evidence workflow using authored noise and an earlier
public speech excerpt.

## Independent coverage comparison

Four previously unused AMI meeting cohorts contributed 96 three-second
native excerpts. Cohorts and recorded participants were disjoint from
those named in the repository at the freeze. Manual 10 ms word timings
defined four strata, with six excerpts per room in each stratum. A word
annotation is an imperfect proxy for audible speech and can omit or
include speech that is masked in the microphone mix.
[AMI downloads](https://groups.inf.ed.ac.uk/ami/download/) ·
[AMI license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

The experiment compared the existing speech gate with the same gate after
up to 6 dB of gain without clipping. Both kept the half-window speech
rule. The comparison also tested admission if either view passed and if
both passed. Component-level research motivated separating this coverage
measurement from speech-origin and background-origin decisions.
[Environment-aware speech benchmark](https://arxiv.org/abs/2606.10791)

| Annotated speech | Original | Gain-adjusted | Either | Both |
| --- | ---: | ---: | ---: | ---: |
| At least 75% | 12/24 | 15/24 | 15/24 | 12/24 |
| 50% to 59% | 4/24 | 4/24 | 4/24 | 4/24 |
| 40% to 49% | 0/24 | 0/24 | 0/24 | 0/24 |
| At most 25% | 0/24 | 0/24 | 0/24 | 0/24 |

The three newly admitted speech-heavy windows came from two rooms. Nine
of 24 speech-heavy excerpts still fell below the minimum after gain.
Gain remains experimental pending a comparison of voice-origin decisions
on newly admitted genuine and synthetic speech under matched noise.
[Measurements and source commitments](evaluation-panels/audio-speech-admission-transfer-2026-11-13.json)

## Replay

The scalar review recomputes the counts from the public ledger. For native
inference, set the paths to the pinned AMI manual timings and the existing
MIT-licensed speech-activity file. Use a new trial path. The acquisition
step fetches only the 96 source ranges and four WAV headers.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python3 -B -S -m scripts.audio_speech_admission_transfer review \
  docs/evaluation-panels/audio-speech-admission-transfer-2026-11-13.json
backend/.venv/bin/python -B -m scripts.audio_speech_admission_transfer freeze "$trial" --annotations "$ami_annotations"
backend/.venv/bin/python -B -m scripts.audio_speech_admission_transfer acquire "$trial"
backend/.venv/bin/python -B -m scripts.audio_speech_admission_transfer measure "$trial" --checkpoint "$speech_activity_file"
```
