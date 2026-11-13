# Same-word short voice replacements under competing speech

This source panel prepares exact short replacement inputs for a local
voice detector comparison. Each edited interval contains the same two or
three words in genuine, text-to-speech and voice-converted versions. The
earlier PhonemeDF study matched complete utterances and scored their
middle three seconds. This intake binds the same words across the three
separate native source clocks. [Earlier whole-utterance comparison](audio-phonemedf-matched-transfer-2026-11-13.md)

## Source qualification

The run used the published version 1.0.1. It verified the entire 386 MB
TextGrid archive against the publisher's MD5, then scanned complete
publisher members from the first 320 MiB of the genuine, Chatterbox TTS
and Chatterbox VC WAV archives. The overlap contained 1,054 utterance
IDs. All 1,054 had complete corresponding word tiers and matched
the full word sequence across the three origins. The source files are
publisher recordings. The study generated no voices. [Publisher release and CC BY 4.0 terms](https://zenodo.org/api/records/19522979)
· [Published alignment methodology](https://aclanthology.org/2026.lrec-1.469/)
· [LibriSpeech publisher](https://www.openslr.org/12/)
· [Chatterbox publisher and terms](https://huggingface.co/ResembleAI/chatterbox)

Seven distinct source readers qualified after excluding the previous
PhonemeDF and ODSS readers and exact audio source identifiers in the
recorded v9 calibration plan. Requiring no declared reader overlap with
any historical v9 role leaves one qualifying group in the bounded
prefix. The frozen panel retains two development and two assessment
readers with no shared source reader across those roles. One held reader
is outside the historical v9 role IDs. The other held reader and both
development readers have historical v9 reader exposure with different
source identifiers. Generator reference identities and prior model
training exposure are unknown. There is one prompt per reader and two
synthesis mechanisms from one provider.

## Frozen deliveries

Each six-second mono recording uses 16 kHz signed PCM. The genuine
source word span sets the replacement duration. Resample the donor
span to that duration, equalize its RMS to the original words, and apply
20 ms fades. The paired genuine controls resample the same genuine
words to the relevant donor duration and back. A separate genuine
control uses a 1.4× local gain change with the same fades. The original
samples outside every authored interval match exactly.

Each set shares a peak-safe gain and one separately selected AMI
competing-speech excerpt at nominal +5, 0 and −5 dB on the full
six-second waveform. The dry recording and isolated background controls
are available separately. The development and assessment use different
AMI recordings and participant groups. Their recordings have prior
project exposure, with newly selected intervals outside the earlier
ODSS windows. The four edits last **720, 860, 870 and 890 ms**.
The competing speech has **480–880 ms** of simultaneous annotated
words within the exact edit ranges. These are controlled mono mixes of
recorded voices. [AMI releases and terms](https://groups.inf.ed.ac.uk/ami/download/)
· [AMI CC BY 4.0 terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

The freeze contains 108 scheduled deliveries, split equally between
development and assessment. There are 96 distinct PCM commitments.
Two prompts have a genuine resampling control that shares the untouched
waveform because its source and destination durations match. A third
prompt has two resampling controls that share the same donor duration
and therefore the same waveform. The receipt preserves all scheduled
controls. Native replay reconstructed
every exact waveform commitment. The standard-library reader verifies
the declared source ranges, conditions, denominators and paired
geometry without media. No classifier ran, so this source panel yields
no localization rate or confidence change. [Frozen source receipt](evaluation-panels/audio-phoneme-short-edit-intake-2026-11-13.json)

## Replay

The runner requires the previous local PhonemeDF and ODSS range caches,
the exact AMI annotation archive and native WAV files, and the recorded
v9 source plan. Keep all publisher audio and annotations in scratch.
Download 8 MiB ranges with exact `Content-Range` and use the numbered
byte offset filenames. The earlier source directory retains the first
96 MiB of `real` and `ChatterboxTTS`. The new cache holds the rest of
their prefixes, the complete first 320 MiB of `ChatterboxVC` and all
386,093,119 bytes of `textgrids`. New response headers use the
`<offset>.range.headers` suffix. The public receipt contains source
commitments and authored ranges.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_phoneme_short_edit_intake replay \
  --cache "$new_cache" --old-cache "$old_phonemedf" \
  --record docs/evaluation-panels/audio-phonemedf-release-19522979.json \
  --odss-cache "$odss_cache" --ami "$ami_inputs" --receipt "$receipt"
python3 -B -IS backend/scripts/review_audio_phoneme_short_edit_intake.py "$receipt"
```
