# Locate speech changes against a clean earlier recording

The earlier two-noisy-copy test found two of six synthetic replacements
at −5 dB and highlighted several unchanged controls. A clean earlier
copy provides a better reference for testing whether the same recorded
speech is still present. This comparison measures where a waveform
component disappears or changes. Ordinary replacements can produce the
same observation. [Earlier two-noisy-copy test](audio-paired-speech-continuity-transfer-2026-11-13.md)

A standalone detector must infer both the location and likely nature of an
edit from one signal. PartialSpoof measures synthetic speech at several
time resolutions. TRACE studies disruptions in speech-model representations.
A separate phoneme-level study ties model attributions to speech sounds.
The test here uses exact same-recording waveform evidence and no model.
[PartialSpoof](https://arxiv.org/abs/2204.05177)
· [TRACE](https://arxiv.org/abs/2604.01083)
· [Phoneme-level attribution](https://arxiv.org/abs/2607.08586)

## Frozen procedure

The supplied recordings are already aligned at the same 16 kHz mono clock.
The earlier recording is a clean six-second copy. The submitted version
adds competing speech at 0, −5 or −10 dB. The two development readers come
from the earlier panel. The three held readers are outside both previous
short-edit panels and the whole-utterance PhonemeDF panel. The input search
excludes the exact source identifiers in the retained calibration plan.
Historical reader exposure through different recordings and model training
overlap are unknown. Development and assessment use disjoint sets of AMI
participants. Each authored change has at least 250 ms of simultaneous
competing speech within its known interval. [PhonemeDF data and terms](https://zenodo.org/records/19522979)
· [AMI corpus and terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

Compare exact sample differences, the earlier normalized correlation, and
the new coherent projection against the same clean reference. The projection
estimates signed least-squares gain on 240 ms windows with 80 ms hops in
three frequency bands. It uses the stronger earlier-copy frames and
requires supported flanks on both sides. Its 720 ms range uses the median deficit across at least two bands with
supported flanks. A common broad-match check declines
unrelated reference recordings before the range scan. Each route uses
the maximum score from the development controls as its threshold. The
held success rule requires at least four of six synthetic replacements
localized at −5 dB across two reader groups, at most one ordinary control
flag there and zero unrelated-reference flags.

The synthetic replacements use published text-to-speech and voice-converted
samples of the same words. A replacement using a different passage from
the original reader is an additional waveform-change positive. The
controls are untouched speech, local gain, two same-speech resampling
variants and global equalization. Re-recorded or unrelated references
test admission. The comparison never uses the true edit coordinates or
the isolated interfering signal to choose a range. The exact overlap
criterion requires half the edit covered and at least 0.30 temporal IoU.

## Held outcomes

| Submitted target-to-background ratio | Coherent projection, TTS / voice conversion | Ordinary replacements located | False flags on unchanged, gain, resample or EQ controls |
| --- | ---: | ---: | ---: |
| 0 dB | 2/3 · 2/3 | 2/3 | 0/15 |
| −5 dB | 2/3 · 2/3 | 2/3 | 0/15 |
| −10 dB | 2/3 · 2/3 | 2/3 | 0/15 |

At −5 dB both synthetic versions of one **1.970–2.690 s** replacement
produced a **2.000–2.720 s** range. For the other detected reader, the
**2.620–3.380 s** edit yielded **2.640–3.360 s** for text-to-speech
and **2.800–3.520 s** for voice conversion. Each flag overlapped
the intended range. The third held reader's replacements were missed
at all three noise levels. Every same-recording comparison was assessed
and all 18 unrelated or regenerated earlier copies were declined.

Exact sample comparison highlighted the entire six seconds in all 90 held
pairs. The correlation route produced zero above-point local ranges on
these same clean-reference inputs. The candidate passed the frozen
follow-up gate for paired waveform-change evidence. The source panel is
three held readers and two synthesis methods from one provider. The
results apply to same-recording, same-clock pairs and measure a local
waveform difference. They do not establish standalone detection, edit
intent, codec tolerance or AI authorship.
[Anonymous scalar receipt](evaluation-panels/audio-clean-reference-transfer-2026-11-13.json)

## Local application verification

The authenticated earlier-audio comparison has an optional **Local speech
comparison** when `VERITAS_AUDIO_REFERENCE_COMPARISON_ENABLED=true` and
`VERITAS_AUDIO_REFERENCE_CLEAN_SPEECH=true` in development. Select the
clean-copy option for a continuous six-second mono 16 kHz PCM16 WAV or
FLAC that uses the same zero-based sample clock. The comparison checks
whether the supplied recordings share enough sound to measure and
reports a distinct reason when the format, time alignment or shared
sound falls outside its scope. The existing decoded sample comparison
is reported separately. Local speech evidence contributes zero weight
to the AI estimate. An ordinary edit can produce the same finding.

The application computation reproduced the frozen decision and rounded
measurements for all 150 development and held pairs. Three held pairs at
−5 dB also ran through the authenticated API, PDF rendering, whole-JSON
replay and a separate native FFmpeg reader.

| Held pair | Local speech result | Native verification |
| --- | --- | --- |
| Synthetic speech replacement | 2.000–2.720 s, ticks `[32000, 43520)` at 16 kHz | Both source excerpts, waveform envelopes and source clocks matched |
| Local gain control | No qualifying local drop | Same-clock sources verified, no local excerpt exported |
| Separately regenerated earlier file with the same words | Insufficient shared sound | Same-clock sources verified, no local excerpt exported |

The positive's evidence commitment is
`52d7edd2aa05c562fe12a4ea0f5c8087082af9b692ee5b867babf7f4efd7e1c2`.
For all three tests the existing comparison covered the full six seconds
and the retained analysis result hash stayed the same. The PDF and JSON
use the same range. The JSON also carries playable source excerpts.

```sh
PYTHONPATH=backend backend/.venv/bin/python -m scripts.verify_audio_clean_speech_export \
  earlier.wav submitted.wav comparison.json
```

## Replay

Use the [earlier source-cache instructions](audio-phoneme-short-edit-intake-2026-11-13.md#replay)
and extend the three relevant WAV prefixes to 448 MiB each. The four
original AMI captures come from the [public AMI replay](audio-gain-public-replay-2026-11-13.md).
All source PCM, annotations and reader IDs stay local. The public receipt
carries sample commitments, fitted points and held measurements.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze fit score replay; do
  backend/.venv/bin/python -B -m scripts.audio_clean_reference_transfer "$phase" \
    --cache "$new_cache" --old-cache "$old_phonemedf" \
    --record docs/evaluation-panels/audio-phonemedf-release-19522979.json \
    --odss-cache "$odss_cache" --ami "$ami_annotations" --gain-ami "$gain_ami" \
    --run "$run" || exit 1
done
python3 -B -S backend/scripts/review_audio_clean_reference_transfer.py "$receipt"
```
