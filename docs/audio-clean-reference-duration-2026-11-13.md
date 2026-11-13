# Local speech comparison on twelve-second audio

The earlier-copy comparison found more short changes when it could
inspect the full twelve seconds. It also highlighted four genuine
recordings after whole-file equalization. The unchanged six-second
score and the new duration result measure waveform differences and
carry zero weight in the AI estimate. [Portable replay](evaluation-panels/audio-clean-reference-duration-2026-11-13.json)

## Frozen comparison

Use two continuous, aligned mono PCM16 recordings on a native 16 kHz
clock. The earlier copy is clean. The submitted one contains recorded
overlapping voices. Evaluate six six-second windows that start 1.2
seconds apart, using the existing gain-drop score, match requirements
and threshold unchanged. Require the first and last windows and at
least four of the six to support comparison with the reference. For a
local flag, two neighboring windows must select ranges that overlap by
at least 360 ms. Select the stronger range from the pair with the
highest lower score. Adjacent windows share samples. Compare with
checking only the first six seconds and taking the strongest result
from any admitted window. [Existing six-second evaluation](audio-clean-reference-transfer-2026-11-13.md)

The source plan selects 15 continuous publisher recordings, one for
each of three edit positions from two development and three held
readers. It uses publisher word alignments and source-clock support.
The readers previously appeared in the six-second study. This test
uses longer recordings and separately chosen continuous twelve-second
background excerpts. Four AMI sessions have disjoint participant groups
between roles. Their new excerpts exclude the prior exact six-second
intervals. At least 260 ms and up to 990 ms of annotated competing
voices overlap the actual edits. The two synthetic mechanisms come
from one provider. A natural different-passage replacement from the
same reader is also a waveform-change positive.
[PhonemeDF and terms](https://zenodo.org/records/19522979)
· [AMI and terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

Each held level has 18 synthetic replacements, nine ordinary
replacements, 45 untouched, gain, resampling or equalization controls,
and 27 wrong-reference controls. The last group includes a different
reader, a 320 ms sample shift, and a recording that shares only the
first half. The true boundaries are withheld during scoring. A
useful range covers half the true edit and achieves 0.30 temporal IoU.
The predeclared −5 dB gate required four synthetic locations across
two readers and two edit positions, zero misplaced positives, at most
two flagged ordinary controls, and zero wrong-reference admissions at
all levels.

## Held results

| Target-to-background ratio | First six seconds, synthetic / ordinary replacements | Any supported window, synthetic / ordinary replacements | Neighboring-window agreement, synthetic / ordinary replacements | Ordinary controls flagged, first / any / agreement |
| --- | ---: | ---: | ---: | ---: |
| 0 dB | 2/18 · 1/9 | 14/18 · 7/9 | 9/18 · 5/9 | 3/45 · 5/45 · 4/45 |
| −5 dB | 2/18 · 1/9 | 13/18 · 7/9 | 9/18 · 5/9 | 2/45 · 4/45 · 4/45 |
| −10 dB | 2/18 · 1/9 | 13/18 · 7/9 | 9/18 · 5/9 | 2/45 · 10/45 · 4/45 |

Neighboring-window agreement found four of nine text-to-speech and
five of nine voice-converted replacements at each level. It found
edits across all three held readers and all three positions, with
zero misplaced positives. It refused all 81 wrong-reference pairs.
The first-six-seconds baseline admitted the 27 half-matched
references as six-second comparisons.

The four ordinary control highlights at each level came from the same
four whole-file equalization deliveries. Untouched, gain-only and
same-speech resampling controls yielded no neighboring-window flags.
At −5 dB, a late voice-converted replacement at **8.940–9.860 s**
produced evidence at **9.120–9.840 s**, exact source ticks
`[145920, 157440)`. A middle replacement at **5.840–6.560 s**
produced a **5.760–6.480 s** range. A gain and frequency profile can
change both recordings globally while only a short interval satisfies
the local support rule. The frozen gate failed on the four equalization
controls.

## Whole-recording frequency balance

A separate frozen rule looks for a consistent frequency-balance change
between the same recordings. It estimates five band gains over three
four-second parts, relative to the lower bands, and requires the first
and last parts to agree on the direction and approximate shape of the
change. It compares the primary equalization with a different 1,900 Hz
low-pass filter and a +5 dB high-frequency shelf. These extra transforms
reuse the same recorded speakers and backgrounds.

The guard removed all four primary EQ highlights at each held noise
level. It also removed all four, four and five local highlights from
the 1,900 Hz controls at 0, −5 and −10 dB. The high-frequency shelf
produced no local flags and the broad measurement identified 24 of 27
of those controls. It preserved all nine synthetic and five ordinary
replacement locations at each level. The wrong-reference comparisons
still declined. The added test does not cover every equalization,
particularly a processing change applied only to a short passage.
[Frequency-balance source and outcome receipt](evaluation-panels/audio-clean-reference-duration-eq-2026-11-13.json)

The authenticated earlier-copy comparison can now enable exactly
twelve-second native PCM16 WAV or FLAC pairs with
`VERITAS_AUDIO_REFERENCE_LONG_CLEAN_SPEECH=true`, in addition to the
existing comparison and clean-speech settings. The six-second lane
keeps its original measurement and export. A matching twelve-second
pair can show a supported local interval, a broad frequency-balance
finding, or a coverage reason. A broad finding carries the normalized
band gains and no local excerpt. For a local interval, the comparison
shows original-file source ticks and both playable excerpts. These
results carry zero AI confidence and do not establish why a change was
made.

## Replay

Use the same pinned archive prefixes and annotated AMI sources as
the [six-second study](audio-clean-reference-transfer-2026-11-13.md#replay).
No raw waveform is part of the public scalar receipt.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze score replay; do
  backend/.venv/bin/python -B -m scripts.audio_clean_reference_duration "$phase" \
    --cache "$new_cache" --old-cache "$old_phonemedf" \
    --record docs/evaluation-panels/audio-phonemedf-release-19522979.json \
    --odss-cache "$odss_cache" --ami "$ami_annotations" --gain-ami "$gain_ami" \
    --run "$run" || exit 1
done
python3 -B -S backend/scripts/review_audio_clean_reference_duration.py \
  docs/evaluation-panels/audio-clean-reference-duration-2026-11-13.json
```

For the EQ follow-up, use the same archive arguments with
`scripts.audio_clean_reference_duration_eq`. Pass the completed
duration run as `--prior` and a new directory as `--run`, then run
`freeze`, `score` and `replay`. The portable reader verifies the
frequency profiles and retained intervals.

```sh
python3 -B -S backend/scripts/review_audio_clean_reference_duration_eq.py \
  docs/evaluation-panels/audio-clean-reference-duration-eq-2026-11-13.json
```

The local authenticated API test reproduced the held late range and
verified both playable excerpts independently against the original
PCM. Its six-second windows started at 4.800 and 6.000 s, with
evidence commitment
`418673cb84f7f6f79cec1d2756138435a40ce8c8d614088dabf60698f768ae17`.
The PDF and JSON both returned the broad frequency profile on
the EQ control and a declined result for the half-matched reference.
The original analysis hash stayed the same in all three comparisons.
The serving comparison reproduced all 585 frozen admissions, retained
ranges and exported frequency profiles.
To verify an app export against
both original sources, use the local [native FFmpeg verifier](../backend/scripts/verify_audio_clean_speech_export.py).
