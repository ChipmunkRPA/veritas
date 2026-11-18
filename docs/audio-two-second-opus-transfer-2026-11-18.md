# Standalone speech edit localization after Ogg Opus encoding

A two-second speech check can examine longer WAV, FLAC and AAC uploads in the local app. It requires a second elevated response on an overlapping interval. Earlier-copy Opus evidence draws on an additional recording, so its validation does not cover uploads supplied alone. This study tests whether the two-second method retains useful, source-specific locations under Opus and competing speech.
[Longer speech windows](audio-two-second-carriers-2026-11-17.md) ·
[Earlier-copy Opus](audio-reference-waveform-opus-2026-11-18.md)

## Fixed comparison

Re-encode all 96 exact existing inputs from the paired AAC and WAV study with FFmpeg and libopus. The two roles each contain 12 edited inputs from six synthesis families, 12 genuine inputs with no speech edit, and 24 isolated room excerpts. Each of the 48 foregrounds appears once at the same nominal −10 dB speech-to-interference ratio from the earlier study. The roles share no declared foreground group, room recording, or exact waveform. The utterances and rooms appeared in other project evaluations. The Opus inputs and measurements are new. The test holds the original 16 kHz source bandwidth constant. The rights and origin commitments come from the earlier data intake. [LlamaPartialSpoof and CC BY 4.0](https://zenodo.org/records/14214149) · [LibriSpeech release](https://www.openslr.org/12/) · [AMI license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

Use continuous mono 48 kHz Ogg Opus at 64 kb/s constant bitrate. Each compressed packet represents 20 ms of audio. Require exact Ogg page sequence, checksums, packet order, pre-skip and end trim. Compare the native proof with the analysis decoder and require an exact contiguous zero-based 16 kHz sample extent. Recheck the original decoded frame clock separately. [Ogg Opus specification](https://www.rfc-editor.org/rfc/rfc7845.html#section-4.3)

The incumbent comparator uses three-second windows on each Opus delivery, with the same schedule as the earlier AAC/WAV study. The supplement uses existing two-second windows as an additional screening clue and requires an elevated second window with a distinct start and at least one second of overlap. Compare two inherited internal points, one for WAV and one for AAC. Neither is refitted on these Opus files. Count a useful location only if it covers at least half of the known edit and at least 30% of the temporal union. The 157 annotated ranges and source sample units are identical across formats. Both studies count ordinary-file flags separately from missed labels and misplaced regions. Short edits need enough support to reach the location counter's minimum overlap. Recent work distinguishes authenticity evidence inside an interval from evidence about its boundaries. Codec sensitivity in clipped speech motivates a measured transport comparison. No research checkpoint or code was imported. [Boundary and segment research](https://arxiv.org/abs/2609.25822) · [Codec comparison](https://arxiv.org/abs/2610.07216) · [Incumbent checkpoint terms](commercial-model-audit.md)

The fixed test requires additional useful same-carrier locations in both roles, at least 80% retention of each role's matched WAV and AAC locations, at most one added genuine-file flag, and zero isolated-room alerts. The comparison changes no calibrated confidence or authorship decision.

## Local results

| Opus method | Useful locations out of 157 | Edited files with a useful location out of 24 | Genuine files flagged out of 24 | Isolated room clips flagged out of 48 |
| --- | ---: | ---: | ---: | ---: |
| Three-second comparator | 20 | 10 | 1 | 0 |
| Add two-second windows using the AAC point | 23 | 11 | 2 | 0 |
| Add two-second windows using the WAV point | 25 | 13 | 3 | 0 |

All 96 inputs passed clock qualification and produced at least one scored
two-second window. The stricter supplement added three correctly located labels
across two affected files from two generator families. The more sensitive
supplement added five across four affected files from four families. Both kept
all 20 locations from the same-carrier comparator. Two edited files per point received supplementary windows without a useful
supplemental location. One of the added genuine
flags appeared at both points. The other appeared only at the more sensitive
point. These are counts from the same paired inputs.

| Opus method | First role, useful / 76 | Second role, useful / 81 | Matched WAV locations retained | Matched AAC locations retained |
| --- | ---: | ---: | ---: | ---: |
| Three-second comparator | 11 | 9 | — | — |
| Add two-second windows using the AAC point | 12 | 11 | 19/25 | 21/25 |
| Add two-second windows using the WAV point | 13 | 12 | 20/25 | 21/25 |

The more sensitive comparison matches the aggregate total of both older
carriers. It keeps 20 of the 25 exact WAV locations and 21 of the 25 exact
AAC locations, with five and four different label locations respectively.
The stricter comparison keeps 19 WAV and 21 AAC locations, gains four and
two different labels, and covers 23 overall. Both fall below the frozen
retention requirement in the first role. The more sensitive threshold
also adds two genuine flags against the one-flag criterion.

The useful gains support a fresh Opus-specific trial. This test uses retained
utterances and inherited thresholds. It leaves the local standalone Opus
configuration unchanged. The 56 labels below 600 ms are too short for the
two-second overlap rule. Even with all candidate windows admitted, the
exact placement and matching rule could reach at most 98 of 157 labels.

The run completed 1,970 classifier forwards. All 1,328 distinct two-second
starts received responses on each carrier. The median absolute difference
in raw response log odds between Opus and the matched source at the same
start ranged from 0.089 to 0.132 across the four source-role and carrier pairs.
A separate event and waveform replay reproduced 96 decoded sample
commitments, all native frame clocks, four operating-point cells and both
advancement decisions.
[Complete local measurements](evaluation-panels/audio-two-second-opus-transfer-2026-11-18.json.gz)

## Fresh Opus holdout

A second trial tested 138 encoded files from 24 foreground readers, six
synthesis families, six ordinary resampling controls and two room
recordings. Readers and rooms were separate from the two earlier Opus
roles. Some sources appeared in other project tests. Each of the three
conditions used the same 12 edited files, 12 genuine files and six
resampling controls. The two background conditions also included 24
room-only files each. The three-second comparator and the stricter
inherited two-second point were fixed before scoring.

| Sound condition | Edit locations out of 78 | Edited files with a correct location out of 12 | Ordinary and resampled files flagged out of 18 | Isolated room files flagged |
| --- | ---: | ---: | ---: | ---: |
| Clean | 24 → 31 | 10 → 12 | 0 → 0 | — |
| Speech 5 dB below interference | 16 → 20 | 8 → 9 | 3 → 3 | 0 → 0 of 24 |
| Speech 10 dB below interference | 8 → 8 | 6 → 6 | 1 → 2 | 0 → 0 of 24 |

The arrows compare the same-carrier three-second schedule with the
combined schedule. The comparison schedule approximates the existing
check and can inspect more windows than the app currently routes on
Opus. All 138 files passed exact native clock and decoded-waveform checks
before scoring. A separate reader reproduced the counts without model
inference. The procedure completed 2,846 classifier forwards.

The supplement retained every baseline label and produced useful clean
and moderate-background gains. The strongest interference condition
added a misplaced edit highlight and a flag on an ordinary resampling
control. It fell short of the predefined minimum of two new locations
across two readers. At most 43 of the 78 target ranges per condition could
be reached by the current matching rule and two-second schedule, even if
every window passed the speech check. The Opus supplement remains
experimental. It carries zero weight in the overall confidence score.
[Fresh Opus comparison receipt](evaluation-panels/audio-two-second-opus-confirmation-2026-11-18.json.gz)

An isolated app test on a previously measured Opus file reproduced the
selected sample ranges, PDF waveform and signed original verification.
The source clock and the decoded clock were checked separately. The
variable-bitrate control stayed outside the profile. This tests evidence
delivery. The held comparison determines accuracy for its declared
conditions. Codec research on asymmetric errors between genuine and
synthetic speech also supports reporting missed edits and false alerts
separately for each background condition. The cited study examined
neural codecs. [Codec robustness study](https://arxiv.org/abs/2609.39651)

Recount the fresh holdout without audio or detector weights.

```sh
review=$(mktemp -d)
panel=docs/evaluation-panels/audio-two-second-opus-confirmation-2026-11-18.json.gz
python3 -B -S - "$panel" "$review" <<'PY'
import gzip
import json
import sys
from pathlib import Path

with gzip.open(sys.argv[1], "rt", encoding="utf-8") as stream:
    receipt = json.load(stream)
out = Path(sys.argv[2])
parts = {name: receipt[name] for name in ("manifest", "truth")}
for name, data in (parts | receipt["readings"]).items():
    (out / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
PY
python3 -B -IS backend/scripts/independent_review.py compare-events \
  "$review/manifest.json" "$review/truth.json" "$review/three-second.json" \
  "$review/with-two-second.json" "$review/paired.json" --match 0.3 0.5
```

## Replay

Use the two source directories and published matching receipts from
[the earlier carrier comparison](audio-two-second-carriers-2026-11-17.md#reproduce-the-comparison).
The scorer needs the incumbent's pinned local voice and activity files. The
separate reader independently reconstructs the event counts. With the two
source directories, it also re-encodes one file at a time and verifies its
original frame clock, signed sample count and decoded PCM hash without weights.
The frozen procedure stores the source and implementation digests with all
case measurements.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
reader=backend/.venv/bin/python
wav=docs/evaluation-panels/audio-two-second-carrier-wav-2026-11-17.json.gz
aac=docs/evaluation-panels/audio-two-second-carrier-m4a-2026-11-17.json.gz
"$reader" -B -m scripts.audio_two_second_opus_transfer freeze "$wav" "$aac" "$first" "$second" "$run/plan.json"
"$reader" -B -m scripts.audio_two_second_opus_transfer score "$wav" "$aac" "$first" "$second" "$run/plan.json" "$run/result.json.gz"
python3 -B -S -m scripts.review_audio_two_second_opus_transfer "$run/result.json.gz" "$wav" "$aac" --native "$first" "$second"
```
