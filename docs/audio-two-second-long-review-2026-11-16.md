# Two-second speech windows in longer recordings

Chipmunk Verify can show elevated speech responses in continuous,
native 16 kHz mono WAV and FLAC recordings from nine to fourteen
seconds long. The optional local Full analysis uses overlapping
six-second excerpts starting every two seconds and at the exact end
of the file. It uses the established two-second calibration and
checks a window when at least one second contains detected speech.
Each excerpt contributes its strongest qualifying response. Duplicate
source intervals appear once.

The normal audio view shows a playable two-second excerpt, a marked
waveform, exact source times, and links to overlapping observations
of the mixed recording when available. The PDF keeps the source
time and waveform together, and the JSON includes the source
positions and checked excerpts. Ordinary editing and other voices
in the mixture can also cause elevated responses. The review
contributes zero weight to the overall AI estimate.

The optional development setting is
`VERITAS_AUDIO_TWO_SECOND_LONG_REVIEW=true` in the local candidate
audio configuration. The earlier long-window review continues to
run with its separate evidence. The option defaults off. The check
reuses the existing commercially permitted voice and speech activity
assets. [Asset terms and hashes](../backend/app/models.yaml)

## Prior results and application replay

On 180 paired deliveries from an earlier panel, adding the
two-second check to the current long review raised useful locations
from 46 to 66 of 115 on original recordings, with zero flags on
18 ordinary files. At −5 dB the useful count rose from 22 to 36 and
ordinary flags rose from 8 to 12 of 18. At −10 dB the useful count
rose from 10 to 20 and ordinary flags rose from 8 to 10 of 18.
One of 36 isolated-background deliveries flagged at each level
after the extension. The same original recordings were reused across
the noise conditions. [Earlier paired results](audio-long-focused-context-2026-11-16.md#completed-paired-comparison)

Five local simulated USD 9.99 purchases exercised the check through
the normal queue. No job queued before payment. The ordinary and
isolated backgrounds stayed clear in the app replay. The clean WAV
and the noisy FLAC each showed two playable windows, including at
least one correctly located edit. The browser decoded all four clips
as two seconds. For each window the PDF keeps the exact time and
exported waveform on the same page. The signed replay compares
the clips and covered samples against the original file. The clean
test produced the same earlier evidence and overall assessment
across the on and off runs.
[Prepaid browser, PDF and signed export receipt](evaluation-panels/audio-two-second-long-app-2026-11-16.json)

## Fresh transfer procedure

The follow-up has 24 known edited recordings from the six published
generator families and 24 untouched recordings. The untouched group
contains ten files from the same corpus and 14 from the LibriSpeech
test-other archive. Exact source IDs and native PCM hashes are compared
with prior receipts. There are 41 different declared foreground readers.
Eighteen appeared in the earlier paired panel. The meetings differ
from the prior paired study and use different participants. They come
from the same AMI corpus as that study. The waveform of each source
stays native before applying the common gain and real competing-speech
mixes at −5 and −10 dB. Every mixture has an exact isolated background
control. [LlamaPartialSpoof and labels](https://zenodo.org/records/14214149)
· [LibriSpeech and its license](https://www.openslr.org/12/)
· [AMI license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

The fixed comparison counts one-to-one detections, ordinary and
background flags, unassessed cases, and genuine time highlighted.
It tests the application rule and a stricter comparison that requires
an elevated, distinct two-second window with at least one second of
overlap. This asks whether a second window placement can reduce
isolated flags. The same detector reads both windows. A recent
multi-speaker localization study describes how aggregating responses
can affect boundary precision and how controlled results degrade
under recordings and generators outside calibration.
[Multi-speaker temporal localization study](https://arxiv.org/html/2609.10051v1)

## Fresh transfer results

The input audit verified all 240 native delivery hashes, six source room
hashes, and 96 exact mixture and isolated-background pairs. The largest
difference between the intended and rendered mixing ratios was 0.00053 dB.
Every delivery had eligible measurements. At each noise level the same
24 edited recordings contain 155 labeled intervals, and 24 ordinary
recordings under the same conditions provide controls. Each of the 48 mixtures
also has an isolated-background control.
[Input clock and mixing audit](evaluation-panels/audio-two-second-long-fresh-input-audit-2026-11-16.json)

| Condition | Edit intervals located out of 155, current / two seconds / with second window | Ordinary flags out of 24, current / two seconds / with second window | Isolated-background flags out of 48, current / two seconds / with second window |
| --- | --- | --- | --- |
| Clean | 64 / 82 / 81 | 0 / 1 / 0 | — |
| Competing speech at −5 dB | 34 / 49 / 47 | 5 / 11 / 5 | 0 / 6 / 1 |
| Competing speech at −10 dB | 19 / 27 / 22 | 1 / 8 / 2 | 0 / 6 / 1 |

The independent event counter requires at least 0.30 intersection over
union and half of the known interval recovered. All three comparisons
keep the current selections. With a second overlapping window required,
additional matches came from 13 clean sources, 10 sources at −5 dB, and
three at −10 dB. The condition at −10 dB still missed 133 of 155 known
intervals with the second-window rule. These are matched comparisons on
three deliveries of the same edited sources.

The results favor using the second window to confirm a focused range.
The two-second window remains useful as local evidence. The confirmation
step needs its own app replay for the saved source samples and captions.
Both comparisons use the same voice detector and are treated as
supplementary evidence. The published receipt stores positions and scalar
responses. The independent export checks the event counts and the amount
of ordinary audio covered by flags. [Numeric transfer receipt](evaluation-panels/audio-two-second-long-fresh-2026-11-16.json.gz)

## Reproduce

Keep the native source archives local. The new Part B ranges each
contain 32 MiB starting at the 1, 1.5, 3, 5, 6, 7.5, 10 and 11.5 GiB
offsets. Save the HTTP 206 receipt and the bytes for each range. The
Part A files match the exact prior checksum list. Use the official
LibriSpeech test-other archive and the pinned AMI annotation release.
The local loader checks the published archive checksum and the source
sample clocks. [Part A and Part B preparation](audio-noise-window-transfer-2026-11-12.md#replay)
· [Official LibriSpeech checksums](https://www.openslr.org/resources/12/md5sum.txt)

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
runner=backend/.venv/bin/python
module=scripts.audio_two_second_transfer
"$runner" -B -m "$module" capture "$ami_annotations" "$run/rooms"
"$runner" -B -m "$module" prepare "$ranges" "$part_b_labels" "$part_a" "$part_a_labels" \
  "$librispeech" "$run/rooms" "$run/inputs"
"$runner" -B -m "$module" score "$run/inputs" models/weights/audio-upgrade "$run/result.json.gz"
python3 -B -S -m "$module" review "$run/result.json.gz"
python3 -B -S -m "$module" export "$run/result.json.gz" "$run/independent"
"$runner" -B -m scripts.audio_two_second_app \
  docs/evaluation-panels/audio-long-focused-context-paired-2026-11-16.json.gz \
  "$prior_native_inputs" "$run/app" --browser
```

The `review` and `export` commands also accept the published numeric
receipt directly. The export supplies a manifest, exact source-sample
truth, and results from the three comparisons for another evaluator.
