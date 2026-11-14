# Locating competing voices in natural speech

Two local tests measured overlapping speech on **128 three-second excerpts
from eight AMI meetings**. A separate replay used **93 archived deliveries**
to measure what would happen if the overlap finding automatically withheld
an AI-leaning voice result. The overlap methods remain in evaluation.

## Sources and measurement

Each fresh meeting contributed eight excerpts with at least 1.2 seconds
of simultaneous annotated words and eight with at most 0.1 seconds. Every
excerpt contains at least 2.45 seconds of annotated speech. We selected
starts by hash from the manual annotation grid before downloading audio.
The eight meeting cohorts share no recorded participant with each other
or with meetings named in the prior repository evidence checked at
`f5080f4`. Pretraining exposure is unknown. Manual word timings can miss
breaths, nonverbal speech, and quieter speakers. The minimal-overlap
counts are measured against those labels. [AMI recordings and annotations](https://groups.inf.ed.ac.uk/ami/download/) ·
[AMI CC BY 4.0 terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

The experiment uses the existing pinned, MIT-licensed speaker segmentation
checkpoint through its public ONNX export. It counts native frames whose
top class represents at least two simultaneous speakers. An interval
needs at least 250 ms of contiguous support. The frame grid is 270 samples
at 16 kHz, and only receptive fields inside the original sample extent
enter the measurement. The first test compared right padding, centered
padding, and the intersection of both. The follow-up passed the exact
48,000 unpadded samples into the same dynamic input and compared them with
right padding on four new meetings. The follow-up also spaced interval
starts by at least ten seconds. Neither view identifies the speaker who
caused a voice-origin response. [Model terms](https://huggingface.co/pyannote/segmentation-3.0) ·
[Public ONNX export](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models) ·
[Speaker segmentation methodology](https://www.isca-archive.org/interspeech_2023/bredin23_interspeech.html)

A useful interval must cover at least half the annotated overlap and
achieve a temporal intersection-over-union of 0.30 across the excerpt.
For the first evidence trial, the fixed rule requires at least four of
eight useful locations per meeting and no more than one marker in eight
minimal-overlap controls per meeting. The follow-up also requires at least
four more useful locations than right padding, with gains in two meetings.

## Results

| Source bank and method | High-overlap locations | Minimal-overlap clips marked |
| --- | ---: | ---: |
| First, right padded | 19/32 | 3/32 |
| First, centered | 15/32 | 2/32 |
| First, agreement | 15/32 | 2/32 |
| Follow-up, right padded | 19/32 | 2/32 |
| Follow-up, exact unpadded input | 18/32 | 0/32 |
| Follow-up, agreement | 16/32 | 0/32 |

Both first-bank alternatives missed the meeting-level evidence criteria.
In the follow-up, removing padding eliminated two extra markings on
minimal-overlap clips and lost a net one correctly localized high-overlap
excerpt. It lost two useful locations in ES2009a, gained one in ES2014d,
and kept the totals in the other rooms. Across both banks the repeated
right-padded estimate varied across rooms even with the same checkpoint
and rule. The candidate did not qualify as a new general-purpose,
customer-visible overlap check.

The archived decision bank contains three false AI flags on genuine
foreground mixtures, one AI flag on isolated genuine room audio, and
sixteen correct synthetic flags. The test does not assume the designated
foreground is the sound that drove any response. Withholding AI results
where the marker appeared removed all three genuine-foreground false
flags, plus the isolated background flag, in every arm. It also removed
**nine of sixteen** true synthetic flags with right padding and **eight
of sixteen** with either centered padding or agreement. The latter two
each removed seven noisy hits and one clean hit. The source decision
rule allows a maximum of one lost hit and zero clean losses. This keeps
the overlap marker outside confidence scoring.
[Archived matched audio and source terms](audio-harmonic-overlap-2026-11-11.md#source-separated-comparison)

The public receipts include only the public-source sample ranges and
hashes, manual overlap ranges, model interval results, and anonymous
archived decision records. They contain no audio, word text, voice
embeddings, or checkpoint bytes.

## Reproduce from public source ranges

Review both ledgers with the standard library. The transfer reviewer's
`--first` argument also checks that the follow-up is
bound to the fixed first-bank source and interval records.

```sh
export PYTHONPATH=backend
first=docs/evaluation-panels/audio-locate-competing-speech-2026-11-13.json
transfer=docs/evaluation-panels/audio-overlap-padding-transfer-2026-11-13.json
python3 -B -S -m scripts.audio_locate_competing_speech review "$first"
python3 -B -S -m scripts.audio_overlap_padding_transfer review "$transfer" --first "$first"
python3 -B -S -m scripts.audio_locate_competing_speech replay-fetch "$first" --cache "$first_cache"
python3 -B -S -m scripts.audio_locate_competing_speech replay-fetch "$transfer" --cache "$transfer_cache"
```

Each fetch uses 64 short native PCM ranges and four WAV headers from the
first-party publisher, about 6.2 MB in total. It checks the native file
extent and PCM SHA-256 against the published commitments. Use empty cache
paths. To rerun the overlap model on these exact samples, obtain
`sherpa-onnx-pyannote-segmentation-3-0.tar.bz2` from the linked ONNX release.
Use the SHA-256 pinned in the existing [runtime freeze](evaluation-panels/audio-overlap-context-freeze-v180.json).
Expand the pinned macOS arm64 ONNX Runtime wheel into `$runtime/site` and
keep the exact wheel in `$runtime`. The runner verifies its installed
files before invoking the local model. Inference needs NumPy from the
project's Python environment. Both receipts can use the same asset.

```sh
backend/.venv/bin/python -B -m scripts.audio_locate_competing_speech replay-model "$first" \
  --cache "$first_cache" --runtime "$runtime" --model-archive "$export_archive"
backend/.venv/bin/python -B -m scripts.audio_locate_competing_speech replay-model "$transfer" \
  --cache "$transfer_cache" --runtime "$runtime" --model-archive "$export_archive"
```

Those commands rerun all 128 public-source measurements. Rerunning
inference on the separate 93 archived cases also requires the retained
earlier matched bank. The public reader reproduces its withholding
arithmetic from the published decision records.
