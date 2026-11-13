# Narrower speech evidence under rain and train noise

The earlier combined speech check found more local edits by adding all narrow
ranges to the existing broad ones. It also highlighted more genuine speech.
This comparison keeps every narrow range supported by two adjacent responses
and retains a broad range only when it contains no entire narrow range. The
reported overlap is an inspection interval on the original sample clock.
The method uses the existing licensed voice and speech-activity weights with
their unchanged calibration. It neither estimates the origin of the background
nor creates a new confidence score. [Earlier overlapping-window results](audio-dense-evidence-confirmation-2026-11-12.md) · [Installed weight terms](commercial-model-audit.md)

Recent research separates evidence near speech transitions from evidence
inside a segment. Separate studies emphasize multiple time scales, explicit
temporal coordinates, and evaluations under unfamiliar noise. This experiment
uses a smaller comparison of existing window responses on the source clock.
It uses no external code or weights from those papers. The VeriSpoof-Base
and VeriSpoof-Plus checkpoints carry noncommercial terms. SPADE lists its
English release as pending and requires gated access to review the terms.
[Boundary and segment learning](https://arxiv.org/html/2609.25822v1) · [Temporal pyramid research](https://arxiv.org/html/2606.16837v1) · [LENS-DF](https://arxiv.org/html/2507.16220v1) · [VeriSpoof weights](https://huggingface.co/AustinXiao/VeriSpoof-Plus) · [SPADE dataset card](https://huggingface.co/datasets/rogertseng/spade)

## Frozen comparison

The retained CC BY 4.0 LlamaPartialSpoof archive contributes 18 edited Part B
source groups across six generator families and 12 genuine Part A groups.
All 30 source file IDs and native PCM hashes are absent from the five compared
ledgers. All 30 speaker IDs appear in that history. The held files are new
exact recordings for this comparison, with prior project speaker exposure and
unknown model-training exposure. Six edited sources use cut and paste and
twelve use overlap and add. One malformed publisher clock was excluded during
scanning before any scoring. [Publisher release and terms](https://zenodo.org/records/14214149) · [Part B preparation](audio-noise-window-transfer-2026-11-12.md)

Each of the 30 clips supplies a gain-only delivery and two matched mixtures
with recorded rain or an arriving train. Each mixture sets the foreground
five decibels below the added background by full-clip RMS before PCM16
quantization. All three deliveries from a source share a peak-safe gain. Four
eight-second excerpts of each background enter alone, for 98 deliveries.
The two recordings were chosen before any inference and are separate from the
traffic, restaurant, construction and market recordings in earlier comparisons.
Repeated excerpts from one recording still describe one environment.
[Rain by Mijesty, CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:Sound_of_light_rainfall.ogg) · [Tampere train by Work With Sounds/Werstas, recorded by Laura Vaara, CC BY 4.0](https://commons.wikimedia.org/w/index.php?title=File:WWS_TrainStoppingAtTheTrainStation.ogg&oldid=1203805579)

Compare the current sparse rule, a fixed dense rule, their full union, and
the narrower evidence rule. The three-second windows, their directions, and
the source samples are shared across methods. One reported interval is useful
when it covers at least half a publisher-labeled edit and reaches 0.30
intersection over union. The frozen gate asks for three newly located noisy
source groups across two families, at most one loss per condition and one
additional genuine group under noise, no new clean or isolated flag, and no
more than two additional points of marked genuine time across the matched
speech files. The last measure includes genuine portions of edited speech.

## Held results

| Condition | Sparse | Dense | Full union | Narrower evidence |
| --- | ---: | ---: | ---: | ---: |
| Gain-only edits localized | 6/18 | 12/18 | 13/18 | 13/18 |
| Rain edits localized | 0/18 | 3/18 | 3/18 | 3/18 |
| Train edits localized | 0/18 | 0/18 | 0/18 | 0/18 |
| Genuine files flagged across three deliveries | 0/36 | 0/36 | 0/36 | 0/36 |
| Isolated background excerpts flagged | 0/8 | 0/8 | 0/8 | 0/8 |
| Genuine speech time marked across matched clips | 9.20% | 6.18% | 11.31% | 9.18% |

The narrower rule passed the frozen gate. It retained all useful sparse
locations and matched the full union's locations in each condition while
marking less genuine speech. The three noisy gains came from two generator
families in the rain recording. None appeared under the train recording.
Four genuine and three edited train mixtures produced no usable response.
All eight isolated excerpts stayed unassessed. The reported edit counts
include these skipped inputs. A separate replay verified all 98 delivery
hashes and all 511 window hashes against their exact PCM.

The result supports a bounded optional review of short native recordings.
The same rule has no demonstrated general rescue under strong noise. The
train result, shared speaker history, and small number of noise recordings
leave confidence changes unsupported. A scoped integration must preserve the
underlying three-second readings and display any narrower waveform as an
interval supported by the two surrounding windows.
[Frozen case ledger](evaluation-panels/audio-focused-window-transfer-2026-11-12.json)

## Replay

Use the pinned source ranges in the [Part A preparation](audio-reduced-background-localization-2026-11-12.md#reproduce-the-frozen-inputs)
and [Part B preparation](audio-noise-window-transfer-2026-11-12.md#replay).
Set `sounds` to a private directory and download the two original files.
The runner binds the exact inputs before measuring any voice responses.

```sh
curl -fLsS --max-filesize 1200000 'https://upload.wikimedia.org/wikipedia/commons/1/15/Sound_of_light_rainfall.ogg' -o "$sounds/rain.ogg"
curl -fLsS --max-filesize 2500000 'https://upload.wikimedia.org/wikipedia/commons/9/9d/WWS_TrainStoppingAtTheTrainStation.ogg' -o "$sounds/train.ogg"
export PYTHONPATH=backend VERITAS_AUDIO_DETECTOR_MODE=candidate VERITAS_BILLING_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_focused_window_transfer prepare "$run" \
  --part-a "$part_a" --part-b "$part_b" --labels "$part_a_labels" --background "$sounds"
backend/.venv/bin/python -B -m scripts.audio_focused_window_transfer score "$run"
python3 -B -S -m scripts.audio_focused_window_transfer review "$run/receipt.json" --native "$run"
```

The reviewer also accepts the published receipt with no native path to
recalculate the result from the saved decisions. Original clips and authored
mixtures stay local.
