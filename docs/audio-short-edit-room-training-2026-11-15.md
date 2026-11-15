# Training a short-edit detector across room and level changes

The exact-voice diagnostic found that the one-second classifier ranked
14 of 16 PhonemeDF edits above their genuine controls before room
rendering and nine after rendering. This study changes the training
inputs and compares the results with a dry-only fit. It tests short
synthetic replacements against genuine edits on the same sample clock.
Room augmentation has also been studied for whole-utterance detection.
[Exact-voice diagnosis](audio-two-voice-isolation-2026-11-15.md) ·
[Room-response augmentation study](https://arxiv.org/abs/2409.14712)

## Source roles and delivery conditions

The original PhonemeDF roster supplies 48 fitting prompts, 16 control
calibration prompts, and 24 held prompts. These roles use separate readers
and separate recordings of competing conversation. A second held bank
uses eight MLS hosts and matched genuine splices, with Piper and KittenTTS
replacements. The speaker and capture audits check the stock bank against
the fitting and calibration roles. The dry source excerpts and backgrounds
come from earlier qualified studies. The room deliveries are newly made
for this comparison.
[PhonemeDF sources and rights](audio-short-edit-detector-training-2026-11-14.md#source-and-fitting-roles) ·
[Stock sources and rights](audio-random-clock-stock-2026-11-13.md#frozen-inputs)

Each source group produces four delivery conditions at 16 kHz on the
same six-second clock. The dry level control reduces the signal by 8 dB.
For the room condition, align the dominant impulse peak and match the
untouched clip's RMS to the dry clip. Keep that level correction constant
for all genuine and synthetic variants in the group. The fourth condition
adds the exact matched competing conversation 5 dB louder than the
room-rendered foreground, measured over the full clip. It also keeps
an isolated version of the louder conversation as a negative control.
Quantized foreground and background samples add exactly in every mix.
Fit and calibration use Surrey responses. PhonemeDF testing uses BBC
responses. The separate stock test uses Huddersfield responses.
[Room response rights and capture geometry](audio-room-assets-v299-method.md) ·
[BBC qualifications](audio-bbc-room-assets-v303-method.md)

## What the three fits test

All fits reuse the same small temporal convolutional architecture, 64
log-Mel bands, speech gate, 16 epochs and matching initial seed. The
loss covers the edited frames and endpoints and ranks the edited interval
above the matched genuine splice and resampled control. The dry-only fit
sees the original dry view. The paired fit sees all four versions and
uses the dry margin as a reference for the other three. A separate paired
fit subtracts the median of each band's detected speech frames from that
band across the clip, to test whether reducing stationary coloration
preserves the discriminative signal. The same transformation applies at
inference. No new pretrained authenticity weight is used.
[Shared localization architecture and analysis windows](audio-short-edit-detector-training-2026-11-14.md#detector) ·
[Speech activity model and terms](https://github.com/snakers4/silero-vad)

Fit one fold on MeloTTS and hold out Chatterbox. Reverse those roles in
the other fold. Both folds test both stock generators. Stock observations
repeat across the folds, so each fold's result stays separate. Derive two
cutoffs per fitted model from the ordinary calibration sources. The strict
point is their maximum. The wider point is their 95th percentile by
nearest rank. Both also require exceeding the highest assessed isolated
background from calibration. The held stage binds the complete fit
receipt and checks each weight file's digest before it scores a held clip.

The short proposals span 0.50, 0.75 or 1.00 seconds. A useful location
covers at least half the known edit and reaches 0.30 temporal intersection
over union. Results also track the same edit clock in genuine controls,
unassessed cases, wrong ranges, and isolated background alerts. The actual
current two-second app scan runs on the held competing-speech inputs
using its existing serving point. Its longer windows can limit overlap on
the shortest edits. The replay also measures a fallback that keeps a
current finding whenever it exists and uses the candidate only when the
current scan has no flagged window.

## Held results

The native audit verified 2,288 distinct waveform hashes with no exact
PCM overlap between source roles. It reconstructed every loud mixture
as the sum of its room-rendered voice and isolated background. The
measured gains were within 0.0002 dB of the three intended differences
of −8 dB for level only, 0 dB for room only, and +5 dB for competing
speech relative to the foreground.

At the strict points, subtracting each band's speech median gave the
largest gains on the held BBC room-only PhonemeDF clips. Each table cell
gives useful locations out of 24 edited files and flags out of 72 ordinary
files. The ordinary files include untouched, genuinely spliced and
same-word resampled controls.

| Fit | Held Chatterbox useful · ordinary flags | Held MeloTTS useful · ordinary flags |
| --- | ---: | ---: |
| Dry-only fit | 1 · 51 | 0 · 3 |
| Paired fit | 1 · 42 | 0 · 2 |
| Paired fit with per-band centering | 3 · 31 | 4 · 10 |

In the held Chatterbox fold the centered fit retained the dry-only
location and added two. It removed 20 ordinary flags and added none.
Two of its three useful locations were clear of ordinary alerts at the
same edit clock. In the other fold all four locations were clear of
matched alerts. The centered fit added 10 ordinary flags and cleared
three. These are gains on the rendered room-only panels, with remaining
ordinary flags in both folds.

On the louder competing-speech inputs, the centered fit trained on
MeloTTS found **3/16** Huddersfield stock-voice edits versus **1/16** for
its dry-only fit. Ordinary flags moved from **4/16 to 5/16** and isolated
background flags from **3/8 to 2/8**. Two of the three locations were
clear of matched alerts. The other centered fold found no stock edits.
Both centered fits found zero useful locations on held PhonemeDF under
louder speech at their strict points. The wider points and all four
conditions are in the numeric receipt.

The current app scan found one useful PhonemeDF interval per fold under
competing speech. In each fold it also flagged 25/72 ordinary controls,
and its useful interval coincided with an ordinary alert at the edit clock.
It found zero stock locations with zero ordinary or isolated-background
flags. Using the stronger stock candidate as a fallback would add all
three locations, five ordinary flags and two isolated-background flags.
In the same fold the fallback adds no PhonemeDF locations and increases
ordinary flags from 25/72 to 40/72 and background flags from 0/24 to 6/24.

## Separate recorded-room transfer

After the initial held comparisons, run all six frozen models on the
same 48 already qualified files from four different meeting rooms.
The set contains 11 edited files, 33 ordinary files and four isolated
backgrounds. It verifies no exact PCM or meeting capture overlap with
the primary study and excludes declared LibriSpeech hosts and donors
from the PhonemeDF source roles. The recorded background has not been
passed through the synthetic room renderer. It is mixed with speech
containing clone, expressive and stock replacements. No model or cutoff
is fitted on these data. [Transfer sources and scope](audio-short-edit-speech-layers-2026-11-14.md#separate-transfer-test)

All six models assessed all 11 edits. None of the strict rules found a
useful location there. The wider paired rule in the MeloTTS-trained fold
found one location that recurred on a matched genuine control, with
24/33 ordinary flags. Neither wider centered rule found an edit. This
keeps all three fits in evaluation and limits the room-only result to its
tested conditions. The separate recorded-room failure does not support
adding the centered model to loud uploaded speech.

## Reproduce locally

The numeric review recomputes calibration, grid-valid decisions, new and
lost correct ranges, and ordinary and background alerts. Recorded audio,
representations and model files stay in the local run. The receipt includes
waveform and model digests and the scalar decisions.
An independent numeric recount reproduced all six calibration pairs and
120 decision rows, including the app fallback comparisons. A separate
recount checked all 12 recorded-room outcomes. A second native pass
using the frozen weights reproduced all six held score ledgers byte
for byte.

```sh
export PYTHONPATH=backend
module=scripts.audio_short_edit_room_training
receipt=docs/evaluation-panels/audio-short-edit-room-training-2026-11-15.json.gz
python3 -B -S -m "$module" review "$receipt"
```

For native reproduction, use the source plan and pinned local asset
folders from the studies above. Choose fresh output paths for each stage.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
"$runner" -B -m "$module" freeze "$run/input" "$phoneme" "$stock" "$rooms"
"$runner" -B -m "$module" fit "$run/input" "$run/fit" "$speech"
"$runner" -B -m "$module" held "$run/input" "$run/fit" "$run/held" "$speech"
"$runner" -B -m "$module" current "$run/input" "$run/app.json" "$voice_weights"
"$runner" -B -m "$module" transfer "$run/input" "$run/fit" "$run/transfer" "$speech" "$phoneme" "$recorded"
python3 -B -S -m "$module" combine "$run/input/freeze.json" "$run/fit/fit.json" \
  "$run/held/held.json" "$run/app.json" "$run/receipt.json" --transfer "$run/transfer/transfer.json"
```
