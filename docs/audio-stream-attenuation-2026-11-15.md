# Turning down one voice at a time in a noisy recording

The earlier comparison often found a cleaner estimated voice without
recovering evidence of the edited words. This study turns down each
estimated voice in turn, keeping both in the submitted mixture. It tests
whether the detector's response follows one estimated voice at the same
time and whether that voice is closer to the known foreground on the
benchmark. Recent research similarly tests detector responses to speech and
background components separately. This experiment uses the project's own
waveform operations and existing pinned assets. [Earlier separation study](audio-mixture-phase-isolation-2026-11-14.md) ·
[Component-wise detector research](https://arxiv.org/abs/2610.17617)

## Inputs and comparison

The AMI development and ICSI transfer panels each contain 16 synthetic
word replacements, 24 ordinary mixtures and eight matched room-only
excerpts. Background speech is 5 dB louder over the six-second mixtures.
The panels reuse the eight foreground source groups and their
Chatterbox and MeloTTS replacements with different meeting recordings.
This measures background transfer with shared foregrounds. A separate
stock-voice panel uses 16 Piper and KittenTTS replacements, 24 ordinary
mixtures and three isolated room excerpts. Its eight host groups, source
recordings and randomized splice times were fixed in the earlier study.
Same-reader and different-reader genuine splices serve as controls.
The three isolated excerpts are from the same meetings as the mixtures
but come from different times. All three panels had previous project exposure.
[AMI/ICSI source preparation](audio-nested-voice-window-2026-11-14.md) ·
[Stock-voice source preparation and terms](audio-random-clock-stock-2026-11-13.md)

The pinned separator estimates two aligned voices. Complementary
time-frequency masks preserve the phase of the original mixture and sum
back to the original samples within `1e-5` before attenuation. Subtract
half of each estimated voice in turn. The two resulting waveforms retain
the original clock. Score one-second windows starting every half second.
Both versions must meet the same speech admission rule in a window.
The separator and independent classifier carry Apache-2.0 declarations.
Use the app's existing pinned speech activity weights.
[Original separator](https://huggingface.co/alibabasglab/MossFormer2_SS_16K) ·
[Pinned ONNX conversion](https://huggingface.co/mhavo/mossformer2-ss-16k-onnx/tree/3be7f95237b5121caf41dec4c62b0af341a640af) ·
[Pinned classifier](https://huggingface.co/Gustking/wav2vec2-large-xlsr-deepfake-audio-classification/tree/f7050b586236dc910d1157f430def2d0647b02b4) ·
[Speech activity weights and MIT terms](https://github.com/snakers4/silero-vad)

Compare the highest of the two responses with a second rule that requires
their difference to reach log(2) in raw log odds. Under that rule the
estimated voice which was turned down less in the stronger reading is the
one the response appears to follow. A third rule also uses the lower of
the stronger response and the original mixture's response at that time.
Each rule uses a separately measured AMI cutoff. It has to exceed the
second-highest ordinary source-group maximum and the highest assessed
isolated background response. The ICSI and stock evaluations use those
cutoffs without fitting. A fallback keeps the current two-second result
when that result has a flag.

A useful location covers at least half of the known change and has at
least 0.30 temporal intersection over union. The same overlap rule
applies to matched control locations at the edit clock, even when their
reported endpoints differ. The acoustic diagnostic compares both estimates
with the exact clean foreground over the known edit, using scale-invariant
signal-to-distortion ratio. Only that diagnostic gets the clean foreground
and known edit bounds. The estimated stream can still contain more than one
voice.

## Local results

On AMI development, none of the three candidate rules usefully located an
edit. The current check and the one-second mixture baseline also had zero
useful locations at their respective points. The three candidate cutoffs
were 2.2521501, 2.1521978 and 2.0848012 raw log odds, in table order.

| ICSI transfer | Useful edits /16 | Useful without an alert at the same edit clock on a matched control | Wrong edit windows | Ordinary flags /24 | Room-only flags /8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current two-second review | 0 | 0 | 1 | 0 | 0 |
| Original mix, one-second windows | 4 | 0 | 6 | 15 | 3 |
| Fully separated mixture-phase estimates | 4 | 1 | 7 | 16 | 2 |
| Half attenuation, highest response | 3 | 0 | 8 | 17 | 1 |
| Half attenuation, require a selective response | 1 | 0 | 9 | 12 | 0 |
| Require both a selective response and support from the mix | 1 | 0 | 0 | 3 | 0 |

The last rule eliminated the one-second baseline's six wrong edit windows
and 12 of its 15 ordinary flags. It also lost three of the four useful
localizations. Its **3.500–4.500 s** location covered the known
**3.550–3.920 s** replacement. It also flagged the same interval and
estimated voice in both the untouched and genuine resampling controls.
The estimate it favored scored −28.82 dB against the clean foreground,
compared with −0.82 dB for the other estimate. The unfiltered
half-attenuation method followed the closer estimate on one of its three
useful locations. The most selective fallback kept the current ranges
and added one useful ICSI location and three ordinary flags.
Across all 96 AMI and ICSI deliveries, 93 had two eligible estimated
streams. A separate standard Python reconstruction verified the bytes of
all 186 attenuated inputs.

| Stock-voice transfer | Useful edits /16 | Wrong edit windows | Ordinary flags /24 | Room-only flags /3 |
| --- | ---: | ---: | ---: | ---: |
| Current two-second review | 1 | 1 | 3 | 0 |
| Original mix, one-second windows | 0 | 2 | 5 | 0 |
| Half attenuation, highest response | 0 | 0 | 1 | 0 |
| Half attenuation, require a selective response | 0 | 2 | 5 | 0 |
| Require both a selective response and support from the mix | 0 | 2 | 4 | 0 |

All 43 stock deliveries produced two estimates. The two selective rules
each lacked a qualifying window on one of the 16 edited files. None of
the three rules added a useful location even as a fallback, while they
added one, five and four ordinary flags respectively. With the result
from ICSI, this leaves no new useful location clear of a matched ordinary
alert and tied to the known foreground. Keep the three methods in evaluation.

The held sets test switching the competing conversation and then switching
the foreground source bank, generators and edit placements together.
Both use genuine backgrounds. To test whether model evidence can identify
the synthetic component, a further benchmark needs matched mixtures with
synthetic content in the primary voice, the competing voice, both, or neither.

## Replay

The numeric receipt exposes sample-clock commitments, raw scalar model
responses and counts. No recording, transcript, voice feature vector or
weight is included. A reviewer can rebuild the cutoffs, coverage, wrong
windows, false alarms and edit-clock collisions with standard Python.
A separate local recount reproduced the three candidate and fallback rows
for all three panels and both sets of held cutoffs.

```sh
export PYTHONPATH=backend
module=scripts.audio_stream_attenuation
receipt=docs/evaluation-panels/audio-stream-attenuation-2026-11-15.json
python3 -B -S -m "$module" review "$receipt"
```

Native reproduction uses the [two earlier mixture-phase views](audio-mixture-phase-isolation-2026-11-14.md#replay),
the original stock-voice archive, the pinned separator and classifier, and
the existing speech activity weights. Use ONNX Runtime 1.24.3 for the stock
preparation and output to fresh folders. The AMI phase finishes and the
cutoffs are fixed before either transfer phase begins.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
"$runner" -B -m "$module" measure ami "$run/ami" "$views/ami-views" "$classifier" "$speech"
"$runner" -B -m "$module" measure icsi "$run/icsi" "$views/icsi-views" "$classifier" "$speech" \
  --development "$run/ami/scored.json"
"$runner" -B -m "$module" measure-stock "$run/stock" "$stock" "$classifier" "$speech" \
  "$separator" "$run/ami/scored.json"
python3 -B -S -m "$module" combine "$run/ami/scored.json" "$run/icsi/scored.json" \
  "$run/receipt.json" --stock "$run/stock/scored.json"
```
