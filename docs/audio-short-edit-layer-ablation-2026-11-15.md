# Testing earlier speech-encoder layers under competing voices

The six-second single-upload comparison tests whether earlier frozen speech
features help locate a brief synthetic replacement when the background is
louder than the target. Earlier-layer readouts are motivated by Pascu and
colleagues’ cross-dataset results on whole-recording screening. This trial
asks a separate question about locating a particular edit under overlapping
speech. No code or checkpoint from that paper is used.
[Earlier-layer research](https://arxiv.org/abs/2606.30780)

## Frozen comparison

The previously fitted temporal head reads projected frame features from
layers 4, 8 and 12. Compare two refits, one keeping layer 4 and one keeping
4 and 8. The omitted input columns are set to zero in fit and inference.
The comparison keeps the same projected columns, head architecture, 16
training epochs, fixed initialization, generator splits, speech admission,
and 0.50, 0.75 and 1.00 second proposals. Each fit has its own two operating
points derived from genuine calibration readers and isolated backgrounds.
All transfer measurements use those fixed points. The ablation changes
both layer depth and the number of nonzero inputs.
[Original feature study and checkpoint terms](audio-short-edit-speech-layers-2026-11-14.md)

Each direction trains with one generator and holds the other out. The
primary panel's 48 synthetic edits, 144 ordinary recordings and 48 isolated
backgrounds have the same −5 dB clock as the prior study. The two directions
share held source groups. Useful locations cover half the actual edit and
reach 0.30 temporal intersection over union. Matched genuine splices,
resampling and isolated backgrounds also retain exact time ranges.

| Check, wider point | Useful, of 48 | Ordinary flags, of 144 | Isolated flags, of 48 |
| --- | ---: | ---: | ---: |
| Current short review | 4 | 64 | 0 |
| All three layers | 2 | 20 | 4 |
| Layer 4 | 1 | 19 | 0 |
| Layers 4 and 8 | 2 | 22 | 4 |

Keeping layers 4 and 8 located 10/48 dry edits at the wider point,
versus 5/48 for all three layers. Both had zero dry ordinary flags in
the primary panel. The dry gain did not persist on the separate stock
bank, where the best two-layer fit found 4/16 and the existing app check
found 11/16.

## Transfer to three source banks

The transfer reads 255 previously committed but separately sourced files.
The 48 room recordings use three additional generator families. The 124
stock recordings use two additional families and separate dry, 0 dB and
−5 dB conditions. The 83 randomized-clock cases use two other stock voice
variants with 32 edits, 48 ordinary files and three isolated backgrounds.
Their eight hosts and three meeting captures are excluded from the other
banks. The randomized placements and exact media had earlier exposure in
the project, independently of these fitted heads and cutoffs. The replay
checks every submitted PCM digest and the recorded reader and room roles.
[Separate room sources](audio-remote-neighbor-replication-2026-11-12.md) ·
[Stock sources and terms](audio-stock-voice-transfer-2026-11-13.md) ·
[Randomized sources and terms](audio-random-clock-stock-2026-11-13.md)

Neither earlier-layer arm located a useful range among the 43 noisy edits
in the room and first stock banks. The current check located eight in
those banks. The three-layer model located the same ElevenLabs edit in
the room bank in both directions, and none of the 32 noisy stock edits.
Every transfer arm assessed all files in both banks.

The stronger result came from layer 4 trained with Chatterbox on the
randomized stock bank. At the wider point and −5 dB, it located **3/16**
and flagged **2/24** ordinary recordings, compared with **1/16** and
**3/24** for the current check. Two of the three results occurred in the
same group on two generator variants. A same-reader splice and a
different-reader splice both flagged the exact **2.400–2.900 s** window
in that group. The other location was **3.300–3.800 s**, inside a Kitten
edit at **2.861–3.821 s**, with no candidate flag at that clock in the
matched genuine files. At 0 dB the same fit located **2/16**, versus
**4/16** for the current check. Neither earlier-layer fit flagged an
isolated background on the three transfer banks.

Keeping both the current check and the strongest layer-4 result would
add three useful windows and two ordinary flags at −5 dB on the last
bank. Only one added window cleared the matched-edit-clock check. At
0 dB it would add one useful window and one ordinary flag. This compares
the two existing decisions at their frozen points. No combined confidence
was fitted. The cross-bank misses and repeated genuine splice ranges
leave both ablations in evaluation.

## Replay

The receipts contain sample intervals, scalar scores, source digests, and
the fit commitments. They carry no recordings, voice representations or
weights. The portable commands rebuild the cutoffs and recount eligible
locations and matched-control collisions. Python 3.12 and 3.14 returned
identical results. Repeating the CPU fit reproduced all four head files
and the fit receipt byte for byte. Two native passes reproduced the
earlier-layer readings for all 255 transfer files.

```sh
export PYTHONPATH=backend
panels=docs/evaluation-panels
base="$panels/audio-short-edit-detector-training-2026-11-14.json"
original="$panels/audio-short-edit-speech-layers-2026-11-14.json"
fit="$panels/audio-short-edit-layer-ablation-fit-2026-11-15.json"
module=scripts.audio_short_edit_layer_ablation
python3 -B -S -m "$module" review "$base" "$original" "$fit"
for bank in transfer random; do
  python3 -B -S -m "$module" review-transfer \
    "$panels/audio-short-edit-layer-ablation-$bank-2026-11-15.json" "$fit" "$base"
done
```

Native replay uses the [original source plan and feature cache](audio-short-edit-speech-layers-2026-11-14.md#independent-replay).
Rebuild the three transfer banks from their linked methods. Set `speech`
to the local `silero_vad.jit` and `encoder` to the pinned WavLM directory.
The original fit directory provides the previously committed head files.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
"$runner" -B -m "$module" fit "$sources" "$feature_cache" "$speech" "$run/fit"
"$runner" -B -m "$module" transfer "$sources" "$rooms" "$stock" "$encoder" "$speech" \
  "$run/fit" "$run/transfer.json" --original-fit "$original_fit"
"$runner" -B -m "$module" transfer "$sources" "$rooms" "$stock" "$encoder" "$speech" \
  "$run/fit" "$run/random.json" --original-fit "$original_fit" --random-stock "$random_stock"
```
