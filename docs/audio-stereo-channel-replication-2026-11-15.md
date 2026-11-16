# Retest short stereo speech on new readers and measured rooms

The [earlier stereo comparison](audio-stereo-regression-2026-11-15.md)
found two additional useful locations with the tighter separate-channel
fallback. Both came from one scene. This replay tests whether the two
frozen operating points transfer to new speaker pairs and rooms. It
keeps the original five two-second windows, classifier, speech admission
and interval overlap rule.

## Source and room design

The pinned English MLS development shard supplies one native six-second
crop from each of 24 declared readers and 24 distinct books. The intake
excludes 24 reader IDs from the earlier MLS stock-voice studies,
including source and donor roles. Fixed hashes order the eligible recordings. The
crop rule checks speech-energy support throughout the clip, at the
destination and at a separate source interval for the genuine splice.
Enrollment uses waveform amplitude alone. Cross-corpus reader aliases and
classifier pretraining overlap are unknown.
[MLS release and CC BY 4.0 terms](https://www.openslr.org/94/)
· [Earlier stock source roster](audio-stock-voice-transfer-2026-11-13.md)
· [Earlier random-clock roster](audio-random-clock-stock-2026-11-13.md)

The two stock voices use fresh neutral prompts, the archived Piper
LJSpeech Medium and KittenTTS Nano 0.8 INT8 weights, and one stock voice
from each. Kitten's runtime starts a new ONNX session after setting the
seed for each prompt. The generator files are checked against the
earlier source receipt. Each insertion uses a 960 ms voiced fragment.
The ordinary splice uses a different fragment from the same genuine
recording. Both use the same 20 ms fades and match RMS to the replaced
segment. A half-amplitude preparation step gives all versions of one
recording equal headroom before splicing. The unchanged samples outside
each insertion are verified before room rendering. The generators
serve only to create controlled test inputs.
[Piper weights and training source](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/ljspeech/medium/MODEL_CARD)
· [Kitten checkpoint and terms](https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8)

Six reader pairs use lateral paths in the Aachen AIR 1.4 meeting room.
Six separate pairs use MAIR's Huddersfield St Paul's Hall on its S03
and S07 paths. Both rooms are held out from the previous separate-channel
comparison. The rendered signal preserves the stored
response clock and accounts for direct arrival in the edit label.
The second voice starts 5 dB louder by mono RMS in the untouched mix.
Each scene holds both gains constant across five deliveries: untouched,
two ordinary splices, a synthetic splice in the quieter or louder voice,
or synthetic splices in both voices. Half the scenes have overlapping
edit schedules. The other half have distinct edit clocks. Each room
gets both generator assignments. These are controlled binaural renders.
[AIR measurement and terms](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/)
· [MAIR assets and CC BY 4.0 terms](https://zenodo.org/records/2639677)
· [Measured Huddersfield paths](audio-blind-huddersfield-transfer-2026-11-12.md)

## Decision procedure

Keep the current mono result whenever it flags. Otherwise compare the
highest admitted response on the original left and right samples, using
each previously frozen channel point separately. Each view needs at
least one second of detected speech within the two-second window. All
reported locations must cover at least half an authored edit and 30%
of the temporal union. A range can cover both edit clocks where they
overlap. Speaker identity remains unknown. All 24 distinct authored edit clocks fit at
least one grid window before speech admission. A separate sample check
confirmed that each single-voice change starts at or after its authored
start, and that both changes add back to the simultaneous-edit delivery
in every scene. The points are internal comparison settings.

## Held results

Each room supplied 18 edited and 12 ordinary deliveries. Every file had
an admitted window. A misplaced flag landed outside the required overlap
with either known edit. Each table cell gives useful locations,
ordinary flags, and misplaced flags in that order.

| Rule | Aachen | Huddersfield | Combined |
| --- | ---: | ---: | ---: |
| Existing mono | 1 / 0 / 0 | 2 / 1 / 2 | 3 / 1 / 2 |
| Tighter channel fallback | 1 / 0 / 0 | 2 / 1 / 2 | 3 / 1 / 2 |
| Wider channel fallback | 1 / 0 / 0 | 3 / 4 / 6 | 4 / 4 / 6 |

All added flags at the wider point came from Huddersfield. Its one
additional useful range came from a file with two simultaneous edits.
On single-voice edits, every rule found one of 12 Piper replacements
and one of 12 Kitten replacements. The tighter fallback made the same
decisions as the existing check across the entire panel. Both channel
fallbacks remain in evaluation. These results describe the staged voice
and microphone positions used here. [Held receipt](evaluation-panels/audio-stereo-channel-replication-2026-11-15.json.gz)

## Replay

The public receipt is limited to threshold provenance, ordinary controls,
exact sample clocks, source commitments and scalar scores. A second
native preparation reproduced the complete frozen plan and all 60
stereo files byte for byte, including both families' generated inserts.
An independently implemented count agreed on the flags, placements
and coverage. The numeric replay produced identical results on Python
3.12 and 3.14.

```sh
export PYTHONPATH=backend
module=scripts.audio_stereo_channel_replication
receipt=docs/evaluation-panels/audio-stereo-channel-replication-2026-11-15.json.gz
python3 -B -S -m "$module" review "$receipt"
```

For native reconstruction, use the full pinned MLS development Parquet
file and the native earlier stock roots identified above. The preparation
reads `freeze.json` and the six verified generation assets from the
first root. It reads `placements-freeze.json` from the second root. It
also takes the qualified MAIR root and the unchanged AIR 1.4 archive.
Use the generator programs in disposable Python directories. The receipt
lists the runtime versions. The test ran the speech assessor on the Mac
CPU using the same three weights and configuration files as the parent.
[Pinned MLS mirror](https://huggingface.co/datasets/parler-tts/mls_eng/tree/faf6604dcf0bb9ec0d9c280a140ad74da9c931b0/data)

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
"$runner" -B -m "$module" freeze "$mls_dev" "$stock_root" "$placement_root" \
  "$mair_root" "$air_zip" "$run/input" --runtime "$kitten_runtime" "$generator_runtime"
"$runner" -B -m "$module" score "$run/input" models/weights/audio-upgrade "$run/receipt.json.gz"
```
