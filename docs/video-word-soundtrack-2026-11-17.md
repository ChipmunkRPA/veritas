# Recover missed word intervals from a single soundtrack

The [whole-word trial](video-word-content-2026-11-17.md) compares a spoken digit with the mouth movement at the same time. Its first soundtrack reader often identified the right digit while returning too little support to begin the comparison. This study holds the mouth reader and both video decision points fixed. It compares three ways to obtain more audible intervals. Recent speech-alignment research motivates counting a wrong label, a misplaced interval, and an unassessed file separately. [Speech alignment benchmark](https://arxiv.org/abs/2609.32396) · [Audio and visual content study](https://arxiv.org/abs/2610.30872)

1. Divide the proposed digit's response by the total response for any digit. A minimum absolute response and separation from other digits still apply.
2. At the first reader's proposed time, confirm the same digit in a separate short recording excerpt and two excerpts shifted by 20 ms. The short reader uses normalized cepstral measurements and their change through the word.
3. Scan the full soundtrack every 40 ms for intervals between 200 and 600 ms. Require the frame-based and short-window readings to support the same digit at the new interval. Confirm nearby readings and decline similarly strong competing proposals. This route can supply a new interval when the first reader does not qualify.

The short-window reader was fitted on the existing genuine Lombard Grid training split. The extra admission points were chosen on 28 new utterances from seven development speakers and fixed independently for PCM and AAC. Selection maximizes the added correct audio times minus added wrong audio readings, with a preference for fewer wrong readings when tied. [Lombard Grid data and licence](https://zenodo.org/records/3736465) · [Short-window fit history](evaluation-panels/video-word-soundtrack-fit-2026-11-17.json)

## Held clips and transfer to a separate recording setup

Two Lombard Grid panels each use 33 previously unused videos, one from each of the same 33 held speakers. No target or donor recording repeats between panels or comes from fitting or development. Original GRID provides 16 further videos from its own capture setup. Exact publisher word boundaries are registered against the decoded GRID soundtrack using the full waveform and individual word matches. Both collections use scripted digit utterances. The study replaces a digit with a different digit of similar duration by the same speaker. Same-digit replacements, unchanged speech, gain changes and ±200 ms whole-track offsets are controls for a *word-content* claim. The edit needs both correct digit readings and at least 0.40 overlap of the union with the known interval. [GRID data and licence](https://zenodo.org/records/3625687)

| Panel and carrier | Correctly admitted replacement words, original → scan | Correct video locations, timing alone → plus original word reading → plus scan | Added flags on untouched or gain controls | Added flags on same-digit replacements | Added flags on whole-track offsets | Added wrong or misplaced readings on different-digit edits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| First Lombard, PCM | 14 → 16 of 33 | 3 → 9 → 10 | 1 | 0 | 1 | 0 |
| First Lombard, AAC | 14 → 16 of 33 | 4 → 10 → 11 | 3 | 0 | 2 | 1 |
| Second Lombard, PCM | 13 → 18 of 33 | 1 → 3 → 4 | 0 | 0 | 0 | 1 |
| Second Lombard, AAC | 14 → 19 of 33 | 1 → 3 → 3 | 0 | 0 | 0 | 2 |
| GRID, PCM | 4 → 9 of 16 | 0 → 0 → 1 | 2 | 1 | 2 | 3 |
| GRID, AAC | 4 → 9 of 16 | 0 → 0 → 1 | 3 | 2 | 2 | 2 |

The table uses the wider video arm, which requires a consistent mouth reading at neighboring times and checks for agreement at an alternate offset. The stricter video arm adds zero correct locations with the independent scan on all three panels. The two approaches that reuse the original interval also add zero correct locations in either video arm on these panels. The independent scan improved acoustic coverage under both encodings. The small gain in complete video findings comes with more incorrect or misplaced findings, particularly in the separate GRID recordings. The word-content check remains experimental. These are staged human speech replacements, so the results measure localization and word disagreement in scripted utterances.

Each Lombard carrier includes 66 untouched or gain deliveries, 33 same-digit deliveries and 66 whole-track offsets per panel. The corresponding GRID denominators are 32, 16 and 32. A control flag is counted when the word-content comparison adds a new highlighted file beyond the existing timing result. The final column counts wrongly supported candidate reports on edited files separately, even where another check already flagged the file.

[Development source plan](evaluation-panels/video-word-soundtrack-development-sources-2026-11-17.json.gz) · [First Lombard source plan](evaluation-panels/video-word-soundtrack-first-sources-2026-11-17.json.gz) · [Second Lombard source plan](evaluation-panels/video-word-soundtrack-second-sources-2026-11-17.json.gz) · [GRID source and word clocks](evaluation-panels/video-word-soundtrack-grid-sources-2026-11-17.json)

## Native video witnesses

The replay remuxed all 792 Lombard deliveries and checked the original picture bitstream, source frame timestamps, and exact presented audio samples. The newly recovered edit on the first panel points to **1.420–1.700 s**, with frames 34, 37 and 41 for surrounding and during-edit context, on both carriers. The additional PCM edit on the second panel points to **1.500–1.780 s**, with frames 35, 39 and 43. Frame indices start at zero. The numeric receipts retain frame hashes so a second machine can check the same positions. The original whole-word run also reproduced byte for byte under the updated evaluation runner. [First native replay](evaluation-panels/video-word-soundtrack-first-native-2026-11-17.json.gz) · [Second native replay](evaluation-panels/video-word-soundtrack-second-native-2026-11-17.json.gz)

## Reproduce

The portable recount requires neither media nor installed weights. It checks the frozen method, development speaker exclusions, the full matched case list, each reported audio interval and the recorded totals.

```sh
export PYTHONPATH=backend
panels=docs/evaluation-panels
prior="$panels/video-word-content-selection-2026-11-17.json.gz"
selection="$panels/video-word-soundtrack-selection-2026-11-17.json.gz"
python3 -B -S -m app.bench.video_word_soundtrack "$selection" "$prior"
for panel in first second grid; do
  python3 -B -S -m app.bench.video_word_soundtrack \
    "$panels/video-word-soundtrack-$panel-2026-11-17.json.gz" "$prior" --freeze "$selection"
done
```

For native inference, use the [publisher archives, weights and capture steps](video-word-content-2026-11-17.md#reproduce). Rebuild the earlier acoustic training source list with `audio-inputs`. Generate the short-window weight from that source list. The checked hash and fit history are published. Keep the short-window and original weights in separate folders.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VERITAS_OFFLINE=true
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
mkdir -p "$run"
"$runner" -B -m scripts.video_word_audio_windows "$fit/acoustic-source.json" "$sources" "$run"

# Use the matching source plan and capture outputs from the earlier study's capture procedure.
# Use the calibration split for the development plan and held for the others.
"$runner" -B -m scripts.video_word_soundtrack_transfer "$fit" "$prior" "$run/audio-word.safetensors" \
  "$sources" "$held_plan" "$held_crops" "$held_timing" "$run/held.json.gz" --phase held --freeze "$selection"
"$runner" -B -m scripts.export_video_word_content "$held_plan" "$selection" "$run/held.json.gz" \
  "$sources" "$run/native" --soundtrack-previous "$prior"

# Materialize the publisher ZIP members in the GRID roster under $grid and name the roster with-audio.json.
"$runner" -B -m scripts.video_word_soundtrack_grid "$grid" "$assets" "$fit" "$prior" \
  "$run/audio-word.safetensors" "$selection" "$run/grid.json.gz"
```

The GRID roster gives every video and audio ZIP member, the expected local filename, and SHA-256. The local measurements retain numeric decisions and source clocks; licensed originals and trained weights are rebuilt on the replay machine.
