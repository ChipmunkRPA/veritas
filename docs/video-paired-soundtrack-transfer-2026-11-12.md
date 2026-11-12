# Locate a soundtrack shift relative to the same pictures

The earlier-video comparison can now find a short piece of matching sound that plays earlier or later in the analyzed copy. It requires the two files to have the same decoded pictures, identical relative native picture clocks and matching continuous decoded audio clocks. It needs no timestamp from the user and no model weights. This measurement describes timing between the supplied copies and carries zero weight in the AI estimate. [Comparison and export guide](video-reference-comparison-2026-11-11.md)

## Fixed held comparison

The test uses 15 retained CREMA-D actor recordings from a pinned publisher release. Four actors supplied development views and 11 different actors supplied held views. The earlier lip-sync study used these recordings, so the test measures transfer to newly authored changes on previously acquired sources. All actors speak the same sentence. Each held actor supplied two local 640 ms shifts, one earlier and one later, with offsets from 160 to 280 ms and a 5 ms crossfade. The two matched deliveries use the same H.264 pictures with PCM in Matroska and AAC in MP4. [Earlier source qualification](video-lipsync-local-delay-transfer-2026-11-11.md) · [Publisher data and content terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt)

The check correlates 160 ms pieces of 16 kHz mono audio every 40 ms within a 360 ms radius. It requires a distinct waveform match, consistent offsets of at least 120 ms and two same-time matches on each side. After the four development actors, the method allowed gaps in the shifted sequence where a waveform match was ambiguous, while requiring matches in at least two-thirds of the windows. Same-time flank windows bound the reported interval. The final method, exact source roster and acceptance criteria were sealed before the held cases ran. A useful result needs at least 0.60 interval overlap, measured as intersection over union, with a lag error of at most 16 decoded samples, or 1 ms. The gate requires at least 16 useful ranges per codec, seven for each direction, eight source groups and no more than one control alert, with none on global shifts or changed pictures.

## Held results

| Case | PCM | AAC |
| --- | ---: | ---: |
| Local audio moved later, correctly localized | 11/11 | 11/11 |
| Local audio moved earlier, correctly localized | 11/11 | 11/11 |
| Alerts on assessed controls | 0/66 | 0/66 |
| Changed-picture controls left unassessed | 11/11 | 11/11 |

All 44 estimates recovered the exact authored signed sample shift. The smallest interval intersection over union was 13/16. Controls cover untouched audio, gain changes, an added local noise bed, an unrelated inserted excerpt, a whole-track shift, a short 40 ms offset and a clip with changed pictures. The existing exact audio comparison can report differing sample ranges in these clips. The new check also measures the direction and amount of local displacement when the matching evidence is present. Independent replay recalculated all 198 delivery decisions, verified 13,104 decoded pictures and 7,201,809 decoded samples, and reconstructed all 44 native PNGs and 88 playable waveform excerpts. [Held decision ledger and replay counts](evaluation-panels/video-paired-soundtrack-transfer-2026-11-12.json)

This study covers local audio changes on picture-identical copies. Differently encoded pictures can fail the exact match requirement. A constant whole-track offset, a newly synthesized replacement with different audio, or a change to the visible mouth requires other evidence. The measured interval covers the matching waveform support, so its boundaries may differ from the edit's full boundaries.

An isolated Full test analysis used one simulated USD 9.99 checkout. The authenticated earlier-copy comparison measured the held AAC excerpt at **0.560–1.160 s** in the earlier clip and **0.760–1.360 s** in the analyzed clip, a shift of **200 ms**. The exported picture at **1.034 s** matched native frame 31 pixel for pixel, and both playable excerpts matched the decoded samples. Chrome loaded the picture and both sound excerpts, showed the two waveforms, downloaded matching JSON and a PDF, and showed the explicit skip message on a changed-picture control. The separate comparison created no new payment, analysis, worker, or object records. Both the current and earlier-format exports replayed from the two original files.

## Replay

The published [intake roster](evaluation-panels/video-paired-soundtrack-sources/intake-freeze.json) lets a reviewer start without the project's earlier private source folder. Download the 15 clips from the pinned CREMA-D release with the existing acquisition command. It checks every upstream Git LFS pointer, clip length and clip SHA-256. The generated receipt must match the [published intake receipt](evaluation-panels/video-paired-soundtrack-sources/intake-receipt.json) byte for byte. Use a local backend environment with FFmpeg and keep source recordings and playable excerpts local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
replay=$(mktemp -d)
prior="$replay/sources"
run="$replay/run"
roster=docs/evaluation-panels/video-paired-soundtrack-sources
mkdir "$prior"
cp "$roster/intake-freeze.json" "$prior/"
backend/.venv/bin/python -B -m scripts.prepare_video_lipsync_local_delay_transfer acquire "$prior" || exit 1
cmp "$roster/intake-receipt.json" "$prior/intake-receipt.json" || exit 1
for phase in freeze development held report; do
  backend/.venv/bin/python -B -m scripts.video_reference_soundtrack_panel "$phase" "$run" "$prior" --workers 2 || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_reference_soundtrack_panel "$run/receipt.json" --run "$run" --workers 2
python3 -B -S -m scripts.review_video_reference_soundtrack_panel docs/evaluation-panels/video-paired-soundtrack-transfer-2026-11-12.json
```

The panel seals the downloaded inputs before authoring the matched views. The native verifier uses a separate waveform-correlation implementation and reopens each output file. A local replay from an empty directory reproduced the published results and verified all 198 held deliveries, 13,104 pictures and 88 playable excerpts. Compare the `results` section of your new receipt with the published one. Exact encoded-file hashes may depend on the FFmpeg build.
