# Compare a video with an earlier copy

On a development instance, set `VERITAS_VIDEO_REFERENCE_COMPARISON_ENABLED=true`. Sign in, open a completed short video analysis, and choose **Compare with an earlier video**. The comparison uses the retained analyzed clip and a temporary upload. It creates no analysis or payment order. The two supplied files establish the comparison inputs, and determining their order or authorship requires other evidence.

Each clip may be up to 48 MiB, 360 decoded pictures, and 12 seconds from its first to its last picture timestamp. The stream headers must declare at most 921,600 pixels per picture before the full frame probe starts. The tool compares every native relative presentation timestamp as an exact fraction and records each source's original time base and timestamps. Different timing, display dimensions, or unverifiable timestamps yield an unassessed comparison.

The comparison reports every consecutive range of differing decoded pictures. The three longest ranges include side-by-side pictures from before, during, and after the range when those pictures exist. The selected in-range picture has the greatest count of changed RGB pixels. An outline shows the exact native bounding rectangle of those pixels. Re-encoding and color conversion can affect every picture, and the report identifies that case explicitly. If the final picture differs, the record gives its timestamp without inferring an end time.

A scoped local comparison can identify one short region after H.264 re-encoding. With an FFV1 RGB earlier copy, the supported picture sizes and cadences are 360 × 288 at 25 pictures per second and 480 × 360 near 30 pictures per second. With a color 4:2:0 H.264 earlier copy, support is limited to 360 × 288 at exactly 25 pictures per second. Both clips must be upright, use identical relative picture timestamps and contain at most 100 pictures. The submitted copy must be color 4:2:0 H.264. A flagged region includes the same native comparison area in the earlier and analyzed pictures before, during and after the range. The check reports when there is insufficient shared picture content or too little support on either side of a potential range. A visual change may extend beyond the measured box, and ordinary processing may produce a local difference. [Lossless earlier-copy results](video-reference-lossy-transfer-2026-11-11.md) · [H.264 earlier-copy reserve results](video-reference-h264-grid-reserve-2026-11-11.md)

The earlier H.264 study recovered 24/32 neural redraw locations in the tested 25 fps group. A separate group localized 31/32 ordinary edits and produced no flags on 96 controls. The 480 × 360 H.264 earlier-copy trial produced three ordinary delivery flags and failed the joint gate. [Original H.264 trial](video-reference-h264-earlier-transfer-2026-11-11.md)

The first soundtrack is compared on the same 16 kHz decoded mono sample clock when both tracks have the same relative start, sample count, and continuous timing. Adjacent 20 ms bins with differences form a range. Counts include only samples that differ. The three longest ranges include waveforms and exact playable samples for the first two seconds or less. The two waveforms share a scale. JSON contains the exact source clocks, SHA-256 commitments, waveforms, and media. The separate PDF includes source identities, all ranges, picture previews, and selected waveforms. These measurements carry zero AI-confidence weight.

## Local verification

Four retained licensed CREMA-D videos supplied moving pictures and speech. The test prepared each pair with lossless RGB video and mono PCM at 25 pictures per second. A rectangular replacement used pixels from other frames of the same recording at frame indices `[12, 24)`. A sound-only replacement within the same pair used samples from elsewhere in that soundtrack at `[12800, 22400)`. All four comparisons recovered both intervals and the exact changed-pixel and sample counts at the measured locations. Four metadata-only remux controls showed no picture or sound differences. All four ordinary H.264 re-encodes changed every decoded picture and preserved the PCM. Four trims were unassessed because picture timestamps differed.

A separate replay compared the two exact MOV files from the [short lip-sync study](video-lipsync-local-delay-transfer-2026-11-11.md). It verified that all 83 pictures matched and that the only reported soundtrack range covered the known delayed interval at samples `[16640, 26880)`. The uploaded source may start sound before the first picture, so that example's audio file-clock range differs from the displayed time relative to its first picture.

The app test compares the exported JSON and media byte for byte to a source replay. It checks PDF times, before/during/after labels, and waveforms. It also verifies that unauthorized access and withdrawn revisions are refused and that comparison creates no persisted source or analysis records. [Publisher terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt) · [Aggregate receipt](evaluation-panels/video-reference-comparison-2026-11-11.json)

## Replay without the app

Use the project Python environment and FFmpeg. Both originals are required. The first command writes a new directory with the comparison JSON and the media it references. The second verifies the browser's JSON download, including the embedded media. It also accepts a directory generated by the first command. The replay re-decodes both files and compares the results against the export without model weights, an account, or a payment.

```sh
cd backend
python -m scripts.compare_video_references earlier.mov analyzed.mov evidence-folder
python -m scripts.compare_video_references earlier.mov analyzed.mov downloaded-comparison.json --verify
```

The replay uses the same measurement implementation as the app. Record your FFmpeg and Pillow versions if two runs produce different decoded bytes. Keep source files, previews, and sound clips in your local workspace.

To verify a downloaded version 2 JSON with a measured local range using only Python 3.11 or later, FFmpeg, and FFprobe, run from the repository root.

```sh
python3 -I backend/scripts/verify_native_video_interval.py analyzed.mp4 downloaded-comparison.json --earlier earlier.mp4
```

This reader opens both original clips and checks file hashes, the full native timestamp arrays, the displayed interval, and all six embedded before, during, and after crops against native RGB pixels. Add `--expect-receipt-sha256` with a separately recorded JSON hash to verify the exact export bytes. The result verifies source correspondence for the recorded range. To reproduce range selection and verify the soundtrack, use the full comparison replay above.

The optional local comparison uses version 2 of the video export. The verifier rebuilds the selected interval, exact native source crops, and both picture clocks. It also verifies earlier version 1 exports using the exact comparison. The initial version 2 replay covered 12 held source pairs, with identical range and tile selections on the five neural redraws, no local finding on six unchanged deliveries, and explicit abstention on the different-source pair. After the H.264 change, three archived version 1 and version 2 packages replayed with identical JSON and evidence bytes.

The authenticated H.264 earlier-copy comparison located a 0.920–1.880 s review interval in a held clip with a known ordinary edit. The PDF showed before, during and after pictures, and all six native JSON crops matched both source files pixel for pixel. The stored analysis result, billing and job records, and stored file list were identical before and after the comparison. An independent re-encode and a tandem re-encode yielded no local finding. The unrelated earlier copy abstained.

An unpaid simulated checkout created no analysis or worker job. Simulated Stripe confirmation of the USD 9.99 purchase preceded creation of the analysis job for a held GRID clip. The first CPU smoke test reached its time budget. An isolated comparison repeat disabled one existing face classifier. A later repeat kept all configured Quick checks enabled and finished the paid CPU worker in 120.24 seconds. The authenticated FFV1 earlier-copy comparison then exported the selected frames, exact native crops and clock range in PDF and JSON. Offline replay verified the export against both original files.
