# Earlier-copy localization of timing-preserving mouth edits

An earlier copy substantially helps locate short edits that preserve speech
timing. This test used two upright H.264 deliveries with the same native
picture clock. It searched the whole frame and did not receive a face location,
a marked event, or the authored time range. A result measures a difference
between copies. Ordinary pixel edits also produce such differences.

## Frozen comparison

Fit a response point for each arm on eight ordinary GRID sources. Compare each
source with itself, a separate CRF32 encoding, a global level adjustment and
a different source with the same picture clock. Use the same picture-content
gates for both arms. The first arm measures local blurred luminance difference.
The second also requires locally normalized fine-texture disagreement. Both
choose a tile and an interval of 8, 12, 16 or 20 pictures with five measured
pictures on each flank. Keep one before, middle and after frame from each file.
No detector weights were used. [GRID sources](https://zenodo.org/records/3625687)
· [Lombard sources](https://zenodo.org/records/3736465)

Transfer used 17 retained native 720 × 480 source pairs from two generator
families, each with two ordinary local controls. These files have earlier
project exposure. The comparison method and both operating points were frozen
before examining these pairwise measurements. Compare the selected tile with
all pixels changed in the corresponding lossless edit at the selected middle
frame. The frozen rule required at least 0.50 temporal and 0.10 spatial
intersection over union, including the separately timing-qualified clip from
the second family. [Native input qualification](lombard-fresh-musetalk-2026-11-13.md)
· [Prior single-upload comparison](video-mouth-blind-prediction-2026-11-13.md)

| Held result | Luminance | Luminance plus texture |
| --- | ---: | ---: |
| Timing overlap reached the minimum | 15/17 | 16/17 |
| Time and changed-pixel overlap both reached the minimum | 11/17 | 10/17 |
| First family, both reached the minimum | 10/11 | 9/11 |
| Second family, both reached the minimum | 1/6 | 1/6 |
| Identical, compression or global level controls flagged | 0/51 | 0/51 |
| Ordinary local resampling or earlier-pixel replacement flagged | 26/34 | 27/34 |
| Wrong-source pairs admitted | 0/15 | 0/15 |

Two sources had no sufficiently long wrong-source match in their family.
The first arm correctly localized nine of ten independently timing-qualified
first-family edits across six source groups. The second arm found eight across
six groups. On the separately timing-qualified second-family edit, both arms
returned the exact interval of 1.038062–1.702422 seconds. All 1,632 pixels in
their chosen tile changed, but the full change covered 18,572 pixels at that
frame. The single tile therefore fell below the frozen spatial overlap rule.
Neither arm passed the complete rule. The trial stays offline. A further test
can fit region extent separately from the time window, using new source groups
for confirmation. The earlier-copy requirement and ordinary local flags also
limit the interpretation of any future result to differences between copies.

## Replay

The public ledger contains anonymous case and source-group indices, file and
frame hashes, exact timing, tile coordinates and overlap counts. It excludes
media and local paths. The standard-library reader recomputes the operating
points and overlap results from those measurements. The native runner needs
the archived H.264 and lossless deliveries. It verifies those hashes and
uses FFmpeg to recover exact source frames. The first held attempt stopped
before scoring when the lossless second-family archive had been moved to
external storage. The rerun used the same frozen files after resolving their
paths to that archive. [Anonymous ledger](evaluation-panels/video-paired-native-mouth-transfer-2026-11-13.json)

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze development held; do
  backend/.venv/bin/python -B -m scripts.video_paired_native_mouth_transfer "$phase" "$carriers" "$run" || exit 1
done
python3 -B -IS backend/scripts/review_video_paired_native_mouth_transfer.py \
  docs/evaluation-panels/video-paired-native-mouth-transfer-2026-11-13.json
```
