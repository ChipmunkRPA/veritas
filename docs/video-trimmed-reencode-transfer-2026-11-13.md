# Compare a re-encoded trim against an earlier video

The local **Compare with an earlier video** action can align separately
encoded clips when the analyzed copy is shorter. Both clips need upright
720 × 480 H.264 4:2:0 pictures at the same verified rate of 24 or 25 fps.
The shorter clip needs at least 40 pictures and the earlier clip can
contain up to 100. The earlier exact-picture trim path still takes priority.

The new path tests each possible frame offset. It compares coarse changes
over two and four pictures and requires a distinct matching offset with
motion support near both ends. It then applies the existing local picture
comparison, which accounts for overall brightness and contrast. Re-encoding
can change decoded values across a whole clip. This path presents only
regions supported by the local comparison. The browser, PDF and JSON
retain both clocks, source frames and crops from before, during and after.
The soundtrack is outside this check. An ordinary edit can produce the
same evidence, and the observation contributes no AI confidence.
[Video copy localization research](https://openaccess.thecvf.com/content/CVPR2022/html/He_A_Large-Scale_Comprehensive_Dataset_and_Copy-Overlap_Aware_Evaluation_Protocol_for_CVPR_2022_paper.html)
· [Existing local comparison](video-paired-local-area-2026-11-13.md)

## Local assessment

Four previously consumed publisher recordings supplied development
deliveries. Eleven distinct CREMA-D recordings supplied held deliveries.
The held recordings had prior project exposure. Their trim-and-encode
deliveries were new. All 77 held pairs used 24 fps. Each source supplied
three edits with exact authored frame and pixel masks and four controls.
The control files include a strong H.264 re-encode of the same pictures, a
full-clip grade, a different source and a whole-picture translation.
The renderer does not use generated media or model weights.
[Publisher recording terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eebd53610a08b1dc/LICENSE.txt)
· [Paired-area sources](evaluation-panels/video-paired-local-area-sources-2026-11-13.json)
· [Picture-return sources](evaluation-panels/video-scene-return-public-sources-2026-11-13.json)

| Held case | Correct offset | Local comparison completed | Useful region or control flag |
| --- | ---: | ---: | ---: |
| Large local grade | 9/11 | 9/11 | 9/11 |
| Small local grade | 10/11 | 10/11 | 9/11 |
| Local texture copy | 9/11 | 9/11 | 9/11 |
| Re-encode | 10/11 | 10/11 | 0/11 flags |
| Whole-clip grade | 11/11 | 6/11 | 0/11 flags |
| Wrong source | 0/11 | 0/11 | 0/11 flags |
| Whole-picture translation | 0/11 | 0/11 | 0/11 flags |

The older exact-picture path aligned none of these separately encoded
pairs. The new path located **27 of 33** known edits and found no region
on the 44 controls. Six matching ordinary controls reached no local
comparison, and all 22 wrong-source or translated controls abstained.
Every admitted offset was correct. One flagged small edit fell below
the required 0.50 temporal IoU. The separate source reader matched all
168 native PTS and RGB hashes in the 28 flagged pairs.
[Decision ledger](evaluation-panels/video-trimmed-reencode-transfer-2026-11-13.json)

The app comparator reproduced all 77 held decisions. An isolated Full
analysis of a held edit used simulated payment, then the authenticated
earlier-copy endpoint produced the two timelines and six native crops.
Chrome loaded all six source previews and six crops and downloaded the
same PDF and JSON as the endpoint. The standalone source verifier
re-decoded all six crops in the downloaded JSON from both files.

## Replay

The development app uses the existing `VERITAS_VIDEO_REFERENCE_LOCAL_AREA`
option on the earlier-video comparison. The caller must supply both files.
The public study acquires 15 exact publisher recordings by checksum and
retains generated carriers locally. The recorded toolchain binds the
encoder and decoder used for byte-for-byte repeatability.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for phase in freeze development held; do
  backend/.venv/bin/python -B -m scripts.video_trimmed_reencode_transfer "$phase" "$run" || exit 1
done
python3 -B -IS backend/scripts/review_video_trimmed_reencode_transfer.py \
  docs/evaluation-panels/video-trimmed-reencode-transfer-2026-11-13.json
backend/.venv/bin/python -B -m scripts.compare_video_references "$earlier" "$analyzed" "$export" --verify
python3 -B -IS backend/scripts/verify_native_video_interval.py "$analyzed" "$export" --earlier "$earlier"
```
