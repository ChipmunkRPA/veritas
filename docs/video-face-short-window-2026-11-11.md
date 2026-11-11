# Short-window face-swap localization study

The shipped paired face check missed a known three-second swap across four versions of the same clip. Its six-frame median suppressed a strong response at one sampled instant. This comparison used the same pinned model weights and added a denser, time-local rule. [Earlier paired result](video-face-pe-publisher-2026-11-11.md)

## Sources

- **Exposed video diagnostic.** Four exact insert deliveries of one FaceSwapAI production swap cover 3.000–6.000 seconds, with identical unmodified pictures outside that interval. An unchanged delivery and an ordinary recompression are matched controls. The publisher made both adult identities synthetically and released the study under CC BY 4.0. [Publisher video study](https://faceswapai.com/resources/video-face-swap-continuity-study) · [Exact matched preparation](video-faceswapai-matched-v186-method.md)
- **Separate photo source.** A different FaceSwapAI production study released exact input and output images of four fictional adults under CC BY 4.0. The exact local images match the publisher's SHA-256 hashes. The leftmost face and its published mapping were fixed before scoring. A 24-fps clip uses a small, deterministic horizontal crop motion over the still. The two positive deliveries insert the production swap for 0.750 or 1.500 seconds, starting at 2.500 seconds. The original, a matched 0.750-second blur and a matched 1.500-second JPEG recode are controls. Lossless encoding and complete native decoding verified every picture and presentation timestamp. This is a separate synthetic source component from the video study. Its two insert durations share one underlying face edit and contain no natural face motion. [Publisher photo study and license](https://faceswapai.com/resources/how-to-map-different-faces-in-a-group-photo) · [Exact source manifest](https://faceswapai.com/datasets/independent-multi-face-mapping-study-v1.json)
- **Genuine camera controls.** Two VCD2 720p source groups were selected by fixed SHA-256 order from the eligible receipt-complete recordings. Each has unchanged, chroma-converted and recompressed deliveries. The latter two change only 3.000–6.000 seconds. The camera datasets supply ordinary-edit negatives. [Microsoft data license](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA) · [Local exact preparation](video-vcd2-matched-controls-v290-result.md)

## Frozen comparison

The sampler took native pictures nearest each quarter-second, with earlier ties. Both models processed the same exact RGB, timestamps and largest-face boxes. Equal decoded RGB inputs shared one response within each arm. The rule required three consecutive scheduled core samples and two scheduled flank samples on each side. All seven needed measured faces, adjacent gaps no greater than 0.375 seconds and an adjacent box overlap of at least 0.45 IoU. This is geometric tracking. The rule used the higher qualifying contrast across the two arms, keeping the earlier development scales and paired operating point fixed. A core median had to exceed the paired point, both flank medians had to stay below it, and the core had to exceed the stronger flank by at least 0.5 in normalized response. Two core samples on the same arm therefore had to corroborate. These are internal detector scores. The range extends between corroborating sampled times.

The whole-clip comparison used the shipped six-frame paired rule. On retained identical bytes, the original native input receipts and scores were replayed. The five new deliveries received actual six-position scoring alongside the dense sampler. The appearance comparison used the exact archived app results for the publisher-video and VCD2 deliveries. The new photo deliveries ran the current appearance model and window calculation on a single observed track with the shared dense cadence. That comparison omits production tracker and fallback behavior on the photo cases.

## Results

| Input | Whole-clip paired flag | Short-window localization | Current appearance overlap or flag |
| --- | ---: | ---: | ---: |
| Exposed publisher-video swap deliveries, one source | 0/4 | 0/4 | 4/4 |
| Separate photo-source 0.750-second swap | 0/1 | 1/1 | 1/1 |
| Separate photo-source 1.500-second swap | 0/1 | 0/1 | 1/1 |
| Synthetic original and ordinary-edit controls | 0/5 | 0/5 | 2/5 |
| Genuine camera original and ordinary-edit controls | 0/6 | 0/6 | 0/6 |

Both appearance control flags came from the same exposed synthetic video source, later in the clip than the known 3–6 second intervention. The two genuine camera groups had no flags in any method. All 17 clips were assessed by all three methods.

The short-window rule localized the 0.750-second photo swap to corroborating samples from **2.500 to 3.000 seconds**. Its exact before, during and after pictures were at **2.250, 2.500 and 3.250 seconds**. The known swapped pictures begin at 2.500 seconds and stop before 3.250 seconds. At 1.500 seconds of swapped material, the complement remained elevated through the edit and the narrow flanks prevented a flag. The exposed publisher video had an isolated strong response at 5.750 seconds inside the insert and did not meet persistence. The appearance check covered both photo insert durations and all four exposed video renderings. On the photo cases, its last supported sample was the first restored frame.

Both Metal workers finished locally at 72.184 and 69.218 seconds, with peak sampled process-tree RSS of 1.33 and 1.54 GiB. They each made 250 model forwards over 609 matched score positions, including 579 dense positions and 30 additional positions for the new six-frame comparison. Independent replay recomputed sampling, matched frame clocks and geometry, window selection and whole-clip values. Twenty-one private PNGs re-decoded to the expected native RGB, covering the six known edit triplets and the detected triplet. [Aggregate and replay commitments](evaluation-panels/video-face-short-window-2026-11-11.json)

The short-window check remains experimental. It found a narrow edit the shipped whole-clip check missed, while the current appearance check found that edit and the five other positive deliveries. The two positive source components came from one publisher. A larger comparison needs distinct generators and moving footage with known boundaries. The tested photo construction cannot establish natural-motion sensitivity.

The first manifest download returned HTTP 403. A subsequent full PNG transfer timed out after receiving 45,067 bytes. The public server advertised byte ranges, and the bounded range retrieval assembled both exact publisher hashes before the freeze. Both scored arms and the independent readout passed.
