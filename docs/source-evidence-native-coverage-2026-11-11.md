# Local validation of expanded source evidence

## Images

The app now searches for a matching same-size patch on images up to 1,024 pixels per side. The highlighted region and a matching crop retain their original image coordinates. Large searches use textured blocks to shortlist candidate locations, then verify the matches at native resolution. A search that exceeds the candidate limit reports inconclusive coverage.

The native-size replay used the eight existing v298 photos at their recorded 800–1,024 pixel long sides. Each source supplied an unchanged image, a tone change, a blur, a translation copy and a reflected copy, each in lossless and JPEG90 delivery. The known edited region was supplied as the search target. The app recovered the exact location and transformation in 24 of 32 copy views. The eight remaining views had insufficient texture. One blur control produced a copy-compatible match in both formats, and one unchanged control produced ambiguous matches in both formats. These observations measure matching within a supplied region. They provide no estimate of how often the app initially locates an unknown edit. [Original source panel and test method](image-copy-transfer-v298-method.md)

A separate local test fixed four previously unused licensed photo groups and their comparison regions before opening the source files. All four photos measured 1,024 pixels on the long side. The product recovered the correct location and transformation in all eight known copy edits, across all 16 lossless and JPEG90 views. No matching patch was reported in the 24 unchanged, tone and blur views. Two of those views, both from the same blur control, were unassessed because they reached the search limit. The two formats agreed on all 20 paired outcomes. [Exact aggregate](evaluation-panels/image-copy-native-v6-fresh-pilot.json)

## Audio

The standalone stereo check now compares the louder source channel with the actual mono signal supplied for analysis. One-second windows in the first 30 seconds flag a drop of at least 12 dB when the source channel has enough level for review. Known opposing-channel controls produced partial or full cancellation. Matching channels, quadrature and quiet controls produced no strong warning.

The production decoder then rechecked all 24 qualified windows from two genuine AMI meetings. The mono samples and source clocks matched all 24 prior commitments. None of 72 one-second windows triggered a strong-loss warning. The largest measured loss was 4.055 dB, and 24 windows were below the review level. The cue describes a potential loss of audible content in the analysis input. [AMI intake and coverage](audio-ami-native-stereo-v291-result.md)

## Documents

The PDF extractor applies local OCR to one dominant image area when that area is separate from native text and passes recognition checks. Readable OCR paragraphs retain word geometry and the exact frozen page raster. A rotated test PDF preserved the native caption, recovered three scanned sentences with one character error, and kept the recognized words inside the source image rectangle. The rendered PDF and signed evidence ZIP carried the source-bound excerpts. Unread areas remain marked as partial coverage. Mixed extraction is outside the validated authorship-confidence scope.

## Video

The low-memory candidate ran offline through the production detector boundary and model loader. It used the same pinned weights, CPU device and 4,608 MiB model budget as the reference. The checks verified native pixels and timestamps against existing input receipts. The full app comparison remains separate.

| Input | Native frames scored by candidate | Raw scores at or above 0.5 |
| --- | ---: | ---: |
| Two unchanged camera recordings | 8 | 0 |
| Matched unswapped control, using a synthetic base clip | 6 | 2 |
| Known swap, sampled inside the changed range | 4 | 0 |
| Known swap, sampled outside the changed range | 2 | 1 |

The reference skipped all four inputs for its activation allowance. The candidate linked all 20 scores to their native timestamps. The guarded worker completed in 195.6 seconds and peaked at 1.38 GiB of sampled process-tree memory. The raw score comparison diagnoses one known swap. It establishes neither calibrated confidence nor a broader error rate. The candidate remains on its test branch while more sensitive face-swap checks are evaluated. [Candidate source](https://github.com/ChipmunkRPA/veritas/commit/b45eb88944bb997c0e3009d416e221eaab1170bb) · [Camera control method](video-vcd2-matched-controls-v290-method.md) · [Matched swap and edit boundaries](video-faceswapai-full-app-v190-result.md)
