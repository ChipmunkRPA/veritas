# Picture-cut context for coarse face changes

An ordinary cut between shots can cause a face-appearance change. This check asks whether stable picture detail outside the tracked face also changes at both ends of the same coarse review range. When it finds support, Chipmunk Verify shows the measured source interval and three full frames, one before, one during, and one after. The report describes a picture cut as one plausible explanation for that face change. The original face evidence and the overall result and confidence are preserved.

## Method and comparison

On upright 1280×720, 30 fps, 120-picture HEVC `hvc1` MP4 deliveries, the check uses the median of two or three observed face boxes to exclude the head area. It reduces the measured pictures to 320×180 and compares 12-pixel textured tiles around the face using normalized local correlation. The match allows up to three pixels of displacement at the reduced size. Each boundary response is adjusted by neighboring motion, and the same tiles must be stable between the outer pictures. A supported pair needs at least five shared changed tiles across two sectors. The check looks within 12 native pictures of each existing coarse boundary, for a change lasting 5–24 pictures with at least half the shorter range overlapping. The fixed response point of 0.30 is a texture comparison value.

The development set contains 61 retained CREMA-D and RTFS deliveries. The held set contains 56 separate GRID and RTFS deliveries. It uses known edit boundaries, original recordings, and both direct and tandem encodes. The genuine speaker cuts are controlled composites of genuine recordings in the same layout. All comparisons use ranges supplied by the existing Full face checks. The development run selected 0.30 from 0.10, 0.20, and 0.30 by requiring zero adverse calls, then maximizing correct cuts and using the higher value to break a tie. It explained all 10 routed development cuts at their exact boundaries across five actor pairs and gave no cut explanation to the four routed edited clips. Commercial-evaluation source terms and limitations are recorded separately. The picture calculation uses no added weights. [Source terms](video-common-source-preparation-v89.md) · [GRID and HEVC source controls](video-face-unrouted-short-transfer-2026-11-12.md) · [Frozen inputs and exact results](evaluation-panels/video-spatial-cut-context-transfer-2026-11-12.json)

| Held category | Files | Existing coarse range | New picture-cut explanation |
| --- | ---: | ---: | ---: |
| Genuine same-layout speaker cuts | 16 | 15 | 13 exact |
| Face-edited deliveries | 16 | 8 | 0 |
| Other ordinary deliveries | 24 | 0 | 0 |

The exact picture-cut boundaries were found in seven of eight genuine source pairs. They occurred at 2.100–2.467 or 2.833–3.200 seconds. An independent readback re-decoded 65 representative pictures and verified their native timestamps and RGB hashes. The 24 other ordinary files did not supply a coarse range, so they did not reach the picture comparison. These are scope-specific results for retained, controlled clips.

A supplemental check kept the same response point and used five additional CREMA-D actor pairs and one separate RTFS recording pair. The unchanged method matched both boundaries on all five CREMA-D cutaways and both directions of the RTFS cut. None of the five ordinary exposure changes received a coarse range, so these controls confirm the app skipped the picture check on them. The two RTFS files were converted from 25 to 30 fps by frame duplication and tested on the resulting HEVC timeline. [Supplemental control receipt](evaluation-panels/video-spatial-cut-context-supplement-2026-11-12.json)

## Local application

Set `VERITAS_VIDEO_FACE_SPATIAL_CUT_CONTEXT=true` to enable the check locally. It defaults to off. It requires a supported delivery, a coarse face-appearance range, adequate face and texture measurements, and available time. The existing brief face-return result takes priority. A clip with a separate brief swap and a coarse cut retains its earlier evidence. A picture-cut finding only addresses its linked range. Review other visual and audio evidence separately.

The paid-flow simulation verified the browser, PDF, API, and JSON on the same held cut. All displayed the range beginning at frame 63 and ending before frame 74, with exact source pictures 62, 68, and 74. The browser sought to frame 68, and the signed verifier remeasured both boundaries from the supplied original and rejected a changed original. A held six-picture face swap retained its existing evidence. The ordinary original received no picture-cut finding, and disabling the flag returned the same coarse finding and overall decision as before. Historical v20 and v21 video packages and the r110 audio package replayed against their original sources. [Paid-flow and browser receipt](evaluation-panels/video-spatial-cut-context-app-2026-11-12.json)
