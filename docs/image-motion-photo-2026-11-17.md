# Compare a Motion Photo’s still image and embedded video

Some JPEG Motion Photos retain a short video with a declared presentation time for the still. Chipmunk Verify can compare the still with the frame at that time and nearby frames from the same file. A surviving difference shows the marked area, an unmarked still crop, the corresponding video crop and a mask. The browser, PDF and JSON give the same pixel coordinates and frame times. This check contributes no weight to the AI assessment.

A comparison requires a complete JPEG Motion Photo 1.0 directory, a supported complete MP4 and a measured presentation timestamp within 6.67 ms of the declaration. The JPEG may include a complete linked HDR gain-map image. Some capture files also contain a short camera data block before the video. A standalone JPEG, an unset time or a file with the video removed cannot supply this comparison. The timing declaration establishes which frame to compare. A signed provenance check runs first and may stop the analysis. [Android Motion Photo format](https://developer.android.com/media/platform/motion-photo-format)

The check aligns visible details on three successive native frame timestamps. It compares local structure at two scales, discounts ordinary changes between the video frames and requires broad agreement across the shared area. It reports the largest remaining supported region. The still crop uses the original oriented pixels. The corresponding video crop and difference mask use the reduced comparison grids declared in JSON. The frame may use adjacent support on one side when the declared time is at the start or end. Camera processing, movement, and ordinary editing can explain a difference.

## Held camera-file comparison

The separate held intake contained 18 source files from 18 Wikimedia contributors with public licences. Twelve contained both a recognized JPEG layout and a complete embedded video. For each of those twelve, the comparison used the original, a JPEG re-encode and a whole-image tone change as ordinary controls. Two other deliveries per source replaced different fixed regions in the still while preserving the camera’s video and the declared time. Each replacement covered 14% of the image width and height and came from nearby content. These staged edits exercise local difference detection.

| Case | Deliveries | Assessed | Result |
|---|---:|---:|---|
| Staged still edits | 24 | 22 | 21 correct locations, 1 misplaced highlight, 2 unassessed |
| Original, re-encoded and tone-adjusted copies | 36 | 31 | 0 highlights, 5 unassessed |
| Source files with an unsupported or missing attachment | 6 | 0 | No comparison |

A correct location requires at least 0.10 intersection over union between the reported mask and the staged region. The comparison point was selected on a separate development panel. Support for the linked gain-map layout was added for files that the first held pass could not parse, with the same comparison point. Deliveries from a source share its camera image and motion. Counts describe these sources and edits. They do not estimate performance on general JPEG editing or on AI object removal.

[Pinned source list](evaluation-panels/image-motion-photo-sources-2026-11-17.json) ·
[Held measurements](evaluation-panels/image-motion-photo-held-2026-11-17.json.gz)

## Local app use and replay

Enable `VERITAS_IMAGE_MOTION_PHOTO_REVIEW=true` on the development backend. Upload a supported JPEG Motion Photo. Open **Motion Photo still and video** in the ordinary image result. The export uses `motion_photo_comparison`. The signed evidence verifier can replay the times, mask and crops from the submitted JPEG. It uses FFmpeg and OpenCV and requires no additional pretrained weights.

The local app replay used simulated $9.99 checkout payments. No job was created before payment. The edited sample showed the four expected evidence images at 1.169489 seconds in the browser, standard PDF and summary PDF. Both packages reproduced the images and timing from the uploaded JPEG. The camera original had no reported region. The edited sample had the same overall decision with this comparison enabled or disabled. [Local app receipt](evaluation-panels/image-motion-photo-app-2026-11-17.json)

Download each original from the pinned list into one folder. Keep its recorded filename and check its SHA-256. Run the comparison from the repository root.

```bash
PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  backend/.venv/bin/python -B -m scripts.image_motion_photo_panel \
  docs/evaluation-panels/image-motion-photo-sources-2026-11-17.json \
  /path/to/source-jpegs /path/to/held-result.json
```

The runner checks every source size and digest before preparing its controls. The receipt includes each delivery’s hash, measured frame times, source regions, and mask overlap. Originals and derived images stay on the machine running the replay.
