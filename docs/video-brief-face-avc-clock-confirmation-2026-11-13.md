# Four-second H.264 face localization across source frame rates

This comparison asks whether the existing bounded face proposal adds useful timestamp ranges to Full analysis on short H.264 MP4 deliveries. The earlier comparisons added six exact locations across three distinct source groups and added no flags on 70 ordinary controls. Each run missed its own criterion for gains across at least three source groups. The newest bank keeps the same face proposal and its fixed operating point. It changes the source captures, the edit positions and their original frame rates. [First carrier comparison](video-brief-face-120-carrier-2026-11-13.md) · [Earlier H.264 confirmation](video-brief-face-avc-confirmation-2026-11-13.md)

Eight source groups contribute two recordings each at 25, 29.97, 30 and 50 fps. The roster excludes known capture components, original links and donor keys present in the previous four related comparisons. The broader project previously used some of the archived recordings. Retime selection maps each delivery picture to the nearest verified source timestamp without blending images. All deliveries use the same exact 120-picture 1280 × 720, 30 fps output clock. The test uses six contiguous output pictures from two different retained publisher-swap families, beginning between frames 62 and 111. Each capture has an unchanged file, a re-encoded unchanged file, a known swap, a re-encoded swap, a local exposure change, a same-recording jump and a cut to another genuine capture. The initial H.264 encodes use CRF 23 and the re-encodes use CRF 35. No new model or learned weights were added. [Source permissions and attribution limits](video-common-source-preparation-v89.md) · [Face assets and commercial licenses](video-appearance-plan-v107-method.md)

A candidate only runs on a delivery without an existing supplemental short range. A useful range overlaps at least half of its temporal union with the known edit. The comparison kept the existing criteria of four added useful ranges across three groups, including at least two exact, at most one new ordinary flag, no new cut or jump flags, at most one wrong range on edited clips and measurements on at least 75% of routed clips. The offline proposal reads the exact source pictures without the serving worker's optional 12-second budget. Paired compression variants share a capture and are counted separately only at the file level.

## Held results

| Outcome | Existing Full | Full with fixed proposal |
| --- | ---: | ---: |
| Useful locations on initial encodes | 5/8 | 5/8 |
| Useful locations after tandem encoding | 4/8 | 5/8 |
| Ordinary uploads flagged | 0/40 | 0/40 |
| Routed clips measured by the bounded proposal | — | 47/47 |

The one added range exactly matched the authored **2.767–2.967 s** change after tandem compression of a 29.97 fps source, from frame 83 through frame 88. The before, during and after images correspond to frames **82, 86 and 89**. There were no misplaced additional ranges. This comparison improves localization by one delivery. The performance on everyday footage remains uncertain beyond these 40 controls, and the default stays off.

The independent checks reopen all 6,720 presentation times and native RGB frames and confirm the three selected original frames. The public receipt preserves source and output commitments, conditions and decisions. It contains no media, source identities or model files. [Case and native timing ledger](evaluation-panels/video-brief-face-avc-clock-confirmation-2026-11-13.json)

## Replay

Recount the ordinary and edited results, frame-rate strata and all published
timestamp commitments using the standard library.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_video_brief_face_avc_confirmation \
  docs/evaluation-panels/video-brief-face-avc-clock-confirmation-2026-11-13.json
```

The signed evidence for the separate WebM application smoke tested the same five-to-seven-picture method using a known range at 2.067–2.267 s. The customer view and report now display the measured appearance change and counts of nearby regions with substantial motion for each endpoint. A simulated paid checkout yielded three matching source frames and crops across Chrome, PDF, JSON and signed original replay. Those measurements carry zero weight in the AI assessment. This WebM evidence improvement is already available through the existing optional short carrier setting. The separate H.264 comparison leaves lip synchronization and generalization to other deliveries unmeasured.

## Local H.264 trial

To try the check locally, use `VERITAS_VIDEO_FACE_REVIEW_MODE=appearance` and `VERITAS_VIDEO_FACE_SHORT_AVC_REVIEW=true` and restart the worker. It runs on Full and Deep analyses of upright 1280 × 720, 8-bit 4:2:0 `avc1` MP4 with 120 verified pictures at exactly 30 fps. The check looks for a five- to seven-picture appearance change that returns toward its earlier appearance. It can supply a distinct range when the current review has no supplemental short range. The ordinary view, PDF and JSON show both boundary measurements and the before, during and after pictures and face crops. The trial carries zero overall AI confidence weight.

The real local worker ran with the flag both off and on using two encodes of one previously evaluated edit, an unchanged clip and a genuine picture-cut control. Both encodes added the same **2.067–2.267 s** range, with source pictures **61, 65 and 68**. Neither control added a face range. All four pairs kept their overall assessment. Chrome loaded both boundary measurements and all six images. The API, JSON and Standard PDF agreed on the six exact exhibits per positive, and signed replay verified the range and both boundary measurements from the submitted video. The face check took **1.25–1.72 s** per clip against a 12-second limit. The standalone picture-return check still found the genuine cut in both modes. This test had no clip with a simultaneous positive face and picture-return result. [Local trial receipt](evaluation-panels/video-brief-face-avc-app-trial-2026-11-13.json)

Use the retained four-second source bank with `media.json` and `media/`. The runner uses an isolated database, verifies the source hashes against the earlier carrier study and simulates payment before running each analysis.

```sh
PYTHONPATH=backend backend/.venv/bin/python -m scripts.video_brief_face_avc_app "$retained_four_second_bank" "$new_private_run"
```

To recreate the H.264 experiment, supply a new local run directory and the archived freezes for the prior eight-second HEVC, eight-second AVC/VP9 and four-second AVC comparisons. The authoring step checks each decoded source timestamp and RGB commitment before writing the exact 30 fps delivery. The verification step checks each delivered picture and timestamp through an independent FFmpeg process.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for phase in freeze prepare score verify; do
  backend/.venv/bin/python -B -m scripts.video_brief_face_avc_clock_confirmation "$phase" "$run" \
    --prior "$hevc_run" "$carrier_run" "$avc_run" --jobs 2 || exit 1
done
```
