# Brief face changes in 720p30 H.264 delivery

The current Full face-appearance review measures longer windows at 720p30. Its supplemental brief-change check admits exact 720p25 and 1080p30, leaving 720p30 outside that check. This experiment tests the unchanged 8 Hz measurement at 720p30 and a second arm with a 15 Hz fallback guarded by background continuity. Both preserve the current tracked face range. Neither contributes to AI confidence. [Earlier resolution transfer](video-short-return-resolution-transfer-2026-11-11.md) · [Existing background guard](video-face-codec-background-transfer-2026-11-12.md)

## Frozen comparison

The 31 deliveries use exact 1280 × 720 H.264 MP4, 8-bit 4:2:0 and 120 verified pictures at 30 fps. Four held RTFS source groups remain after the previous 77 short-face and scene study exclusions and the subsequent six-source codec study. Original links and documented donor keys separate these groups from that earlier roster. They appeared in other project work, and unknown participant or pretraining overlap remains possible. The 1080p inputs are resized to 720p. Three have an exact 30 fps source clock. One 59.94 fps source uses nearest native pictures with an earlier tie break and no interpolation. [Retained face-swap source terms](video-common-source-preparation-v89.md) · [VCD2 data terms](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA)

Each paired group supplies one six-picture and one ten-picture publisher swap, unchanged video, tandem recompression, local brightness, a same-source jump cut and a cross-recording whole-picture cut. The edited intervals are **2.400–2.600 s** and **2.900–3.233 s**. Local controls retain the original picture assignments outside their edits. The tandem control recompresses the full clip. Three extra recordings come from different retained VCD2 camera sources. The test uses the installed face detection, tracking and appearance assets under their existing license terms. [Appearance assets and terms](video-appearance-plan-v107-method.md)

The comparator runs the actual Full probe, decode, shot, track, face and detect stages with the serving face allowlist. The supplemental arms call the same native measurement code and retain the fixed 0.10 separation point, half-second duration limit, exact timestamps and distinct-range rule. The denser arm runs every other native picture only if the sparse pass measured a candidate and had no distinct range. It applies the existing 0.60 lower-quartile background requirement at both boundaries. The two passes share the existing 12-second short-face budget. A correct range needs at least half temporal intersection over union with the authored interval. The gate requires three newly localized swaps in at least two groups, at least 80% measured coverage for both swaps and ordinary controls, at most one newly flagged ordinary file and zero newly flagged whole-picture cuts.

## Held results

| Outcome | Current Full face check | With 8 Hz | With guarded 15 Hz fallback |
| --- | ---: | ---: | ---: |
| Useful six-picture locations | 0/4 | 0/4 | 3/4 |
| Useful ten-picture locations | 0/4 | 0/4 | 0/4 |
| Ordinary alerts | 0/19 | 0/19 | 0/19 |
| Whole-picture cut alerts | 0/4 | 0/4 | 0/4 |

All three new locations match **2.400–2.600 s** exactly and come from distinct held groups. Each packet uses native pictures **70, 74 and 78** as its before, during and after anchors. The existing ten-picture ranges had 0.444 IoU with their authored intervals. The new pass left those ranges in place. The complete held set had eight abstentions from the existing multiple-face guard. One source accounted for two swaps and four ordinary abstentions. Two cross-recording cuts also contained multiple detected faces. All other cases completed both passes. The independent reviewer checked all 3,720 native frame hashes and timestamps, 45 selected sample bindings and nine aligned anchor crops. [Public case ledger](evaluation-panels/video-face-720p30-transfer-2026-11-12.json)

The original advancement gate failed on coverage. The new passes assessed **6/8 swaps** and **15/19 ordinary controls**, below the prespecified 80% for each. After reviewing that shortfall, the project approved a narrower, optional release for single-face evidence on the exact tested profile. This decision prioritizes the three additional exact locations and the unchanged observed control flag rate. The sample leaves same-background subject switches, other encoding profiles, fractional frame rates, longer clips and lip-sync performance unmeasured.

## App scope

When brief face-change review is enabled, Full and Deep can use the fallback on the tested upright 1280 × 720 H.264 MP4 profile with 120 native frames, an exact 30 fps clock and 8-bit 4:2:0. The 8 Hz pass supplies context and the 15 Hz pass supplies any new published range. Every new range needs the stronger background guard. A multiple-face input stays unassessed by this fallback. Existing face ranges and overall confidence are preserved. The plain view, JSON and Standard PDF show the same half-open source interval and original before, during and after crops.

The local Full app recovered the held **2.400–2.600 s** range under simulated checkout, with the three exact source images and crops in its API and PDF. The signed export replayed all source images and both guarded background boundaries from the upload. The previous VP9 and HEVC signed packages also replayed with the updated verifier. The ordinary original and cut showed no added range, and the multiple-face swap recorded an unassessed result. Chrome rendered all three full images and all three crops and navigated to their source positions.

## Reproduce

The arithmetic check uses the public case rows. Rebuilding and replaying the native clips also requires the retained local archive and the existing camera, face-swap and model qualification receipts. Source identities, pictures and detailed native bindings stay local.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_face_720p30_transfer prepare "$run"
backend/.venv/bin/python -B -m scripts.video_face_720p30_transfer score "$run"
backend/.venv/bin/python -B -m scripts.review_video_face_720p30_transfer "$run/public.json" --study "$run"
```
