# Short face-swap localization with local texture

The current face appearance check localized all 24 short swap views in the moving-source holdout and flagged none of the 42 matched genuine and ordinary-edit controls. A new local texture check localized zero swaps at its frozen point. The camera and separate publisher diagnostic also produced zero texture flags. The texture check stays experimental. [Counts and run commitments](evaluation-panels/video-face-texture-return-2026-11-11.json)

## Comparison

Four genuine moving source groups supplied 28 fitting views. Six separate moving source groups supplied 24 test swaps and 42 genuine controls. Each group uses a six-second HD excerpt with an exact native clock, an original and ordinary lighting, blur and time-step interventions. The held groups also have two publisher-provenance swap families. Each intervention uses a nominal 0.75- or 1.50-second span centered at 3 seconds. The half-open boundaries use the first native presentation time at or after each scheduled edge. The runner selects the exact altered pictures inside the interval and the exact original pictures outside it. Existing independent command receipts bind the RGB inputs. Sources were selected by the smallest total archive bytes in disjoint prior roles. The archived data publisher declares CC BY-SA 4.0. Portrait attribution and independent pretraining separation remain incomplete. [Source permission review](video-common-source-preparation-v89.md) · [Native construction and verification](video-native-inserts-v95-method.md)

The new check measures luminance texture in three face regions and three adjacent scene regions at 8 Hz. It compares the ratio of high- to mid-frequency energy through short intervals. A candidate range needs a contiguous face track, four measured samples on both flanks and a consistent shift in at least two face regions after subtracting common surrounding change. Immediate agreement at both edges can contribute up to one quarter of the response. The fixed comparison point was the largest measured genuine fitting response, from an ordinary 0.75-second blur. Fourteen of the 28 fitting views had enough texture coverage. The current comparator used its unchanged window calculation and point at 4 Hz with neighboring native frames on a single observed face. It represents the appearance method in isolation from the full app tracker and fallback. Both weights have commercial licenses. Recent work on explicit facial texture features and temporal frequencies motivated the independently implemented cue. [October 2026 temporal features study](https://arxiv.org/html/2610.03380v1) · [Temporal frequency study](https://openaccess.thecvf.com/content/ICCV2025/papers/Kim_Beyond_Spatial_Frequency_Pixel-wise_Temporal_Frequency-based_Deepfake_Video_Detection_ICCV_2025_paper.pdf) · [Face detection weights](https://huggingface.co/opencv/face_detection_yunet) · [Face appearance weights](https://huggingface.co/opencv/face_recognition_sface)

| Source panel and result | Current appearance | Local texture |
| --- | ---: | ---: |
| Moving holdout, 0.75-second swaps localized | 12/12 | 0/12 |
| Moving holdout, 1.50-second swaps localized | 12/12 | 0/12 |
| Moving holdout, genuine and ordinary-edit false flags | 0/42 | 0/42 |
| Moving holdout, unassessed swaps | 0/24 | 11/24 |
| Moving holdout, unassessed controls | 0/42 | 21/42 |
| Two separate camera groups, ordinary delivery false flags | 5/10 | 0/10 |
| One separate publisher source, short swaps localized | 4/8 | 0/8 |
| Same publisher source, short swaps flagged at the wrong range | 4/8 | 0/8 |
| Same publisher source, original and recompression flags | 3/3 | 0/3 |

Successful localization required at least 0.20 intersection-over-union with the known interval. The current check reached 0.50 on all 24 moving views and 0.75 on 14. Median absolute onset and end errors were 0.12 and 0.11 seconds for the 0.75-second edits. They were approximately 0.037 and 0.033 seconds for the 1.50-second edits. The two swap variants at each duration and the ordinary controls from one source share content. No texture finding expanded coverage.

The separate diagnostic reused the exact same frozen points. It included two genuine Microsoft VCD2 camera recordings with unchanged, color-converted and recompressed deliveries. Five appearance false flags came from one of the two camera groups. The FaceSwapAI diagnostic used two exact short spans cut from one previously tested swap across four color interpretations and its ordinary recompression. Its base video uses synthetic still states and crossfades. All 21 external views were assessed. The first external plan placed the camera center outside its documented intervention. It was retained and corrected before any external inference. [Camera dataset license](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA) · [Camera input construction](video-vcd2-delivery-v206-method.md) · [Publisher input construction](video-faceswapai-matched-v186-method.md)

The CPU runs verified 2,631 native RGB inputs against their exact timestamps and archived receipts. The fit, held and external stages took 77.5, 142.7 and 38.3 seconds. Maximum worker RSS was about 335 MB. Independent readout checked source separation, case counts, exact interval endpoints, anchor order and both frozen points. The public aggregate contains no media, individual fingerprints or identifying paths.

## Local replay

Run from the repository root with the existing exact private archives mounted and a new output directory.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_face_texture_return freeze "$run_dir" --media "$archive/video-native-inserts-v95"
backend/.venv/bin/python -B -m scripts.video_face_texture_return fit "$run_dir" --media "$archive/video-native-inserts-v95"
backend/.venv/bin/python -B -m scripts.video_face_texture_return held "$run_dir" --media "$archive/video-native-inserts-v95"
backend/.venv/bin/python -B -m scripts.video_face_texture_return readout "$run_dir"
backend/.venv/bin/python -B -m scripts.video_face_texture_external freeze "$run_dir" "$archive"
backend/.venv/bin/python -B -m scripts.video_face_texture_external run "$run_dir" "$archive"
```
