# Fine-detail correspondence across mouth contours

The experiment measured every held clip. Neither method localized any of the ten neural redraws with an independently verified source-aligned mouth event. The fine-detail method flagged 14 of 93 ordinary clips. Adjusting for coarse detail reduced that to four, with zero correct redraw locations. The check stays experimental. [Replayed decisions and native evidence audit](evaluation-panels/video-mouth-contour-phase-2026-11-12.json)

## Method and held sources

Earlier tests measured radial texture energy, temporal agreement after optical flow, or fine detail at fixed native picture positions. This test filters decoded luminance at two native spatial bands. An eye-based similarity transform and observed lip corners position four rails across a family of mouth contours. It compares the signed phase of tangential gradients on opposite sides of each contour sector against two offset pairings and a nearby reference. The coarse-adjusted arm subtracts the simultaneous positive coarse-band change. This tests whether neighboring detail stops aligning across a contour during one short edit. It supplies no camera identification. [Face X-Ray boundary study](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Face_X-Ray_for_More_General_Face_Forgery_Detection_CVPR_2020_paper.html) · [Video residual research](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Media%20Forensics/Verdoliva_Extracting_camera-based_fingerprints_for_video_forensics_CVPRW_2019_paper.pdf) · [Earlier radial comparison](video-mouth-rim-transfer-2026-11-11.md) · [Earlier fixed-coordinate comparison](video-mouth-fixed-residual-2026-11-12.md)

The detector scans 8-, 12-, 16- and 20-picture intervals inside one observed shot, with five-picture flanks. Both flanks must show consistent fine detail in at least three sectors. The proposal needs a positive loss in at least three of those same sectors. The submitted clip alone supplies the pixels and clocks. Authored edit boundaries and the earlier copy enter evaluation only. A correct range needs at least 0.50 temporal intersection over union. The third-largest ordinary development maximum set each internal point before held scoring, using 56 genuine and conventionally edited views from eight previously measured GRID sources in four source components. Only the existing commercially admitted YuNet geometry weights were used. [GRID data and terms](https://zenodo.org/records/3625687) · [YuNet licence](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

The 104 held clips include 11 retained neural redraws and 33 paired ordinary copies from six Lombard speaker groups. Ten redraws kept one independently measured release aligned to original audio. This qualification establishes the time of a local speech event. The study did not independently label every phoneme or test perceived synchrony over the full clip. There are 48 additional genuine recordings from 12 other speaker groups and 12 earlier GRID delivery controls. The 48 genuine originals were fresh for this method, and the neural positives were reused, all from one renderer. [Lombard release and terms](https://spandh.dcs.shef.ac.uk/avlombard/) · [Earlier synchronized redraw audit](lombard-neural-mouth-redraw-2026-11-12.md) · [Exact delivery controls](video-mouth-jaw-transfer-2026-11-11.md)

## Held results

| Outcome | Fine detail | Fine detail adjusted by coarse band |
| --- | ---: | ---: |
| Measured clips | 104/104 | 104/104 |
| Event-aligned redraws localized | 0/10 | 0/10 |
| Event-aligned redraws with correct overlap before applying the point | 2/10 | 3/10 |
| Misplaced flagged redraw ranges | 1/11 | 0/11 |
| Untouched originals flagged | 9/59 | 3/59 |
| Ordinary resamples and retimes flagged | 2/22 | 0/22 |
| H.264 recompressions flagged | 3/8 | 1/8 |
| Whole-picture cut controls flagged | 0/4 | 0/4 |

The proposed range with the best fine-detail overlap was **0.667–1.335 s**, compared with the authored **0.709–1.376 s** interval. Its overlap was 15/17, but the score remained below the frozen point. The raw scores had AUC **0.424** and **0.438** for ranking event-aligned redraws over ordinary clips. A smaller point alone would not resolve the observed ordering. The frozen worker verified 6,597 native presentation timestamps and observed face geometry in every held picture. A separate process replayed decision arithmetic and reopened 71 exact frame/crop exhibits for 25 selected ranges across 22 files.

## What remains untested

None of the new positives needed for generator-independent validation were admitted. DeepSpeak has a separate commercial grant. The published BabelFake, LipSync-A, and MAVOS-DD terms restrict commercial use. TalkingHeadBench has mixed upstream source terms. DigiFakeAV offers examples of generated talking heads but does not supply this test’s required matched local boundaries and independently verified speech event. The next test should compare short audio-conditioned, source-timed redraws from a second generator with same-geometry ordinary processing. It should measure whether a cue separates a neural redraw from conventional local processing after both pass through the same export, with new speaker groups and independently confirmed audio and native picture intervals. [DeepSpeak terms](https://www.ischool.berkeley.edu/research/publications/2025/deepspeak-dataset) · [BabelFake release conditions](https://arxiv.org/html/2610.06339v1#S7) · [LipSync-A terms](https://huggingface.co/datasets/AnsonShe/LipSync-A) · [MAVOS-DD terms](https://huggingface.co/datasets/unibuc-cs/MAVOS-DD) · [TalkingHeadBench rights review](video-transfer-research-and-rights-20261110.md) · [DigiFakeAV method](https://hubeiwuhanliu.github.io/DigiFakeAV.github.io/)

## Local replay

Use the retained private Lombard redraw folder as `lombard`, the independent short timing intake folder as `fresh`, and an unused private `run` directory. Source clips, crops and detailed records stay local.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze development held; do
  backend/.venv/bin/python -B -m scripts.video_mouth_contour_phase "$phase" "$run" "$lombard" "$fresh" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_mouth_contour_phase "$receipt" --assemble "$run"
python3 -B -S -m scripts.review_video_mouth_contour_phase docs/evaluation-panels/video-mouth-contour-phase-2026-11-12.json
```
