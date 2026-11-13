# Color residuals in short synchronized mouth redraws

The two color checks and the luminance comparator each localized **0 of 10** neural redraws that retained an independently measured speech event. The color checks flagged **0 of 117** ordinary videos and the luminance comparator flagged one untouched original. The measured score ordering favored ordinary videos over the neural redraws. The checks remain experimental. [Held decisions and native replay](evaluation-panels/video-mouth-chroma-coupling-2026-11-13.json)

## Frozen comparison

Color-space correlation has separated some generated and genuine still images in prior work. This experiment asks a narrower question. Does a native mouth region change its fine luminance–chroma relationship relative to nearby skin, and does that local contrast persist against the pictures on both sides? The implementation uses no external detector code or weights beyond the existing rights-reviewed face geometry. It measures decoded RGB and does not establish the camera's sensor pattern or the source of an individual color channel. [Inter-channel color correlation research](https://arxiv.org/abs/2606.08864) · [Local color residual research](https://arxiv.org/abs/2608.20558) · [YuNet licence](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

Two mouth-centered ellipses supply eight pairs of native neighborhoods. Both need comparable luminance texture. The arms compare local luminance residual energy, how much chroma variation luminance explains, and how much chroma residual remains after accounting for luminance. Each searches 8-, 12-, 16- and 20-picture intervals with five-picture flanks in one observed shot. At least three sectors must change consistently against both flanks. The third-largest ordinary development maximum fixes each internal operating point before opening held results. A useful proposal overlaps at least half of the exact authored native interval by temporal intersection over union. Audio and earlier copies do not enter the detector.

The 56 ordinary development files come from eight GRID recordings in four speaker components. The 128 held files comprise 11 neural redraws and 33 paired originals/resamples/retimes, 48 genuine recordings from 12 additional Lombard speaker groups, 12 earlier recompression and cut controls, and 24 newly constructed controls from those additional groups. These last controls are 12 whole-video H.264 transcodes and 12 fixed mild local chroma denoise operations. The positive files use one LivePortrait workflow and retain their earlier exposure. Ten of those redraws had an independently observed local source-time speech event. [Source and edit verification](lombard-neural-mouth-redraw-2026-11-12.md) · [GRID CC BY 4.0 release](https://zenodo.org/records/3625687) · [Lombard CC BY 4.0 release](https://zenodo.org/records/3736465)

## Held results

| Arm | Correct aligned locations | Ordinary flags | Aligned edits with sufficient raw overlap before the operating point | ROC AUC among measured clips |
| --- | ---: | ---: | ---: | ---: |
| Luminance control | 0/10 | 1/117 | 2/10 | 0.441 |
| Luminance–chroma coupling | 0/10 | 0/117 | 3/10 | 0.376 |
| Residual chroma after accounting for luminance | 0/10 | 0/117 | 3/10 | 0.379 |

All three arms measured 9/10 event-aligned redraws, 30/33 paired ordinary files, 36/48 additional originals, 21/24 new ordinary derivatives, and all 12 earlier codec controls. The strongest useful residual-chroma proposal spans **0.290–0.952 s** against the authored **0.331–0.993 s**. Its score was 0.177, below the frozen point of 0.256. The replay matched **8,011 held native picture timestamps** during scoring, reopened **29 native frame/crop commitments** across 10 selected ranges, checked the 1,414 timestamps of the new derivative controls and independently reconstructed every pixel in the 707 denoised pictures. Exactly 192 of those pictures changed, all in the intended intervals. The low held score ranking and narrow source diversity do not support enabling the check or adjusting customer confidence.

## Separate source qualification with a second generator

The retained MuseTalk V1.5 assets have publisher-granted commercial use. The adapted pipeline uses pinned, reviewed auxiliary assets and rights-cleared GRID recordings. The published demonstration media are excluded. The earlier full-trajectory audit found **0 of 16** nominal source-time MuseTalk videos compatible with the intended visible trajectory. A new separately frozen check measured only a stop-like local speech event in the same roster of 16 clips from eight speaker groups. [Asset review and adapted pipeline](mouth-multifamily-v7-evaluation.md) · [Earlier whole-trajectory audit](grid-mouth-motion-v158-result.md) · [Publisher commercial-use terms](https://github.com/TMElyralab/MuseTalk#disclaimerlicense)

An audio-only pass verified a distinct original waveform burst in **14 of 16** clips. A separate muted native landmark readout found a matching genuine reopening event in **2 of 16**. In both clips the ordinary mouth resample and brightness deliveries retained that event, the eight-picture retime moved it, and the audio-only control moved the measured burst by 320 ms. One MuseTalk redraw independently retained the local event. Its original and matched ordinary reopening bracket is **1.240–1.280 s**. The redraw's reopening bracket is **1.280–1.320 s** around the original waveform burst midpoint at **1.3175 s**. The earlier-conditioned version had two plausible visual reopenings and failed the uniqueness condition. This establishes one local event at native picture resolution. Phone identity, perceptual synchronization over the complete clip, and generator-independent detector performance remain unmeasured. The one qualified source component contributed ordinary clips to the color study's development point, so it was excluded from that study's held detector result. [Separate event receipt](evaluation-panels/grid-musetalk-local-release-2026-11-13.json)

## Replay

Use the retained native video bank and an unused private directory for each run. Media, per-source geometry and native crops remain local. The public readers need only Python's standard library.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze prepare development held; do
  backend/.venv/bin/python -B -m scripts.video_mouth_chroma_coupling "$phase" "$run" "$lombard" "$fresh" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_mouth_chroma_coupling "$chroma_receipt" --assemble "$run"
python3 -B -IS backend/scripts/review_video_mouth_chroma_coupling.py "$chroma_receipt"
for phase in freeze run; do
  backend/.venv/bin/python -B -m scripts.qualify_grid_musetalk_local_release "$phase" "$event_run" || exit 1
done
python3 -B -IS backend/scripts/review_grid_musetalk_local_release.py "$event_receipt" --assemble "$event_run" "$run/freeze.json"
```
