# Video fuser `fuser-video-2.1.0` and audio fuser `fuser-audio-2.1.0` on vb-1.1 (H-8)

Bench `vb-1.1`, manifest sha256 `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`, scope sha256 `5059434a74cf…`.
2026-09-15. Automated record (ralph loop), `reviewed_by: ralph-loop (automated; pending human review)` (D-17). The video gate
fails on G9a; G9b passes. The audio gate fails on G10a; G10b passes. Both are in `../shortfalls.md`.

## Summary

- **Frames channel.**
  - Community Forensics and DDA are admitted for images (H-4). They now run on native video frames, on by default under
    `full` and `deep` (BUILD_SPEC H-8), so clips without faces or sound are assessed.
  - Out of fold on fit ∪ cal they raise the video fuser's AUC from 0.9302 to 0.9629 (Community Forensics) and from 0.9539 to
    0.9629 (DDA).
  - The ateeqq and dima806 frame detectors are switched off: both were rejected for images, and on video they do not improve
    the fuser.
- **D3 threshold on compressed real video** (`../d3_recalibration.md`): σ_0.05 0.8706 over 834 fit ∪ cal real rows. As a fuser
  feature (`vd.d3.sigma`) it lowers t2v recall, so it is rejected and D3 stays off.
- **Video, vb-1.1 test:**
  - AUC 0.9668 (0.9454–0.9842), TPR at τ_0.05 87.4 % (118/135), real-video FPR 3.1 % (5/162, Wilson 1.3–7.0 %) (G9b PASS);
  - vendor-LGO macro TPR 63.9 % against 80 % (G9a FAIL): Seedance 26.3 %, Sora 42.1 %.
- **Audio, vb-1.1 test:** EER on ASVspoof 5 17.4 % against 10 % (G10a FAIL); in-the-wild bona fide FPR 1.1 % (1/95) (G10b PASS),
  with 1 of 90 in-the-wild fakes flagged at that threshold.

## Video admissions on fit ∪ cal (decided before the lockbox run)

- **Command:** job `h8-ablate`, `bash scripts/h8_ablate.sh` (`calibrate --ablate`, `--context scored --scope …vb-1.1.parquet`,
  paired grouped bootstrap B 2,000, cross-fit 5 folds).
- **Items:** 353 (188 real, 165 fake).
- **Baseline:** the frames channel `vid.synth.frames.commfor384`, `vid.synth.frames.dda`, `vid.frames.dfdc_b7`, plus the
  v2.0 codec and voice lanes `vid.codec.ffprobe`, `aud.gustking_xlsr` and the lip-sync correlation `vid.lipsync.landmark_xcorr`.
- **Threshold rule:** τ follows ADR 0007: V-talking-head (6 fit ∪ cal reals) is pooled with V-general.
- **An earlier run** with V-talking-head weighted as its own stratum (`data/fusion/h8-ablation-weighted/`, log
  `data/jobs/h8-ablate.run1.log`) is superseded. It blocked Community Forensics on V-general, +6.0 pts [+2.2, +9.9], an
  artefact of the 6 talking-head reals carrying half of the threshold mass (ADR 0007).
- **Ablation tables:** `vb-1.1/video/ablation-<id>.md`.

| Candidate | Out-of-fold AUC without / with | Δ LGO macro, pts [95 % CI] | t2v recall Δ pts [95 % CI] | talking_head Δ pts | V-general ΔFPR pts [95 % CI] | Pooled ΔFPR upper | A3a | A3b | A4 | Decision |
|---|---|---|---|---|---|---|---|---|---|---|
| − `vid.synth.frames.commfor384` | 0.9302 / 0.9629 | +4.0 [-8.0, +16.8] | +14.6 [+8.8, +21.1] | +50.0 [+12.5, +87.5] | +0.5 [-2.7, +4.4] | +3.7 | fail | pass | fail (pooled bound) | on (frames channel) |
| − `vid.synth.frames.dda` | 0.9539 / 0.9629 | +0.0 [-6.7, +6.7] | +5.1 [+1.3, +9.1] | +37.5 [+0.0, +75.0] | +0.5 [-1.6, +2.7] | +2.7 | fail | fail | fail (pooled bound) | on (frames channel) |
| − `vid.frames.dfdc_b7` | 0.9619 / 0.9629 | -2.5 [-8.0, +0.0] | +0.6 [+0.0, +1.9] | +0.0 | +0.0 [-1.6, +1.6] | +1.6 | fail | fail | fail (pooled bound) | on (frames channel, image-admitted with has_face) |
| − `aud.gustking_xlsr` | 0.9668 / 0.9629 | +2.2 [+0.0, +6.7] | -1.3 [-3.3, +0.0] | +0.0 | +0.0 [-1.6, +1.6] | +1.6 | fail | fail | fail (pooled bound) | kept (v2.0 set) |
| − `vid.codec.ffprobe` | 0.9628 / 0.9629 | +0.0 [+0.0, +0.0] | +0.6 [+0.0, +2.0] | +0.0 | +0.0 [+0.0, +0.0] | +0.0 | fail | fail | pass | kept (v2.0 set) |
| − `vid.lipsync.landmark_xcorr` | 0.9629 / 0.9629 | +0.0 [+0.0, +0.0] | +0.0 | +0.0 | +0.0 [+0.0, +0.0] | +0.0 | fail | fail | pass | skipped by the fit (no usable scores on fit items) |
| + `vd.d3.sigma` | 0.9629 / 0.9634 | -2.2 [-6.7, +0.0] | -2.5 [-5.1, -0.6] | -25.0 [-62.5, +0.0] | +0.0 [-1.6, +1.6] | +1.6 | fail | fail | fail | rejected, D3 off |
| + `vid.frames.hf.ateeqq` | 0.9629 / 0.9618 | -3.9 [-10.3, +0.0] | +0.6 [+0.0, +2.0] | +0.0 | +0.0 [+0.0, +0.0] | +0.0 | fail | fail | pass | rejected, off |
| + `vid.frames.hf.dima806` | 0.9629 / 0.9626 | +2.7 [-3.6, +10.0] | -1.3 [-3.2, +0.0] | -12.5 [-37.5, +0.0] | -0.5 [-1.6, +0.0] | +1.6 | fail | fail | fail | rejected, off |
| + `vid.swap.gend_clip_l14` | 0.9629 / 0.9625 | -2.2 [-6.7, +0.0] | -1.3 [-3.3, +0.0] | +0.0 | +0.0 [+0.0, +0.0] | +0.0 | fail | fail | pass | rejected, off |

A "−" row leaves a baseline member out ("without") and keeps it ("with"); a "+" row adds the candidate.

- **Pooled A4 bound cannot pass at 188 reals.**
  - D-44 keeps A4's pooled paired upper bound at ≤ 1.0 pt. One real is 0.53 pts of 188.
  - Every candidate that changes which reals are flagged has a bootstrap interval of about ±1.6 pts or wider. Only a candidate
    that changes nothing passes.
  - No stratum shows established harm (every V-general lower bound ≤ 0).
  - Spec question in PROGRESS.
- **Decisions.**
  - The frames channel members are on by the H-8 directive for image-admitted detectors. Their cards say `pending`, not
    `admitted`, because A4 as written fails.
  - Rejected candidates are off (V-8 / I-14 practice).
- **Card changes** (`backend/app/models.yaml`):
  - `vid.synth.frames.commfor384`, `vid.synth.frames.dda`: `default_enabled: true`.
  - `vid.frames.hf.ateeqq`, `vid.frames.hf.dima806`: `default_enabled: false`, `rejected`.
  - `vid.synth.d3_xclip`: `rejected`, threshold numbers in its limitations.

## Fits

| Artefact | Detectors | fit_n / cal_n | λ (CV log-loss) | cal AUC (in-sample) | sha256 after `--write-artefact` |
|---|---|---|---|---|---|
| `fuser-video-2.1.0.json` | commfor384 frames, dda frames, dfdc_b7 frames, ffprobe, gustking (landmark_xcorr skipped: no usable scores on fit items) | 214 / 139 | 3.0 (0.2206) | 0.9640 | `8f20914fd3c5…` |
| `fuser-audio-2.1.0.json` | `aud.gustking_xlsr` | 562 / 318 | 3.0 (0.5048) | 0.7903 | `65a637ad5e56…` |

- **Command:** job `h8-fit`, `bash scripts/h8_fit.sh`, with `--context scored --scope …vb-1.1.parquet`, seed 7, bootstrap 200.
- **Vendor-LGO refits:** one per excluded fold (V-Veo, V-Sora, V-Kling, V-Seedance, V-WanHunyuan) in `data/fusion/lgo-video-2.1.0/`.
- **Audio refit:** it replaces the 07:58Z fit at 178cefc, with the same fit and cal numbers.
- **Sub-fusers:** the video fuser's h2 face-swap, partial-edit and voice sub-fusers are skipped (fewer than 10 items of a class
  in fit or cal).
- **Loader:** `full` and `deep` load the artefact while its five detectors are enabled. `vid.frames.hf.ateeqq` and dima806
  are now off, so `fuser-video-2.0.0-alpha` no longer loads.

## Results on `test` (lockbox; job `h8-report`, `bash scripts/h8_report.sh`)

| Set | Reals | Fakes | AUC (95 % CI) | TPR at τ_0.05 | FPR at τ_0.05 | ECE | Inconclusive |
|---|---|---|---|---|---|---|---|
| `video/test` | 162 | 135 | 0.9668 (0.9454–0.9842) | 87.4 % (118/135; Wilson 80.8–92.0 %) | 3.1 % (5/162; Wilson 1.3–7.0 %) | 0.0595 | 18.9 % |
| `audio/test/asvspoof5` | 86 | 90 | 0.8786 (0.8221–0.9283) | 38.9 % (35/90) | 4.7 % (4/86) | 0.2027 | 73.3 % |
| `audio/test/in_the_wild` | 95 | 90 | 0.9065 (0.8083–0.9753) | 1.1 % (1/90) | 1.1 % (1/95; Wilson 0.2–5.7 %) | 0.1534 | 81.6 % |

- **Video thresholds:** τ_0.05 0.7163 from 74 `cal` reals; τ_0.01 not established (the rank rule needs 99).
- **Audio thresholds:** τ_0.05 0.8041 and τ_0.01 0.8089 from 118 `cal` reals. EER: `asvspoof5` 17.4 %, `in_the_wild` 16.8 %,
  all audio 24.3 %.
- **Vendor LGO** (TPR at each refit's τ_0.05; AUC):

  | Fold | Fakes | TPR | AUC |
  |---|---|---|---|
  | V-Kling | 19 | 78.9 % (15/19) | 0.9740 |
  | V-Seedance | 19 | 26.3 % (5/19) | 0.7654 |
  | V-Sora | 19 | 42.1 % (8/19) | 0.8616 |
  | V-Veo | 18 | 72.2 % (13/18) | 0.9688 |
  | V-WanHunyuan | 18 | 100.0 % (18/18) | 1.0000 |

  Macro 0.6392.
- **Gate:** `../gate-v2.1.txt` (36 PASS, 14 FAIL, 8 MISSING of 58 checks, image rows re-measured under ADR 0007);
  `check_shortfalls.py` PASS.
- **Access log:** `test_access.log` 2026-09-15T09:38:28Z, git `6af1e93` plus the uncommitted H-8 changes committed with this
  record (the ADR 0007 rule and the card defaults). Systems `bcf4c139…+8f20914f…+65a637ad…`.

## Limits of this record

- **Card eval rows:** `--update-cards` rewrote the `eval` rows of the 8 fitted image, video and voice detector cards from this
  run. Seven changed: `img.dda` and `img.dfdc_b7` (TPR at 1 % FPR under ADR 0007), and the vb-1.1 stand-alone rows of
  `vid.codec.ffprobe`, `vid.synth.frames.commfor384`, `vid.synth.frames.dda`, `vid.frames.dfdc_b7` and `aud.gustking_xlsr`.
- **Scores:** from the H-2 video pass under an allow-list policy (`h2-score-video`). The default profile runs the same frames
  channel, but latency is not re-measured (H-10).
