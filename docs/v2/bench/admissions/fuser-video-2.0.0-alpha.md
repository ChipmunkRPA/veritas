# Video fuser `fuser-video-2.0.0-alpha` on vb-1.0-core (V-8, D-39)

Bench `vb-1.0-core`, manifest sha256 `d0224343a5c0ad38605ade2b5225deba3cb607ace950e2ae3a127ba9dab26d25`. 2026-09-14. Automated record
(ralph loop), `reviewed_by: ralph-loop (automated; pending human review)` (D-17). Measured on core with 95 % intervals. The video
gate FAILS (G9a), and G9b passes only degenerately; both are in `../shortfalls.md`.

## Decisions (on fit ∪ cal, before the lockbox run)

- **Fitted set.** The commercial default video set with usable core scores, minus `vid.frames.univfd`: `vid.frames.hf.ateeqq`,
  `vid.frames.hf.dima806`, `vid.frames.dfdc_b7`, `vid.codec.ffprobe`, `aud.gustking_xlsr` (fitted) and
  `vid.lipsync.landmark_xcorr` (skipped: no usable scores on fit items). The model-free v0.3 heuristics
  (`vid.forensic.face_noise`, `vid.temporal.*`) and `vid.tracks.yunet_dinov2` report no probability in the score cache, so they
  cannot be fitted.
- **Admission ablations** (`ablation-vid.*.md`; 80 items, 37 real, 43 fake; cross-fit 5 folds; paired grouped bootstrap B 2000):

  | Candidate | Out-of-fold AUC with / without | t2v recall at τ_0.05, Δ pts [95 % CI] | A3a | A3b | A4 | Decision |
  |---|---|---|---|---|---|---|
  | + `vid.synth.frames.commfor384` | 0.8718 / 0.7593 | +34.1 [+18.4, +51.3] | fail | pass | fail (V-general) | rejected, off |
  | + `vid.synth.frames.dda` | 0.8906 / 0.7593 | +24.4 [+10.0, +39.5] | fail | pass | fail (V-general) | rejected, off |
  | + `vid.swap.gend_clip_l14` | 0.7838 / 0.7593 | -7.3 [-15.9, +0.0] | fail | fail | fail (V-general) | rejected, off |
  | − `vid.frames.univfd` | 0.7593 / 0.7744 | +0.0 | fail | fail | pass | rejected, off |

- **A3a is not informative on core video.** Every vendor-LGO refit sets τ_0.05 from 15–16 `cal` reals; the rank rule needs 19,
  so no refit flags an item and every Δ is 0.0 pts. A4 fails on its confidence bound (ΔFPR +0.0 pts on V-general, CI [-10.0, +10.0]).
  As for images (I-14), candidates that do not pass the criteria as written stay off; the measured out-of-fold gains of the two
  frame detectors are recorded for v2.1 (spec question in PROGRESS).
- **Defaults flipped:** `vid.frames.univfd` `default_enabled: false`. The other cards keep their defaults; their admissions are recorded.
- **Profiles.** `full` and `deep` load this artefact (loader check: "fuser-video-2.0.0-alpha.json matches the enabled detectors",
  sha256 `b86c4746e184…`). `quick` keeps the uncalibrated fallback ("detector vid.codec.ffprobe is not enabled").
- **Parameterisation.** The artefact is the M-1 monotone stacker over the detectors' clip aggregates with per-context Platt
  (context `video`). ARCHITECTURE §4.4's lane sub-fuser logits → union fuser (its open item for the ML owner) is not built:
  the h2 face-swap, partial-edit and voice sub-fusers are skipped (fewer than 10 fakes of a class in fit or cal: 2 talking-head
  fakes, no edit or voice fakes on core video). At runtime the `condition` stage sets `face_px` from the face tracks (on 97 of
  the 150 cached core video runs), while core video manifest rows have no `face_px_max`, so no face term is fitted; `min_side`
  terms match the runtime context.

## Fit

`fit_n` 48, `cal_n` 32; λ 3.0 (CV log-loss 0.6684); cal AUC 0.7539 (in-sample). Refits without one vendor fold each (V-Veo,
V-Sora, V-Kling, V-Seedance, V-WanHunyuan) in `data/fusion/lgo-video-2.0.0-alpha/`.

## Results on `test` (lockbox)

| Set | Reals | Fakes | AUC (95 % CI) | TPR at τ_0.05 | FPR at τ_0.05 | ECE / Brier | Inconclusive |
|---|---|---|---|---|---|---|---|
| `video/test` | 38 | 32 | 0.7599 (0.6301–0.8754) | 0.0 % | 0.0 % (0/38; Wilson upper 9.2 %) | 0.1471 / 0.1963 | 100 % (70/70) |

- **τ_0.05 and τ_0.01:** not established (16 `cal` reals).
- **Vendor LGO (refits):** V-Kling AUC 0.7000 (5 fakes), V-Seedance 0.5368 (5), V-Sora 0.9632 (5), V-Veo 0.7171 (4), V-WanHunyuan
  1.0000 (4); TPR at τ_0.05 0/4–0/5 in every fold; macro 0.0000, macro ECE 0.3708.
- **v0.3 baseline** on the same items (`baseline-v0.3.md`): video AUC 0.6933 [0.5663, 0.8167], τ not established.
- **Conformal:** coverage 100 % per class, and the α = 0.05 set holds both classes for all 70 test items (inconclusive 70/70), so the
  artefact separates items (AUC) without yet supporting a decision at a known error rate.

## Gate

`gate-video-2.0.0-alpha.txt` (image `fuser-image-2.0.0-alpha` + this video artefact): 15 PASS, 19 FAIL, 24 MISSING of 58 checks
(exit 1). G9a FAIL (macro 0.0000); G9b PASS (0/38) is degenerate. Image rows are those of I-14; audio (G10) and latency (G13) are MISSING.

## Limits of this record

- The video card `eval` rows are not written: `--update-cards` covers image detectors only.
- The report ran without `--write-artefact` (image metrics stay those of I-14; video `metrics.test` not filled).
- `test_access.log` holds this access (git `ecc512b`, systems `3486cde0…+b86c4746…`).

## Reproduce

```bash
cd backend
V=vid.frames.hf.ateeqq,vid.frames.hf.dima806,vid.frames.dfdc_b7,vid.codec.ffprobe,vid.lipsync.landmark_xcorr,aud.gustking_xlsr
B=vid.frames.univfd,$V
for c in vid.synth.frames.commfor384 vid.synth.frames.dda vid.swap.gend_clip_l14; do .venv/bin/python scripts/calibrate.py --manifest data/vbench/core/manifest.parquet \
  --scores data/vbench/scores --modality video --seed 7 --out ../docs/v2/bench/admissions/ --bench vb-1.0-core --detectors $B,$c --ablate $c; done
.venv/bin/python scripts/calibrate.py --manifest data/vbench/core/manifest.parquet --scores data/vbench/scores --modality video --seed 7 \
  --out ../docs/v2/bench/admissions/ --bench vb-1.0-core --detectors $B --ablate vid.frames.univfd
C="--manifest data/vbench/core/manifest.parquet --scores data/vbench/scores --modality video --seed 7 --bench vb-1.0-core --detectors $V"
.venv/bin/python scripts/calibrate.py $C --out data/fusion/
for f in V-Veo V-Sora V-Kling V-Seedance V-WanHunyuan; do .venv/bin/python scripts/calibrate.py $C --out data/fusion/lgo-video-2.0.0-alpha/ \
  --exclude-folds $f --version 2.0.0-alpha.lgo-$f; done
.venv/bin/python scripts/vbench_report.py --manifest data/vbench/core/manifest.parquet --scores data/vbench/scores \
  --artefact data/fusion/fuser-image-2.0.0-alpha.json --lgo-artefacts data/fusion/lgo-image-2.0.0-alpha \
  --video-artefact data/fusion/fuser-video-2.0.0-alpha.json --video-lgo-artefacts data/fusion/lgo-video-2.0.0-alpha --eval-split test \
  --bench vb-1.0-core --repro data/fusion/repro-image-2.0.0-alpha/fuser-image-2.0.0-alpha.json data/fusion/repro2-image-2.0.0-alpha/fuser-image-2.0.0-alpha.json
.venv/bin/python scripts/vbench_gate.py --results ../docs/v2/bench/results.json --goals ../docs/v2/bench/goals.yaml \
  --manifest data/vbench/core/manifest.parquet > ../docs/v2/bench/gate-video-2.0.0-alpha.txt
```

Job logs: `v8-ablate`, `v8-fuse`.
