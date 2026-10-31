# Audio fuser `fuser-audio-2.0.0-alpha` on vb-1.0-core (A-1, D-39)

Bench `vb-1.0-core`, manifest sha256 `d0224343a5c0ad38605ade2b5225deba3cb607ace950e2ae3a127ba9dab26d25`. 2026-09-14. Automated record
(ralph loop), `reviewed_by: ralph-loop (automated; pending human review)` (D-17). Measured on core with 95 % intervals. The audio
gate FAILS G10a; G10b passes on its point estimate only. Both are in `../shortfalls.md`.

## Decisions (on fit ∪ cal, before the lockbox run)

- **Fitted set.** `aud.gustking_xlsr` alone (the commercial default audio detector with usable core scores).
- **Admission ablation** (`ablation-aud.mo_thecreator_w2v2.md`; 149 items, 75 real, 74 fake; cross-fit 5 folds; paired grouped
  bootstrap B 2000): adding `aud.mo_thecreator_w2v2` lowers the out-of-fold AUC (0.7822 with, 0.7904 without) and tts recall at
  τ_0.05 by −8.1 pts [−16.7, +0.0]. A3a / A3b not established (no audio LGO folds); A4 fails on its CI (A-bonafide ΔFPR +0.0 pts,
  CI [−4.1, +4.2]). Rejected: `default_enabled: false`, so `deep` now enables `aud.gustking_xlsr` only.
- **Profiles.** `quick`, `full` and `deep` load this artefact (loader check: "fuser-audio-2.0.0-alpha.json matches the enabled
  detectors", sha256 `2ea7921de5b4…`). The standalone audio pipeline (`aud-2.0-a1`) uses it through `common.select_fusion(ctx,
  "audio")`; recordings gated out as music- or silence-dominated stay `not assessable` with the fallback decision.
- **Parameterisation.** The M-1 monotone stacker over the detector's recording aggregate with per-context Platt (context `audio`,
  from the global fit: 25 real, 24 fake `cal` items, below `min_class_n` 100). Sub-fusers h1, h2 face swap and h2 partial edit
  are skipped (no items of those threats); the `voice` sub-fuser is fitted.
- **Music / noise thresholds.** Not re-fitted here: the content gate `sms-heuristic-v1` keeps the A-1 thresholds measured in
  `docs/v2/qa/a1_audio_content.md` (1.00 % [0.62, 1.62] of speech gated out). Core audio has no music or noise stratum, so the
  false-positive behaviour on music and background noise is not measured on core (card `fp_modes`).

## Fit

`fit_n` 100, `cal_n` 49; λ 1.0 (CV log-loss 0.5936); cal AUC 0.9067 (in-sample); disagreement d95 not defined (one detector).
No audio LGO refits: G10 is judged on source sets (`asvspoof5`, `in_the_wild`), not on generator folds.

## Results on `test` (lockbox)

| Set | Reals | Fakes | AUC (95 % CI) | EER | TPR at τ_0.05 | FPR at τ_0.05 | ECE / Brier | Inconclusive |
|---|---|---|---|---|---|---|---|---|
| `audio/test` | 75 | 76 | 0.8670 (0.7993–0.9283) | 22.4 % | 5.3 % (4/76) | 1.3 % (1/75; Wilson 0.2 %–7.2 %) | 0.0890 / 0.1555 | 61.6 % |
| `audio/test/asvspoof5` | 17 | 19 | 0.8545 (0.7023–0.9606) | 23.5 % | 5.3 % (1/19) | 0.0 % (0/17) | 0.2369 / 0.1805 | 80.6 % |
| `audio/test/in_the_wild` | 37 | 19 | 0.8485 (0.7000–0.9786) | 26.3 % | 0.0 % (0/19) | 2.7 % (1/37; Wilson 0.5 %–13.8 %) | 0.1458 / 0.1635 | 55.4 % |

- **τ_0.05** = 0.8336 (n = 25 `cal` reals); τ_0.01 not established.
- **v0.3 baseline** on the same items (`baseline-v0.3.md`, `v1.audio_fused`): audio test AUC 0.8637 (0.7908–0.9247), TPR at τ_0.05
  5.3 %, FPR 1.4 %, ECE 0.1278; G10a 0.2353, G10b 0.0270. G10 is unchanged: the fused score is a monotone transform of the single
  detector, so the ranking (EER) and the count above τ are the same; the fit improves calibration (ECE 0.1278 → 0.0890).

## Gate

`gate-audio-2.0.0-alpha.txt` (image `fuser-image-2.0.0-alpha` + video `fuser-video-2.0.0-alpha` + this audio artefact): 16 PASS,
20 FAIL, 22 MISSING of 58 checks (exit 1). G10a FAIL (EER 23.5 %, target ≤ 10 %); G10b PASS (1/37, Wilson upper 13.8 % above the
target). Image rows are those of I-14, video rows those of V-8; latency (G13) is MISSING.

## Limits of this record

- Small sets (17–37 reals per G10 set); the intervals above are wide.
- The audio card `eval` rows are not written: `--update-cards` covers image detectors only.
- `test_access.log` holds this access (git `01ce315`, systems `3486cde0…+b86c4746…+2ea7921d…`).

## Reproduce

```bash
cd backend
.venv/bin/python scripts/calibrate.py --manifest data/vbench/core/manifest.parquet --scores data/vbench/scores --modality audio --seed 7 \
  --out ../docs/v2/bench/admissions/ --bench vb-1.0-core --detectors aud.gustking_xlsr,aud.mo_thecreator_w2v2 --ablate aud.mo_thecreator_w2v2
.venv/bin/python scripts/calibrate.py --manifest data/vbench/core/manifest.parquet --scores data/vbench/scores --modality audio --seed 7 \
  --bench vb-1.0-core --detectors aud.gustking_xlsr --out data/fusion/
.venv/bin/python scripts/vbench_report.py --manifest data/vbench/core/manifest.parquet --scores data/vbench/scores \
  --artefact data/fusion/fuser-image-2.0.0-alpha.json --lgo-artefacts data/fusion/lgo-image-2.0.0-alpha \
  --video-artefact data/fusion/fuser-video-2.0.0-alpha.json --video-lgo-artefacts data/fusion/lgo-video-2.0.0-alpha \
  --audio-artefact data/fusion/fuser-audio-2.0.0-alpha.json --eval-split test --bench vb-1.0-core \
  --repro data/fusion/repro-image-2.0.0-alpha/fuser-image-2.0.0-alpha.json data/fusion/repro2-image-2.0.0-alpha/fuser-image-2.0.0-alpha.json
.venv/bin/python scripts/vbench_gate.py --results ../docs/v2/bench/results.json --goals ../docs/v2/bench/goals.yaml \
  --manifest data/vbench/core/manifest.parquet > ../docs/v2/bench/gate-audio-2.0.0-alpha.txt
```

Job logs: `a1-ablate`, `a1-fuse`, `a1-ci`.
