# Baseline image detectors on vb-1.0-core: face channel (I-9) and I-14 inputs

Criteria ML §3.2 A1–A6. Bench `vb-1.0-core` (D-39), manifest sha256 `d0224343a5c0ad38605ade2b5225deba3cb607ace950e2ae3a127ba9dab26d25`,
fit ∪ cal only (930 items in the ablations: 379 real, 551 fake). 2026-09-14. Automated record (ralph loop), pending human review.

The baseline is the commercial default-enabled image set: img.univfd, img.hf.ateeqq, img.hf.dima806, img.dfdc_b7, five forensic
checks, four metadata rules and the watermark decoder. For a detector already in it, the ablation compares the baseline without
the detector ("without") against the full baseline ("with"). Wvolf, which is not in it, is added.

**Decisions.** No detector below passes every criterion as written. This record changes no `models.yaml` default: I-14 decides
the image fuser's set on the fused gate. The recommendations are judgments on the measured values, labelled as such. Recorded in
`models.yaml` on 2026-09-14 (I-9): img.hf.wvolf `retired`; face.gend_clip_l14 `rejected` (`face.gend_clip_l14.md`); img.dfdc_b7
and img.hf.dima806 stay `pending`, kept in the baseline set until I-14's fused gate.

## Summary

| Detector | A1 licence | A2 AUC [95 % CI] on declared threats | A3a Δ LGO macro [CI] | A3b | A4 | A5 | A6 p95 per call |
|---|---|---|---|---|---|---|---|
| img.dfdc_b7 (face_swap) | commercial, MIT | 0.7959 [0.6911, 0.8809]; 60 of 61 fakes, 156 of 379 reals scored | fail, +1.3 [-3.2, +5.6] | pass | fail (R1, R3) | not decided: 4 paired fakes | 222 ms (n 905) |
| img.hf.dima806 (gan_face) | commercial, Apache-2.0 | 0.7410 [0.6141, 0.8566]; 35 of 35 fakes | pass, +3.5 [+0.8, +6.7] | fail | fail (R5) | not decided: 3 paired fakes | 73 ms (n 2,000) |
| img.hf.wvolf (gan_face) | **fail**: unknown, NOASSERTION (denied by the commercial policy, D-04) | 0.7547 [0.6257, 0.8677]; 35 of 35 fakes | fail, 0.0 [0.0, 0.0] | fail | pass (no change) | not decided: 3 paired fakes | not established (cached rows carry no runtime) |
| img.univfd (full_synthesis, gan_face) | commercial, MIT | **0.5065 [0.4382, 0.5875]**; 452 fakes, 379 reals | fail, -4.1 [-9.4, +0.8] | fail | fail (R1, R3) | pass: ΔTPR +3.2 pts on jpeg75 and social_ig (63 paired) | 437 ms (n 2,000) |
| img.hf.ateeqq (full_synthesis) | commercial, Apache-2.0 | 0.7472 [0.6878, 0.8021]; 452 fakes, 379 reals | pass, +16.0 [+9.0, +22.7] | pass | fail (R1, R3, R5) | pass: ΔTPR +3.2 pts on jpeg75 and social_ig (63 paired) | 35 ms (n 2,000) |

A2 subsets follow the card's `threat` list through the sub-fuser groups (`scripts/admission_evidence.py`). A5 is decided only
with at least 30 paired robustness fakes (assumption; ML §3.2 names no minimum), so the face detectors' ΔTPR on 3–4 items is
reported in the JSON and not decided.

## Face channel (I-9)

- **DFDC-B7.** It is the baseline's face-swap signal: without it, threat faceswap recall at τ_0.05 falls from 55.7 % to 16.4 %
  (Δ +39.3 pts, CI [+25.8, +53.2]); gan_face recall moves -5.7 pts (CI [-17.1, +5.7]). A4: R1 12.2 % → 15.9 % (+3.7, CI [0.0,
  +9.7]), R3 +1.3 (CI [0.0, +4.0]); R5 falls 12.9 % → 8.2 %. Recommendation: keep it enabled until a face-swap detector is
  admitted, because GenD adds +13.1 pts on face-swap stills on top of it but fails A3a / A4 (`face.gend_clip_l14.md`).
- **dima806.** A small marginal gain (+3.5 pts LGO macro) with no family or threat gaining 10 pts; without it, t2i recall is
  3.1 pts higher (CI [-4.8, -1.3] for with − without), and R5 FPR is 2.4 pts higher with it (CI [0.0, +5.9]).
  Recommendation: I-14 decides on the fused gate; no retirement from this record alone.
- **Wvolf.** Not admissible under the commercial policy (A1). Adding it changes no out-of-fold decision (every stratum and slice
  identical), which suggests the monotone stacker gave it zero weight; the ablation JSON records no coefficients, so this is an
  inference. Recommendation: retire from the face channel (it is already disabled).
- **GenD face.** See `face.gend_clip_l14.md`: A3a -0.6 pts, A3b fails on the macro clause, A4 fails (R1).

## I-14 inputs

- **UnivFD 0.4.0 does not discriminate on core.** A2 AUC 0.5065 on its declared threats, and the baseline without it is 4.1 pts
  better on LGO macro (CI [-9.4, +0.8] for with − without). The QuickGELU fix (B-4a, `docs/v2/qa/b4a_univfd_quickgelu.md`) matched
  OpenAI's features within 7.9e-06, so this is not a loader mismatch on the tested crop. I-14 must decide whether img.univfd stays in the default set.
- **Not a norm-variant artefact.** A direct score-cache check (core fit ∪ cal reals plus t2i / personalised / gan_face fakes, status
  ok, without the audit and C2PA exclusions, so n differs from A2) gives UnivFD AUC 0.4956 on norm rows (n 913, 527 fakes; mean
  p real 0.209, fake 0.188), 0.3822 on norm and 0.4317 on native rows of the 167 robustness items (74 fakes). The Ateeqq control on
  the same items: 0.7447, 0.7930, 0.8067.
- **Ateeqq carries most of the baseline's synthesis recall:** without it, t2i recall falls from 63.1 % to 35.4 % (Δ +27.7 pts,
  CI [+22.6, +32.8]) and edit recall from 23.7 % to 7.9 %. A4 fails on R3 (1.3 % → 5.3 %, +4.0, CI [0.0, +9.3]).
- **A4 power.** Real strata hold 49–85 items (R6: 4); one false positive moves an FPR by 1.2–2.0 pts, above A4's 0.5 / 1.0 pt
  limits (spec question in PROGRESS).

## Reproduce

```bash
B=img.univfd,img.hf.ateeqq,img.hf.dima806,img.dfdc_b7,img.forensic.spectrum,img.forensic.ela,img.forensic.noise,img.forensic.jpeg_dq,img.forensic.copymove,img.meta.generator,img.meta.exif,img.meta.consistency,img.meta.qtables,img.watermark.dwtdct
for c in img.dfdc_b7 img.hf.dima806 img.univfd img.hf.ateeqq; do make calibrate BENCH=vb-1.0-core DETECTORS=$B ABLATE=$c; done
make calibrate BENCH=vb-1.0-core DETECTORS=$B,img.hf.wvolf ABLATE=img.hf.wvolf
cd backend && .venv/bin/python scripts/admission_evidence.py --manifest data/vbench/core/manifest.parquet --scores data/vbench/scores \
  --detector img.dfdc_b7 --detector img.hf.dima806 --detector img.hf.wvolf --detector img.univfd --detector img.hf.ateeqq
```

Raw outputs in this directory: `ablation-<id>.{md,json}` and `admission-evidence-<id>.json`.
