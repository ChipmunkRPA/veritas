# Admission record: img.commfor384 (Community Forensics ViT-S/16 384)

BUILD_SPEC I-6; criteria ML §3.2 A1–A6. Bench `vb-1.0-core` (D-39), manifest sha256
`d0224343a5c0ad38605ade2b5225deba3cb607ace950e2ae3a127ba9dab26d25`. 2026-09-14. Automated record (ralph loop), pending human review.

**Decision: not admitted on this record.** A4 fails (R1, R3, R5) and A5 fails (ΔTPR +16.7 pts on jpeg75, +21.7 pts on social_ig;
ML §3.2 then allows admission only with context interaction terms); the A1 parity run is not yet established. The detector stays
`default_enabled: false`; `models.yaml` admission `rejected` (2026-09-14). I-14 decides the image fuser's detector set.

## Criteria

| Criterion | Result | Evidence |
|---|---|---|
| A1 licence and parity | **pass** | model `commfor.vit_s16_384`: licence class commercial, SPDX MIT (`python -m app.cli models list`). CPU vs MPS float32 on 50 vb items with a short side ≥ 384 px max \|Δp\| 3.60e-06, mean 3.71e-07 (`pytest -m slow tests/test_detector_parity.py -k commfor`, 2026-09-14) |
| A2 stand-alone AUC | **pass**: 0.8440, 95 % CI [0.8012, 0.8843] | card threats `full_synthesis` → bench t2i, personalised, gan_face, t2v, plus every real; fit ∪ cal; scored 424 of 452 fakes and 305 of 379 reals (`not_applicable` below the 384 px short side) |
| A3a marginal fused value | **pass**: Δ LGO macro TPR@5%FPR +25.9 pts, CI [+19.1, +33.0] | `ablation-img.commfor384.md`; macro without 46.8 %, with 72.7 %; every fold F1–F7 complete |
| A3b coverage gap | **pass** | 18 families or threats gain ≥ 10 pts with CI lower bound > 0, e.g. threat edit +68.4 [+52.6, +84.2], family midjourney +44.7 [+29.7, +62.2], family seedream4 +52.0 [+32.0, +72.0] |
| A4 no harm to reals | **fail**: R1, R3, R5 | out-of-fold FPR at τ_0.05: R1 15.9 % → 18.3 % (+2.4, CI [-4.0, +10.0], n 82); R3 5.3 % → 6.7 % (+1.3, CI [0.0, +4.0], n 75); R5 8.2 % → 4.7 % (-3.5, CI [-8.2, +1.2], n 85; fails on the CI upper bound); R2, R4, R6 unchanged |
| A5 robustness | **fail**: ΔTPR +16.7 pts on jpeg75, +21.7 pts on social_ig | fixed τ_0.05 = 0.4897 from the A2 reals; 60 paired robustness fakes in fit ∪ cal (native variant): jpeg75 42 → 32 flagged, social_ig 42 → 29 (`admission-evidence-img.commfor384.json`) |
| A6 latency | **pass**: per call p50 38 ms, p95 63 ms, max 537 ms | 1,705 scored norm rows on core images (warm MPS, cached `runtime_ms`); the whole-analysis p95 belongs to X-4 |

## Items and exclusions

- fit ∪ cal image rows: 930 in the ablation (379 real, 551 fake; C2PA and failed-audit rows dropped).
- Contamination: the card lists `OwensLab/CommunityForensics-Eval`; no core item comes from it, so 0 items were dropped.

## Power of A4 on core

The real strata of core fit ∪ cal hold 49–85 items (R6: 4). One additional false positive moves a stratum's FPR by 1.2–2.0 pts,
above A4's 0.5 pt rise and 1.0 pt interval bound, so A4 can pass only where the candidate adds no false positive at all. The
criterion is applied as written; the question is recorded for the supervisor (PROGRESS "Spec questions").

## Reproduce

```bash
make calibrate BENCH=vb-1.0-core ABLATE=img.commfor384 \
  DETECTORS=img.univfd,img.hf.ateeqq,img.hf.dima806,img.dfdc_b7,img.forensic.spectrum,img.forensic.ela,img.forensic.noise,img.forensic.jpeg_dq,img.forensic.copymove,img.meta.generator,img.meta.exif,img.meta.consistency,img.meta.qtables,img.watermark.dwtdct,img.commfor384
cd backend && .venv/bin/python scripts/admission_evidence.py --manifest data/vbench/core/manifest.parquet --scores data/vbench/scores \
  --detector img.commfor384
```

Raw outputs: `ablation-img.commfor384.{md,json}` and `admission-evidence-img.commfor384.json` in this directory.
