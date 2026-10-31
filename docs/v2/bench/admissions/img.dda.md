# Admission record: img.dda (Dual Data Alignment, DINOv2 ViT-L/14 + LoRA)

BUILD_SPEC I-7; criteria ML §3.2 A1–A6. Bench `vb-1.0-core` (D-39), manifest sha256
`d0224343a5c0ad38605ade2b5225deba3cb607ace950e2ae3a127ba9dab26d25`. 2026-09-14. Automated record (ralph loop), pending human review.

**Decision: not admitted on this record.** A4 fails (R1, R3, R4, R5) and A5 fails (ΔTPR +33.3 pts on jpeg75 and social_ig; ML §3.2
then allows admission only with context interaction terms). The detector stays `default_enabled: false`; `models.yaml` admission `rejected` (2026-09-14).
I-14 decides the image fuser's detector set.

## Criteria

| Criterion | Result | Evidence |
|---|---|---|
| A1 licence and parity | **pass** | models `dda.dinov2_vitl14_lora` and `dinov2.vitl14`: licence class commercial, SPDX Apache-2.0 (`python -m app.cli models list`). Card: CPU vs MPS float32 on 50 vb items max \|Δp\| 1.22e-05 (fp16 not safe, max \|Δp\| 2.21e-02; float32 on every device) |
| A2 stand-alone AUC | **pass**: 0.8721, 95 % CI [0.8395, 0.9028] | card threats `full_synthesis` → bench t2i, personalised, gan_face, t2v, plus every real; fit ∪ cal; scored 425 of 452 fakes and 324 of 379 reals (`not_applicable` below the 336 px short side) |
| A3a marginal fused value | **pass**: Δ LGO macro TPR@5%FPR +27.7 pts, CI [+20.8, +34.6] | `ablation-img.dda.md`; macro without 46.8 %, with 74.5 %; every fold F1–F7 complete |
| A3b coverage gap | **pass** | 19 families or threats gain ≥ 10 pts with CI lower bound > 0, e.g. threat edit +68.4 [+52.6, +81.6], family long-tail +39.0 [+24.4, +53.5], family seedream3 +60.0 [+30.0, +90.0] |
| A4 no harm to reals | **fail**: R1, R3, R4, R5 | out-of-fold FPR at τ_0.05: R1 15.9 % → 13.4 % (-2.4, CI [-14.0, +5.5], n 82); R3 5.3 % → 6.7 % (+1.3, CI [0.0, +4.0], n 75); R4 0.0 % → 1.2 % (+1.2, CI [0.0, +3.6], n 84); R5 8.2 % → 8.2 % (0.0, CI [-4.7, +4.7], n 85); R2, R6 unchanged. R1 and R5 fail on the interval bound |
| A5 robustness | **fail**: ΔTPR +33.3 pts on jpeg75 and on social_ig | fixed τ_0.05 = 0.7497 from the A2 reals; 60 paired robustness fakes in fit ∪ cal (native variant): jpeg75 42 → 22 flagged, social_ig 42 → 22 (`admission-evidence-img.dda.json`) |
| A6 latency | **pass**: per call p50 396 ms, p95 423 ms, max 3,829 ms | 1,755 scored norm rows on core images (warm MPS, cached `runtime_ms`); the whole-analysis p95 belongs to X-4 |

## Items and exclusions

- fit ∪ cal image rows: 930 in the ablation (379 real, 551 fake; C2PA and failed-audit rows dropped).
- Contamination: the card lists `MSCOCO train2017`, matched exactly against `source_dataset`, so 0 items were dropped. Core fit ∪ cal
  holds 21 reals from `detection-datasets/coco`. Whether they come from train2017 was not checked; until it is, DDA's FPR on the
  COCO reals may be optimistic. Open item for I-14.

## Power of A4 on core

The real strata of core fit ∪ cal hold 49–85 items (R6: 4). One additional false positive moves a stratum's FPR by 1.2–2.0 pts,
above A4's 0.5 pt rise and 1.0 pt interval bound. The criterion is applied as written; the question is recorded for the supervisor
(PROGRESS "Spec questions").

## Reproduce

```bash
make calibrate BENCH=vb-1.0-core ABLATE=img.dda \
  DETECTORS=img.univfd,img.hf.ateeqq,img.hf.dima806,img.dfdc_b7,img.forensic.spectrum,img.forensic.ela,img.forensic.noise,img.forensic.jpeg_dq,img.forensic.copymove,img.meta.generator,img.meta.exif,img.meta.consistency,img.meta.qtables,img.watermark.dwtdct,img.dda
cd backend && .venv/bin/python scripts/admission_evidence.py --manifest data/vbench/core/manifest.parquet --scores data/vbench/scores \
  --detector img.dda
```

Raw outputs: `ablation-img.dda.{md,json}` and `admission-evidence-img.dda.json` in this directory.
