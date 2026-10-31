# Admission record: face.gend_clip_l14 (GenD CLIP ViT-L/14 on face crops in stills)

BUILD_SPEC I-9; criteria ML §3.2 A1–A6. Bench `vb-1.0-core` (D-39), manifest sha256
`d0224343a5c0ad38605ade2b5225deba3cb607ace950e2ae3a127ba9dab26d25`. 2026-09-14. Automated record (ralph loop), pending human review.

**Decision: not admitted on this record** (`models.yaml` admission `rejected`, 2026-09-14). A3a, A3b and A4 fail; A5 is not
decided (4 paired fakes); the N5 crop check is not established. The detector stays `default_enabled: false`. The face-channel
decisions for DFDC-B7, dima806 and Wvolf are in `baseline-image.md` (DFDC-B7 and dima806 kept pending I-14; Wvolf retired).

## Criteria

| Criterion | Result | Evidence |
|---|---|---|
| A1 licence and parity | **pass** | model `gend.clip_l14`: licence class commercial, SPDX MIT (`python -m app.cli models list`). Card: CPU vs MPS float32 on 50 vb items with a scorable face max \|Δp\| 1.28e-05 |
| A2 stand-alone AUC | **pass**: 0.8729, 95 % CI [0.7935, 0.9395] | card threats `face_swap` → bench faceswap, talking_head, plus every real; fit ∪ cal; scored 60 of 61 fakes and 156 of 379 reals (the rest have no scorable face) |
| A3a marginal fused value | **fail**: Δ LGO macro TPR@5%FPR -0.6 pts, CI [-1.9, 0.0] | `ablation-face.gend_clip_l14.md`; macro without 46.8 %, with 46.3 %; F7 -4.0 pts, other folds unchanged |
| A3b coverage gap | **fail** | threat faceswap and family faceswap-stills gain +13.1 pts, CI [+5.4, +23.1], but the LGO macro falls (-0.6 pts) |
| A4 no harm to reals | **fail**: R1 | out-of-fold FPR at τ_0.05: R1 15.9 % → 17.1 % (+1.2, CI [0.0, +3.1], n 82); R5 8.2 % → 7.1 % (-1.2, CI [-3.5, 0.0]); other strata unchanged |
| A5 robustness | not decided | 4 paired robustness fakes with a scorable face in fit ∪ cal (ΔTPR 0.0 pts on jpeg75 and social_ig, 3 of 4 flagged before and after); A5 is decided only with at least 30 paired fakes (`admission_evidence.py`) |
| N5 crop policy | not established | ML N5 asks for face-subset AUC within 0.02 of the RetinaFace-crop AUC on 200 faces; no RetinaFace detector is installed (InsightFace is non-commercial, V-2), so YuNet crops are the only policy measured |
| A6 latency | **pass**: per call p50 42 ms, p95 131 ms, max 5,189 ms | 905 scored norm rows on core images (warm MPS, cached `runtime_ms`); the whole-analysis p95 belongs to X-4 |

## Items and exclusions

- fit ∪ cal image rows: 930 in the ablation (379 real, 551 fake; C2PA and failed-audit rows dropped).
- Contamination: the card lists Celeb-DF v3, DeepSpeak v1.1, DeepSpeak v2, FFIW and FaceForensics++; no core item comes from them,
  so 0 items were dropped.
- Core fit ∪ cal holds 61 face-swap stills. GenD adds recall on them, and its t2i recall contribution is negative (threat t2i -1.5
  pts, CI [-3.4, 0.0]).

## Reproduce

```bash
make calibrate BENCH=vb-1.0-core ABLATE=face.gend_clip_l14 \
  DETECTORS=img.univfd,img.hf.ateeqq,img.hf.dima806,img.dfdc_b7,img.forensic.spectrum,img.forensic.ela,img.forensic.noise,img.forensic.jpeg_dq,img.forensic.copymove,img.meta.generator,img.meta.exif,img.meta.consistency,img.meta.qtables,img.watermark.dwtdct,face.gend_clip_l14
cd backend && .venv/bin/python scripts/admission_evidence.py --manifest data/vbench/core/manifest.parquet --scores data/vbench/scores \
  --detector face.gend_clip_l14
```

Raw outputs: `ablation-face.gend_clip_l14.{md,json}` and `admission-evidence-face.gend_clip_l14.json` in this directory.
