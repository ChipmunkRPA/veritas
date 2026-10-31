# H-3 real-photo false-positive analysis (vb-1.1 fit ∪ cal)

Automated record pending human review. The numbers come from `h3_real_fp_tables.md` / `h3_real_fp.json`, which this command
generated (2026-09-14T19:52:35Z, manifest sha256 `8314119e487a…`, fuser `fuser-image-2.0.0-alpha` sha256 `3486cde0d4bf…`):

```
cd backend && .venv/bin/python scripts/real_fp_analysis.py --manifest data/vbench/manifest.parquet --scores data/vbench/scores \
  --fuser data/fusion/fuser-image-2.0.0-alpha.json \
  --detectors face.gend_clip_l14,img.commfor384,img.dda,img.dfdc_b7,img.faces.yunet,img.forensic.copymove,img.forensic.ela,\
img.forensic.jpeg_dq,img.forensic.noise,img.forensic.spectrum,img.hf.ateeqq,img.hf.dima806,img.hf.haywoodsloan,img.meta.consistency,\
img.meta.exif,img.meta.generator,img.meta.qtables,img.watermark.dwtdct,loc.mesorch,prov.c2pa
```

Scores: the H-2 image pass on the vb-1.1 scope (`h2-score-image-v11` exit 0), 20 image detectors under the H-2 licence policy.
R1 ∪ R2 holds 1,036 fit ∪ cal reals: `commons-r1-camera` 310, `commons-r2-smartphone` 272, `vision-natives` 340,
`synthbuster-plus-test` 114. Contact sheet: `backend/data/vbench/audit/h3_camera_fp.png` (sha256 `b5bbcfdd4501…`, gitignored).

## Findings

1. **No detector triggers the D-41 camera-original rule.** No stand-alone R1 ∪ R2 FPR at τ_0.05 exceeds 10 % with n ≥ 300.
   Only `face.gend_clip_l14` exceeds 10 %, on fewer reals: 11.4 % (23/201; 7.7–16.6). `img.dfdc_b7` gives 8.5 % (17/201;
   5.3–13.1) on the same faces. Both are recorded as "not established".
2. **The v2.0 fused false positives are a format shortcut.** The fused R1 ∪ R2 FPR is 4.2 % (44/1,036; 3.2–5.7), and 42 of the
   44 flagged reals come from `synthbuster-plus-test` (42/114, 36.8 %). The other R1 ∪ R2 sources give 2/922.
   - **Format:** these reals are 1,024 px PNGs with no JPEG quantisation tables and no camera EXIF (sizes 1024×678, 1024×680 and portrait
     equivalents), the same format as the generated images of that source.
   - **Where the flag comes from:** the fuser's context prior (Stage 2, before any detector term) has a median of 0.886 for
     these reals, against 0.090 for the three JPEG camera sources.
   - **Detectors do not separate the classes:** the median fused s is 0.813 for these reals and 0.833 for the 386 fakes of
     the same source. The mean detector contributions on the 44 flagged reals lie between −0.1 and +0.9 pp.
   - **Where the prior comes from:** in this artefact, "lossless, no camera EXIF" is the baseline context (every context
     term off). Inference, not measured here: the fit data held mostly fakes in that context.
3. **The stand-alone false positives come from other sources than the fused ones.**
   - `img.hf.ateeqq` flags 44/310 `commons-r1-camera` reals (its R1 rate is 10.6 %, 45/424), but 1/114 synthbuster reals.
   - `img.dda` flags 43/272 `commons-r2-smartphone` reals (R2 7.2 %).
   - `img.forensic.spectrum` and `img.commfor384` rise at ≥ 8 MP: spectrum 10.3 % (60/585; 8.1–13.0) against 4.2 % below 8 MP;
     commfor384 8.5 % (50/585) against 0.9 % (4/450).
4. **Copy-move and JPEG double quantisation have no false-positive rate at τ.** These two checks score only when they find a
   pattern, and then give the capped value 0.75. τ is that value, so every scored real counts as flagged (100 % of 13 and 12).
   With abstentions counted as not flagged they fire on 1.3 % (13/1,036) and 1.2 % (12/1,036) of R1 ∪ R2 reals.
5. **Recompression strata are outside the camera rule, but the rates are recorded for H-4.** `img.forensic.ela` flags 12.9 % of R3
   (76/587) and `img.forensic.noise` 12.3 % (72/587). `img.hf.dima806` flags 15.1 % of R5 (83/549).

## Contact sheet: causes (60 items)

The sheet shows the 60 R1 ∪ R2 reals with the highest fused s: the 44 flagged items (s ≥ τ 0.8252) and 16 unflagged near misses
(s 0.813–0.825, captioned "below tau"). Tiles are about 200 px, so heavy denoising and fine demosaicing traces cannot be judged
at this scale. Item 1 (`08c4d92ca878`) was also read at full size (1024×678). It shows no smearing, halo or ringing.

| Cause | Items | Evidence |
|---|---|---|
| Computational photography | 0 | No smartphone tone mapping, portrait blur edges or night-mode texture visible. The smartphone source `commons-r2-smartphone` has 0 fused flags (0/272). |
| HDR | 0 | No halos along high-contrast skylines at tile scale (for example items 37, 51, 53: bright sky, sun, snow). |
| Heavy denoise | 0 established | Not judgeable at tile scale. Item 1 at full size shows no denoise smearing. |
| Platform recompression | 0 | All 58 synthbuster items are lossless PNG. The two Commons items are q94 JPEGs at 1.2–1.3 MP with camera EXIF (a downscaled upload, not a platform re-encode). |
| Other: format shortcut (lossless, no EXIF, ≈ 0.7 MP, the fakes' format) | 58 | Everything except items 4 and 13. Ordinary daylight, museum-interior and landscape photographs from `synthbuster-plus-test` (licence field: RAISE, research use). Median context prior 0.886; detector terms near zero. |
| Other: face-swap detector on an occluded face | 2 | Items 4 (`0b88de624ecc`, s 0.944) and 13 (`ce7c4ed70ece`, s 0.894) share one `group_key`. Both show a person wearing a face mask (item 4 at a podium, item 13 receiving an injection). The manifest records no face (`face_px` unset), yet `img.dfdc_b7` gives 0.979 / 0.803 (+41.5 / +13.6 pp); item 13 also gets `img.hf.ateeqq` 0.999 (+10.6 pp). Context prior 0.618 (q94, min side 925–982). |

## Decisions (D-41: retire, context-gate with named features, keep)

The decisions apply to the H-4 admissions and the H-7 refit. Named features are `Context` terms that the fuser already computes
(`has_face`, `face_px:*`, `jpeg_q:*`, `min_side<512`, `min_side>1024`, `has_camera_exif`, `upscaled`, `video_frame`).
`min_side>1024` is the nearest existing term for "≥ 8 MP" (8 MP at 3:2 is a min side of about 2,300 px), so H-7 may need a
finer resolution term before these interactions are fitted. A context gate becomes a fuser gate (`calibrate --gates`): the
face detectors use `has_face`, so they contribute nothing where the context has no face.

| Detector | R1 ∪ R2 FPR at τ_0.05 | D-41 check | Decision | Named features and reason |
|---|---|---|---|---|
| `face.gend_clip_l14` | 11.4 % (23/201; 7.7–16.6); ≥ 8 MP 14.7 % (21/143) | not established (n < 300) | context-gate | `has_face`, `face_px:*`. It scores only on faces, and 22 of 23 flags are `commons-r1-camera`. Refit only with face terms; the ≥ 8 MP rise goes to H-7 as a `min_side>1024` interaction. |
| `img.dfdc_b7` | 8.5 % (17/201; 5.3–13.1) | not established (n < 300) | context-gate | `has_face`, `face_px:*`. It drove both Commons false positives on masked faces while the context had no face, so its term must drop when no face is found. |
| `img.commfor384` | 5.2 % (54/1,035; 4.0–6.7) | clear | keep | The ≥ 8 MP rate (8.5 %) goes to H-7 as a `min_side>1024` interaction. |
| `img.dda` | 4.8 % (50/1,035; 3.7–6.3) | clear | keep | Flags concentrate on `commons-r2-smartphone` (43/272). H-4 reports the R2 stratum ΔFPR. |
| `img.hf.ateeqq` | 4.6 % (48/1,035; 3.5–6.1) | clear | keep | The R1 rate of 10.6 % (45/424) is not the D-41 unit. `commons-r1-camera` 44/310 goes into the H-4 per-stratum A4 report. |
| `img.hf.dima806` | 2.0 % (21/1,035; 1.3–3.1) | clear | keep | R5 15.1 % is judged by the H-4 A4 rule, not the camera rule. |
| `img.hf.haywoodsloan` | 3.2 % (33/1,035; 2.3–4.4) | clear | keep | none |
| `img.forensic.spectrum` | 7.6 % (79/1,035; 6.2–9.4); ≥ 8 MP 10.3 % (60/585; 8.1–13.0) | clear | keep | The high-resolution sub-slice holds ≥ 300 reals and its point estimate is above 10 %. Revised at H-4: this row first said "context-gate `min_side>1024`". A hard gate on that term would switch spectrum off for most camera originals (a 1,024 px min side is far below 8 MP). Its fused contribution on the flagged reals is +0.0 pp, so H-7 tests `min_side>1024` as an interaction instead. |
| `img.forensic.ela` | 0.6 % (6/1,035; 0.3–1.3) | clear | keep | R3 12.9 % goes to the H-4 A5 context terms (`jpeg_q<70`, social). |
| `img.forensic.noise` | 1.7 % (18/1,035; 1.1–2.7) | clear | keep | R3 12.3 % handled as for ELA. |
| `img.forensic.copymove` | fires on 1.3 % (13/1,036; 0.7–2.1) | n < 300 scored | keep (partial-edit lane) | The τ tie makes the conditional rate meaningless. H-6 uses the derived `pe.copymove.pairs`. |
| `img.forensic.jpeg_dq` | fires on 1.2 % (12/1,036; 0.7–2.0) | n < 300 scored | keep (partial-edit lane) | Same tie. H-6 uses `pe.jpeg_dq.support`. |
| `loc.mesorch` | no probability score (aggregate is a score without a probability scale) | not measurable | keep (partial-edit lane) | H-6 derived features `pe.mesorch.*`; localiser FPR on reals is measured in H-6. |
| `img.meta.*`, `img.watermark.dwtdct`, `prov.c2pa`, `img.faces.yunet` | no numeric score on any real | not measurable | keep | Metadata, provenance and face-finding roles; no p(AI) on reals to threshold. |
| **Fuser (v2.1 refit, H-7)** | 4.2 % (44/1,036; 3.2–5.7); `synthbuster-plus-test` 36.8 % (42/114) | n/a | context-gate the prior | Lossless images without camera EXIF (`jpeg_q` absent, `has_camera_exif` off) must not carry the verdict. Refit with reals of that context present in fit (the synthbuster reals) and report "PNG / lossless" as its own EA-02 slice. If the lossless-context prior still exceeds τ on reals of that slice, the slice stays outside the validated envelope. |

## Limitations

- **Source mix in R1:** R1 mixes Commons camera uploads with 114 synthbuster reals whose provenance is research-use RAISE
  imagery. The fused R1 rate (10.4 %, 44/424) is mostly a property of that one source.
- **Tile scale:** the denoise and demosaicing classification is limited by the tile scale. Only one item was read at full size.
- **Stale prior:** the v2.0 fuser was fitted on vb-1.0-core. Its context prior may shift once H-7 refits on vb-1.1 fit.
- **Leaked items:** the two test ↔ fit / cal duplicates noted under H-2 (OpenFake ↔ Rapidata ideogram) are fakes and do not affect these real-photo rates.
