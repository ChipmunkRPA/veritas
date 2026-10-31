# Admission record: Veritas probe head v1 on vb-1.1 (BUILD_SPEC H-5)

Bench `vb-1.1`, manifest sha256 `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`, 2026-09-15. Automated
record (ralph loop), pending human review. Criteria ML §3.2 A1–A6 with the D-44 A4 / A5 rules. The I-13 record on
vb-1.0-core (CLIP only, no fused ablation) is `veritas-probe-v1.md`.

**Decision: rejected (A4 fails; A1 not established).** The PE-Core head adds +8.1 pts to the fused LGO macro TPR@5%FPR
(95 % CI +5.5 to +10.6) and lifts Seedream 4 (F5) from 75.0 % to 95.0 % on held-out refits. Adding it also establishes harm
on the R4 web / stock reals: fused out-of-fold FPR +4.6 pts (1.2 % → 5.9 %, paired 95 % CI +2.6 to +6.9, n 495). Established
harm in a stratum of ≥ 150 reals blocks (D-44), so the head does not enter the v2.1 image fuser. `test` was not touched.

## What was run

| Step | Job and command | Output |
|---|---|---|
| Features, pool | `h5-features` (2026-09-15 00:22–01:30Z): `probe_features.py --backbone <b> --manifest data/vbench/manifest.parquet --splits fit,cal --views 3 --fp16` for siglip2_so400m, pe_core_l14, clip_l14 | D-12 permissive pool of vb-1.1 fit ∪ cal: 1,823 items × 3 views; siglip2 5,469 extracted, pe_core 5,469, clip 4,788 + 681 cached |
| Backbone selection | `h5-eval` (01:30–01:31Z): `eval_probe.py --inner auto` per backbone → `data/vbench/probe-eval/vb-1.1-<b>.json` | nested LGO heads (below); selected `pe_core_l14` = highest LGO macro TPR at τ |
| Features, scored bytes | `h5-scored` (03:43–06:14Z): `probe_score.py extract --backbone pe_core_l14 --manifest data/vbench/manifest.parquet --scope data/vbench/scope/vb-1.1.parquet --robustness --views 3 --fp16` | 17,166 tasks × 3 views: norm/none 29,052 views; native none and 15 transforms 1,800 views each (jpeg95, launder, rephoto_proxy, screenshot 228; exif_forge 1,749; upscale2x 1,728 + 57 not applicable) |
| Heads and scores | `h5-score` (06:15Z): `probe_score.py score --backbone pe_core_l14 --manifest data/vbench/manifest.parquet --scope data/vbench/scope/vb-1.1.parquet --robustness` | heads production (1,823 items, C 1), LGO F1–F7, out-of-fold 0–4 (nested LGO inner CV); `img.veritas_probe.pe_core_l14@0a7c2ea015d5`: 17,147 ok, 19 not applicable. Rows of a family's fakes come from that family's LGO head, pool items from their out-of-fold head, every other row from the production head (ADR 0006) |
| A2, A5, A6 | `admission_evidence.py --manifest data/vbench/manifest.parquet --scope data/vbench/scope/vb-1.1.parquet --scores data/vbench/scores --detector img.veritas_probe.pe_core_l14 --threats full_synthesis --out data/fusion/h5-evidence` | `vb-1.1/admission-evidence-img.veritas_probe.pe_core_l14.json` |
| A3a, A3b, A4 | `h5-ablate2`: `calibrate.py --manifest data/vbench/manifest.parquet --scores data/vbench/scores --modality image --seed 7 --out data/fusion/h5-ablation --bench vb-1.1 --context scored --scope data/vbench/scope/vb-1.1.parquet --detectors img.commfor384,img.dda,img.dfdc_b7,img.veritas_probe.pe_core_l14 --ablate img.veritas_probe.pe_core_l14 --ablate-bootstrap 2000 --gates "img.dfdc_b7=has_face"` | `vb-1.1/ablation-img.veritas_probe.pe_core_l14.md`; baseline = the H-4 admitted set |
| Decision | `admission_record.py` (inputs `vb-1.1-inputs.yaml`) | row in `vb-1.1-admissions.md` |

- **Training safeguards** (ADR 0006, `train_probe.py` / `probe_score.py`): nested LGO inner CV; class-blind format-parity views
  (scale 0.5–1.0, 448 px crop, JPEG q 60–100); refusal of any `test` item and of LGO / out-of-fold overlap with the scored rows.
- **Scope.** The ablation reads only vb-1.1 scope items (`calibrate --scope`). A first run on every scored fit ∪ cal item
  (`h5-ablate`, history `data/fusion/h5-ablation-unscoped/`) included 223 fakes of the interim scoring pass outside the scope.
  The probe has no rows for them while every real has one, so its missing indicator encoded the label (A3a +9.4 pts there).

## Backbone selection (nested LGO on the permissive pool; `cal` pool held out: 66 reals)

TPR at τ_0.05 of each held-out head, fakes detected / held-out fakes.

| Backbone | F1 Midjourney | F2 FLUX.2 | F3 GPT-image-1.5/2 | F4 nano-banana | F5 Seedream 4 | F6 Qwen-Image | F7 Imagen 4 | no family | LGO macro AUC | LGO macro TPR at τ | macro TPR at zero held-out FP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| siglip2_so400m | 47/48 | 25/25 | 20/20 | 45/45 | 42/42 | 56/56 | 31/32 | 437/439 | 0.9982 | 0.9926 | 0.9613 |
| **pe_core_l14** | 48/48 | 25/25 | 19/20 | 45/45 | 42/42 | 56/56 | 32/32 | 438/439 | 0.9984 | 0.9929 | 0.9869 |
| clip_l14 | 46/48 | 23/25 | 12/20 | 38/45 | 37/42 | 50/56 | 28/32 | 379/439 | 0.9878 | 0.8531 | 0.6256 |

The pool's reals and fakes come from the same permissive sources, so these numbers show separation inside the pool, not
generalisation to other real sources (66 held-out reals; τ_0.05 there admits 3 false positives). The fused ablation below is
the evidence that decides.

## Criteria (vb-1.1 fit ∪ cal, scope items: 5,001 items, 2,697 real, 2,304 fake)

| Criterion | Result | Evidence |
|---|---|---|
| A1 licence and parity | not established | backbone `pe_core.large_p14_336` class commercial (Apache-2.0); training pool D-12 permissive only; CPU vs MPS parity of the backbone not recorded (I-13 measured fp16 vs fp32 parity: min cosine ≥ 0.99999) |
| A2 stand-alone AUC | pass: 0.9216 [0.9061, 0.9352] | full-synthesis threats (t2i, personalised, gan_face) vs every real of fit ∪ cal |
| A3a marginal fused value | pass: +8.1 pts, CI [+5.5, +10.6] | LGO macro TPR@5%FPR 78.3 % → 86.3 % |
| A3b coverage | pass | families gpt-4o-image-1 (+17.1 pts [+7.3, +29.3]), gpt-image-1.5-2 (+12.3 [+1.8, +22.8]), seedream4 (+13.8 [+6.2, +21.5]) |
| A4 no harm to reals | **fail** | pooled ΔFPR −0.0 pts, upper bound +0.91 (passes ≤ +1.0); R4 blocks: +4.6 pts [+2.6, +6.9], n 495 (established harm, D-44); R5 +2.9 pts [+1.1, +4.9] without established harm (lower bound ≤ +2.0) |
| A5 robustness | pass | ΔTPR jpeg75 +1.7 pts, social_ig +0.0 pts (n 120 each) |
| A6 latency | pass: p95 836 ms (n 9,684) | cached per-call runtime of the scored norm rows (backbone forward pass plus head) |

## Per-family recall

Fused system (H-4 admitted set) without and with the probe. LGO: refits without F, TPR@5%FPR on F's fit ∪ cal fakes.
Out-of-fold: 5-fold cross-fit, τ_0.05 from the out-of-fold reals.

| Family | LGO n | LGO without | LGO with | Out-of-fold n | without | with | Δ pts [95 % CI] |
|---|---|---|---|---|---|---|---|
| F1 Midjourney | 153 | 98.7 % | 98.7 % | 153 | 98.7 % | 98.7 % | +0.0 [+0.0, +0.0] |
| F2 FLUX.2 | 70 | 58.6 % | 62.9 % | 70 | 60.0 % | 68.6 % | +8.6 [+0.0, +17.3] |
| F3 GPT-image-1.5/2 | 57 | 52.6 % | 64.9 % | 57 | 57.9 % | 70.2 % | +12.3 [+1.8, +22.8] |
| F4 nano-banana | 118 | 83.9 % | 92.4 % | 118 | 86.4 % | 94.9 % | +8.5 [+3.4, +13.7] |
| F5 Seedream 4 | 80 | 75.0 % | 95.0 % | 80 | 85.0 % | 98.8 % | +13.8 [+6.2, +21.5] |
| F6 Qwen-Image | 106 | 84.9 % | 92.5 % | 106 | 85.8 % | 94.3 % | +8.5 [+1.9, +15.0] |
| F7 Imagen 4 | 106 | 94.3 % | 98.1 % | 106 | 97.2 % | 99.1 % | +1.9 [+0.0, +4.9] |
| GAN faces | – | – | – | 153 | 64.7 % | 58.8 % | −5.9 [−11.1, −0.7] |
| partial edits | – | – | – | 161 | 84.5 % | 82.0 % | −2.5 [−6.2, +1.2] |

Midjourney (F1) is already detected by the admitted set; among the LGO families Seedream 4 (F5) gains most. GAN faces lose recall.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 camera originals | 424 | 9.7 % | 6.8 % | −2.8 | [−7.8, −0.6] |
| R2 smartphone | 612 | 0.3 % | 0.0 % | −0.3 | [−1.1, +0.0] |
| R3 social | 589 | 14.8 % | 10.4 % | −4.4 | [−7.3, −1.5] |
| R4 web / stock | 495 | 1.2 % | 5.9 % | **+4.6** | [+2.6, +6.9] |
| R5 faces | 549 | 3.6 % | 6.6 % | +2.9 | [+1.1, +4.9] |
| R6 hard negatives | 28 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |

## Open items

- The R4 harm needs a cause before a v2 probe: R4 is COCO / Flickr30k / OpenFake `imagenet` and `docci` reals, and the pool's
  reals are camera and smartphone photographs. Adding R4-like reals to the permissive training pool is the next lever.
- A1 needs a CPU vs MPS parity run of the PE-Core backbone (`tests/test_detector_parity.py` covers detectors, not backbones).
