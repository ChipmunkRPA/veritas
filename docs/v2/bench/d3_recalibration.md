# D3 σ threshold on compressed real video (vb-1.1, fit, cal)

- Command: `scripts/d3_recalibrate.py --manifest data/vbench/manifest.parquet --scores data/vbench/scores --eval-dir data/vbench/eval/d3_lane --bench vb-1.1 --out ../docs/v2/bench --constants app/engine/fusion/d3_sigma.json`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Score key: `vid.synth.d3_xclip@252607e85631`; V-4 runs: `data/vbench/eval/d3_lane` (native, crf)
- Real rows: 834 from 218 items (native file plus its re-encodes); reals without a σ 19, fakes without a σ 152 (not counted)
- Native σ source: score cache 353, V-4 native run 441; cache vs V-4 on 281 items: max |Δσ| 4.446449, median 0.210570; median σ 1.882691 vs 1.868137. Both runs score the same shots and clip counts, but the V-4 script seeded the sampling plan from sha256(item_id) and the production path seeds it from the media sha256, so the clips sit at other positions and σ differs per item; the threshold reads only the distribution of σ.
- Threshold: rank rule on s = −σ over every real row, encodes weighted equally; a row is flagged when σ ≤ σ_α. Fitted on these reals and reported on them (in-sample); no fake is used for fitting.

| Quantity | Value |
|---|---|
| σ_0.05 | 0.870586 |
| σ_0.01 | 0.568481 |
| s_σ = IQR / 1.349 | 0.719749 (IQR 1.4726–2.4436) |
| Stand-alone AUC of −σ, native reals vs fakes | 0.5815 |

Derived feature `vd.d3.sigma`: z = clip((σ_0.05 − σ) / s_σ, ±8).

## False-positive rate at σ_0.05 per encode

| Encode | n | median σ | at or below σ_0.05 |
|---|---|---|---|
| crf23 | 154 | 1.9032 | 3.9 % (6/154; 1.8–8.2 %) |
| crf28 | 154 | 1.9069 | 5.8 % (9/154; 3.1–10.7 %) |
| crf28-720p | 154 | 1.8487 | 5.2 % (8/154; 2.7–9.9 %) |
| crf35 | 154 | 1.9420 | 4.5 % (7/154; 2.2–9.1 %) |
| native | 218 | 1.9247 | 5.0 % (11/218; 2.8–8.8 %) |

## False-positive rate at σ_0.05 per real stratum (native files)

| Stratum | n | median σ | at or below σ_0.05 |
|---|---|---|---|
| V-general | 212 | 1.9247 | 5.2 % (11/212; 2.9–9.1 %) |
| V-talking-head | 6 | 1.8519 | 0.0 % (0/6; 0.0–39.0 %) |

## Recall at σ_0.05 per threat (native files)

| Threat | n | median σ | at or below σ_0.05 |
|---|---|---|---|
| t2v | 568 | 1.7153 | 7.2 % (41/568; 5.4–9.6 %) |
| talking_head | 8 | 2.4297 | 0.0 % (0/8; 0.0–32.4 %) |

All real rows: 4.9 % (41/834; 3.6–6.6 %). All fakes: 7.1 % (41/576; 5.3–9.5 %).
