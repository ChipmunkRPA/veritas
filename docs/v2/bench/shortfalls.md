# Shortfalls (D-02)

Bench `vb-1.1` (D-40), manifest sha256 `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`, scored scope sha256
`2041bf678721…` since J-4 (H-2 scope `5059434a74cf…`; `vb-1.1_scope.md`). The B-5 … Q-5 state entries below refer to `vb-1.0-core`
(manifest `d0224343a5c0…`); every goal table refers to vb-1.1. Automated record (ralph loop), pending human review.

D-02: every goal of `goals.yaml` either passes the final gate or appears here with the measured value and its interval, every
mitigation attempted (named BUILD_SPEC items) and the product consequence (the family or slice is "not validated" and its
decisions are capped at `inconclusive`).

## State of this file

- **2026-09-14, B-5.** Created with the bench data shortfalls and the D-39 deferrals. The v0.3 baseline gate on core (24 PASS,
  23 FAIL, 11 MISSING of 58 checks) is in `baseline-v0.3.md`: it is the reference the v2 system is compared with, not a v2
  shortfall. Goal entries are added from the v2 gate runs by I-14 (image), V-8 (video), A-1 (audio) and Q-5 (final gate).
- **2026-09-14, I-14.** Image goal entries from the `fuser-image-2.0.0-alpha` gate run (`gate-image-2.0.0-alpha.txt`: 14 PASS,
  18 FAIL, 26 MISSING of 58 checks; video, audio and latency goals are MISSING in this run by design). Record:
  `admissions/fuser-image-2.0.0-alpha.md`.
- **2026-09-14, V-8.** Video goal entries from the `fuser-video-2.0.0-alpha` gate run (`gate-video-2.0.0-alpha.txt`: 15 PASS,
  19 FAIL, 24 MISSING of 58 checks; the image rows are those of I-14, audio and latency stay MISSING). Record:
  `admissions/fuser-video-2.0.0-alpha.md`.
- **2026-09-14, A-1.** Audio goal entries from the `fuser-audio-2.0.0-alpha` gate run (`gate-audio-2.0.0-alpha.txt`: 16 PASS,
  20 FAIL, 22 MISSING of 58 checks; image and video rows as in I-14 and V-8, latency stays MISSING). Record:
  `admissions/fuser-audio-2.0.0-alpha.md`.
- **2026-09-14, X-4.** Latency goals from `make bench` (`perf.md`, passed to the report with `--latency ../docs/v2/bench/perf.json`):
  G13a PASS (1.10 s), G13b PASS (8.23 s) on the `full` profile, host not idle (1-minute load 9–19 on 18 CPUs). Gate run
  `gate-x4-perf.txt`: 18 PASS, 20 FAIL, 20 MISSING of 58 checks; the other rows are those of the A-1 run. No latency shortfall.
- **2026-09-14, Q-5.** `make final-gate` runs `make ci`, the goal table (FAIL and MISSING are allowed, a refusal is not) and
  `scripts/check_shortfalls.py`. The checker requires, for every goal below target, a row here with a value, an interval or
  "no interval", a mitigations paragraph naming BUILD_SPEC items and a consequence, plus an entry in
  `backend/app/engine/fusion/not_validated.yaml` with its cap and its statement in `docs/limitations.md`. Rows without a
  computed interval now say so; the stale G13 MISSING row is removed (X-4 measured PASS). Field test (Q-3,
  `docs/field_test.md`): no AI or authentic decision was published for the 23 field images and 8 field videos; all were
  inconclusive, 10 images capped by the envelope rule, the others with both classes in the conformal set.

- **2026-09-15, H-7.** Image goal entries from the `fuser-image-2.1.0` gate run on vb-1.1 (`gate-image-2.1.0.txt`: 36 PASS,
  10 FAIL, 12 MISSING of 58 checks). Video and audio rows are MISSING in this run by design until H-8, and the latency rows until
  H-10. The I-14 image table is replaced; G1a, G1b, G2, G4a, G4c, G6b, G7b, G8 and G11 pass on vb-1.1. Record:
  `admissions/fuser-image-2.1.0.md`.
- **2026-09-15, H-8.** One report of the three v2.1 fusers on vb-1.1 `test` (`gate-v2.1.txt`: 36 PASS, 14 FAIL, 8 MISSING of
  58 checks; job `h8-report`). The threshold rule now pools real strata below 19 `cal` reals (ADR 0007): image R6 (11 `cal`
  reals) joins R3, video V-talking-head (4) joins V-general. Image rows are re-measured with the same artefact:
  - G1b (F3 47.7 %) and G7b (69.7 %) fall below target;
  - G5c jpeg50 is now +4.7 pts;
  - the other image verdicts are unchanged.
  Video (G9) and audio (G10) rows replace the V-8 and A-1 tables. Records: `admissions/fuser-video-2.1.0.md`, ADR 0007.
- **2026-09-15, H-9.** Field test v2.1 (`docs/field_test.md`, D-42): on 163 items through the live API the decision rate is
  ≥ 50 % for images and videos in both sets (new stills 71/80, new clips 19/32), so D-42 adds no shortfall row. Every
  inconclusive had both classes in the conformal set; no decision was capped by the envelope.
- **2026-09-15, H-10.** `make bench` re-run on v2.1 (`perf.md`, job `h10-bench`) and the vb-1.1 report re-run with
  `--latency` (`gate-v2.1-final.txt`, job `h10-report`): 38 PASS, 14 FAIL, 6 MISSING of 58 checks. G13a and G13b pass, so the
  latency rows say PASS and the `latency` entry leaves `not_validated.yaml`. The other rows are unchanged from H-8. The bench
  data table now describes vb-1.1.
- **2026-09-17, J-3c1.** Image admissions under D-53 (`admissions/v2.1.1-passes.md`, ledger `admissions/v2.1.1c1-elimination.json`):
  - the D-51 runs collapsed: the all-at-once bases went {commfor384, dda, dfdc_b7 has_face} → four members → {gend has_face} →
    {ateeqq !has_face} → ∅ (harness LGO macro 44.1 → 53.8 → 3.3 → 39.5 %), and the removal-order run reached a two-member base
    whose "without" system ran at a 16.27 % served-rule FPR. Both runs stay committed as diagnostics;
  - the D-53 elimination stopped at step 4 with the end set {dda, dfdc_b7 has_face, ateeqq !has_face, dima806}: gend removed at
    step 1, ateeqq gated `!has_face` at step 2, commfor384 removed at step 3. The joint check against the served set passes
    (Δ LGO macro +14.3 pts [+10.7, +17.8], A4 pass, end-set served-rule FPR 4.96 %). J-5 fits `fuser-image-2.1.1` on that set;
  - disclosed there: dima806 is admitted on A3b only; the joint check reads R3 +2.90 pts [+1.03, +4.86] and R6 +14.3 pts
    [+3.33, +29.17] on 28 reals. R6's interval excludes +2.0 pts; its harm counts as "not established" only because R6 holds
    fewer than 150 reals (D-51), so it is left to the G3 test FPR. fit ∪ cal figures are optimistic after selection;
  - levers outside the fixed pool: `img.veritas_probe.pe_core_l14` (A1 CPU vs MPS parity not recorded) and `img.hf.haywoodsloan`
    (A1 licence class unknown). Recording the parity or clearing the licence would make each a candidate for a later pool.
- **2026-09-17, J-5.** One report of the v2.1.1 fusers on vb-1.1 `test` (`gate-v2.1.1.txt`: 26 PASS, 26 FAIL, 6 MISSING of 58
  checks; jobs `j5-chain` at ac9ca5c, `j5-score-dima806`, `j5-report2`). Every v2.1 image figure is superseded.
  - **Size shortcut (D-45).** `fuser-image-2.1.0` learned "large image means real" (J-2 probe: `min_side>1024` −6.30,
    `min_side<512` −4.32; context-only logit on `cal` AUC 0.7897 in the J-5 pre-check). `fuser-image-2.1.1` carries no additive
    size or face term, one effective Platt intercept and resolution-balanced weights: gate 2b 0.5000, no remaining additive term
    for gate 2c. Without the shortcut the full `image/test` AUC is 0.9014 (2.1.0: 0.9632), and the full-view LGO macro TPR at
    τ_0.05 falls to 0.5908 (2.1.0 under ADR 0007, full view: 0.7716); the size-matched macro, which G1a reads, is 0.5837.
  - **Views.** G1, G2, G4, G6 and G7 read the size-matched `image/test` view (958 reals, 820 fakes); the full view (2,152 reals,
    1,843 fakes) is secondary; G3 reads the full real strata.
  - **New image rows below target:** G1a, G3 R2 / R4 / R5, G4a, G4c, G6a, G6b, G7a, and G5a / G5b / G5c on more transforms.
    G3 R1 and G6c now pass. The 2.1.0 image shortfalls table is in git history (commit 0aab936).
  - **Video.** The served `fuser-video-2.1.0` failed gate 2b (0.8160 > 0.55): video fits now use `policy.VIDEO_2B` (no additive
    size or face terms, one effective intercept), and `fuser-video-2.1.1` reads 0.5000. Audio (0.5000) is refitted on the J-4 scope
    without a policy change.
  - **Robustness scoring.** The J-4 pass ran without `img.hf.dima806`, so the first J-5 report (job `j5-chain`) left every G5,
    G6c and G11 row MISSING; job `j5-score-dima806` scored it on the 600 robustness items (none and 11 transforms: 600 each,
    upscale2x 588 of 598, exif_forge 573 of 573) before `j5-report2`.
  - **Field test** (`docs/field_test.md`, job `j5-field`, in-distribution sample): decision rates stay ≥ 50 % (images 16/23 and
    42/80, videos 7/8 and 18/32), so D-42 adds no row. Image decision accuracy fell (original 13/16, v2.1 set 35/42; H-9: 22/22,
    66/71): 8 real camera originals were called AI (short side 1,080–4,160 px) and 2 FLUX.2 stills authentic. No finished analysis
    published without an estimate (D-49).
- **2026-09-22, K-1.** Decision figures under the served abstention policy. ad26e7a (2026-09-18, after the J-5 report) made a
  runtime slice lacking either class abstain. The published report is now the logged J-11 lockbox run (`test_access.log` line 19,
  served code `b4a3b2e`, artefact shas unchanged, no `--write-artefact`), not a new read of `test`. AUC, TPR, FPR and ECE are
  unchanged; G6b reads 32.7 % (was 28.3 %), G6c 34.3 % (was 32.4 %, still PASS); the tally stays 26 PASS, 26 FAIL, 6 MISSING.
  Video: 5 of 19 runtime leaves hold both classes, and every 480–719 px leaf has 0 reals, so 58 of 135 test AI videos are always
  inconclusive (48.5 % of `video/test`). Field test re-run through the live API (job `k1-field`, `docs/field_test_k1.json`): the
  v2.1-set decision rates fall below 50 % for images and videos, so D-42 rows are added below. Record:
  `docs/v2/qa/k1-served-policy/README.md`.
- **2026-09-22, K-4.** Documentation honesty sweep of the release review (`docs/v2/qa/v211-final-ml-audit.md`); no model,
  artefact or goal row changed. Latency rows carry the J-11 figures; the reuse of `test` is counted below ("Post-hoc rule
  changes"), and the J-5a probe's read is added to `test_access.log`, marked retroactive; the video face-presence prior gets its
  own section with the vb-1.2 lever; the 2a row no longer calls the field test a check of container geometry. Next-phase item:
  the video and audio fuser detectors are fitted and on by default while their admission records are pending, and
  `vid.synth.frames.commfor384` fails A3a and A4 while its image twin was rejected. Record: `docs/v2/qa/k4-docs/README.md`.
- **2026-10-10, N-20a.** Decision figures under the served policy with the decision rules of section N (D-64, D-65, D-76,
  D-77). One new read of `test`: `scripts/j5_report.sh --figures` (job `n20a-report`, `test_access.log` line 21, source
  `ffcca70b`, the content of the scored image files measured for D-65), artefact shas unchanged (no `--write-artefact`). AUC,
  TPR, FPR, ECE, Brier and the conformal coverage (G6a) are unchanged: the rules change which items are decided, not the
  scores or the sets. G6b reads 39.7 % (was 32.7 %), G6c 39.1 % (was 34.3 %, still PASS); `video/test` is 50.2 % inconclusive
  (was 48.5 %), `audio/test` unchanged at 72.8 %. The decision error rate (selective risk) of `image/test` falls from 10.4 %
  (125/1,197) to 9.2 % (99/1,073); video 4.7 % (7/148, was 7/153); audio unchanged. The tally stays 26 PASS, 26 FAIL,
  6 MISSING. What each rule changed: "Post-hoc rule changes" below. Record: `docs/v2/qa/n20a-record/README.md`.

## Field decision rate below 50 % (D-42, K-1)

Live API, `full` profile, served v2.1.1 artefacts, 2026-09-22 (job `k1-field`). Small samples, no intervals. Every decision the
envelope rule newly capped was correct in the J-5 run (same p(AI) to six decimals); capping only suppresses decisions.

| Kind, set | Decision rate | J-5 (before ad26e7a) | Causes of inconclusive | Next levers |
|---|---|---|---|---|
| image, v2.1 set | 38/80 (48 %) | 42/80 (52 %) | conformal band 38; envelope cap 4, all real camera originals in `image/q70-89/side>1024` leaves, which hold no AI validation images | AI images at JPEG q 70–89 above 1,024 px in `fit`/`cal` (vb-1.2); slice the bench decision figures by the submitted bytes as the runtime does (see below) |
| video, v2.1 set | 13/32 (41 %) | 18/32 (56 %) | conformal band 14; envelope cap 5: four AI clips at 480–719 px (0 reals in those leaves) and one real clip in `video/res>=720/dur<4s/face>=256` (0 fakes) | real videos at 480–719 px and short AI clips at ≥ 720 px in `fit`/`cal` (vb-1.2) |

The original set stays at or above 50 % (images 15/23, videos 5/8; 1 and 2 envelope caps). The envelope rule is not relaxed
(D-42). **Bench–runtime slicing gap:** the bench report places test images in the runtime slice of the scored `norm` re-encode
(always JPEG q ≥ 90; ADR 0004/0005), while a live upload is sliced by its own bytes. The bench inconclusive rates therefore do not
include the envelope caps that camera originals at JPEG q 70–89 receive in the field.

## Post-hoc rule changes (disclosed, J-5)

The rules below changed after results on vb-1.1 `fit` ∪ `cal` were seen. Each was decided by the supervisor (D-##) and applied to
`fit` ∪ `cal` only; nothing was fitted on `test`. `test` itself is not a fresh lockbox (count below), so the v2.1.1 `test` figures
are re-used-test, post-selection figures, and vb-1.2 needs a fresh lockbox.

Reads of vb-1.1 `test` (`test_access.log`, one line per report run unless stated):
- 8 v2.1 report runs (lines 9–16, 2026-09-15);
- the v2.1 ML audit's size-free refit, evaluated on `test` (`docs/v2/qa/v21-ml-audit.md`: G1a 0.5648, R2 25/409), before D-45;
  not logged (a supervisor-side read);
- the two J-5 report runs (lines 17–18, 2026-09-17), after the v2.1.1 design had been chosen with the v2.1 test figures known;
- the J-5a operating-point probe (`scripts/v2/probes/j5a_operating_point.py`, job `j5a-probe`, 2026-09-17T12:48Z), which read
  `test` reals without the report CLI: logged retroactively by K-4 (line 20);
- the J-11 report run (line 19, 2026-09-22), the source of the K-1 served-policy figures;
- the release audit's class cross-tabs on the score cache (`docs/v2/qa/v211-final-ml-audit.md`), not logged (a review-side read);
- the N-20a report run (line 21, 2026-10-10), the source of the figures under the decision rules of section N below.

**Decision rules of section N (N-20a).** Four rules decided after the v2.1.1 `test` figures were known change which results
are decided; none changes a score, a set, a fit, a threshold or an artefact byte (PR-6). The counts are the N-20a report's
(`results.md`, "Decision rules"): items whose set at α = 0.05 holds one answer inside the slice envelope and that the rule makes
inconclusive. An item under two content conditions is counted under each; an item that D-76 or D-77 takes is counted under the
one rule that is its result's reason (D-77 is asked first, then D-76, then D-76 as amended).

- **D-64** (N-18, N-19f): a strong code needs two checks that count (that entered the estimate); with none that entered the
  result is not assessable. The report's decisions are at α = 0.05, the level of the moderate codes, so the first rule changes
  none of them: it concerns 184 `image/test` items with fewer than two checks, of which none would have had a strong code,
  and the 706 audio items, whose model has one check (69 of them would have had one). No item of `image/test` or
  `video/test` is without a check that entered, and one audio item is, which was not decided before either: the second
  rule changes no decision.
- **D-65** (N-18): content the bench does not hold (flat blocks above 53.46 % of the 8 × 8 blocks, the screenshot rule,
  greyscale) is inconclusive. `image/test`: 37 items under a condition (34 real, 3 AI), 23 decisions capped; full view 68
  items, 41 decisions capped; under `screenshot_jpeg` 305 items, 44 decisions capped.
- **D-76** (N-19i; amended 2026-10-05, built by N-19m): a decision needs one check that counts with a row of at least ln 2
  toward its answer, and the rows of the checks together at least ln 2 toward it. `image/test`: 39 decisions taken by the
  first condition, 19 by the second; full view 62 and 28; `video/test` 4 and 1; audio none.
- **D-77** (N-19m): a decision needs the estimate in its answer's part of the scale (34 or less for the clean answer, 65 or
  more for the AI answer, in whole points). `image/test`: 46 decisions (49 items, 41 real, 8 AI); full view 75; video and
  audio none.
- **Together:** `image/test` 705 of 1,778 inconclusive where K-1 read 581 (124 more), the full view 1,656 of 3,995 where it
  read 1,454 (202 more), `video/test` 149 of 297 where it read 144 (5 more); audio unchanged.
- **How D-76 and D-77 were chosen:** each was decided and measured on `fit` and `cal` with the packaged fusers and the recorded
  scores, without a read of `test` (`docs/v2/01-DECISIONS.md`, D-76 and D-77, "Measured"; `docs/v2/qa/prod-readiness-2026-09-29/supervisor-checks.md`,
  checks 18 and 25). The N-20a report is the first read of `test` under them. D-65's flat-block limit is the 99.5th percentile
  of the share over the `fit` ∪ `cal` images of both classes (`content_kinds.json`); `test` was not read to set it.


- **D-44** (H-4) amends the D-41 A4 / A5 rules: a stratum blocks only on established harm, and a point excess asks for a runtime term.
- **D-51** (J-3c): the ablation threshold merges strata with fewer than 19 `cal` reals (R6 → R3); only strata with ≥ 150 reals ask
  for an interaction term, and a term is kept only when ≥ 20 `fit` and ≥ 20 `cal` items of each class carry it; a candidate with
  established harm in a stratum with a learnable term is ablated again gated off where the term holds (gated retry).
- **D-53** (J-3c1): one change per elimination step, a lone member is never dropped, reals tied at τ are weighted to the rank rule's
  share (operating point at τ), and pooled reals block only on a paired lower bound above +1.0 pt.
- **Gated retry, both ablations disclosed:** `img.hf.ateeqq` at elimination step 2, ungated
  `admissions/v2.1.1c1-step2/ablation-img.hf.ateeqq.md` and gated `!has_face`
  `admissions/v2.1.1c1-step2-gated/ablation-img.hf.ateeqq.md`. The `img.dda` retry at the same step (same two directories) was
  admitted as well, but D-53 (b) applies only the admitted retry with the highest gated A3a lower bound (ateeqq, +4.6 pts), so
  dda stays ungated.
- **`policy.VIDEO_2B`** (J-5, under D-45 (a)): video stackers drop the additive size and face terms after gate 2b measured 0.8160 on
  `fuser-video-2.1.0`; the refit reads 0.5000. It keeps the missing-indicator terms, so a face-presence prior survives (section
  "Video face-presence prior" below).
- **D-52** (J-3d/J-5): on vb-1.1, a weighted gate 2a above 0.60 is disclosed instead of refused. The rule was set after 2a had
  measured a weighted AUC of 0.8490 on the `fuser-image-2.1.0` population (0.8582 on the served 2.1.1 artefact); the `D-45 2a` row
  below is that disclosure.

## Audio goals (J-5, `fuser-audio-2.1.1`, vb-1.1 `test`)

Mitigations attempted: A-1 fitted the voice fuser on `aud.gustking_xlsr` and rejected `aud.mo_thecreator_w2v2` (lower
out-of-fold AUC, `admissions/ablation-aud.mo_thecreator_w2v2.md`); H-2 scored every vb-1.1 audio row; H-8 refitted
`fuser-audio-2.1.0` with the scored context (`admissions/fuser-video-2.1.0.md`, "Audio"); J-5 refitted `fuser-audio-2.1.1` on
561 fit and 313 cal rows of the J-4 scope. Consequence: synthetic-voice detection is "not validated" for v2.1.1 (D-02).

| Goal | Measured (95 % interval) | Target | Note |
|---|---|---|---|
| G10a EER, `asvspoof5` | 17.4 % (86 reals, 90 fakes; no interval for the EER; AUC 0.8786 (0.8221–0.9283)) | ≤ 10 % | one learned detector; TPR at τ_0.05 38.9 % (35/90) |
| G10b in-the-wild bona fide FPR at τ_0.05 | PASS: 1.1 % (1/94; Wilson 0.2 %–5.8 %) | ≤ 10 % | TPR on the same set 1.1 % (1/89) at τ_0.05; EER 16.9 % |

## Video goals (J-5, `fuser-video-2.1.1`, vb-1.1 `test`)

Mitigations attempted: H-1 added OpenVid-1M reals and H-2 scored 350 video reals and 300 fakes (D-43), so τ_0.05 is established
from 74 `cal` reals; H-8 put the image-admitted Community Forensics and DDA detectors on native frames (on by default in
`full` and `deep`), recalibrated D3 on compressed real video (`d3_recalibration.md`, rejected as a fuser feature), ran the
video ablations under D-44 and ADR 0007 and refitted `fuser-video-2.1.0` with vendor-LGO refits
(`admissions/fuser-video-2.1.0.md`); J-5 refitted `fuser-video-2.1.1` under `policy.VIDEO_2B` (gate 2b 0.8160 → 0.5000) with
vendor-LGO refits (`admissions/fuser-image-2.1.1.md`, "Video and audio"). Consequence: video synthesis is "not validated" for
v2.1.1 (D-02).

| Goal | Measured (95 % interval) | Target | Note |
|---|---|---|---|
| G9a vendor-LGO macro TPR@5 %FPR | 0.5953 (no interval for the macro): V-Kling 84.2 % (16/19; Wilson 62.4 %–94.5 %), V-Seedance 36.8 % (7/19; Wilson 19.1 %–59.0 %), V-Sora 21.1 % (4/19; Wilson 8.5 %–43.3 %), V-Veo 55.6 % (10/18; Wilson 33.7 %–75.4 %), V-WanHunyuan 100.0 % (18/18; Wilson 82.4 %–100.0 %) | ≥ 0.80 | per-fold AUC 0.8268–0.9983; the pooled `video/test` TPR at τ_0.05 is 80.7 % (109/135); 2.1.0 read 0.6392 |
| G9b real-video FPR at τ_0.05 | PASS: 2.5 % (4/162; Wilson 1.0 %–6.2 %) | ≤ 5 % | τ_0.05 0.7330 from 74 `cal` reals |

### Video face-presence prior (K-4, release audit P1-1)

`fuser-video-2.1.1` has no additive size or face term (gate 2b 0.5000), but its missing-indicator term for DFDC-B7 is a
face-presence prior: `miss:vid.frames.dfdc_b7` = +0.3695 logit (bias +1.8719; the largest detector value weight is 0.3884), and a
video without a detected face leaves DFDC-B7 not run. "No face found" alone separates the video classes with AUC 0.7180 on `fit`
(55/100 AI, 13/114 real videos without a face) and 0.7007 on `cal` (34/65, 9/74; probe `scripts/v2/probes/k4_video_face_prior.py`,
record `docs/v2/qa/k4-docs/video_face_prior.md`, `test` not read), and 0.711 on `test` (81/135, 29/162) as measured by the release
audit. Gate 2b tests additive context terms only and cannot detect a prior carried by a missing-indicator term. D-50 names and
removes exactly this residual for images. vb-1.2 levers: apply D-50's face rule to video (a `has_face` gate on DFDC-B7, or its
missing weight fixed at 0), and extend gate 2b to missing-indicator terms. Consequence: part of the video fuser's separation may
come from face presence rather than from detector evidence; disclosed in `docs/limitations.md` ("Videos").

## Deferred to v2.1 (D-39)

| Item | State | Consequence |
|---|---|---|
| I-8 SPAI adapter and admission | `img.spai` (Apache-2.0) is built and `default_enabled: false`, deep profile only; no admission record was run | SPAI does not feed the image fuser; its spectral signal is not part of any calibrated verdict |
| Q-3 new field-test source collection | only the existing field manifest is re-run | field behaviour is known only on the sources already collected |

## Image goals (J-5, `fuser-image-2.1.1`, vb-1.1 `test`)

Mitigations attempted for every image goal below:
- H-1 grew the real strata (VISION natives and social copies, Commons top-up, HF face reals).
- H-2 scored every image real, 5,000 fakes and 11 robustness transforms.
- H-3 analysed the camera-original false positives.
- H-5 trained the Veritas probe head: A3a +8.1 pts, rejected on established R4 harm.
- H-6 fed the partial-edit derived features: rejected, no gain.
- H-7 and H-8 fitted and re-measured `fuser-image-2.1.0` (superseded).
- J-2 removed the image-size prior from the fit (D-45) and J-3c set the D-50 fit policy.
- J-3c1 re-ran the admissions under D-51 and D-53: end set DDA, DFDC-B7 on faces, ateeqq where no face is detected, dima806.
- J-4 drew the robustness subset from `test` only (600 items).
- J-5 fitted `fuser-image-2.1.1` on that set (`admissions/fuser-image-2.1.1.md`), τ_0.05 0.7150 and τ_0.01 0.9607 from 850 `cal` reals.
- J-5a measured the operating-point transfer against 2.1.0 like for like and its cause (`scripts/v2/probes/j5a_operating_point.py`).

Consequence for each FAIL or MISSING row: the goal's stratum, transform or family is "not validated" for v2.1.1 (D-02).

| Goal | Measured (95 % interval) | Target | Note |
|---|---|---|---|
| G1a LGO macro TPR@5 %FPR | 0.5837 (no interval: the report computes none for the fold macro) | ≥ 0.75 | size-matched folds: F1 Midjourney 81.2 % (26/32), F2 FLUX.2 40.0 % (6/15), F3 GPT-image 25.0 % (7/28), F4 nano-banana 64.7 % (22/34), F5 Seedream 4 83.3 % (55/66), F6 Qwen-Image 42.1 % (8/19), F7 Imagen 4 72.2 % (13/18); full view 0.5908, against 2.1.0's full-view 0.7716 with the size prior |
| G1b LGO minimum family TPR@5 %FPR | 0.2500: F3 gpt-image-1.5-2 25.0 % (7/28; Wilson 12.7 %–43.4 %) | ≥ 0.50 | F2 FLUX.2 40.0 % (6/15; Wilson 19.8 %–64.3 %) |
| G3 operating-point transfer (all strata) | τ_0.05 on test reals: 7.5 % pooled (162/2,152; Wilson 6.5 %–8.7 %), 7.6 % in the size-matched view (73/958; Wilson 6.1 %–9.5 %), 7.3 % stratum-weighted (R6 with R3; no interval); τ_0.01: 1.3 % pooled (29/2,152; Wilson 0.9 %–1.9 %), 1.8 % size-matched (17/958; Wilson 1.1 %–2.8 %), 1.4 % stratum-weighted | 5 % / 1 % | τ_α is set on `cal` reals with equal stratum weights and does not transfer to `test`: "TPR at 5 % FPR" names the calibration target, not the realised test rate. Like for like with 2.1.0 at its τ_0.05 (`docs/v2/qa/j5a_operating_point_probe.md`): full view 7.5 % vs 5.3 %, size-matched view 7.6 % vs 10.1 %, stratum-weighted 7.3 % vs 5.0 %; 2.1.0's lower full-view rate came with the size prior (D-45) and is higher on the size-matched view. Cause as measured: the `cal` reals hold unusually few high-scoring reals, mostly saturated `img.hf.ateeqq` scores (groups reshuffled between `cal` and `test` reals within strata: P(≥ 7.32 %) = 0.0070, B = 2,000); per-context Platt slopes add about 0.5 pt; no code, context or composition cause was found. vb-1.2 lever: set τ on out-of-fold scores of fit ∪ cal reals and report τ's grouped-bootstrap interval; τ is not re-tuned on `test` |
| G3 R2 FPR at τ_0.05 | 6.1 % (25/409; Wilson 4.2 %–8.9 %) | ≤ 5 %, upper ≤ 7.5 % | smartphone photographs; 0.0 % under 2.1.0 with the size prior; R1 passes at 3.6 % (14/387; Wilson 2.2 %–6.0 %) |
| G3 R3 | 11.7 % (56/479; Wilson 9.1 %–14.9 %) | ≤ 5 %, upper ≤ 7.5 % | social-platform native bytes (So-Fake-OOD reals, VISION Facebook / WhatsApp copies); 9.6–15.8 % in every min-side bucket with ≥ 95 reals |
| G3 R4 | 9.1 % (37/405; Wilson 6.7 %–12.3 %) | ≤ 5 %, upper ≤ 7.5 % | web and stock photographs; 11.6 % (20/172) below 384 px, 15.2 % (12/79) at 1025–1536 px |
| G3 R5 | 5.8 % (26/450; Wilson 4.0 %–8.3 %) | ≤ 5 %, upper ≤ 7.5 % | portraits; 8.6 % (21/245) below 384 px |
| G3 R6 | 18.2 % (4/22; Wilson 7.3 %–38.5 %) | ≤ 5 %, upper ≤ 7.5 % | Commons fetch frozen at 50 of 500 (bench data row below) |
| G3 screenshot | MISSING | ≤ 5 % | the `screenshot` transform is outside the D-40 scoring scope (11 transforms) |
| G4a ECE, test | 0.0537 (0.0432–0.0724) | ≤ 0.05 | size-matched view; full view 0.0519 (0.0421–0.0661) |
| G4b ECE, LGO macro | 0.1651 (no interval: the report computes none for the fold macro) | ≤ 0.08 | fold ECE 0.0794 (F5) to 0.2914 (F2) |
| G4c Brier, test | 0.1270 (no interval: the report computes none for the Brier score) | ≤ 0.12 | full view 0.1281 |
| G5a ΔTPR jpeg75 / webp80 / resize0.5 | 16.7 / 12.5 / 23.5 pts (no interval: the report computes none for fixed-threshold deltas) | ≤ 10 pts | 264 fakes; crop70 4.2 pts passes; jpeg95 MISSING (not in the D-40 scope) |
| G5b ΔTPR jpeg50 | 28.0 pts (no interval: fixed-threshold deltas) | ≤ 20 pts | social_ig 16.7, social_x 12.1, screenshot_jpeg 13.3, noise3 −0.8, upscale2x 0.0 pass |
| G5c ΔFPR resize0.5 / crop70 / screenshot_jpeg / noise3 | +3.0 / +2.0 / +21.1 / +22.8 pts (no interval: fixed-threshold deltas) | ≤ +2 pts | 298 reals; jpeg75 −1.3, jpeg50 −2.0, webp80 +1.0, social_ig −0.7, social_x −1.0, upscale2x +1.0 pass; exif_forge −0.3 is trivially satisfied (no metadata features); jpeg95, screenshot, launder, rephoto_proxy MISSING (not in the D-40 scope) |
| G6a conformal coverage, reals / AI | 91.5 % (877/958; Wilson 89.6 %–93.1 %) / 93.9 % (770/820; Wilson 92.1 %–95.3 %) | ≥ 94 % per class | the conformal thresholds come from `cal` like τ and share its transfer gap |
| G6b inconclusive, test | 39.7 % (705/1,778; Wilson 37.4 %–41.9 %), served policy with the decision rules of section N (N-20a) | ≤ 20 % | full view 41.5 % (1,656/3,995); G6c social_ig 39.1 % (220/562; Wilson 35.2 %–43.2 %) passes; K-1 read 32.7 % (581/1,778) before D-64, D-65, D-76 and D-77, and J-5 28.3 % (504/1,778) before ad26e7a made a slice without both classes abstain |
| G7a face-swap AUC vs R5 | 0.7819 (0.7384–0.8217) | ≥ 0.90 | full view 0.8139 (0.7774–0.8486); GenD, the face-swap detector, was removed at J-3c1 step 1 |
| G7b face-swap TPR@5 %FPR vs R5 | 51.4 % (95/185; Wilson 44.2 %–58.5 %) | ≥ 70 % | full view 54.5 % (115/211) |
| G7c GAN face TPR@5 %FPR vs R5 | 63.0 % (58/92; Wilson 52.8 %–72.2 %) | ≥ 85 % | AUC 0.8662 (0.8226–0.9073); full view 66.7 % (82/123) |

## Shortcut gate on vb-1.1 (D-45, D-52)

Gate 2a measures container geometry on the vb-1.1 population: no refit can pass it, and the remedy is data (vb-1.2). D-52
discloses a weighted 2a above 0.60 on vb-1.1 with this row (artefact sha12, weighted and unweighted out-of-fold AUC, each feature
group's weighted AUC). Record: `shortcut_gate.json` (job `j5-shortcut-gate`). The field test cannot check container geometry:
its v2.1 set is itself size-confounded (32 of 33 stills at 768–1,024 px are AI, 33 of 35 above 1,536 px are real;
`docs/field_test.md`), and no geometry-matched field set exists yet.

| Shortfall | Measured | Effect on the gates |
|---|---|---|
| D-45 2a | `fuser-image-2.1.1` sha12 `746ada66f73d`: weighted AUC 0.8582, unweighted AUC 0.9526; feature groups (weighted): min_side 0.6467, megapixels 0.5217, aspect 0.7634, div64 0.6272, face_context 0.4761 | disclosed under D-52 on vb-1.1; container geometry (square and divisible-by-64 sides are far more common among fakes) may be part of every vb-1.1 detection figure; no geometry-matched field set exists to check it (vb-1.2 removes the imbalance) |

## Latency (J-11)

Mitigations attempted: X-4 measured v2.0 (G13a 1.10 s, G13b 8.23 s on `full`); H-7 enabled `img.commfor384` and `img.dda` under
`full` and `deep`, and H-8 their native-frames twins; H-10 re-ran `make bench` with them; J-4 added a 24 MP camera original to G13a
and re-ran `make bench` (job `j4-bench`: G13a 4.12 s, G13b 9.22 s, v2.1 detector set); J-11 re-measured the served v2.1.1
full-profile detector sets under a predeclared coverage contract (`docs/v2/qa/j11-latency/`, source `b4a3b2e`). Consequence: none,
both rows pass. G13 specifies an idle host; the J-11 figures were measured on a loaded host (1-minute load 14–16 from other
projects), so they are loaded-host figures, not idle-host ones.

| Goal | Measured (95 % interval) | Target | Note |
|---|---|---|---|
| G13a image p95 | PASS: 4.697 s (no interval: nearest-rank p95 of 3–5 valid warm runs per workload) | ≤ 25 s | `full`, `image_24mp_camera` (6000 × 4000 R1 camera original, 3 runs); `image_800px` 1.575 s, `image_256px_face` 1.061 s (5 runs each); served v2.1.1 detector set; loaded host (1-minute load 14–16) |
| G13b video p95 | PASS: 11.877 s (no interval: 2 valid warm runs) | ≤ 120 s | `full`, `video_9s_1080p`; loaded host (1-minute load 14–16); `video_60s_1080p` was not re-measured (J-4, v2.1 set: 50.79 s) |

## Bench data (vb-1.1)

The `vb-1.0-core` table of B-5 is in git history (commit 1abee83 and earlier).

| Shortfall | Measured | Effect on the gates |
|---|---|---|
| Wikimedia Commons fetch frozen (B-4b, H-1; D-38, D-40) | planned vs fetched: R1 camera 649 / 1,300 (HTTP 429 at the 4 h time box), R6 hard captures 50 / 500, R2 smartphone 324 / 800, video camera 0 / 200 (`sources.yaml` `frozen` entries) | R6 holds 22 `test` reals: even 0/22 has a Wilson upper bound of 14.9 %, so G3 R6 cannot pass on vb-1.1 |
| Test reals below 300 (D-38) | audio A-bonafide 265, R6 22, video V-general 189, V-talking-head 9 (`vb-1.1_summary.md`) | wider intervals on G3 R6, G9b and G10b; ADR 0007 pools strata below 19 `cal` reals in τ_α (image R6 → R3, video V-talking-head → V-general) |
| Label audit exclusion (H-1) | `commons-video-general`, 15 reals, failed the agent audit | excluded from every split |
| Video scoring cap (D-43) | 350 of 435 audited video reals scored, 85 unscored (`vb-1.1_scope.md`) | video `test` holds 162 reals and 135 fakes; vendor-LGO folds hold 18–19 fakes, and their Wilson intervals are up to 41 pts wide |
| Transforms outside the D-40 scoring scope | `jpeg95`, `screenshot`, `launder` and `rephoto_proxy` have no rows for `img.dda` | G3 screenshot, G5a jpeg95 and G5c jpeg95 / screenshot / launder / rephoto_proxy are MISSING |
| Perturbation errors (J-4, J-5) | `upscale2x` 598 tasks, 588 scored, 10 failed (80 MP decode limit after 2×); `exif_forge` 573 tasks, all scored (13 webp natives refused by the perturbation) | the robustness table pairs 552 `upscale2x` and 538 `exif_forge` items with their native reference, against 562 for the other transforms |
| Cross-split duplicates (H-1) | 7 byte-identical groups (14 items, no label conflict); two OpenFake ideogram prompts span splits: test `edaa431e20` ↔ cal `009e00a005`, test `38c9223316` ↔ fit `c6c9d08273` | the two test items may score optimistically; no resplit (a namespace-free prompt key is planned for vb-1.2) |
| OpenVid-1M grouping (ML audit P2) | the video group key is the clip, not the YouTube source video: three sources (0R1JsNFK5nw, AwfJkaTUTtw, FT7mNxMMoi0) have clips in more than one split | clips of one source video may sit on both sides of the video `fit` / `test` boundary; vb-1.2 keys OpenVid clips by YouTube id |
| Disk (D-40) | 163–166 GiB free on the shared volume against "≥ 250 GB"; the fetcher stops storing below 100 GB; cache 40.54 GB of 55 GB | the Commons top-up ended at its time box under HTTP 429 throttling, not at the disk floor |
