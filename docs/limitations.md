# Limitations and responsible use

**Verdicts are probabilistic evidence, not proof of authorship or manipulation.** Every report says so and lists the
caveats and limitations that applied. Use a qualified human reviewer before any consequential decision (HR, legal, academic
integrity, journalism, fraud, insurance).

## Validation status (v2)

- **Benchmark.** Image goals are judged on `vb-1.1` (D-40), the superset of vb-1.0 with more real photographs. Its scored scope
  (`docs/v2/bench/vb-1.1_scope.md`) holds every image real and up to 5,000 image fakes. On `test` the manifest's real strata hold
  R1 403, R2 412, R3 479, R4 405, R5 450 and R6 22 items (`docs/v2/bench/vb-1.1_summary.md`); the rates below use the scored
  counts (R1 387, R2 409, R3 479, R4 405, R5 450, R6 22). Video (650 items, D-43) and audio (1,600 recordings) are judged on the
  same bench.
- **Re-used test split.** vb-1.1 `test` is not a fresh lockbox. It was read by 8 v2.1 report runs (`test_access.log` lines 9–16),
  by the v2.1 ML audit's size-free refit (`docs/v2/qa/v21-ml-audit.md`, unlogged), by the J-5a operating-point probe (logged
  retroactively), by the two J-5 report runs (lines 17–18), by the J-11 report run (line 19) and by the release audit's class
  cross-tabs on the score cache (`docs/v2/qa/v211-final-ml-audit.md`, unlogged). Nothing was fitted on `test`, but the v2.1.1
  design (D-45, D-50, D-52, D-53) was chosen after test-derived numbers had been seen, so every v2.1.1 `test` figure is a
  re-used-test, post-selection figure. vb-1.2 needs a fresh lockbox. Results, confidence intervals and shortfalls: `docs/evaluation.md` and
  `docs/v2/bench/shortfalls.md`.
- **Uncalibrated until fitted.** Until a fuser artefact is fitted and loaded for a kind, its decision comes from the fallback
  fuser: the probability is labelled uncalibrated and the verbal scale is capped. Image analyses under `full` and `deep` load
  `fuser-image-2.1.1`, video under `full` and `deep` loads `fuser-video-2.1.1`, and audio under every profile loads
  `fuser-audio-2.1.1`. `quick` image and video analyses and documents use the fallback (the image fuser needs `img.dda`, which
  `quick` does not run). The three artefacts ship with the source, so a fresh install (and an upgraded one that kept an older
  artefact in its data directory) serves them once the detector weights are installed; a server whose weights are missing, or whose default profile is `quick` (the default on CPU outside the
  Compose file), serves the fallback, and `GET /api/ready` with an API key says which serves per kind (`docs/runbook.md`, "Calibrated fusers").
- **Fusion models pending human review.** The three served fusers were fitted and checked by automated records; no person has
  signed a review of them yet. Every result and report that uses one says "Fusion model pending human review" until the owner
  records a review (`docs/v2/fuser-review-2.1.1.md`). A review changes no number. Lane probabilities on the video timeline and the voice lane of a recording are smoothed window
  probabilities, not calibrated probabilities.
- **The number a result shows.** A calibrated estimate is printed as a number, with a marker on the scale, only for a decisive
  result in which at least two checks scored. An inconclusive result, and a result in which one check scored, print the estimate
  and its range as text ("Estimate 53–78: inconclusive"). A result of the uncalibrated fallback prints "score n / 100, not
  calibrated", has no marker on the scale, and lists one sentence under NOT VALIDATED: no detection rate, false-alarm rate or
  calibration error was measured for the fallback, so none of the calibrated model's figures is quoted beside it. A result capped
  because the input is outside what was validated prints no number, no range and no marker; the reason is printed instead. The
  result page and the four report profiles follow the same rule. A report made before report template set `r21` keeps the text it
  was made with (a point estimate on the scale and in the sign-off block); request a new report revision for the current wording.
  - **A number that a provenance record set.** Where a provenance record decides a result and its value is the number the
    result publishes (the floor of 85 that self-declared evidence sets, the cap of 10 of a trusted capture record), that
    number is not an estimate of the checks. It is printed in words with what set it ("Set by the record: 85 out of 100, the
    floor of self-declared evidence"), without a marker on the scale, and never as an estimated likelihood, as calibrated,
    or on the fallback as a score of hand-set weights. Beside it stands what the statistical checks alone give: their
    estimate, on the fallback their score, or "no estimate". A marker or a capture record that did not move the estimate
    leaves the statistical number as the result's number. A trusted declaration shows no number of its own.
  - **The number under a record that did not set it.** That statistical number is shown as the result of the statistical
    checks alone would show it. It is the large numeral, with a marker on the scale, only where the checks alone decide.
    Where they alone are inconclusive, by the calibrated levels or because no check gives at least weak support for the
    answer of the estimate (D-76), it is printed in words with its range ("Estimate of the statistical checks alone 83–96:
    they alone do not decide"), without a marker, and "What this means" says after the record's own sentence which way
    the statistical checks alone point, that no check supports that on its own, and where their estimate comes from. The
    result is the record's: nothing says that it is inconclusive, and its headline and its code do not change. The result
    says what the checks alone decided (`decision.statistical_alone`: their code, and the side that D-76 took away); a
    result published before that field keeps the form it had, the large numeral included. On the fallback the number is a
    score in words in every case, and the field is not published.
  - **The statistical number beside a record that decides.** Wherever that number is printed beside the record (the line
    under the headline, the sentence of the full reports, what a capture's result means, the declaration's sentence), it
    has its unit and says what the statistical checks alone decided. Where they alone are inconclusive it is never
    printed bare: "statistical checks alone: 90 out of 100, they alone do not decide", and as a sentence "The statistical
    checks alone do not decide: their estimate is 7–56 out of 100." (the range where the result holds one, else the
    estimate). Where they alone decide it reads "98 out of 100". Where D-76 took their decisive code, its sentence in its
    form beside a record is said too, whether the record set the result's number or not. The limitation that holds the
    number is shown by the same rule ("Statistical estimate without the provenance record: 83–96 out of 100; the
    statistical checks alone do not decide."); the result itself stores the caveat as it was written ("Statistical
    estimate before the provenance override: 90%."), so a consumer of the stored caveats reads the bare number and has
    to read `decision.statistical_alone` beside it. Under a self-declared marker the result's confidence is low where
    D-76 took the decisive code of the statistical estimate: an unsigned marker does not resolve statistical
    uncertainty. Under a trusted declaration and a trusted capture the confidence is the record's. A result published
    before `decision.statistical_alone` does not say what the checks alone decided: its number stands with its unit and
    without those words.
  - **Which condition took the statistical code, beside a record (D-77; D-76 as amended).** `decision.statistical_alone`
    names the condition (`rule`: `scale`, `no_check`, `together_less`, `together_against`), with the side of D-76 where
    D-76 took the code and none where D-77 did, and the sentence beside the record follows it. Its three forms open
    alike: "The estimate of the statistical checks alone points away from AI generation or manipulation, but no check
    gave at least weak support for that on its own." (then "Their estimate comes from …"), and where a check supports
    the answer and the checks together do not, "…, but taken together the checks gave less than weak support for that:
    what one check adds, another takes away." or "…, but taken together the checks point the other way.", then "Their
    estimate rests on the starting point of 8 out of 100 for inputs like this one."; the second condition's sentence no
    longer says that no check gave support, which was false there. Under a trusted capture the sentence follows the
    sentence of the number and begins with the pronoun ("Their estimate points away …", then "It comes from …" or "It
    rests on …"). Where the result holds fitted terms of checks that could not score on the side of the answer (the
    video model: a check that is not applicable to the input), "rests on" names them beside the start ("… the starting
    point of 69 out of 100 for inputs like this one and 2 fitted terms of checks that could not score."), alone and
    beside a record. Where D-77 took the code the number stands with "they alone do not decide" and nothing of D-76 is
    said. Under a self-declared marker the confidence is low whichever condition took the statistical code. A result
    stored before the field named a condition reads as the first form, with the same opening.
  - **What an examiner's position is compared with under a provenance record (D-15; design review D4-01).** Under a
    trusted capture record and under a trusted declaration the steps of a conclusion are counted from the step of that
    record on the scale, the first and the fifth, on both fusion paths. With a fusion model that is the step of the
    decision code. On the uncalibrated fallback the code stands one step inside (the fallback issues no strongest code)
    while the scale fills the record's segment: there a conclusion on the record's own step has 0 steps and is signed as
    concurring, and its `machine_code` holds the step beside it. A conclusion says what its steps are counted from in
    `steps_from` (`capture_record`, `declaration`), a key that is present only there: in the API's conclusion, in its
    signed body and in its custody event; agreement (`departs_from_machine`, `examiner_departs`) is read from the stored
    steps. Under a capture the page and the reports name the step ("Examiner's opinion: Inconclusive, 2 steps from the
    step of the capture record"; "…, on the step of the capture record"; in a list "…, away from the record's step"),
    because they head such a result by the record's sentence and print the decision code nowhere; the signed decision
    keeps its words ("concurs with the machine assessment"): under a capture the machine assessment is the record's
    result. Under a declaration the words do not change. A conclusion recorded before 2.1.1's migration 0023 keeps its
    steps and its signature; one recorded on the fallback under a capture, whose steps were counted from the code's
    step, keeps "agrees with the machine assessment" / "departs from the machine assessment by 1 step" with them.
  - **Outside what was validated** the statistical number is printed beside no record, as it is not printed alone: the line
    and the limitation say "not shown (outside what was validated)". The stored result keeps the number.
  - **A record that decides alone.** While a check did not run, where no check that scored has weight for the input, or on
    the fallback where no check of the pixels, frames, sound or text scored, a record publishes its own value and no
    statistical estimate. The result then says that the statistical checks gave no estimate, prints no account of a
    fusion and no breakdown, and reports no agreement.
  - **Results stored before these rules** are shown by them from what they store; the sentences of their findings keep
    the words they were published with. A report made before report template set `r35` keeps its text.
- **Two conditions on a verdict (D-64).** The two strongest headlines ("Strong indications …", "No indications … found")
  need at least two checks that count. A check counts when it entered the estimate: its row in the result's breakdown is
  not zero, which is also what the line "only one check scored" counts and what "Calibrated fusion of k of n detectors"
  states. A check that scored but that the fusion model gates off for the input, or gives no weight on the side of its
  score, does not count. With one check that counts, the result is the headline of the same side one step weaker, and its
  reason says that one check alone does not give the strongest result. The served audio model holds one detector, so a
  recording never gets either of those two headlines. A result of the uncalibrated fallback needs a check of the content:
  when only metadata or provenance records scored (a camera's EXIF block under `quick`, container tags of a video), the
  result is "not assessable". Both rules only weaken a result; neither changes an estimate or its range.
  - **What the count is not.** It does not ask that the checks point the same way. A row can be the step from a check's
    neutral point on a side where its score has no weight, and it counts. A row near zero counts.
  - **A third condition (D-76): a headline of one answer needs a check that supports it.** A calibrated result is
    decisive when its estimate reaches a calibrated level. In some contexts the model's starting point alone stands near
    such a level (53 out of 100 for a large photograph with a small face), so an estimate could reach it on checks that
    each "had little effect". A headline of one answer, on either side and strong or not, therefore needs one check
    that counts whose row in the breakdown is at least "weak" toward that answer: a factor of two on the odds, ln 2 in
    log-odds. One check, not the sum of several: three rows of +0.27, +0.33 and +0.10 give none. The starting point and
    the fitted term of a check that could not score are not checks. Without such a check the result is "Inconclusive",
    its estimate, range and breakdown are published as before, and "What this means" says which way the estimate
    points and where it comes from. The rule changes only a result that the statistical checks would decide inside what
    was validated: an input outside it keeps the reason of its cap, a result without a check that counts stays "not
    assessable", a provenance record decides whatever the checks support, and the uncalibrated fallback, which
    publishes no rows, is left as it is. It weakens a result and changes no estimate. Like the first two conditions it
    reads a check's row from its neutral point, so the three lines of "What the count is not" hold here too.
  - **And the checks together (D-76 as amended on 2026-10-05).** One check that supports the answer is not enough where
    another check takes the support away: small AI pictures read "No significant indications of AI generation or
    manipulation" at estimates of 10 to 23 on rows such as +2.36 beside −1.29, so that the headline rested on the
    model's start of 13 for such inputs. The rows of all the checks that count must therefore also sum to at least
    "weak" (ln 2) toward the answer. Where they do not, the result is "Inconclusive" and says "…, but taken together
    the checks gave less than weak support for that, so the result is inconclusive: what one check adds, another takes
    away." where the rows sum toward the answer or to zero, and "…, but taken together the checks point the other way,
    so the result is inconclusive." where they sum toward the other answer, each followed by "The estimate rests on the
    starting point of 13 out of 100 for inputs like this one." Because the rows sum to the estimate less the model's
    start, this condition is a level on the estimate that the start sets: an AI headline needs at least twice the odds
    of the start, a clean headline at most half. It is strict where a context's start stands near one of its
    calibrated levels: a picture with a face starts at 53, so an AI headline needs an estimate of 69.3 where the level
    is 62.9; a small picture with its three checks starts at 13, so a clean headline needs 6.8 or less where the level
    is 49.6 (supervisor check 28, `probes/supervisor/r415_part3_states.py`; run again on the built tree:
    `docs/v2/qa/n19m-capture/r415_part3_states.part3.txt`). The rows are the same under every configured prior, so
    this condition does not depend on `VERITAS_PRIOR_AI`.
  - **A fourth condition (D-77): a headline of one answer needs the estimate in its part of the scale.** A calibrated
    headline follows the calibrated levels for inputs like the one analysed, and in one context of the image model
    (small pictures, a shorter side under 512 px) the clean level reaches an estimate of 49.6: a 448 px crop of an AI
    picture read "No significant indications of AI generation or manipulation" at "49 % estimated likelihood" with a
    plausible range of 24 to 85. A headline of one answer therefore also needs the estimate itself, as the result
    prints it in whole points, in that answer's part of the scale: 34 or less for no indications, 65 or more for
    indications. Otherwise the result is "Inconclusive", prints its number in words ("Estimate 24–85: inconclusive"),
    and says: "The estimate itself, 49 out of 100, stands in the middle part of the scale (35 to 64), where Veritas
    gives no answer. The calibrated levels for inputs like this one would have read no significant indications; a
    headline of that answer also needs an estimate of 34 or less." The scale's note says the same. This condition is
    asked before D-76, and a result gives one reason: a result that fails both gives this one, also where no single
    check supports the answer and the estimate prints 35 to 64. The bound is read on the published estimate (after a
    configured prior, clamped, six decimals, a half upwards), so no calibrated result prints "35" under a clean
    headline. Both conditions act where D-76 acts and nowhere else: an input outside what was validated keeps the
    reason of its cap (its reason quotes the headline of the levels only where both conditions would have left it), a
    result with a check that did not run keeps that reason, and a provenance record decides whatever the estimate.
    Section 7 of the full reports states the rule with both conditions and no longer says that the headline follows no
    fixed range of the estimate. The rule bounds the level of the small-picture context and does not repair it: at an
    estimate of 34 or less that level still decides wherever D-76 lets it. The uncalibrated fallback follows the
    segments already, cut on the score and not on the printed number: a score of 34.7 prints "35" under its clean
    headline (known issues).
  - **Under a configured prior (`VERITAS_PRIOR_AI`) the setting also decides which results are decisive.** The code is
    taken from the calibrated levels before the prior's shift, and D-77 reads the number that the reader sees after it.
    A prior under one half therefore takes AI headlines away and one over it clean headlines: at a prior of 0.1 an AI
    headline needs an estimate of 94.2 or more on the scale of the levels, at 0.9 a clean headline one of 5.5 or less
    (an estimate of 83 with no prior is published as 35 at 0.1; one of 10 as 50 at 0.9:
    `backend/tests/test_n19m_rules.py`). Under a self-declared marker the result's confidence is then low wherever the
    rule took the statistical code, also over checks that themselves point to AI. On the code reviewer's grid of score
    combinations (not a sample of files; `code-review-round2.md`, "The read of the third part of N-19m"; not run again
    by the loop) the AI codes fall at 0.1 from 5,929 to 4,151 for images, from 2,755 to 511 for video and from 8 to
    none for audio, and at 0.9 the clean codes from 2,996 to 2,359, from 187 to 104 and from 9 to none. The benchmark
    figures and the measurements of D-77 are those of the default, no prior.
  - **Under a configured prior an inconclusive result can stand with its estimate in the part of the scale of the other
    answer.** With no prior a result that D-77 takes has its estimate in the middle part (35 to 64). Under a prior it
    need not: at 0.1 a picture with a face to which the levels give the AI answer at an estimate of 70 is published at
    21, and at 0.9 a small picture to which they give the clean answer at 30 is published at 79
    (`backend/tests/test_n19n_prior.py`). Such a result is "Inconclusive" and names the part in which its estimate
    prints: its reason ends "AI generation or manipulation is the answer of the calibrated levels for inputs like this
    one, but the estimate stands in the other answer's part of the scale, so the result is inconclusive."; the note
    under the scale reads "The estimate stands in the clean answer's part of the scale, although the calibrated levels
    for inputs like this one would have given the AI answer, so the headline is inconclusive."; and what the result
    means says "The estimate itself, 21 out of 100, stands in the clean answer's part of the scale (34 or less). The
    calibrated levels for inputs like this one would have read indications; a headline of that answer also needs an
    estimate of 65 or more." (on the other side "the AI answer's part of the scale (65 or more)"). Section 7 of the
    full reports opens "…: the calibrated levels give the AI answer, and its estimate stands in the clean answer's part
    of the scale." These texts follow where the published estimate prints, so a result stored before they existed,
    whose reason ends with the words of the middle part, reads the same. A result that D-77 or D-76 took under a
    configured prior also says why the levels and the estimate part ways, with the share of the result's own caveat:
    "The estimate is adjusted for the share of AI-generated or manipulated inputs that this server is set to assume,
    10 %; the calibrated levels are not." (after a sentence of D-76, alone or beside a record: "The estimate and its
    starting point are adjusted …", because the starting point that the sentence names holds the shift too: a start of
    13 reads 2 at a prior of 0.1). The sentence is said wherever the result holds the caveat of a configured prior,
    also at a prior of 0.5, which shifts nothing. Beside a record it follows D-76's sentence only: where D-77 took the
    statistical code under a record, the number stands with "they alone do not decide" and no sentence of a condition
    is said.
  - **One count of the checks.** A check that addresses the file's provenance and integrity (the container check of a
    video) is outside every count: "k of n detectors", the line "only one check scored", the count "in the estimate"
    and the sentence on how many checks entered the fusion model count the same checks. Its row stays in the breakdown,
    where the result says that it is shown and not counted, as it says of a row of zero.
  - **No check that scored has weight for the input.** Where every check ran or was not applicable, checks scored, and the
    fusion model gives none of them a row that is not zero, the result is "not assessable" and says so in those words:
    "For an input like this one the fusion model gives the checks that scored no weight." No estimate, range or breakdown
    is published, and the findings of the checks that scored are listed as checks outside the estimate, never as reasons.
    With the served models this is a video on which no frame check applies (a short side under 336 px, or no clip in the
    plan) while the container check or the voice check scored; before, such a result published the model's starting point
    as its estimate. While a check did not run, the cap for that answers first, as before. A provenance record over a
    result in which no check counts decides alone: the record's own figure, with no statistical estimate beside it, and a
    self-declared marker then has confidence low.
- **Image-size shortcut (D-45).** The v2.1 image fuser learned that large images are real: its size terms alone separated the
  classes (context-only AUC 0.7897 on the calibration split). `fuser-image-2.1.1` holds no image-size or face-size term, so its
  figures are lower and are the ones to use. On vb-1.1 `test`, full view against full view: AUC 0.9014 (0.8900–0.9122) against
  0.9632 for v2.1, and 59.1 % average detection of a generator family left out of the fuser's training against 77.2 %. The primary,
  size-matched view reads AUC 0.9054 (0.8905–0.9192) and 58.4 %. "Left out" holds for the fuser only: the training data of the
  ateeqq and dima806 classifiers is not established, so they may have seen these families. Every v2.1 image figure is superseded.
- **Container geometry (D-52).** The benchmark's fakes differ from its reals in container geometry: square images and sides
  divisible by 64 are far more common among fakes. A classifier on geometry alone reaches an out-of-fold AUC of 0.8582 (weighted;
  unweighted 0.9526), and no refit can remove that from the data. Every vb-1.1 detection figure may therefore include geometry
  cues that do not hold outside the benchmark. The field test cannot check this: its v2.1 set is size-confounded as well (32 of
  the 33 stills at 768–1,024 px are AI, 33 of the 35 above 1,536 px are camera originals; `docs/field_test.md`). No
  geometry-matched field set exists yet; vb-1.2 is planned to remove the imbalance.
- **Operating point.** "TPR at 5 % FPR" names the calibration target, not the realised test rate. The thresholds are set on
  calibration reals with equal stratum weights and do not transfer exactly: on vb-1.1 `test` reals the image fuser flags 7.5 % at
  τ_0.05 (162/2,152; Wilson 6.5–8.7 %; size-matched view 7.6 %, 73/958; stratum-weighted 7.3 %) against 5 %, and 1.3 % at
  τ_0.01 (29/2,152; Wilson 0.9–1.9 %) against 1 %. Like for like, the v2.1 image fuser flagged 5.3 % on the full view, 10.1 % on
  the size-matched view and 5.0 % stratum-weighted at its τ_0.05: its lower full-view rate came with the size prior that D-45
  removed, and on the size-matched view its rate is higher. The measured cause (`docs/v2/qa/j5a_operating_point_probe.md`) is
  the calibration sample: the 850 calibration reals hold unusually few high-scoring reals, mostly saturated `img.hf.ateeqq`
  scores (reshuffling groups between calibration and test reals within strata gives a test rate this high with p = 0.007; v2.1
  gives p = 0.40), and per-context Platt slopes add about 0.5 pt. No code, context or composition cause was found. Selection
  optimism from the D-53 admission path could not be measured without re-running it and cannot be excluded
  (`docs/v2/qa/j5_operating_point_diagnosis.md`, (f)). The thresholds are not re-tuned on `test`. Per-stratum rates are under "Real photo strata" below.
- **Detectors pending admission.** For images, a learned detector enters the default profile only after its admission record on
  the benchmark (`docs/v2/bench/admissions/`); until then it is off by default or carries weight 0, and the Methods section says
  "not established" for its error rates. The v2.1.1 image fuser holds the four detectors admitted on vb-1.1 under D-53: DDA,
  DFDC-B7 on faces, the ateeqq classifier where no face is detected, and the dima806 classifier (`v2.1.1-admissions.md`).
  Community Forensics was rejected on images and is off in the image profile. The admission rules changed after fit and
  calibration results were seen (D-44, D-51, D-52, D-53, `policy.VIDEO_2B`; `docs/v2/bench/shortfalls.md`, "Post-hoc rule
  changes"). The SPAI adapter is deferred (D-39): built, off.
- **Video and audio detectors are not admitted.** The rule above does not hold for video and audio. Every detector of the video
  and audio fusers is on by default and fitted, by the H-8 decision, while its admission record is pending: Community Forensics on
  native frames (its image twin was rejected; A3a and A4 fail), DDA on native frames (A3a, A3b and A4 fail), DFDC-B7 on faces, the
  codec lane and the voice detector `aud.gustking_xlsr`, the sole audio detector. Admitting or removing them is a next-phase item.
- **Field test.** The field set is an in-distribution sample (same sources as vb-1.1): held out by group key and file hash, but
  its AI items come from the generators and datasets vb-1.1 trains on, and its real clips from OpenVid-1M. It is a small sample
  without intervals, not a validation. The v2.1 run (H-9, `docs/field_test.md` history) was its second run: the first capped every
  video because the served artefacts had no slices. On that run, new stills: 71 of 80 decided, 66 correct; new clips: 19 of 32
  decided, all correct. The v2.1.1 run under the served abstention policy (K-1, `docs/field_test.md`), without the size prior: new
  stills 38 of 80 decided, 31 correct; the 7 wrong decisions called 5 real camera originals AI and 2 FLUX.2 stills authentic. On
  the original manifest 3 real camera originals were called AI (12 of 15 decisions correct). New clips: 13 of 32 decided, all
  correct. Both new-set decision rates are below 50 % (D-42 rows in `shortfalls.md`): 4 real camera originals at JPEG q 70–89 and
  4 AI clips at 480–719 px were capped because their slice lacks validation examples of one class. The J-5 run (42 of 80 and
  18 of 32 decided) predates that rule. Large camera originals are now more often flagged or left inconclusive than under v2.1.
  The N-20b run under the served rules (D-64, D-65, D-76, D-77; `docs/field_test.md`, 2026-10-11): new stills 36 of 80 decided,
  30 correct; new clips 10 of 32 decided, all correct; on the original manifest 12 of 23 images decided, 11 correct. D-76 took
  4 image and 3 video decisions, D-77 none. **Camera originals and the R1 figure:** real camera originals received an AI
  decision in 5 of 52 cases (9.6 %, Wilson 4.2–20.6 %; 8 of 52 in the K-1 run) against the benchmark's R1 false-positive rate of
  3.6 % (14/387, Wilson 2.2–6.0 %); Fisher exact p = 0.061, so the field no longer shows camera originals flagged more often
  than the benchmark does. The three that K-1 decided at p(AI) below τ_0.05 (0.683–0.706 against 0.715) are inconclusive under
  D-76; the five still decided AI stand at p(AI) 0.77–0.96.
- **Families not validated** on the benchmark carry no measured error rate; the report states the family as not validated
  instead of borrowing another family's numbers.
- **Coverage date.** Model cards record the newest generator families a detector covers; most say "not established". Newer
  generators may evade every detector (PRD VG-05), and the report states the coverage date.

## Not validated in v2.1.1

These goals were missed on `test` (values, intervals, mitigations: `docs/v2/bench/shortfalls.md`; D-02). A generator family or a
photo stratum cannot be identified from an input, so for those the cap is this disclosure: analyses still run, and a report
never borrows another family's validated numbers. Where the engine applies a cap, the entry names it. The machine-readable list
is `backend/app/engine/fusion/not_validated.yaml`, checked by `make final-gate`.

- **Generator families (not validated).** Leave-one-generator-out TPR at τ_0.05 (the family is left out of the fuser's training
  only; the classifiers' own training data is not established) averages 58.4 % over seven families in the size-matched view (full
  view 59.1 %; target 75 %); the weakest family, GPT-image-1.5 / 2, reaches 25.0 % (7/28, Wilson 12.7–43.4 %) against 50 %, and FLUX.2 40.0 % (6/15).
  A report on an image from a family absent from training does not borrow the average.
- **Real photo strata (not validated).** False-positive rate on vb-1.1 test reals at τ_0.05 (target 5 %) and τ_0.01 (target 1 %):
  R1 camera originals 3.6 % (14/387, Wilson 2.2–6.0 %) and 0.5 % (2/387); R2 smartphone photos 6.1 % (25/409, Wilson 4.2–8.9 %)
  and 0.2 % (1/409); R3 social-platform bytes 11.7 % (56/479, Wilson 9.1–14.9 %) and 1.7 % (8/479); R4 web and stock photos 9.1 %
  (37/405, Wilson 6.7–12.3 %) and 3.7 % (15/405); R5 portraits 5.8 % (26/450, Wilson 4.0–8.3 %) and 0.7 % (3/450); R6 hard
  negatives 18.2 % (4/22, Wilson 7.3–38.5 %) and 0.0 % (0/22), too few to bound the rate. On the benchmark only R1 meets the 5 %
  target; in the field 5 of 52 camera originals were decided AI (9.6 %, Wilson 4.2–20.6 %; Fisher exact p = 0.061 against R1;
  see "Field test" above). For R6 the admission joint
  check measured +14.3 pts [+3.3, +29.2] for the end set; that harm counts as "not established" only because R6 holds fewer
  than 150 reals, not because the interval includes the +2.0 pt limit.
  Screenshots are not measured: the `screenshot` transform was not scored.
- **Calibration (not validated).** Expected calibration error on test is 0.0537 against 0.05, and 0.1651 over the
  leave-one-generator-out folds against 0.08; the Brier score is 0.1270 against 0.12. The conformal sets cover 91.5 % of real and
  93.9 % of AI test images against 94 %, and 39.7 % of test images (705/1,778) are inconclusive against 20 % under the served
  policy with the decision rules D-64, D-65, D-76 and D-77 (41.5 % in the full view; 32.7 % before those rules). Of the test
  images that get a decision, 9.2 % are wrong (99/1,073; 10.4 % before those rules). The runtime envelope (which slices abstain) is read from the artefact's `test`-split
  slice metrics, so the `test` decision figures are evaluated with an envelope derived from the same data. A runtime condition slice without both classes in its validation data abstains
  (ad26e7a); the J-5 figure of 28.3 % predates that rule. Since L-1, an input smaller than the smallest size at which the
  vb-1.1 test split holds both classes is also capped at inconclusive, whatever its pooled slice says: images with a shorter side
  below 256 px (the smallest AI test image; non-AI test images go down to 81 px) and videos below 480 px high
  (`backend/app/engine/fusion/validated_sizes.json`). The benchmark figures above do not apply this floor; it affects 20 of the
  1,871 non-AI image test items and 3 of the 162 non-AI video test items (`scripts/v2/validated_sizes.py`). The dima806
  classifier is face-oriented and enters the image fuser without a gate; on an image with no detected face it scores the whole
  image, which its stand-alone admission evidence (AI-generated faces) does not cover, and the result says so. Transformed copies fall into such slices more often: 100 % of JPEG 50
  copies, 85.9 % of screenshot + JPEG copies and 46.6 % of JPEG 75 copies are inconclusive.
- **Transformed images (not validated).** At a fixed threshold, detection falls by 16.7 pts under JPEG 75, 12.5 pts under WebP 80
  and 23.5 pts under a 0.5× resize (target ≤ 10), and by 28.0 pts under JPEG 50 (target ≤ 20). The false-positive rate rises by
  21.1 pts under screenshot + JPEG, 22.8 pts under added noise, 3.0 pts under a 0.5× resize and 2.0 pts under a 70 % crop (target
  ≤ 2). JPEG 95, PNG screenshots, laundering and the re-photograph proxy were not scored. The runtime condition slices are cut by
  JPEG quality, image size and face presence only; noise, resizing, cropping and screenshots are not slice dimensions. So noisy,
  resized and cropped copies are decided like clean ones despite their raised false-positive rates (inconclusive under the served
  policy: noise 27.4 %, 154/562; 0.5× resize 29.4 %, 165/562; 70 % crop 28.3 %, 159/562). The cap reaches mainly copies whose
  JPEG quality lands them in a slice without validation examples of both classes: 100 % of JPEG 50 and 85.9 % of screenshot + JPEG
  copies are inconclusive.
- **GAN faces (not validated).** TPR at τ_0.05 63.0 % (58/92) against 85 %. Face-swap stills reach 51.4 % (95/185) against 70 %,
  with AUC 0.782 against 0.90. The GenD face detector is not in the fitted set.
- **Localisation maps (not validated).** The Mesorch and RADAR localisers show regions but are not in the fitted set, so they do
  not move the decision. On the benchmark's scored bytes Mesorch reports a region on 35.7 % of real photographs (963/2,697, vb-1.1
  fit ∪ cal), so a region is not evidence of an edit on its own. Partial edits are detected through the learned detectors (TPR at
  τ_0.05 78.4 %, 40/51 in the size-matched view).
- **Video synthesis (not validated).** Vendor leave-one-out TPR at τ_0.05 averages 59.5 % against 80 %: Sora 21.1 % (4/19),
  Seedance 36.8 % (7/19), Veo 55.6 % (10/18), Kling 84.2 % (16/19), Wan / Hunyuan 100 % (18/18). Real videos are flagged at
  2.5 % (4/162, Wilson 1.0–6.2 %). The fitted set reads Community Forensics and DDA on native frames, DFDC-B7 on faces, the
  codec lane and the voice detector. The D3 X-CLIP statistic separates generated from camera footage weakly on its own (AUC
  0.58 on vb-1.1 fit ∪ cal) and is not in the fitted set, nor are UnivFD frames.
- **Synthetic voice (not validated).** Equal error rate 17.4 % on the ASVspoof 5 test set (86 reals, 90 fakes) against 10 %.
  On in-the-wild clips the bona fide false-positive rate is 1.1 % (1/94), but only 1 of 89 cloned voices is flagged at that
  threshold. The second voice detector (`aud.mo_thecreator_w2v2`) is not in the fitted set.

## Provenance and watermarks

- Content Credentials (C2PA) can be stripped, and a valid manifest describes what its signer declared, not what happened to the
  pixels. Absence of a manifest is uninformative.
- **SynthID is not detectable offline.** Google's detector runs only in Google services and a cloud API; SynthID *Text* needs
  Google's keys. Veritas cannot say whether an image, video, recording or text carries SynthID.
- **The metadata checks read an XMP packet up to its first 64 KB** (SEC4-01); a real packet is under it. What stands beyond
  that in a longer, crafted packet is not read.
- Metadata can be stripped or forged. Absence of metadata is treated as uninformative, never as evidence of authenticity;
  camera EXIF is only weak evidence. On the uncalibrated fallback a result in which no check of the pixels, frames, sound
  or text scored is "not assessable", whatever the metadata says (D-64).
- **Which record a result shows (D-15).** A file can carry more than one provenance record. One decides, by this order: a
  declaration of AI generation that validates against the trust list; a trusted capture record that covers the current
  bytes; a self-declared marker. The result is headed by the record its decision applied and names it. A file with a
  trusted capture record and an AI marker in its metadata is the capture's result: its headline is the capture's, and
  the marker is listed as a finding without weight, with a caveat that it was not applied. Signed evidence outranks an
  unsigned marker; that is a rule of precedence, not a finding that the marker is false.
- **A result that a trusted capture record decides (D-15).** It is headed "The file's Content Credentials record a capture
  of these bytes", on the first step of the scale, with the basis "Documentary · Content Credentials (capture record)":
  the result is the record's, not what the statistical checks found. The page and the reports print no agreement and no
  count of checks under it. The record states what the capture device declared. It does not show what was in front of
  the device: a screen or a print can be captured, and a signing key can be misused. The decision code of such a result
  is still the clean one ("no indications"), with the provenance hypothesis as the one that drives it; the lists of the
  product read "Capture record" for such a row. A consumer that reads the decision code alone (the `decision_code`
  column of a batch summary's CSV, the label of the v1 view) cannot tell it from a clean statistical result and has to
  read the driving hypothesis or the decision's record beside the code. Where the statistical checks alone give an AI
  result, the next step says so: weigh the record's signer against them.
- **A self-declared marker is worded by its kind:** unsigned metadata that names a generator; a public watermark message;
  or Content Credentials that are no trusted declaration (a generator named under a signer on the trust list without a
  declared AI source type, or a declaration or a named generator under a signer that is not on the list). Each raises the
  result to "Indications …" at most, and each can be attached to a genuine file: metadata by anyone, a public watermark
  message by anyone, credentials of an unlisted signer by anyone who signs them.

## Images

- Learned detectors generalise imperfectly to generators released after their training data.
- Heavy JPEG recompression, screenshots, resizing below 224 px, cropping and social-media re-encoding degrade pixel-level signals.
  The condition profile flags these inputs, and the fuser abstains in slices without validated performance.
- **A small or re-saved copy of an AI picture can read "No indications of AI generation or manipulation found".** The
  browser QA's second pass with the packaged models (`docs/v2/qa/prod-readiness-2026-09-29/browser-qa.md`, "Second pass
  on 9fdbe83"; the reviewer's sample, not run by the loop) cut 24 small lossless pictures from five AI pictures: 14 read
  a headline of no indications. The two conditions on a headline of one answer (D-77; D-76 as amended) take five of
  them. Of the nine that stay, eight read the strongest clean headline (seven of the sample's sixteen 448 px crops
  that had been reduced or re-saved once, and one reduced picture), because the checks themselves read them as
  genuine: in seven no check that counts gives the AI answer even weak support. No rule on the headline reaches such a
  result; the benchmark holds few pictures of this kind and understates how often it happens.
- **Screenshots, flat graphics and greyscale images are capped at inconclusive** (D-65). An image is outside what was
  validated, whatever its slice says, when
  - more than 53 % of its 8 × 8 pixel blocks are flat (one value in each colour channel). The limit is the 99.5th percentile
    of that share over the 6,546 images of vb-1.1 the checks were fitted and calibrated on (`fit` and `cal`; the median is
    0.04 %, and 33 of those images are above the limit);
  - it is greyscale: one channel, or three equal ones (55 of the 6,546 are);
  - or its file matches the screenshot rule (lossless, a screen resolution, no camera metadata).

  The result then reads "Inconclusive" with that reason, on the calibrated result and on the uncalibrated fallback; the
  estimate is still published beside it. **What the rule does not catch:** a screenshot that is mostly a photograph (a post
  with a picture) can stay below the limit and is then assessed as the photograph would be; a screenshot saved as JPEG,
  cropped or resized does not match the screenshot rule either. The benchmark report applies the same rule to the files it
  scored and states, for every set, how many scored items each condition holds for and how many decisions it takes away
  (`scripts/vbench_report.py --content`). **The figures of this page were published before the rule and are those without
  it**; they are published again under it when the report is next run on the test split (BUILD_SPEC N-20a;
  `backend/app/engine/fusion/content_kinds.json`; `docs/v2/qa/n18-validated-content/README.md`).
- Face detectors need a detectable, reasonably frontal face of at least 48 px.
- Localisation maps (Mesorch, copy-move) show where a localiser responded. They are not masks of manipulation, and attention maps
  (Grad-CAM) show where a classifier looked.
- Classical forensics (spectrum, ELA, noise, JPEG history) are explanatory heuristics with known false positives (HDR
  processing, heavy denoising, flat backgrounds) and are capped in influence.
- Region coordinates, image dimensions and the overlay are in the pixels of the image as it is shown, with its EXIF orientation
  applied. A photograph stored rotated (EXIF Orientation 5–8) that was analysed before this rule keeps the finding sentence it
  was published with, whose coordinates have the axes swapped; its report tables and overlay are drawn correctly. Re-run the
  analysis to get a corrected sentence.

## Videos

- Sampling depends on the profile: `quick` seeks 16 frames; `full` samples up to 4 fps over the first 120 s and tracks every
  face. Short clips give few observations; the Sampled lane shows what was analysed.
- **Lip-sync evidence is one-directional.** Good sync is informational and never evidence of authenticity. An offset step or a
  loss of correlation is explanatory and capped: voice-over, dubbing, occluded mouths, several speakers and variable frame-rate
  recordings produce the same readings. Well-synchronised lip-sync deepfakes are not detected by the sync lane. The
  SyncNet-based lane is research-only (D-37).
- Fully synthetic (text-to-video) content relies on the synthesis lane and whole-frame detectors, several of which were trained on
  images.
- **Video fuser.** `full` and `deep` load `fuser-video-2.1.1`, fitted on Community Forensics and DDA on native frames, DFDC-B7
  on faces, the codec lane and the voice detector, without additive size or face terms (`policy.VIDEO_2B`, a post-hoc rule set
  after gate 2b failed on the v2.1 video fuser). On vb-1.1 test it separates videos
  with AUC 0.9613 (0.9374–0.9806); τ_0.05 comes from 74 calibration reals and flags 2.5 % of real videos (4/162). 50.2 % of test
  videos (149/297) are inconclusive under the served policy with the decision rules of section N (48.5 % before them; the J-5
  figure of 27.3 % predates ad26e7a), and 4.7 % of the decided ones are wrong (7/148). Recall on a
  vendor absent from training is lower (see "Video synthesis" above).
- **The sound track of a video carries no weight in its estimate.** The served video model holds the voice detector with a
  fitted weight of zero, and its stand-alone fit there says nothing at any score (fitted on eight clips; on the 29 videos of
  the vb-1.1 test split it scored, the check's AUC is 0.52). Real footage whose sound was replaced by a synthetic voice therefore gets the
  headline its frames give. When the headline is clean and the voice check's own score is at least 0.91 (10 to 1, not
  calibrated for video sound tracks), "What this means" and the reports say that the sound track was flagged by a check
  that carries no weight in the estimate. Below that score nothing is said, and the sentence is a notice, not a measured
  rate: how often it appears on genuine videos was not measured. Analyse the sound track as a recording to get the audio
  model's result for it.
- **The video fuser keeps a face-presence prior.** When DFDC-B7 finds no face, its fitted missing term adds +0.37 logit towards
  AI (`miss:vid.frames.dfdc_b7` = +0.3695, one of the largest terms; bias +1.87). "No face found" alone separates the video classes
  with AUC 0.718 on `fit` (55/100 AI and 13/114 real videos without a face) and 0.701 on `cal` (34/65 and 9/74;
  `docs/v2/qa/k4-docs/video_face_prior.md`), and 0.711 on `test` as the release audit measured (81/135 and 29/162). Gate 2b tests
  additive context terms only, so it reads 0.5000 and cannot see a prior carried by a missing-indicator term. D-50 removed the
  same residual for images; applying it to video is a vb-1.2 lever (`docs/v2/bench/shortfalls.md`).
- **480–719 px videos are always inconclusive.** Every runtime video slice at 480–719 px has no real videos in its validation data,
  so the envelope rule caps every such video at inconclusive, whatever its score. On vb-1.1 test these slices hold 58 of the 135 AI
  videos and none of the 162 real ones; only 5 of 19 video slices contain both classes. More real videos at that resolution are
  needed before the system can decide on them.
- **Profiles and lanes.** `quick` uses the fallback fuser. The lip-sync lanes and the face-swap lane of a whole face track (`vid.lipsync.*`,
  `vid.swap.gend_clip_l14`): these lanes are shown with their segments but are not in the fitted set, so they weigh 0 in the decision. The synthesis
  frame checks and the per-frame face-swap check are in the fitted set by their one score for the whole video; the segments
  flagged on any lane are not validated.

## Audio

- Recordings are analysed up to 600 s as 16 kHz mono. Each second is labelled speech, music, noise or silence by a heuristic
  classifier, and only windows of up to 4 s at least half inside speech are scored.
- A recording with less than 1 s of speech, or with more music than speech, is not assessable: the voice detectors are trained on
  speech, and music is a known source of false positives.
- Anti-spoofing classifiers were trained on ASVspoof-style and in-the-wild synthetic speech. Heavy compression, background noise,
  telephone bandwidth and non-English speech reduce reliability (`docs/eval_audio.md`).

## Documents

- **Frontier-model text (2025–2026): mitigated, not solved.** The shipped public classifiers flagged 0 of 8 frontier-model
  paragraphs written in a concrete register; after fine-tuning on a modern corpus (v0.2) the fused detector flags 8 of 8 with 0
  Wikipedia false positives, and halves false positives on real business emails. But the held-out paragraphs share an author with
  the training ones, the corpus contains one frontier model plus Qwen-3B and ChatGPT-2022, and recall on RAID GPT-4 text dropped
  from 0.97 to 0.85. Human business email remains the hardest register (14 % false positives). A "human-like" verdict on polished
  prose is still weaker evidence than an "AI" verdict. Details: `docs/evaluation.md`, `docs/eval_text_ood.md`,
  `docs/finetune_text.md`.
- **The fine-tuned classifier is installed separately.** The fine-tune (`desklib-ft-v2`, 1.7 GB) is not part of the source or
  of the downloaded weights: it is copied to a server by hand (`docs/runbook.md`, "Local models"). Without it the base public
  classifier scores documents, the result says so ("The fine-tuned text classifier … is not installed on this server, so the
  base classifier … scored this document"), and `GET /api/ready` with an API key reads `fine_tune: not_installed`. What that changes is the
  "before" column of the v0.2 comparison (`docs/evaluation.md`; held-out run in `docs/finetune_text.md`): on the 8 held-out
  frontier-model paragraphs the base classifier alone reached TPR 0.25 and the full pipeline flagged 0 of 8, against 8 of 8
  with the fine-tune (16 held-out samples: TPR 1.000, FPR 0.000); false positives on Enron business email were 28 % against
  14 %; recall on RAID GPT-4 text was 0.97 against 0.85. A document result that names the base classifier and reads "no
  significant indications" is therefore weak evidence about text from 2025–2026 models.
- English only. Other languages, code, tables and lists reduce reliability.
- Texts under ~120 words yield low-confidence verdicts; under 15 words are rejected.
- Known false-positive modes: formulaic or templated human writing, non-native English, memorised or famous passages (low
  perplexity), text edited with grammar tools.
- Known false-negative modes: paraphrased or lightly human-edited machine text, prompts that instruct a model to write "unlike an
  AI", and newer models not represented in classifier training data.
- Binoculars uses a 1.5B-parameter model pair for laptop feasibility; the original paper's 7B pair is stronger.
- The RoBERTa secondary detector was trained on HC3, so HC3 evaluation numbers for it are optimistic; it is weighted accordingly.

Sensitive regional screening (AI authorship not established). The optional
native PNG localizer improves region recovery in a limited, same-editing-family
comparison while increasing false flags on untouched photographs. Ordinary
spatial edits have not been ruled out as a cause. Its uncalibrated regional
responses contribute no weight to overall authorship confidence. The older
recompression check and its weights remain available for comparison and rollback.

## Data handling

- Originals are stored write-once and re-hashed before reports, exports and downloads; the custody log is hash-chained.
- Retention (`make purge`, which runs `python -m app.cli retention purge`) works on a whole revision chain: an analysis with its
  re-runs, reports and packages.
  - Never purged: a chain with a queued or running revision, or with a report or package still being produced; a chain in a case
    under legal hold; a chain in a case whose retention date lies in the future.
  - A case whose retention date has passed loses every chain, signed ones included.
  - Without a retention date, a named case is kept. A quick analysis (no named case) is purged when its newest revision is older
    than `VERITAS_RETENTION_DAYS` (default 30) and no revision is signed.
  - Each purged revision leaves one `retention_purge` event in the custody log: ids and hashes, no file name. Earlier events,
    including the file name recorded at intake, are never rewritten. The event is stored in the same database transaction as
    the deletion, so an interrupted purge cannot delete a revision without its event. An interrupted purge can leave the
    directories and bytes of the chain it had just deleted; the next purge removes the directories and `store gc` the bytes.
  - The purge and `DELETE /api/analyses/{id}` also unlink the stored bytes (original, exhibits, PDFs, packages) that no remaining
    record references. Nothing is unlinked while an analysis, report or package is in progress; those objects, and objects that
    deletes before release 2.1.1 left behind, are freed by `python -m app.cli store gc`.
  - Nothing purges on its own: an operator schedules `make purge`. Deleted bytes are unlinked, not overwritten; backups and
    file-system snapshots keep them until they expire.
- Nothing is sent to third parties: all models run locally, and production defaults are offline (PL-01). The only outbound path is
  administrator-configured webhooks, off by default.

## Known coverage gap measured in v0.3: small GAN-generated portraits

Fully synthetic StyleGAN-style faces at ≤ 256 px were under-detected by the v0.3 still-image verdict: the face-fused ranking was
good (AUC 0.886) but at the 0.5 threshold only ~65 % were caught, and the whole-image classifiers (trained mainly on diffusion
output) scored them as real. The gap persists in v2.1.1. On vb-1.1 `test` the image fuser flags 63.0 % of GAN faces at τ_0.05
(58/92; Wilson 52.8 %–72.2 %; AUC against real faces 0.8662 (0.8226–0.9073)), against a G7c target of ≥ 85 % (`docs/evaluation.md`).
