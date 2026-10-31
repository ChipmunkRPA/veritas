# Changelog

One entry per release. The version is the constant of `backend/app/version.py`; `frontend/package.json` carries the same
number. A running instance states it, with the commit it was built from, at `/api/ready` (with an API key), in the metric
`veritas_build_info`, in every result manifest and in every report.

## 2.1.1 (not tagged yet)

The first release of Veritas v2 that is prepared for production use on one host. Its number is that of the three fusion
models it serves (`fuser-image-2.1.1`, `fuser-video-2.1.1`, `fuser-audio-2.1.1`). Earlier builds of this repository stated
`0.1.0` whatever they contained; no earlier version was tagged.

This entry describes the release as the repository holds it. The release is tagged by the repository's owner once the
production-readiness work is complete, and three steps are the owner's alone: to review and sign the three served fusion
models (until then every result that uses one says "Fusion model pending human review"), to choose a licence for the
repository (there is none: see `THIRD_PARTY_NOTICES.md` for the material of others), and to authorise any publication or
deployment.

### What the release is

- **A local, offline service** that examines images, videos, recordings and documents and reports what it found as
  evidence: a fused estimate with its uncertainty, the signals behind it, exhibits, plain-language findings and the
  limitations that applied. Models run on the host, and production defaults are offline.
- **Detection (unchanged from the build measured as v2.1.1 on the benchmark `vb-1.1`).** Calibrated fusion for images and
  videos under the profiles `full` and `deep` and for recordings under every profile; documents and the profile `quick`
  for images and videos use the uncalibrated fallback, and a result says which it is. A result outside what was validated
  abstains or is capped and says why. The measured rates, their intervals and every goal that is below its target are in
  `docs/evaluation.md`, `docs/v2/bench/shortfalls.md` and `docs/limitations.md`; this release claims no more than they state.
- **Evidence and reporting.** Cases with roles, browser pre-hash at intake, a write-once evidence store, a hash-chained
  custody log, annotations, conclusions, sign-off and peer review; four PDF report profiles (summary, standard, forensic,
  newsroom), signed, with evidence packages that `verify` and `reproduce` check offline. Under a provenance record an
  examiner's position is compared with the record's step on the scale: the steps of a conclusion count from it (the
  first under a trusted capture, the fifth under a trusted declaration), the conclusion, its signed body and its custody
  event carry `steps_from`, and a position two steps away needs a justification (refused with 422 without one).
- **Deployment.** One backend image for the API, the worker and the model fetch, nginx in front, SQLite by default and
  Postgres with a separate application role as a Compose file (`docs/runbook.md`). `docs/settings.md` lists every setting.

### Since the v2.0 build

- **Benchmark and fusion (v2.1, v2.1.1).** The benchmark `vb-1.1` (more real photographs; video and audio on the same
  bench); an image fusion model without image-size or face-size terms, whose figures are lower than those of v2.1 and
  are the ones to use; abstention where performance was not validated; the three served models tracked in the source
  (`backend/app/engine/fusion/release/`) and loaded from there.
- **What a result says.** The number, the scale and the wording of a result follow what was measured for it: no number for a
  capped result, text instead of a marker for an inconclusive one, "not calibrated" for the fallback. The page, the API,
  the JSON export and the PDFs state the same.
- **Explanations that match the decision.** A check's contribution is measured from its neutral point (the score at which
  that check alone favours neither answer), so a check that reads "against" no longer stands under an AI headline it
  raised; the starting point is named as a property of the model, not evidence about the file. Direction and strength of a
  check in the estimate come from its contribution, in one vocabulary on the page and in the reports. The reports state
  the headline rule in words, with what each level measured, in place of a table of fixed ranges. The check behind an
  audio or video estimate is the check the ledger shows. A result that could not be assessed says "No estimate was
  formed: …" and speaks of no estimate elsewhere. Results published earlier keep their rows and their wording.
- **No verdict outside what was validated.** Four rules, each of which only weakens a result; no detector and no fusion
  model changed.
  - An image of a kind the validation set does not hold is capped at "Inconclusive" with the reason: more than 53 % of its
    8 × 8 pixel blocks flat (screenshots of text and interfaces, flat graphics), a file that matches the screenshot rule,
    or a greyscale image. The limit is the 99.5th percentile of that share over the 6,546 images the checks were fitted
    and calibrated on. A screenshot that is mostly a photograph can stay below it (`docs/limitations.md`, "Images").
  - The two strongest headlines need two checks that scored. The served audio model holds one detector, so a recording
    reads at most "No significant indications …" or "Indications …".
  - A result of the uncalibrated fallback in which only metadata or provenance records scored (a camera's EXIF block
    under `quick`) is "not assessable".
  - Under a clean headline, "What this means" and the reports name a check that flagged the file and carries no weight in
    the estimate ("The sound track was flagged by a check that carries no weight in this estimate …").

  The v1 view and `export.json` carry the band, the label and the confidence of the published decision; no label says
  "authentic": the clean labels read as the headlines do, and a result that could not be assessed is labelled "Not
  assessable". The benchmark report applies the content rule to the files it scored and counts what each condition caps
  (`scripts/vbench_report.py --content`). The benchmark figures were published again on 2026-10-10 under the served
  rules D-64, D-65, D-76 and D-77 (`docs/v2/bench/shortfalls.md`, "Post-hoc rule changes"): `image/test` is 39.7 %
  inconclusive (705 of 1,778; 32.7 % before the rules) and 9.2 % of its decisions are wrong (99 of 1,073; 10.4 %
  before), `video/test` 50.2 % inconclusive (149 of 297; 48.5 % before), audio unchanged (72.8 %). Results published
  earlier keep their headline.
  Pipelines `img-2.0-p15`, `vid-2.0-p20`, `aud-2.0-a14`, `doc-2.0-p15`; report template set `r31`.
- **Reports and records carry what the examiner and the custodian said.**
  - A report that no examiner has signed does not read as cleared: its header says "Draft · not reviewed" (the newsroom
    explainer "… · not for publication"), a redacted report is headed "Redacted copy", and a draft conclusion is printed as
    "Draft opinion, not signed" without the declaration and the signature lines.
  - On a signed record the examiner's opinion stands directly under the machine headline, at its size, on page 1 of all
    four report profiles and on the result page: "Examiner’s opinion: Inconclusive, departs from the machine assessment by
    2 steps", with the justification in full. A finding's examiner status ("Disputed", "Needs check") is printed on its
    row in every profile and on the page. An examiner cannot sign "concur" or "concur with changes" beside a conclusion
    that departs; a record that pairs them prints the departure.
  - The standard and forensic reports print the intake record: on the cover the hash the custodian stated and how it
    compares, what was stated of the acquisition and the time Veritas took the file in (no longer under "Received"), and a
    legal hold; in the custody appendix the method, custodian, source, date acquired and notes. A stated hash that differs
    is a limitation in those reports and one line in the summary and the newsroom explainer. The case page shows the date
    acquired, the notes and the comparison on the exhibit's row.
  - A file name is stored and shown as it was received (any Unicode; control characters removed): intake no longer turns
    "фото_свидетеля_№4.jpg" into "4.jpg". Only the upload on disk and a download carry a sanitised name, which keeps its
    extension. Records made before keep the name they have.
  - An evidence bundle takes the forensic report that already holds the record as it stands and makes one only when
    there is none (`reuse` in `POST /api/v2/analyses/{id}/reports`).
- **Roles, the case and the session in the browser.**
  - The owner of a case adds and removes its members, with no case role above a member's role on the instance, and reads
    the custody log of the case (`GET /api/v2/cases/{id}/custody`, `GET /api/v2/cases/{id}/members/candidates`). "Edit
    case…" changes the title, the client or matter reference, the description and the classification, and closes or
    reopens the case (`status` in `PATCH /api/v2/cases/{id}`; a closed case takes no new exhibit); an administrator sets
    the second-review rule, the legal hold and the retention date there, and the legal hold at "New case". "Withdraw…"
    withdraws a signed revision with a reason. The custody log words each change ("legal hold on", "case closed",
    "retained until 01 Mar 2031") and names a removed member.
  - On a case that requires a second review, a revision the examiner has signed reads "awaiting second review" with the
    half glyph wherever its state is shown, the home lists what awaits a second review, and "Record peer review…" is in
    the header. The line of a twice-reviewed record names both reviewers. The reports of a case are listed in two
    groups, "Issued" and "Drafts", under the ids their PDFs carry.
  - The header of a case exhibit names the case, the exhibit and the revision and leads back to the case; the analyses
    list shows them under the file name. The page of a cancelled run says who cancelled it, when and why, and a
    cancelled run is run again from its retained source ("Run again").
  - What an account cannot do is not offered: nothing that starts work for a viewer or a reviewer, no delete on a
    signed analysis, neither a re-run nor an export on a superseded revision. The device, the count of checks and the
    queue in the sidebar are shown to administrators. `/design` opens for a signed-in person.
  - A session that ends while the application is open shows the sign-in form in place: the page stays, and what was
    typed is kept and can be sent after the sign-in.
  - A note is added on a sentence in the manuscript. The intake form refuses a mistyped custodian hash and a source that
    is not a web address before the upload, in plain words, and says what is missing while "Begin analysis" is switched
    off. The requests of one view of an original are one "original viewed" event.
- **Tables, lists and section links, at 1440, 900 and 375 px.**
  - The home case library is as wide as its card: a long case title takes two lines inside its cell with the whole title
    on hover, exhibits and analyses are one cell ("3 exhibits · 5 analyses"), the status and the classification stand
    under the title, and "Last activity" is on screen.
  - The analyses list has fixed columns and rows of one height. The file name and the examiner's position are one line
    each with the whole text on hover; at 900 px the date and the row menu are on screen; at 375 px each analysis is one
    stacked block.
  - At 375 px the evidence ledger shows the name of every check, and the ledger, the methods tables, the model
    catalogue, the measurements of a model card and the batch table are stacked blocks that name each value: none
    scrolls sideways.
  - The address of a section (`/analyses/<id>/review`) opens on that section once the page has its height, and the
    index marks the section the reader is in without changing the address. A reader who scrolls is never moved back.
  - The custody log shows the whole hash. A long file name stays in its cell with the name on hover. The home's
    review-state bar has a swatch and a fill of its own for each state. A dialog is no taller than the window and keeps
    its action row on screen. Placeholders are in the muted ink, fields and ghost buttons have their border in the
    graphic ink, and a field has one height in dialogs and on intake.
  - At 375 px the evidence viewer and the video workspace show their whole title. In the Expert view the strip of
    detector estimates is as wide as its section at every width (it was an 800 px drawing, and the page scrolled
    sideways on a phone), and the two ends of the contribution axis no longer print over each other.
- **The PDF reports, structure and finish** (report template set `veritas-report-2.0.r32`; reports issued before keep
  theirs).
  - Figures are numbered from 1 in the order they are printed, which is the stored order of the result: two analyses of
    one file and a re-run print the same figures under the same numbers. A table is "Table n", not a figure; the sampled
    frames of a video are one figure; the table of a map's regions stands under its map; every reference names a figure
    as the exhibits chapter does. `GET /api/v2/analyses/{id}/exhibits` lists in the same order (it listed by id), so the
    evidence viewer and the report agree; a cursor of that list from before the upgrade is refused with 400.
  - A chapter follows the one before it on the page: the full reports are three to five pages shorter and no page is left
    mostly blank. The headline of the summary and "What this means" stand on one page. A table of fewer than six rows is
    not split over two pages.
  - The summary, standard and forensic profiles carry one header centre, with the classification.
  - Appendix A says how to run the analysis again with its pinned manifest ("Re-run…"), as the last row of the manifest
    table and without an API path. The report of a result without a fusion model no longer points at a weight column.
  - The contributions have a column for the change in log-odds and one for the probability out of 100. The evidence
    ledger runs in the order of the S-ids and gives the reference room. A word is no longer broken inside a table cell.
  - One limitation for one fact; the sign-off counts the acknowledged limitations against those listed and has a
    signature line for each signer; the custody table has a detail column; the file size is in bytes in every profile; the
    glossary holds the terms the report uses; chart axes have round ticks; a spectrogram says that its brightness is
    relative to the recording.
- **Wording, the guided tour and the model cards** (report template set `veritas-report-2.0.r33`, sentence templates
  `tpl-2.12`; reports and results issued before keep theirs).
  - No vocabulary of the build on a page or in a PDF: a validation slice is said in words ("JPEG quality 90 or more,
    shorter side over 1,024 px, face under 120 px"), also inside a sentence a result stored; the values of the analysis
    record are words ("commercial licences only, with an allow list", "Apple GPU · 16-bit floats", "version 2.12", "no
    fusion model (hand-set weights, not calibrated)"); a check is named by its model card. The result document, the JSON
    export and the manifest attached to a PDF keep the recorded values.
  - One word for the role, "examiner" (the review status line keeps its decided wording). The review of a fusion model
    reads "pending human review" or "reviewed by <name> on <date>".
  - The count beside a result says what it counts ("14 checks ran, 3 in the estimate"); the key findings count findings
    ("2 findings point toward AI · 1 against"); a printed sentence writes "83 %" as the page does; the note of a check that
    answers another question than the estimate names both questions. Plain view holds no log-odds and no name of a
    calibration method. A lane and a segment are named as the check that drew them ("face-swap check").
  - The summary page of an image shows the examined picture, reduced; a copy that withholds the file's identity withholds
    the picture. The area of a region is one number in the viewer and in the PDF.
  - The guided tour: the card stands beside what it highlights at every step; a repeat demo opens only analyses a demo
    took in; a case that holds only demo samples is marked (`demo` in the case list) and is left out of "Needs review".
  - A model card's "Source" row names no file of the repository. The Models list has the toolbar and the table of the
    other lists; the lens rail of the viewer has one letter for each lens (O for Regions).
- **Which checks count in a result** (code review of a511ecf; pipelines `img-2.0-p16`, `vid-2.0-p21`, `aud-2.0-a15`, report
  template set `veritas-report-2.0.r34`; results and reports issued before keep theirs).
  - A check counts for the two strongest headlines, and in "Calibrated fusion of k of n detectors", when its row in the
    breakdown is not zero. A check with a score that the fusion model gates off for the input no longer counts, so a
    result that rested on one check and such a score is one step weaker than before, and "k" is the number of rows.
  - A result in which checks scored and the fusion model gives none of them weight for the input is "not assessable",
    in words of its own, where it published the model's starting point as an estimate before (with the served models: a
    video on which no frame check applies while the container check or the voice check scored). The page and the four
    reports list the findings of those checks as checks outside the estimate, never as reasons.
  - A provenance record over a result in which no check counts decides alone: the record's own figure, no statistical
    estimate beside it, and confidence low for a self-declared marker.
  - The benchmark report counts the same way (`bench/report.py`, the mark "fewer than two checks").
  - A result without a fusion model for an image or a video whose condition step failed is capped at inconclusive, as a
    result below the size floor is: nothing compared the input with what was validated. Before, it could read "Likely
    AI-generated" with a score.
- **A result that a provenance record decides is headed by that record** (N-19h; pipelines `img-2.0-p17`, `vid-2.0-p22`,
  sentence templates `tpl-2.13`, report template set `veritas-report-2.0.r35`).
  - The decision names the record it applied (`provenance_record`; a trusted declaration, then a trusted capture, then a
    self-declared marker), and the page and the reports take the basis, the headline and the reasons from that record,
    never from the findings: a file with a trusted capture and a generator named in its metadata is headed as the
    capture's. A marker is worded by its kind (unsigned metadata, a public watermark message, Content Credentials by
    their signer), and the record's findings rank first as reasons.
  - A number that the record set is shown as the record's, in words ("Set by the record: 85 out of 100, the floor of
    self-declared evidence"), beside what the statistical checks alone give: no large numeral, no marker on the scale,
    no fact "Calibrated". Outside what was validated no statistical number stands beside a record. A record that decides
    alone has no account of a fusion: the breakdown is one sentence, the agreement "not reported (no statistical
    estimate)".
- **A decisive headline needs a check that supports it (D-76), and a video counts its checks one way** (N-19i; pipelines
  `img-2.0-p18`, `vid-2.0-p23`, `aud-2.0-a16`, report template set `veritas-report-2.0.r36`).
  - On the calibrated path a headline of one answer needs one check that counts with a row of at least ln 2 (a factor of
    two on the odds) toward it; otherwise the result is inconclusive, and says which way its estimate pointed and that
    it rests on the model's starting point (the QA's camera original read AI at 68 % on a start of 53). The benchmark
    report applies the rule to its decision figures. N-19m adds a second condition (below).
  - Each row of a check says whether it counts (`counted`), and the count "in the estimate", the agreement and the
    sentence on the checks that entered read it: a video's container check, which tests provenance, is shown and not
    counted.
- **The record of a review says who declared what** (N-19j; report template set `veritas-report-2.0.r37`).
  - "Do not concur" and "concur with changes" need a comment, and a conclusion equal to the machine's cannot be signed
    "do not concur". A decision names its object ("concurs with the machine assessment", "… with the examiner’s
    opinion"); beside a result that is not assessable a sign-off is "Opinion recorded, no comparison with the machine
    assessment".
  - The forensic report prints each signer's statement with that signer's own signature line ("Statement of the peer reviewer").
    A signed report that is not issued says why in its running header ("Draft · second review outstanding", "Draft ·
    second review: does not concur"), with the refusal under the examiner's opinion; "with changes" prints the changes.
  - Only the owner or an administrator changes a case's fields, a case role is limited by the member's instance role, the
    custody log names its actors, and an original downloads under the name of the caller's own exhibit. The report
    counts the acknowledged limitations against those put to the examiner; a time of receipt later than the intake is
    refused; a conclusion typed while another tab saved one is kept (made to hold after a refused save by N-19n).
- **What a result and its reports say** (N-19k; pipelines `img-2.0-p19`, `vid-2.0-p24`, `aud-2.0-a17`, `doc-2.0-p16`,
  sentence templates `tpl-2.14`, report template set `veritas-report-2.0.r38`).
  - A check that scored and is not counted says why ("This check scored 0.91 and is not counted in this estimate: …":
    at chance in testing, a closed gate, or not in the fitted model), and its lane is drawn for inspection only. The
    breakdown of a result outside what was validated is "What the checks returned": bars from zero in log-odds, no
    running total, no percent.
  - Under a result that is not assessable the scope table says that checks ran without a usable score, and a camera's EXIF
    block is "A record of the file, not evidence of authenticity", no key finding. Checks that could not run on the
    server are one caveat; checks, steps and figures are named in words, not by ids.
  - A draft package verifies with "DRAFT package: the report in it is not issued"; numbers and dates in report tables do
    not wrap; a one-page report that runs over is set tighter until it fits.
- **A stored result, a record that was not applied, the number beside a record** (N-19l; pipelines `img-2.0-p20`,
  `vid-2.0-p25`, sentence templates `tpl-2.15`, report template set `veritas-report-2.0.r39`).
  - A result stored before N-19h is read as a fresh one: its record is worked out from its stored signals.
  - Under a trusted capture a self-declared marker that the file also carries is named as outranked ("The decision did
    not apply that marker: the signed capture record outranks it, …"); the row of a self-declared record names its kind.
  - The decision says what the checks alone decided beside a record (`statistical_alone`); where they do not decide, the
    number reads "Estimate of the statistical checks alone 83–96: they alone do not decide". Outside what was validated
    nothing prints the estimate, and the record's row carries no amount.
- **A capture's result, and the two conditions of a decisive headline** (N-19m; D-15, D-76 as amended, D-77; pipelines
  `img-2.0-p23`, `vid-2.0-p28`, `aud-2.0-a18`, report template set `veritas-report-2.0.r41`).
  - A result that a trusted capture decides is headed by the record (D-15): the capture record is its basis, its
    headline stands on the first step of the scale, no agreement is given, and the lists read "Capture record". Beside
    any record, where the checks alone do not decide, their number says so ("13 out of 100, they alone do not
    decide"); under a self-declared marker the confidence is then low.
  - D-76 needs one check that counts with a row of at least ln 2 toward the answer, and the rows of the checks that count
    together at least ln 2 toward it. D-77 needs the estimate, as printed, in its answer's part of the scale: 34 or less
    for a clean headline, 65 or more for an AI one. Otherwise the result is inconclusive and names the condition. On
    `image/test` D-76 takes 39 decisions by its first condition and 19 by its second, D-77 46.
- **The rest of the reviews of 9fdbe83** (N-19n; pipelines `img-2.0-p29`, `vid-2.0-p31`, `aud-2.0-a20`, `doc-2.0-p17`,
  sentence templates `tpl-2.16`, report template set `veritas-report-2.0.r42`).
  - AI mentioned in a metadata text field (a description, a PNG Comment) no longer fails the step that writes the
    findings: it is a finding of its own that sets no number. A result whose findings could not be written says so and
    asks for a re-run. The IPTC digital source type is read under any prefix and in a compressed PNG packet, headed "The
    file's metadata declares AI generation"; the reader is linear and reads a packet up to 64 KB.
  - A marker's finding is "Self-declared: …" (or "Self-declared, not applied: …") with an open diamond and no strength
    bars; records that did not decide are listed as "Provenance findings that did not decide this result"; Content
    Credentials that do not validate are titled so and marked "invalid".
  - Moved steps: under a fallback declaration the steps of a conclusion are counted from the declaration's step (the
    fifth), so "Strong indications" agrees and "Inconclusive" is two steps away, and an API client that records it
    without a justification is refused with 422. Under a trusted capture the examiner's line reads "on the step of the
    capture record" or "2 steps from the step of the capture record", and the server refuses with 422 a concurring
    sign-off away from that step, a not-concurring one on it, and "No opinion" signed as concurring.
  - Beside a record the statistical number says when the checks alone point the other way ("98 out of 100, they point
    the other way") or only one check scored. The caveat that shows it has one form ("Statistical estimate without the
    provenance record: 98 out of 100."; the fallback's "Statistical score without the provenance record: 90 / 100, not
    calibrated."; the stored caveat stays), and in the machine's line of the sign-off section and dialog the number's
    phrase is in the line's case ("… · estimate 53–78: inconclusive").
  - Outside what was validated no starting point is given: the breakdown holds what the checks returned, the key
    findings say that none is a reason, and "What this means" says why the input is outside.
- **What the re-checks of N-19n found** (N-19o; pipelines `img-2.0-p30`, `vid-2.0-p32`, sentence templates `tpl-2.17`,
  report template set `veritas-report-2.0.r43`; checks `img.meta.generator` 0.4.5, `img.meta.consistency` 0.4.2,
  `prov.c2pa` and `prov.c2pa.video` 0.4.2).
  - Under a declaration on the fallback a conclusion's steps count from the step of the declaration (the fifth), while
    the decision's code is the fourth: a conclusion on `decision_code` ("Indications …") is one step from it, and is
    refused as concurring. The form, the lines, the lists and case rows, the custody log, the dialog, the server's
    refusals and the four reports say "the step of the declaration", never "the machine decision".
  - Text from a file reaches no record or report with a control or direction character (the cells of an exhibit table,
    a check's `observed`; a C2PA claim generator and signer's name are also cut to 200 characters). The PNG text that the
    metadata checks read is bounded (64 chunks, 64 KB in all), ExifTool reads a copy so bounded, and the XMP history is
    read in one forward pass.
  - A mention of AI in a metadata text field is no marker in the evidence ledger (no direction, no P(AI), no dot), and
    its finding names the field ("the EXIF image description", "the PNG text chunk “Comment”"). Under a configured prior
    the share is said wherever the number of a result is printed. A check whose row is not counted (the container check of
    a video) is titled by what it tests, and no contribution prints as "0.00 log-odds".
  - The owner's step gives a member of the owner of the tables the sentence of that case, refuses a superuser as the
    application role by name, and answers a connection lost at the role statement with one line (exit code 1).
- **What the re-checks of N-19o found** (N-19p; pipelines `img-2.0-p35`, `vid-2.0-p34`, sentence templates `tpl-2.18`,
  report template set `veritas-report-2.0.r45`; checks `img.meta.generator` 0.4.7, `img.meta.consistency` 0.4.5,
  `img.meta.exif` 0.4.1, `prov.c2pa` and `prov.c2pa.video` 0.4.3, `vid.meta.container` 0.3.1, `vid.codec.ffprobe` 0.4.1).
  - A result that a trusted declaration decides on the fallback reads "AI declaration" in the Analyses list and on a
    batch page and the declaration's sentence on the case row, by what the stored result names (`deciding_record` on the
    list, the detail and the batch items); it read "Indications".
  - The PNG keys that the metadata checks read by name (Software, parameters, prompt, workflow, Comment, the InvokeAI
    keys) are read wherever they stand, past the bound on PNG text, so a generator marker past the 64th text chunk decides
    as before; the generator check says how many text chunks the bound left unread. ExifTool reads a copy of a PNG through
    IEND with one XMP chunk outside the bound, also where a chunk runs past the end, and reads the thumbnail from it.
  - Text from a file keeps its words apart (a tab or a newline is a space) and is filtered in the title and caption of
    every exhibit, a C2PA thumbnail's title (cut to 200 characters) and the signal of a video's container tags.
  - Beside a record a configured prior is said of the statistical checks' estimate; a mention's expanded ledger row says
    what the finding says and that it sets no number; a justification typed on the machine's step is "Justification",
    never "of the departure"; the chart's starting point reads "out of 100"; a row that the decision does not count
    reads "not counted" with no bar, and the coverage line counts the statistical checks only.
  - The owner's step refuses as the application role a member of a superuser role and a role with CREATEROLE, and a
    role statement cancelled by a lock timeout says its cause, not a lost connection.
- **What N-20b's read of the reports found** (N-19q; pipelines `img-2.0-p36`, `vid-2.0-p35`, `aud-2.0-a21`,
  sentence templates `tpl-2.19`, report template set `veritas-report-2.0.r46`; no decision, number or fit changes).
  - The inconclusive result of D-76 says what the rule reads: "no check's part in this estimate reached weak support (a
    factor of 2) toward that answer", with one sentence more where a check taken alone or a check outside the estimate
    supports that answer and why that does not decide the result; the note of the scale and the headline rule say the same.
  - The reports agree with themselves: the glossary gives the estimate's range as 2 to 98; a Content Credentials check that
    found no manifest "ran and found no manifest in the file"; an opinion whose sign-off concurs with changes says so;
    Veritas's own rules are no publication; the findings on the analysis carry no miniature; an inconclusive range is the
    "Plausible range", in the summary and in the last row of the breakdown; a heading whose two readings differ names its
    reading; an inconclusive result of the levels says why.
  - On the fallback the score is called the score (its caveats, the re-run note, the one-page foot); the scope row counts
    the checks that ran; the envelope sentence names the input's slice; a recording's lane, its H1 row and its spectrogram
    caption are in the report's words; a heuristic outside the fitted model prints no P(AI); a regions table names its
    values (a logit with its unit, a score, a probability).
  - The agreement line counts the checks that count in the estimate. On a video, the face-swap check that one detector
    carries is named by the score that its finding quotes (its mean over the face frames), and the ledger says why P(AI)
    differs from that score; the earlier method's per-frame series and the lanes disclosure no longer read as said of
    that check; a video's per-frame face boxes are named by frame and box, not as faces; the GOP strip of one key frame
    prints its one tick.
- **PostgreSQL and intake** (N-19e, N-19g, and the owner's step of N-19n).
  - An upload whose media record or stored original a purge removed meanwhile is taken in again, or answered with the
    500 "… retry the upload.", never an unhandled error; `retention purge` removes the media records of uploads that were
    never recorded. A refused start logs `startup.refused` with its sentence, and a command prints `error:` and the
    sentence without a traceback. The people who can be added to a case are in name order on every engine.
  - The owner's step and the restore of a Postgres backup do what they say ("Database" and "Restoring a Postgres
    backup" below).
- **Records.** Retention purges cases past their retention date and unsigned quick analyses past
  `VERITAS_RETENTION_DAYS`, with their bytes, and keeps held, signed and in-progress records; an action and its custody
  event are written together or not at all; reports and packages are issued only for a state that may be issued; backups
  (`backup create`, `verify`, `restore`) and `store verify`. An analysis that was interrupted keeps what its finished
  stages wrote while it waits (`store gc` and a purge leave it); a stage whose objects are missing runs again, and no
  result is published while an object it names is missing. A step of a run that cannot read an object of the evidence
  store fails the analysis; a step that fails for another reason and does not end the run is named in the result ("The
  step that marks regions failed: this result shows no regions from it."), on the page, in the reports and in the
  coverage finding.
- **Operations.** Liveness (`/api/health`) and readiness (`/api/ready`) as two probes, a worker heartbeat, `/api/metrics`,
  JSON logs with a request id, a floor on free disk space (HTTP 507 at intake below it), a migration lock for processes
  that start together, a worker that gives its job back when it is stopped, memory limits per service, and a capacity plan
  that says at start which checks a host is too small for (`python -m app.cli models plan`).
- **Security.** Keys of at least 32 characters in production and no derived signing key; request input checked at one
  place; a content security policy; session and proxy-address handling; every container without capabilities, and the
  backend containers with a read-only root file system.

### What an operator has to know when upgrading a build of this repository

- **Version and commit.** The version is `2.1.1`, not `0.1.0`, and `VERITAS_VERSION` is no longer read. An image is built
  from a named commit: use `make up`, or set `VERITAS_BUILD_COMMIT=$(git rev-parse HEAD)` for any other build; a backend
  image without a full commit fails at build time. The configuration hash of a result no longer includes the version; a
  re-run of a result made by an earlier build reports the version and the configuration hash as changed.
- **Keys.** Production refuses an API key or a signing key shorter than 32 characters and placeholder values.
- **The signing key of an instance that ran in production before this build.** Without `VERITAS_SIGNING_KEY` such an
  instance signed its conclusions, sign-offs and webhook secrets with a key derived from the API key. This build never
  derives the key in production, and its first start on that data directory is refused with a sentence that names the
  way out: `python -m app.cli keys adopt-derived`, run once, writes the derived key to `data/keys/signing.key` when
  every signed record verifies under it, and records the act in the custody log (`--api-key-env NAME` when the API key
  has changed since). Then start again, take a backup (the one taken before the upgrade does not hold the key file)
  and change the API key (`docs/runbook.md`, "Upgrades", step 4). No release has run in production: the command was
  run on records that this build made outside production, which derives the key the same way, and not on the data of
  an earlier build.
- **Compose.** The worker's memory limit defaults to 28 GB (a 32 GB CPU host runs every check of `full`); the web
  container listens on 8080 as an unprivileged user; Postgres is `docker-compose.postgres.yml` with two roles and two
  passwords; `VERITAS_WARMUP_ON_START` no longer exists. The files need Docker Compose 2.24.4 or later, and an older
  one is not always refused: 2.24.0 to 2.24.3 start the TLS stack with port 8080 published as well (`make deploy-lint`
  compares the version). The backend image runs as uid and gid 10001; the web image's base is nginx 1.30, the stable
  line. Both images were built from the release commit `a876a43a` and run as a stack on SQLite, on PostgreSQL and with
  the TLS file (`docs/v2/qa/prod-readiness-2026-09-29/supervisor-checks.md`, check 50).
- **Backups in the Compose stack.** Make the host directory for the image's user first (`sudo install -d -o 10001 -g
  10001 -m 0700 /srv/veritas-backups`) and run the backup commands with `--no-deps`. Before the first start of this
  build, back up with this build's command: it leaves the database of an earlier build at its revision
  (`docs/runbook.md`, "Upgrades", step 3). These commands ran as written on Docker Desktop on the release commit
  (check 50); they were not run on a Linux host.
- **Stops and holds.** An analysis whose job is given back at a stop reads `queued` again, not `running`; a worker
  that waits for free space logs `worker.on_hold` once per hold.
- **Database.** Migrations run at start (`backend/alembic/versions`); SQLite runs in WAL mode; on Postgres run
  `db setup` after every upgrade so that the application role is granted new tables (the Compose file does it at every
  `up`). Run that owner's step as the role that is to own the tables. Two steps started together run one after the
  other; the role statement goes first, with the password only as a SCRAM-SHA-256 verifier. Before anything is changed
  it refuses (exit code 2) a connection without the rights of the database's owner or without CREATE on the schema,
  tables of another role, and an application role that owns a table or the function `audit_events_append_only()`
  of the triggers that keep the custody log append-only, or is a member of a role that does; it ends by reading back
  the schema's rights. A refused role statement prints `error:` and one sentence (exit code 1); `db grants --apply`
  refuses the same.
- **Restoring a Postgres backup.** `backup create` prints the procedure and `docs/runbook.md` ("Backup and restore")
  holds it word for word: a new database (`createdb -O <owner>`) before anything is dropped, then
  `pg_restore --exit-on-error --no-owner` (with or without `--no-acl`), run with the server's own program as the role
  that is to own the tables, then the owner's step. Do not restore with `--clean --if-exists` into the existing database:
  measured on a dump at revision 0018, it left three newer tables, and the owner's step then failed. The Compose form of the restore was not run.
- **API clients.** The API refuses more requests: "do not concur" or "concur with changes" without a comment (422); a
  case's fields changed by a member who is not its owner (403); a case role above the member's instance role (422); a
  time of receipt later than the intake (422); under a fallback declaration "Inconclusive" without a justification
  (422); under a trusted capture a sign-off decision that does not fit the conclusion's step (422).
- **Install from source.** `make setup` installs the versions of `backend/requirements/tested-versions.txt` (with the
  Postgres driver), the frontend from its lock file (`npm ci`) and the browser of the browser tests; run again, it
  brings an existing environment back to those versions.
- **Reports.** A report made by an earlier build keeps the text and the version it was made with; request a new report
  revision for the current wording.

### What the release is not

- **Not the v2.2 detection programme:** the benchmark `vb-1.2` with a fresh lockbox and balanced container geometry; the
  operating point on out-of-fold real items; real videos at 480 to 719 px and AI images natively at JPEG quality 70 to 89
  above 1,024 px, so that those slices stop abstaining; the video face-presence prior; admission of the video and audio
  detectors; a stronger voice detector; a calibrated document fuser on a provenance-clean text benchmark. The test split of
  `vb-1.1` has been read, so a change to a served model cannot be evaluated honestly before `vb-1.2`.
- **Not a cloud deployment and not GPU images.** The Google Cloud bundle under `infra/gcp` passes its offline checks and
  nothing was deployed with it; no build or run of a CUDA image is on record.
- **No user-administration screens and no quotas per principal.**
- **Not proof.** A verdict is probabilistic evidence with stated error rates and limitations; a qualified person reviews
  it before any consequential decision (`docs/limitations.md`).
