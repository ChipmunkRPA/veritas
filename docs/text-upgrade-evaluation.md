# Document and email detector comparison

The current document pipeline publishes inspection scores and source locations. This study
tests which text check can support a reference-calibrated tendency for the assessed writing.
The existing source positions, email body boundaries and evidence exports are separate
from the document-level decision.

## Frozen inputs

The local manifest SHA-256 is
`e10769c188e69889c54f1116c5a4f8f03cbb58f4194a19a3a1ea9e969a92d28c`.
The preparation script records source revision, archive hash, record identity, source
group, authorship label, generator and text hash. Text and weights stay in ignored local storage.

| Input | Role | Samples |
| --- | --- | ---: |
| [LLMTrace](https://huggingface.co/datasets/iitolstykh/LLMTrace_classification/tree/252e21e3dca713bc82bc3c1ef73bf533d7c9fc7e), Apache-2.0 dataset declaration | English articles and news, matched by topic, with 15 represented generator labels | 160 reference fit, 240 held out |
| [PhishingSpamDataSet](https://github.com/HarmJ0y/PhishingSpamDataSet/tree/d13e9bb73d03f458e6d5b6cb6c7b0a708ea47e3d) | Legitimate messages paired with AI rewrites by original message number, across 11 generator labels | 240 evaluation only |
| [AITDNA](https://huggingface.co/datasets/UKPLab/AITDNA/tree/21d77f7d44833507750c0da309295c4147c1bda5), CC BY-SA 4.0 dataset declaration | Human–AI coauthored texts with source span annotations, one per author | 40 evaluation only |

The [6 November source/position audit](writing-source-preflight-v2.md) found that
the pinned AITDNA card's metadata declaration above conflicts with its prose
GPL-2.0 declaration. The earlier metadata record does not resolve those terms.
No new training or source-text/annotation redistribution is authorized by that
record. The new audit checks all 362 rows and inventories unused authors without
running a detector or changing the historical results below.

Selection happens before inference. Binary sets use one original and one AI text per
topic or original message group, with normalized exact-text deduplication within each
domain. Text length is 50–750 words by whitespace count, and email selection uses a
predominantly ASCII-letter filter. This is an English-oriented sampling heuristic.
Mixed writing stays outside the binary fit and evaluation. Email labels describe original
versus AI-rewritten body text. The study measures neither sender identity nor phishing.
Explicit commercial training rights for the email corpus and its upstream sources
are unestablished. It is used for local evaluation only. A separate language
identification pass marked 236 of the 240 email bodies English and four Norwegian.
The fixed email transfer set retains all 240 and records that limitation.

## Model comparison

- Incumbent [Desklib checkpoint](https://huggingface.co/desklib/ai-text-detector-v1.01/tree/5fdea974cd4287c61674951ec78803aa274e2fb7).
- Candidate [four-class DeBERTa checkpoint](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615).
  The publisher explicitly declares Apache-2.0 and commercial use. The study
  independently implements the published mean pooling and sums the three non-human classes.
  Weight SHA-256 `4a1561fadf44ec72934edd6158ff8c76e9388ade1384dbeeea6eb15f93251087`.
- Candidate [ModernBERT checkpoint](https://huggingface.co/rasbt/ai-text-detector-modernbert/tree/a6efd4bf6ecd852f332b2569ae0f98cae13b7985).
  Apache-2.0 license file. Uses published temperature and label mapping.
  Weight SHA-256 `e7f7a617ad4af3abd8bcb939d564d2a2ea36fdb24b893177ae10c12ebe869361`.
- Existing Binoculars pair, measured separately and in matched combinations to
  quantify whether it helps enough to keep it in the decision.

All sequence classifiers use the current application's sentence segmentation,
roughly 350-word chunks and complete tokenizer window scoring. ModernBERT is evaluated
under this integration policy with the same 768-token window limit. Article calibration
fits monotone logistic mappings on the 80 fit topics. Each method uses the same
balanced reference prior and 80% tendency boundary. Compare held-out Brier score,
correct and incorrect conclusions in both directions, and abstentions.
Paired uncertainty resamples complete source groups. A length-only control detects
whether differences in sample length can explain apparent performance.
Email transfer and mixed-author scores characterize failure modes outside that fit.
Training-corpus disjointness for the checkpoint publishers is unverified.
The optional interval run resamples fit topics to measure uncertainty in the
calibration parameters. This interval describes the fit, with distribution shift
and checkpoint errors outside its scope.

The passage comparison uses the 40 annotated mixed-authorship files. It checks
ground truth for the full input window and the highlighted sentence separately.
Both fractions are projected through the same extraction offsets that the
application publishes. A focus sentence can differ in authorship from its
neighbours. The source-coordinate smoke check covered 674 windows.

Replace an incumbent when the matched results show a meaningful improvement in
the intended scope, with the practical error and coverage tradeoff reported.
Keep the prior implementation available for rollback. Whole-document scoring
does not convert passage-level responses into calibrated authorship probabilities.

## Reproduce locally

From `backend`, with the three pinned archives in `data/text-upgrade/sources`
under the filenames in the preparation script, and candidate snapshots in
`data/text-upgrade/models/{tropa,modernbert}`:

```sh
.venv/bin/python scripts/prepare_text_upgrade.py data/text-upgrade/sources data/text-upgrade/manifest.json
.venv/bin/python scripts/score_text_upgrade.py desklib data/text-upgrade/manifest.json data/text-upgrade/desklib.jsonl --device mps
.venv/bin/python scripts/score_text_upgrade.py tropa data/text-upgrade/manifest.json data/text-upgrade/tropa.jsonl --device mps --checkpoint data/text-upgrade/models/tropa
.venv/bin/python scripts/score_text_upgrade.py modernbert data/text-upgrade/manifest.json data/text-upgrade/modernbert.jsonl --device mps --checkpoint data/text-upgrade/models/modernbert
.venv/bin/python scripts/score_text_upgrade.py binoculars data/text-upgrade/manifest.json data/text-upgrade/binoculars.jsonl --device mps
.venv/bin/python scripts/calibrate_text_upgrade.py data/text-upgrade/manifest.json data/text-upgrade/comparison.json data/text-upgrade/{desklib,tropa,modernbert,binoculars}.jsonl --intervals
.venv/bin/python scripts/evaluate_text_passages.py desklib data/text-upgrade/manifest.json data/text-upgrade/sources/ait-span.parquet data/text-upgrade/passages-desklib.json --device mps
```

Choose fresh output filenames for repeats. Each prediction file commits the input
manifest and every scored sample's text hash. The scoring program does not make
network calls or submit customer analyses.

## First completed run

The four-class candidate scored all 680 files. Its saved prediction SHA-256 is
`adb6437b096020ffe398181adcd88300af68a0a6d27423bb9e5f33058fb0ca00`.
With the article fit and the 80% tendency threshold:

| Held-out set | Correct AI | False AI | Correct original | False original | Inconclusive | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Articles, 120 per class | 86 | 2 | 87 | 10 | 55 | 0.094 |
| Email bodies, 120 per class | 8 | 1 | 82 | 49 | 100 | 0.275 |

The article fit transfers poorly to this email source. A separate, suitable
email calibration and independent evaluation are needed for an email tendency.
The incumbent comparison was still in progress when this result was recorded.
This run alone leaves the model replacement decision open.

## Article pilot and second-source decision

The three sequence classifiers completed the same 680-file evaluation. The first
candidate gave 173 correct and 12 incorrect directional decisions on the 240 held-out
articles, versus 127 correct and 4 incorrect for the incumbent. The other candidate
gave 49 correct and 2 incorrect. Article Brier error was 0.094, 0.113 and 0.170,
respectively. The first candidate's paired Brier difference against the incumbent
was -0.019, with a group-bootstrap 95% interval from -0.041 to +0.005. The local
selection reflects the combination of better estimated calibration and 46 more
correct conclusions with eight additional incorrect conclusions. The observed
improvement supported an isolated pilot while its magnitude remained uncertain.

The pilot article fit was pinned by SHA-256 and used 200 topic-bootstrap draws
for a 90% reference-fit interval. It was evaluated for English news and articles
of 50–750 words. Its paired worker and PDF verification below predates the
separate email calibration.

Local worker verification used the same held-out article as a TXT file and as the
body of an EML container. The TXT produced 96.8% toward AI-generated or AI-rewritten
text with a 94.6–98.1% reference-fit interval. The EML stayed inconclusive. The
worker response, JSON export and rendered 15-page report agreed. Model identifiers
were absent from those customer outputs. The report used the recorded 86/120 and
2/120 counts at the applicable 80% threshold. The running app returned HTTP 402
for a signed-in unpaid document request without adding an analysis or job.

The [Defactify](https://arxiv.org/abs/2510.22874) transfer test used 144 articles
from a second publisher. It froze 72 length-matched pairs across six generators
before inference, excluded exact and near duplicates, and reused the two original
fits without adjustment. The manifest SHA-256 is
`1aa08df98620fbe9d251f978ee6f61042cd7aa9d2f98937ec0f35f16a3b48d41`.

| Frozen fit | Correct AI | Correct original | False original | False AI | Inconclusive | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Incumbent | 18 | 60 | 0 | 0 | 66 | 0.087 |
| Four-class candidate | 8 | 68 | 19 | 0 | 49 | 0.212 |

The candidate's paired Brier difference was +0.125 (95% interval +0.098 to
+0.153). All 19 incorrect directional decisions labeled generated text as
original. The pilot calibration is retired for new analyses and re-runs.
New local document analyses keep source-linked passage evidence and withhold
whole-document confidence. The incumbent comparison used an offline fitted
mapping and its figures do not describe the legacy production verdict.

The first candidate is installed at its pinned model revision. The local
`VERITAS_TEXT_DETECTOR_MODE=candidate` setting enables the passage check and the
separate email decision below. Use `legacy` and restart the API to restore the
prior classifier. The repository defaults to legacy.

## Separate email calibration

The fixed federal email source is the public
[Kaggle export of released State Department messages](https://www.kaggle.com/datasets/kaggle/hillary-clinton-emails).
The archive SHA-256 is
`e82e01f8b045818a069715c5574df18bcc73b009798e839d94c63ddc32511caf`.
The selection keeps fully released records with federal sender addresses and
traceable document IDs. It removes quoted threads, release artifacts and calendar
lists, excludes visible contact details, and deduplicates exact and near-duplicate
prose. This operational filter focuses on official correspondence. [Federal government copyright guidance](https://www.usa.gov/government-copyright)
distinguishes official duty text from third-party material in a government release.

The originals manifest SHA-256 is
`5442c89d86f7c8d3a052df7f39ed6df8fbb6f0f32dd7fb286cff240ee2745f93`.
It fixes 60 fit originals across 16 authors and 60 held-out originals across
13 different authors. Pinned Qwen2.5 and SmolLM2 models produce direct and light
rewrites locally, balanced across the two splits. Every excluded or accepted
generation is retained in the journal with its exact source hash. The
independent 240-item business-email challenge is reserved for transfer evaluation.
The federal source is from one period and institutional setting, so transfer
performance matters before enabling a broader email confidence.

Both generators finished the original and revised prompt runs. The first run retained
45 matched pairs after the fixed output checks. Before detector scoring, a second
prompt specified a word range and allowed removal of one leading subject heading.
That run retained 93 matched pairs, including 49 fit and 44 held-out originals.
It excluded 24 out-of-range bodies, one assistant wrapper and two outputs containing
contact details. The pilot and revised journals remain separate. The 186 retained
samples are now being scored by the same three classifiers, with the independent
business-email results reused only for the transfer evaluation.

### Email rollout decision

On the 88 held-out local emails, the selected candidate reduced Brier error from
0.179 to 0.131 versus the incumbent and correctly flagged 24 AI rewrites instead
of 15. Each falsely flagged two human messages. On the separate 240-item business
email challenge, the fit reduced Brier error from 0.241 to 0.224 and flagged 19
AI rewrites and two human messages. The incumbent flagged four AI rewrites and
three human messages. The improvement is practical, with uncertainty in its
magnitude. Adding the smaller sequence classifier improved the local set but
worsened both Brier error and false AI flags on the independent set relative to
the selected single candidate.

The email calibration is separate from the article fit. It applies to English
plain-text EML bodies of 50–220 words whose source map contains one message body
without detected quotes or signature blocks. Reconstructing EML files and
applying this runtime eligibility rule retained 157 business-email items with
unchanged normalized inference text. At 80%, the new fit flagged 12/79 AI rewrites
and 1/78 human emails. The same subset contained 12 rewrites below the old
original-writing boundary. Accordingly the email policy publishes only an AI
tendency at the existing 80% boundary and leaves other responses inconclusive.
Headers, attachments and sender identity are outside this scope.

The frozen email bundle
`a03d90c9d3d342000e742505743e30d4847017775b57c5b50fb9373c09f8db2b`
contains the fit and its 200 author-group resamples, with commitments to the fit
and transfer datasets. Local worker verification used one known rewrite, one
low-response original and a quoted email. The rewrite produced a 99.2%
reference estimate with a 98.4–99.7% interval. The original stayed inconclusive
and the quoted sample stayed outside the fit. The API and JSON agreed, and
the rendered r70 PDF preserved the evidence and measured test counts without
publishing detector names.

### Mixed-writing evidence

All 674 source windows in the 40 annotated coauthored texts aligned between
the two model runs. AUC on purely human or AI focus sentences rose from 0.797
to 0.878 (paired document-bootstrap 95% difference interval +0.012 to +0.162).
For wholly human or AI surrounding contexts it rose from 0.826 to 0.919
(interval +0.004 to +0.197). The context and focus results remain inspection
scores. Their source links show which words were scored without assigning an
authorship probability to each highlighted word.

The pinned article result reproduced through the normal re-run endpoint during the pilot.
An isolated worker in legacy mode successfully ran the incumbent on the same
article. Both calibration bundles leave rollback available through the existing
text mode setting.
