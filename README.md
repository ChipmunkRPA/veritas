# Chipmunk Verify / TrueTrace

Open-source, locally runnable screening of potentially AI-generated or manipulated
images, video, audio, documents, and email. **TrueTrace** is the project name used
for the Human-Scale AI proposal; the application is branded **Chipmunk Verify**.

The application links findings to inspectable source evidence and reports the
scope and uncertainty of each assessment. It is under active development and
has known detection gaps. Automated SaaS reports are preliminary screening and
are not prepared for use as court evidence or expert testimony.

## What the application provides

- **Images:** original crops, region overlays, provenance and metadata observations,
  and visual-generation and editing checks. An overlay is an inspection location,
  not proof that every highlighted pixel was edited.
- **Video:** sampled frames, tracked-face review intervals with before/during/after
  context where available, voice evidence, and audio/picture timing observations.
- **Audio:** selected source-bound speech windows, waveforms, spectrograms, and
  separate background/acoustic context. Background labels do not establish AI origin.
- **Documents and email:** source-positioned writing observations and passage links.
  Whole-article authorship currently lacks a supported calibrated conclusion;
  the limited English plain-text email-body calibration has a separate scope.
- **Findings:** observed evidence, AI-leaning/original-leaning/inconclusive/context-only
  interpretation, confidence only where the stated scope has suitable calibration,
  and whether a check enters the overall estimate. Finding percentages are not averaged.
- **Exports:** PDF, JSON, and evidence packages with corresponding source coordinates
  and integrity records. Ordinary customer outputs omit internal model names.

The tested local configuration uses 16 primary model entries and 34 enabled
model-based and forensic checks. These counts depend on selected modes, profiles,
installed files, licensing policy, and available memory; all checks do not run on
all uploads. Source and weight pins are in [the manifest](backend/app/models.yaml).

## Run locally

Use Python 3.12, Node.js 20+, ffmpeg/ffprobe, and platform libraries for WeasyPrint
(Pango). ExifTool and Tesseract are useful for metadata and OCR. Apple Silicon
CPU/Metal is the locally exercised configuration; other hosts need their own verification.

```sh
cd backend
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev,eval,postgres]" -c requirements/tested-versions.txt
cp .env.example .env
```

```sh
cd frontend
npm ci
```

From the repository root, start the API and web interface in separate terminals:

```sh
make backend
```

```sh
make frontend
```

Open <http://127.0.0.1:5230/>. Keep configuration and credentials in ignored local
environment files. Production deployment needs authentication, HTTPS, protected
storage, retention configuration, and a review of dependency licenses.

Pretrained neural checkpoints are **not included**. Small research decision-head
and projection parameters are published with evaluation capsules. Follow [MODELS.md](MODELS.md) to provision
permitted, pinned weights. Missing or resource-limited checks must be disclosed.
Repository defaults retain legacy selections; the locally verified candidate
modes are explicit opt-ins. See [audio selection and rollback](docs/local-audio-rollout.md)
and [document/email evaluation](docs/text-upgrade-evaluation.md). Local-only
weight mirrors may require separate provisioning; do not expect every model to
be recoverable through one automatic download command.

After provisioning, `VERITAS_OFFLINE=true` prevents model downloads. Runtime
content and original results are stored under the operator's data directory;
user uploads do not automatically train models. Deletion and retention currently
have operator and case constraints; this is not a promise of immediate erasure
from every backup. See [architecture](docs/architecture.md) and [security](SECURITY.md).

## Billing and expert work

Optional Stripe Checkout charges USD 9.99 per automated analysis. When billing is
enabled, acknowledged limitations and verified payment are required before a job
is created. Configure test mode first using [the local Stripe guide](docs/local-stripe-billing.md).
Never commit processor or webhook secrets. Publishing this source does not enable
live charging. Private expert work is separately agreed through sales@chipmunkrpa.com.

## Evaluation and limitations

Start with [the current evidence/report policy](docs/standard-saas-report-policy.md),
[pre-purchase limitations](docs/pre-purchase-analysis-limitations.md),
[the frozen cross-media evaluation](docs/rule80-fresh-evaluation.md), and
[the commercial model audit](docs/commercial-model-audit.md).

The [revised model-selection review](docs/model-selection-revision-v63.md)
reopens useful image, video, audio and passage candidates and their actual
combinations. Earlier frozen gate failures remain historical results; a modest
extra false alarm or lost hit is not an automatic reason to reject a new
comparison. Shared backbones and agreeing decisions do not provide independent
corroboration or a calibrated combined confidence.

The [complete shared audio comparison](docs/audio-shared-confirmation-v64-result.md)
tests 11 fixed methods on 1,080 cases from 120 recordings. A two-head combination
adds 91 correct conclusions with no extra false AI alarms, but also adds 28 AI
cases incorrectly leaning original. Its coverage gain supports further combined-
confidence calibration; the serving model and confidence remain unchanged. The
report retains all methods, conditions, source limits and the interrupted run.

The [next confidence-calibration source panel](docs/audio-confidence-source-qualification-v75.md)
has 400 verified original speech files and 48 backgrounds; 64 matched synthetic
files remain to generate. A pre-model ancestry audit corrected shared dialogue
connections while retaining the selected recordings. No new confidence has
been fitted or enabled from this panel.

The historical cross-media study completed 288 analyses. It exposed missed local
image edits and face swaps, noise-related audio uncertainty, and inconclusive
article results. It is an evaluation of its frozen decision rules, not a universal
accuracy claim. Later presentation changes do not improve those measured outcomes.
Private datasets, source media, credentials, customer reports, and downloaded
weights remain excluded from this repository.

New source-separated [image-localizer](docs/iml-vit-localization-evaluation.md)
and [small-edit tiling](docs/tiled-splice-evaluation.md) comparisons retain
complete candidate outcomes and their original admission decisions.
The [active improvement plan](docs/detection-improvement-plan.md) tracks the next
video, audio and writing evaluations. Independent contributors can use the
[clean-checkout evaluation replays](docs/independent-testing.md#replay-recent-evaluations-from-a-clean-checkout),
[portable testing contract](docs/independent-testing.md), and evaluation issue form.
The [complete-video temporal experiment](docs/mouth-temporal-v11-evaluation.md)
located more edits but failed its ordinary-footage false-alarm requirements;
its complete fitting and comparison records are published for review.
The [source-separated audio experiment](docs/audio-guard-v7-evaluation.md)
passed development, but its [1080-input fresh comparison](docs/audio-guard-v7-confirmation-evaluation.md)
added wrong conclusions despite better aggregate probability scores. It remains
disabled under the frozen error limits; complete fitting, fresh outcomes and
numerical reproduction materials are available for outside review.
The [v8 audio experiment](docs/audio-multisource-v8-execution.md) was stopped before
fitting after an [additional publisher-identity audit](docs/audio-source-identity-v9.md)
found cloned and genuine versions of the same speakers across its data roles.
Its source-separation claims are superseded; waveform reproduction remains valid.
The corrected source plan retains the fixed fresh recordings and joins their
known parent identities. This repair supports a new experiment, not a measured
detection improvement.
Its [prospective execution freeze](docs/audio-identity-v9-execution.md) records
the unchanged learner, corrected roles and native cache-verification requirements
before new extraction or fitting; it does not establish a passing replacement.
The [completed corrected comparison](docs/audio-identity-v9-development-evaluation.md)
reduced false AI calls but lost correct conclusions and increased abstention,
failing five unchanged requirements. It remains disabled; complete numerical
reproduction and 99 native replay receipts are available for outside review.
The [completed conditional response comparison](docs/audio-conditional-v10-development-evaluation.md)
separately tested nonlinear mapping and added acoustic-quality inputs. Both
failed the same five requirements: fewer false alarms came with fewer useful
conclusions. Neither was enabled or sent to fresh confirmation. Complete public
arithmetic, fitting-only refit and 121 native replay receipts support outside review.
The [completed full-vector comparison](docs/audio-full-representation-v11-development-evaluation.md)
recovers 72 correct conclusions against the rejected projection control but still
loses 60 against service, failing four fixed requirements. It remains disabled;
all outcomes, small parameters and 125 native receipts support public-only
numerical/refit review without pretrained-model downloads.
The completed [fixed-layer comparison](docs/audio-layer-v12-development-evaluation.md)
also failed its unchanged usefulness requirements. Lossless public audio feature
packets and fitted objects now support [portable arithmetic/refit review](docs/audio-public-portable-review-v13.md)
without model weights or recordings. External Linux arithmetic reproduction
passed; strict exact cross-platform refitting remains failed, with diagnostic
differences that changed no directional outcomes. Automated external-host
reproduction is distinct from outside human review; no outside human has participated.
The [two-source audio extraction](docs/audio-separation-v14-execution.md) stopped
on a decoder failure before fitting. Its partial journal remains preserved and
does not establish detection quality. A [separately frozen full replication](docs/audio-v14-full-replication.md)
completed all 4,248 inputs from the first input with resource guards and private
decoder-error capture. Its
[completed development comparison](docs/audio-separation-v14-development-evaluation.md)
rejected both frozen arms; the original failure remains preserved. These results
are distinct from a newly frozen comparison of actual decision combinations.

The [completed general-video study](docs/video-demamba-v13-result.md) rejected
its candidate: all192 development cases were inconclusive after its own
calibration. A complete guarded baseline repeated all84 earlier responses
exactly; both interrupted attempts remain preserved. Lossless public responses
support a separate NumPy-only arithmetic review. Four of16 independent native
RGB checks failed and remain explicit; no serving check was replaced.
The completed [small nonlinear mouth-window experiment](docs/mouth-temporal-v12-evaluation.md)
located26/32 edits versus14/32 for its retained reference, but flagged12/56
ordinary clips versus0/56. It was rejected under the frozen false-flag rules;
complete JSON trees, every development window and standalone replay/refit
materials are public. Unused actors were not scored to rescue the result.

Some historical release records and Makefile targets refer to tests, development
scripts, and benchmark inputs absent from the imported archive. Those records are
not a claim that the original full release suite can be rerun from this checkout.
[CONTRIBUTING.md](CONTRIBUTING.md) lists regression checks available here.

## Open development and licensing

Project-authored source is available under the [MIT License](LICENSE).
Third-party code, fonts, trust data, model weights, datasets, and runtime packages
retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
In particular, the separately installed PyMuPDF/MuPDF dependency is dual AGPL/commercial;
the project's MIT license does not waive dependency obligations.

Use issues and pull requests for proposals, reproducible bugs, and evaluation
improvements. Follow [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).
Public evaluation resources should use licensed, redistributable or consented
material. Never publish private uploads as examples.
