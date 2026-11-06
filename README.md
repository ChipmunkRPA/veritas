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

The historical cross-media study completed 288 analyses. It exposed missed local
image edits and face swaps, noise-related audio uncertainty, and inconclusive
article results. It is an evaluation of its frozen decision rules, not a universal
accuracy claim. Later presentation changes do not improve those measured outcomes.
Private datasets, source media, credentials, customer reports, and downloaded
weights remain excluded from this repository.

New source-separated [image-localizer](docs/iml-vit-localization-evaluation.md)
and [small-edit tiling](docs/tiled-splice-evaluation.md) comparisons retained
unsuccessful candidates instead of changing the service to weaker checks.
The [active improvement plan](docs/detection-improvement-plan.md) tracks the next
video, audio and writing evaluations. Independent contributors can use the
[portable testing contract](docs/independent-testing.md) and evaluation issue form.
The [complete-video temporal experiment](docs/mouth-temporal-v11-evaluation.md)
located more edits but failed its ordinary-footage false-alarm requirements;
its complete fitting and comparison records are published for review.
The [source-separated audio experiment](docs/audio-guard-v7-evaluation.md)
passed development, but its [1080-input fresh comparison](docs/audio-guard-v7-confirmation-evaluation.md)
added wrong conclusions despite better aggregate probability scores. It remains
disabled under the frozen error limits; complete fitting, fresh outcomes and
numerical reproduction materials are available for outside review.

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
