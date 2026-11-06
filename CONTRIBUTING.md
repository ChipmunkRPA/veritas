# Contributing to Chipmunk Verify / TrueTrace

Contributions are welcome through issues and pull requests at
[ChipmunkRPA/veritas](https://github.com/ChipmunkRPA/veritas).

Explain the user problem, affected media type, expected behavior, and observed result.
Use synthetic or explicitly redistributable examples. Never attach private uploads,
customer reports, credentials, model checkpoints, or copyrighted evaluation datasets.
For security-sensitive issues, follow [SECURITY.md](SECURITY.md).

## Development

Follow the setup in [README.md](README.md). The source distribution excludes model
weights and private evaluation inputs. Model-free regression checks include:

```sh
cd backend
PYTHONPATH=. .venv/bin/pytest -q tests/test_customer_assessment.py tests/test_finding_assessment.py
```

```sh
cd frontend
npm run typecheck
npx vitest run src/features/analysis/customerAssessment.test.ts src/features/analysis/findingAssessment.test.ts
npm run build
```

Some historical release records refer to tests, scripts, and raw datasets absent
from the imported source archive. Do not represent those records as newly reproduced results.

## Evidence and evaluation requirements

- Preserve source hashes, frame timestamps, PCM sample ranges, text offsets, and original stored results.
- Keep observations, interpretations, inspection scores, calibrated estimates, and uncertainty distinct.
- Missing, unsupported, skipped, or inconclusive checks must not become evidence of originality.
- A whole-file calibration must not be assigned to a crop, frame, sound interval, or passage without suitable evaluation.
- Evaluate candidate checks against the current implementation using declared splits, fixed rules, and licensed data.
  Report false alarms, misses, inconclusive rates, coverage, memory, and runtime. Keep rollback available.
- Changes to models or thresholds require their own evaluation; presentation changes do not establish improved accuracy.
- Preserve third-party notices and check code, checkpoint, and dataset permissions separately.
- Keep model identifiers out of ordinary customer-facing results, reports, and projected API output.

Maintainers review proposed changes and may request additional evidence before adopting a detector.
An issue or pull request does not automatically change a serving calibration or benchmark.

For independent testing, use the [portable evaluation contract](docs/independent-testing.md)
and the repository's **Independent evaluation or detection failure** issue form.
The contract checks source separation, complete denominators, and frozen system
identities across all four media types. The [active improvement plan](docs/detection-improvement-plan.md)
lists the current priorities and the evidence needed before adopting a change.

Project-authored contributions are provided under the [MIT License](LICENSE).
Identify any third-party material and retain its original license and attribution.
