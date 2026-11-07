# Writing focus/context comparison execution

6 November 2026. Prospective source-bound comparison, before new classifier
forwards on project text. The [protocol](writing-focus-v3-protocol.md) and
[public freeze](evaluation-panels/writing-v3-focus-freeze.json) bind the complete
12-file method, installed implementations, numerical runtime, pinned model
assets and the original consumed 40-document/674-focus input packet.

The 45 selected tests pass, covering invalid/omitted responses, inclusive
research references, character denominators, rewriting-class semantics,
single-window native replay aggregation, source-position binding and refusal
of changed freeze metadata. Independently audited native token IDs cover every
input without omission, duplication or reordering. Current model:52,385 context
and18,163 focus content tokens. Candidate:53,735 context and18,648 focus tokens.
Each view has674 token windows; no actual panel input overflows, although
complete-overflow behavior is exercised by focused tests. No real detector score
has been used to select or revise this protocol.

The completed private freeze was accepted unchanged, and thirteen altered
copies (source/history/model revisions, method/code/runtime/assets, forward and
adoption state) were rejected before model loading. Secret-scan findings were
thirteen verified tokenizer-file SHA-256 fields; every value was recomputed from
the pinned assets/runtime before exact field/value exceptions were added. The
directory scan then revealed that global path exceptions skipped entire files in
the installed scanner. Converted them to rule-specific AND conditions, retaining
exact path/field/value scope. Fifteen publication checks verify that generic and
provider credential sentinels remain detected in metadata files, unrelated
paths and all previously excepted files. The skipped-file scan was not accepted
as a complete result. The configuration itself is scanned under a separate text
filename because the scanner's defaults otherwise skip its configuration file.

The locally cached candidate has a consistent MIT weight declaration; the
installed baseline retains Apache-2.0 notices. Neither checkpoint is published.
The original source-data terms conflict remains documented; no new training or
source-text/annotation redistribution is authorized. Input text, author groups
and per-passage records stay local. The public result will contain aggregates.

## Execution

Acquire the exact pinned sources and model assets separately. Regenerate the
private input packet using [preparation instructions](writing-focus-v3-preparation.md).
With the existing numerical environment and offline model loading, create new
private lock and public freeze paths:

```sh
PYTHONPATH=backend backend/.venv/bin/python -m scripts.run_writing_focus_v3 freeze \
  /path/to/inputs.jsonl /path/to/passages-tropa.json \
  /path/to/baseline-snapshot /path/to/tmr-snapshot \
  /path/to/new-lock.json /path/to/new-public-freeze.json
```

Publish and verify the prospective public freeze before the following scoring
commands. Use a separate process for each model to bound memory. Both views are
retained in each document record. Preserve the existing journal if a process is
interrupted; it can resume under the same freeze. A retained failure blocks
resumption and continuation rather than silently omitting the case.

```sh
PYTHONPATH=backend backend/.venv/bin/python -m scripts.run_writing_focus_v3 score \
  /path/to/lock.json baseline /path/to/baseline.jsonl
PYTHONPATH=backend backend/.venv/bin/python -m scripts.run_writing_focus_v3 score \
  /path/to/lock.json candidate /path/to/candidate.jsonl
PYTHONPATH=backend backend/.venv/bin/python -m scripts.run_writing_focus_v3 native \
  /path/to/lock.json baseline /path/to/baseline.jsonl /path/to/baseline-native.json
PYTHONPATH=backend backend/.venv/bin/python -m scripts.run_writing_focus_v3 native \
  /path/to/lock.json candidate /path/to/candidate.jsonl /path/to/candidate-native.json
PYTHONPATH=backend backend/.venv/bin/python -m scripts.run_writing_focus_v3 summarize \
  /path/to/lock.json /path/to/baseline.jsonl /path/to/candidate.jsonl \
  /path/to/new-summary.json
```

Primary continuation gates are frozen before responses. Neither a passing
consumed-development comparison nor a native numerical replay enables serving,
restores whole-document confidence or establishes independent human validation.
The broader image/video/audio/writing improvement program remains unfinished.
