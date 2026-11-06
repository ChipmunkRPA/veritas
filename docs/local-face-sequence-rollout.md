# Local consecutive-face evidence release

This integrates the primary raw-crop method from the [frozen v6 comparison](video-sequence-v6-evaluation.md).
It is an inspection method, not a calibrated face-swap classifier. No weights,
calibration coefficients, overall confidence rules, or public screening claims change.
The secondary brightness-normalized method remains disabled.

## Selection and rollback

`VERITAS_VIDEO_FACE_REVIEW_MODE=sequence` enables the new review adapter for new
full/deep video analyses. `legacy` disables it and retains the existing review
path; restart the backend after changing the setting. The repository default is
`legacy`. Neither setting rewrites saved analyses or removes the old models.
Commercial licence and deny-list checks still apply to both reused checkpoints.

The adapter uses the existing detected largest-face sample positions, with three
native consecutive crops per center. No old classifier response enters its
statistic. An old classifier skipped for resources therefore does not prevent
this smaller independent check from running. The overall result still discloses
missing classification checks and withholds confidence where required.

At most 64 centers and twelve 112-pixel crops per feature batch are allowed.
The regular model manager enforces model/activation budgets and deadlines.
Ambiguous tracks, inferred timestamps, missing source frames, unsuitable cadence,
and insufficient surrounding context do not become negative findings.

## Evidence contract

- One response describes a window, not its individual frames or pixels.
- The highest eligible range is shown first. Existing evidence remains available
  as fallback and context; the new cue has zero influence on overall confidence.
- The normal view, PDF and JSON show the same review range and before/during/after
  frames. Original face-context crops are not manipulation masks.
- Each retained window binds all three native inputs around each center to exact
  source PTS and decoded RGB hashes. Exact alignment coefficients are also stored
  as hexadecimal float strings, avoiding loss under six-decimal JSON formatting.
- Unselected retained windows keep assessed coverage without creating an
  original-file conclusion. Native decoding coverage is separately identified.
- Evidence version 7 retains older versions, including their nested audio and
  timing channels. Model names and internal identities remain filtered.

## Verified engineering results

These are replays of consumed evaluation inputs, not a new accuracy study or
external independent review.

- The production adapter, including normal resource acquisition, matched all
  120 saved v6 cases. Maximum-response tolerance was 0.000002 in feature units;
  all selected ranges and review/no-review choices matched. There were 108 cases
  with eligible windows and 12 unassessed cases. No operating point was refitted.
- A full prepaid-flow test on `rtfs-106-uniface` produced review range
  **1.791667–3.083333 source seconds**. Before/during/after source frames were
  **37 / 62 / 80**, at **1.541667 / 2.583333 / 3.333333 seconds**. Their native
  PTS were **18944 / 31744 / 40960**, time base **1/12288**. Thirty native frames
  supported the selected window. The cue linked to an ordinary finding.
- The API, JSON export, frozen report and signed evidence package agreed. All
  eleven presentation assets were verified; every PNG appeared with exact pixels
  in the PDF. Public implementation-name scans and offline package verification
  passed. PDF evidence pages 8–9 were rendered and visually inspected.
- The matching original and legacy rollback passed the same prepaid/export/
  report/package checks. The original produced no additional change range.
  New and rollback modes produced identical overall decisions on the same swap.
- In these default-budget runs the older large face classifier was skipped for
  memory. The new cue ran independently; the combined result remained
  inconclusive with no overall probability. No memory safeguard was bypassed.
- Chrome rendering passed at 1440-pixel desktop and 390-pixel mobile widths
  without horizontal overflow or page errors. Actual source and silent-context
  players both navigated to the recorded frame 62 at 2.583333 seconds.
- The current targeted video/evidence/customer regression suite passed 203 tests.
  The payment/preflight suite passed 100 tests, including 23 shared new tests.
  Frontend evidence/support checks passed 37 tests, and the production UI built.

The verification uses a simulated Stripe transport, real signed payment events,
the normal queue, actual local weights and the report pipeline. It creates no
real charge. Creation routes refuse unpaid submissions, and checkout creates no
analysis or inference job before confirmed payment. The current purchase notice
must be accepted before checkout.

## Reproduce

The source catalogues and accuracy comparison are linked from the v6 study.
With those consumed receipts and model caches present, replay adapter parity:

```sh
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/validate_face_sequence_rollout.py \
  BASELINE.jsonl FROZEN_REFERENCE.jsonl MEDIA_DIRECTORY OUTPUT.jsonl --device mps
```

`--device cpu` is supported for engineering verification; the completed replay
above used the same Mac GPU device as the frozen reference. Choose a new output
file. A partial `--limit` run cannot report full-panel completion.

For the isolated paid video/report check, use
`backend/scripts/validate_local_audio_rollout.py` with `--mode candidate`,
`--cases video`, `--video-control CLIP`, `--video-evidence-only`, and
`--require-face-sequence` for a known positive engineering fixture. Set
`VERITAS_VIDEO_FACE_REVIEW_MODE=sequence`. Optional `--models-dir` and
`--research-dir` allow an isolated checkout to reuse licensed research fixtures
and pinned caches without copying customer data or credentials.

The v6 accuracy limits remain: 19/60 swaps localized versus 7/60 for the prior
check, 21/60 genuine/control review flags, and 41/60 missed localizations.
These are review-cue counts, not whole-file AI error rates. Long clips, newer
swap generators, lip-sync edits and unfamiliar identities need further testing.
No outside reviewer has independently validated this release. GCP remains deferred.
