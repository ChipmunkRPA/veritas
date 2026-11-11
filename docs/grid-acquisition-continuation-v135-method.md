# Explicit continuation of the original GRID source acquisition

The original v131 stage failed at its 900-second limit after fully verifying one
archive and retaining a 337,641,472-byte prefix of the second. Preserve that
attempt, its failure/profile/journal, the complete first archive and the partial
file unchanged. Do not rerun its command or report it as successful. The v134
preflight independently reverified the complete first archive and two native
video/audio clips. This new plan continues **data acquisition only**; no detector
result, source response or model behavior changes the chosen cohort.

Keep the original sixteen sources, publisher metadata/CC-BY-4.0 attribution,
source ordering and intended two distinct utterances per source. No substitution
or reduction of the cohort. Bind the original failures, source journal and exact
prefix size/SHA-256 into a new private plan and public whole-plan commitment.

The first new stage issues one HTTP Range request for the **missing tail only**
of the second archive. Require 206 and the exact Content-Range and body size;
reject encoded, short, oversized or ignored-range responses. There is no retry
or fallback to downloading the already held bytes. Save the new tail separately,
copy the frozen original prefix and new tail into a **new** archive, then require
the publisher's whole-file size and MD5. Record observed whole-file SHA-256 and
the complete safe directory inventory before admitting the archive. The original
prefix is never appended to or modified. Lack of a publisher ETag is explicit:
whole-body coherence is checked against the publisher's legacy MD5/size, not a
claimed signed identity or modern cryptographic publisher signature.

For the other fourteen **previously unattempted** sources, download one full
archive per new guarded stage, using the same exact URLs/sizes/checksums and
original sequential download/inventory functions. No parallel transfers. This
changes acquisition scheduling from two archives per stage to one: fifteen new
stages are explicitly planned. It increases the aggregate acquisition time
allocation; it does not pass or rewrite the original eight-batch procedure.
Per-stage data-only limits remain 900 seconds, 1.5 GiB sampled descendant RSS,
3 GiB initial RAM/project/archive disk and 1.5 GiB running reserves. Neural limits
and all detection/calibration procedures remain unchanged. Samples are not
guaranteed resource peaks.

Freeze/publish the tested new implementation and plan before any new network
request. Run sequentially outside the sandbox with the exact qualified monitor:

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.grid_acquisition_continuation_v135 run 1
```

Inspect the actual receipt/profile before subsequent original ordinals 2–15.
Stop on a failure or admission denial. Once attempted, preserve every new stage,
failed tail or reconstructed file; no automatic retry, overwrite, relaxed limit
or source replacement. Successful individual stages do not establish a complete
sixteen-source cohort, native clip/clock eligibility or detector performance.

All source IDs/pairings, raw media and individual fingerprints stay private;
publish only code/methods, aggregate availability/bytes and whole-artifact
commitments. No model call, learned parameter publication, serving/default,
Actions, outreach, payment or GCP deployment is part of this acquisition.
