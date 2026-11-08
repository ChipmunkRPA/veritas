# Paired-data acquisition: explicit cache and monitor amendment

The [397-source plan](image-positive-pairs-v21-protocol.md), selection, exact
pair truth and two ordinary controls stay unchanged. This amendment changes
only HTTP caching and resource observation before continuing acquisition.
No model forward or fitting has occurred.

Actual first-phase journals show repeated valid byte ranges during gzip backward
seeks: 189 requests cover only 78 unique ranges. A 64-MiB least-recently-used
cache now retains previously validated blocks. Every cache miss still uses the
original exact-range/ETag/size/body checks. The original reader and frozen pair
code remain untouched; a separately named entry point makes the change explicit.

Original phase 0 completed with a clean resource profile and retained 12 selected
edits. Phase 1's worker exited zero without a resource guard event, but its old
monitor recorded one opaque sampling error. That strict resource verification
remains failed; it is not relabeled as a clean run or assumed to be a proven
exit race. The scheduling parent was held while the child finished, then stopped
only after the child was authoritatively terminal. All original records remain.

The successor execution root copies only fully verified phase 0, preserving its
actual source checkpoint/profile commitments and identifying it as copied,
not re-executed. The affected phase will be explicitly re-read under this published
amendment. It does not seed from the phase 1 checkpoint or omit any planned source.
This is a resource/transport repair, not a post-result change to detection gates.

The new monitor retains process-exit observation events separately, and classifies
one as terminal only when authoritative child polling returns an exit status.
A missing sample while the child remains live still stops the job. Nonzero child
exits remain failures. The same 900-second, 6-GiB sampled-tree and 1536-MiB reserve
limits apply. Sampled RSS is still not a true instantaneous peak guarantee.

All 26 authored tests pass, including original-to-cached checkpoint continuation,
exact returned bytes, cache eviction, changed-object refusal and live/terminal
monitor distinctions. Two real short data-worker help processes also exit zero,
without a guard or unresolved sampling error. No project image/model is used by
those smokes. Code, tests, seed commitments, the retained old failure and both
actual smokes are in the [amendment packet](evaluation-panels/image-positive-pairs-v21-cache-monitor-amendment.json).

All media, individual source pairings and gzip dictionaries remain private.
GitHub Actions stays disabled; verification and publication are local checks and
direct pushes. The complete acquisition, paired-data qualification, independent
source replay, training and fresh model comparison remain unfinished.
