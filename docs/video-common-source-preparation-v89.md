# Shared video comparison: new source preparation

The metadata plan reserves **144 source groups**: 96 new paired face-swap
targets and 48 new recorded audiovisual-speech actors. Each of four provisional
roles contains 24 swap targets and 12 speech actors. There are **336 planned
source assets**, not 336 verified clips or completed detector tests. The
[aggregate receipt](evaluation-panels/video-common-source-preparation-v89.json)
contains counts and whole-artifact commitments; identities and pairings stay
private.

The [planner](../backend/scripts/video_common_sources_v89.py) excludes the 162
source-video groups and known donors from six earlier RTFS acquisitions. It
normalizes YouTube URL forms and known portrait-source aliases. The fixed greedy
metadata ordering finds 460 additional eligible paired targets; the first 96
are planned. It also scans 33 earlier video/lip manifests and plans, excluding
28 known CREMA-D actor codes. Forty-eight of the remaining 63 codes are chosen
by a fixed hash order. No detector response selects a source.

## Permissions and source limitations

- RTFS is pinned to revision `bca865e675852666a426368b2f5614abf1c1926a`.
  Its [publisher card](https://huggingface.co/datasets/godmodes/rtfs-10k/blob/bca865e675852666a426368b2f5614abf1c1926a/README.md)
  declares CC-BY-SA-4.0 and describes Creative Commons YouTube source videos.
  Existing generated clips are the planned swap inputs; no swap-generator
  weights or raw donor portraits are downloaded by this preparation.
- CREMA-D is pinned to commit `1658cd342dff90010aa843eaeebd53610a08b1dc`.
  Its [publisher license](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt)
  supplies ODbL1.0 and DbCL1.0 database/content grants. The complete 91-actor
  demographic inventory is checked against the 7,442 source filename records.
- **Portrait attribution/identity linkage is incomplete.** Although the RTFS
  card describes source links, the actual pinned portrait CSV contains only
  tags and filenames: zero source URLs for 2,000 records. Those donors can be
  linked by filename only. Publisher grants and provenance are not independent
  guarantees of all third-party rights, distinct people or unknown pretraining
  separation. These limits cannot be repaired by assuming missing links exist.

All nine pinned metadata/license files were fetched and hashed. Acquisition
took 2.03 seconds, with a maximum per-download observed process RSS of
34,799,616 bytes. These discrete observations are not continuous or peak
resource measurements. No raw media, neural model or customer upload was read
by this preparation.

## Required before scoring

Acquire the exact media with publisher digests, check native frame/audio clocks
and decoded aliases, then finalize connected source roles. The roles above are
provisional until these checks finish. Keep known edit intervals and source
correspondence when preparing swap/mouth inserts and ordinary controls. Silent
clips cannot validate lip sync. Every retained current/appearance/mouth/video
method must run on the same applicable inputs, preserving missing checks.

Any new mouth generation and detector study needs its own fixed methods,
weight-specific permission review and source/evaluation commitments. No model,
confidence contribution or serving default changes from this metadata plan.
Ten local regression cases cover URL aliases, old-source exclusion, metadata
tampering, paired-role integrity, privacy of the aggregate summary, unsafe paths,
insufficient sources and refusal to overwrite. This is software/source planning,
not new detection accuracy or outside reviewer participation. No workflow,
outreach, real payment or GCP action occurred.
