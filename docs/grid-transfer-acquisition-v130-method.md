# Fresh GRID source acquisition, before model use

Use the [original authors' GRID corpus record](https://zenodo.org/records/3625687)
as a new local-study source. The complete official API metadata snapshot declares
CC-BY-4.0 and open access. Bind that exact snapshot before acquisition. Its
publisher inventories 33 available video archives; talker 21 video is absent.
This is separate from the VidTIMIT record, whose full metadata explicitly declares
CC-BY-NC-ND-4.0 and is excluded. No restricted media is downloaded.

Select sixteen available video archives by SHA-256 order of the fixed salt
`chipmunk.grid-transfer/130:` plus publisher archive name. Keep source identities,
individual file checksums and subsequent clip pairings private. Freeze the
implementation, tests, this method, exact publisher metadata and whole private
plan before downloading any archive. This selection uses no picture, audio,
face geometry or detector response.

Run eight sequential batches of two archives, **one network download at a time**.
Store full archives on Seagate in exclusive directories. Stream bounded 1 MiB
blocks with 30-second network timeouts; require exact publisher byte count and
complete MD5 match, and record observed SHA-256. MD5 is the publisher's integrity
checksum, not proof of authenticity or a modern cryptographic signature. Retain
the complete publisher metadata/CC-BY attribution and original corpus paper with
its publisher checksum. Keep all material private; public outputs are methods,
aggregate availability/bytes and whole-journal commitments.

Inspect only each complete ZIP's directory. Require bounded unique member paths,
no absolute/traversal paths, no encryption, standard stored/deflate compression
and bounded real video entries. Ignore Apple resource forks under `__MACOSX`
and `._` names; they are metadata, not additional videos. Each archive must
inventory 1,000 real `.mpg` files. Do not extract every file or execute anything
from the archive.

The later preparation recipe, fixed now, selects two real video entries per
source by a salted hash order with globally unique utterance stems across the
selected sources. Freeze the resulting complete clip plan before model use.
Audio/clock/geometry failures remain unavailable; do not choose a replacement
from detector responses. Full clip/PCM/clock qualification is not performed in
this acquisition stage. New source groups and different sentences are not proof
of distinct people, unknown ancestry or absence from pretrained encoder data.

Use the exact qualified owned-descendant monitor and unchanged data-only limits
outside the sandbox: 900 seconds per batch, 1.5 GiB sampled descendant RSS, 3 GiB
initial available RAM/project/archive disk and 1.5 GiB running reserves. No neural
network, decoder, generator, fit, customer job or source substitution is run.
Preserve failed/partial downloads and attempted-stage receipts; no automatic
retry, overwrite or relaxed limits. Samples are not guaranteed peaks.

This is acquisition for a frozen-head external transfer study, not a new
performance result or a serving-model training admission. Subsequent evaluation
must preserve the source license/attribution, group related recordings/utterances,
retain coverage failures, and keep parameters fixed. No outreach, Actions,
real payment or GCP deployment occurs.
