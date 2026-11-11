# Fix the complete fresh clip cohort from directory metadata

Use the same original sixteen GRID source archives and CC-BY-4.0 publisher
metadata already bound by v131/v135. Fix the full **32-clip** selection before
any fresh detector or generator response. This stage does not fetch full archives
or establish their checksums: those remain separate required acquisition steps.

Read each publisher ZIP directory using the Python standard library and a
bounded seekable HTTP range reader. Require 206, exact Content-Range, byte count
and identity encoding for every range. Maximum 2 MiB per read, eight requests
and 5 MiB transferred per archive, 30-second request deadlines. Cache complete
ranges locally in worker memory. Do not open an archive member, decompress media
or decode pictures/audio. ZIP footer/directory readahead can include unused bytes
near those structures; it is not full media acquisition or a whole-body checksum.
No retry or fallback to a full-file response.

Run two sequential guarded batches of eight source directories. Require bounded
unique safe paths, permitted unencrypted compression, bounded real video sizes
and exactly 1,000 real `.mpg` members per archive; exclude Apple resource forks.
Save original source correspondence and member names/sizes/CRCs privately.
These directory CRCs are planning metadata, not independently verified media
content. Full checksum-verified acquisition must later match the inventory and
CRC-verified extracted bytes before media eligibility is assessed.

After both actual metadata batches pass, run a third data-only planning stage.
Process original source order. Apply the already fixed v134 salted utterance-stem
order, taking two unused stems per source and carrying the used set across all
sixteen sources. Require 32 distinct stems, all sixteen original sources, and
unchanged first-source choices from v134. Role is `external_transfer`; no media
is marked qualified. The existing heads/calibrations remain fixed for this study.
Distinct corpus source codes/stems do not prove people, recording ancestry or
pretraining independence. No replacement from content/face/detector results.

Freeze/publish the tested code and source-plan commitment before requests. Use
the exact qualified owned-descendant monitor outside the sandbox for all three
data stages: 900 seconds, 1.5 GiB sampled descendant RSS, 3 GiB initial RAM/project/
archive disk and 1.5 GiB running reserves. Sampling is not a peak guarantee.
Inspect actual profile/receipt before the next stage. Stop on failure/admission
denial; preserve attempts and do not retry, overwrite or relax limits.

Publish code/methods, aggregate counts and whole-journal/plan commitments only.
All source/utterance identities, pairings, individual CRCs/fingerprints and raw
metadata packets stay private. No neural call, fitting, detector accuracy claim,
serving/default change, Actions, outreach, payment or deployment occurs here.
