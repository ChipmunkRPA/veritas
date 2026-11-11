# BG-REAL PNG pair acquisition and exact edit qualification

Use the exact v218 private roster bound to the published v218 aggregate.
It names 24 primary recipients and 12 ordered backups at the publisher
release `aaba6cd1ced4ce3c846be9641fed092adb4919d1`. The source photos
and edits are BG-REAL F5 LaMa object removals with the matched original
and per-photo attribution. Source roles and editing provenance stay as
recorded. [Metadata roster](image-bg-real-roster-v218-method.md) ·
[Published roster](evaluation-panels/image-bg-real-metadata-roster-v218.json)

## Exact requests and pixel truth

Request only the named A0 PNG, F5 PNG and semantic mask for each attempted
recipient. Each request starts at the pinned Hugging Face revision. Follow
at most three HTTPS redirects to the exact repository cache path or
recognized publisher asset hosts. Verify the resolver revision and bound
asset length. Verify both image SHA-256 values from the publisher manifest.
When the publisher serves a linked SHA-256 or Git blob digest, verify it
for every asset, including the semantic mask. When there is no publisher
asset digest, record the exact mask bytes obtained from the pinned resolver
and bind their SHA-256 to the later pixel qualification. The aggregate
distinguishes these two mask commitments.

The limits are 36 attempted recipients, three named assets per attempt,
4,000,000 bytes per asset and a 900-second acquisition deadline, with
5-second connect and 10-second read timeouts. There are no retries or
snapshot requests. Failed assets receive a recorded reason. Complete
publisher-bound files are saved privately, including files that subsequently
fail pixel qualification. Failed responses or signed redirect URLs are not
retained. An interrupted run keeps its journal without a successful
completion receipt.

Decode each asset as one upright PNG with the exact declared width and
height and opaque pixels. The semantic mask must be binary, nonempty and
partial. Its pixel count must agree with the publisher area to within one
pixel. Compare the decoded original and edit without resizing, recoloring
or JPEG compression. Use the exact nonempty RGB difference as truth and
reject any changed pixel outside the semantic support. The supplied mask
describes allowed edit support. The exact mask describes changed pixels.

## Prior exposure and backup order

Use the bound v39 source graph and v218 Flickr identities. Compare new
original and edited fingerprints against the prior encoded and RGB hashes,
including the exact 397 prior v26 edit files and 590 v189 input byte/RGB
records. Use the existing v22 informative thumbnail and aspect-ratio rule
with the prior graph's recorded source thumbnails. Use exact encoded and
RGB digests, the same thumbnail rule and canonical source identities to
group the attempted candidates. The graph includes decoded fingerprints
from rejected pairs where available. This lets a rejected pair expose a
transitive link between other candidates. Any component with a known
exposure is excluded. Otherwise, retain the first valid pair in the
original order. These checks cover recorded files, creators and detectable
near duplicates. Unknown aliases and unseen model pretraining remain
outside the inventory.

Attempt all primaries in order. If fewer than 24 qualify, attempt backups
in their committed order until 24 qualify or the 36-candidate pool is
exhausted. Reevaluate the connected components after each added backup.
The offline seal repeats all pixel and exposure checks from the saved
bytes, reproduces the stop point and emits exact truth masks for the
retained sources. Keep every attempted rejection, report any shortfall and
leave unneeded backups unopened. The public summary contains only counts
and whole-artifact commitments. The private journal and qualified roster
contain the media references and source identifiers.

## Commands

From the repository root, prepare a new ignored trial and choose an empty
private destination name on a writable local volume. The cache directory
must not already exist. Preparation opens no images and makes no requests.
Review and record the plan digest before acquisition.

```sh
roster="$PWD/backend/data/independent-evaluation/image-bg-real-roster-v218"
trial="$PWD/backend/data/independent-evaluation/image-bg-real-acquisition-v221"
cache="/absolute/private/path/bg-real-v221-media"
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.image_bg_real_acquisition_v221 prepare "$roster" "$trial" "$cache"
```

Set `plan_sha` to the printed SHA-256. Run the bounded acquisition from a
shell that can reach the public publisher host. The seal uses local
filesystem reads and writes.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.image_bg_real_acquisition_v221 acquire "$trial" "$plan_sha"
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.image_bg_real_acquisition_v221 seal "$trial" "$plan_sha"
```

The subsequent JPEG experiment needs its own freeze of the admitted files,
truth masks and fixed comparison settings. This step runs no detector and
establishes no localization accuracy or whole-image authorship estimate.
