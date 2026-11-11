# Continue the frozen pilot scores after the monitor stop

The v151 clip 0 stage completed and supplies its eleven verified score records.
During the next clip, the owned-descendant monitor recorded a `PermissionError`,
terminated the worker with exit −15, and retained its profile and resource
journal. The worker log is empty and the stage has no worker output. The v155
freeze binds the completed clip 0 receipt and score journal together with the
actual clip 1 failure artifacts.

The continuation accepts the original pilot ordinals 1 through 15. The worker
invokes the v151 scoring code object with new v155 freeze and cache paths. It
uses the existing features, heads, calibrations, research settings, binary
bindings and zero-forward hook. The v151 reader verifies the new stage and
substitutes the v155 caller name for its receipt check. Each worker result
uses the existing scoring schema and binds the new freeze. The v151 paths
remain historical inputs to this continuation.

Freeze and publish the reviewed continuation, then use
`python -m scripts.grid_native_mouth_scores_v155 run <ordinal>` with the project
interpreter and backend on `PYTHONPATH`. Run each explicit stage in an
environment that permits the v108 monitor to sample its own descendants.
The worker requires the outer resource journal and its attached monitor log
before creating output. A terminal profile or existing output prevents entry. The runner
uses its existing data-only bounds of 900 seconds, 1.5 GiB sampled descendant
RSS, 3 GiB initial RAM and disk, and 1.5 GiB running reserves.

After all fifteen continuation stages complete, `aggregate` verifies the
original first clip and all continuation receipts, then applies the v151 panel
aggregate to the 176 ordered records. The public result contains panel and
condition counts and a SHA-256 commitment to the ordered stage receipt
collection. The hash uses compact sorted-key JSON for the collection.
`aggregate_completed(include_private=True)` also returns the source-slot and
component breakdowns and the stage commitment collection for private storage.
