# GRID pilot comparison against the full video pipeline

Run the current video service with an explicit full profile and the commercial
manifest. Use the local appearance review and voice-candidate modes. Keep the
production tracking, sampled-frame face detection, appearance fallback,
audio/picture timing, voice, sound context, provenance, fusion and evidence
assembly. Run the real application pipeline with its registry and model manager.
Its inputs are the verified v149 Matroska clips. The application receives each
as `sample.mkv` with unchanged bytes and no condition label or ground truth.

Freeze a reviewed compute choice before running any pilot clip. The default is
`auto`, resolved by the application on the worker. An explicit CPU benchmark
also uses full. CPU auto-profile ordinarily chooses quick. Record the requested
device, resolved device, manager budget, actual detector devices and precision,
including any production fallback. The full profile omits the deep-only second
pass. Report resource and model coverage for the actual measured configuration.

Bind all application and migration code, the manifest, packaged fusion and
calibration files, installed package versions, the qualified decoder binaries
and the nonsecret configuration. Bind the commercial detector roster and every
required model's pinned file status. Preserve missing files as missing. Bind
available pinned files by full SHA-256 and any local fuser files under
`backend/data/fusion`. The fresh private database starts with the shipped
version-one policy. Record the actual policy and fuser used by each result.
Custom live policy records and model footprint history are outside this run.

Each source has a new private SQLite database, object store, upload area and
cache on Seagate. The app settings skip the dotenv source and the worker
environment excludes inherited app credentials. Preserve the existing home
directory while directing model and plotting caches to the owned stage.
The isolated asset layout contains file-level links to verified weights.
Verification records, extracted repositories and other mutable caches use
new stage paths. A Python audit hook protects external paths from application
writes and blocks application network connections. The only subprocesses the
worker permits are the bound local decoders. No service process, Stripe request
or webhook is started. The real prepaid worker flow is a separate check.

Run one exclusive `clip-OO/authored` stage per pilot clip in frozen order.
Use the unchanged v108 neural monitor with 900 seconds, 6 GiB sampled descendant
RSS, 6 GiB initial RAM, 3 GiB initial project and archive disk and 1.5 GiB
running reserves. Preserve partial work and resource failures without retries.
The private roster includes all eleven matched cases. Process ready clips in
blind numeric filename order. Preserve every unavailable input with its reason.

For each attempted case, save actual stage statuses, registry routes and reasons,
persisted detector runs, emitted observations and evidence, sampled PTS and RGB
commitments, tracking coverage, audio sample support, and the app's current
customer assessment. Compare observed sampled inputs to the matched v149
timestamps and pixels. Keep inspection cues and soundtrack interpretation
available separately from the combined conclusion. A skipped, failed,
inconclusive or unavailable check does not establish an original-file verdict.
The full production pipeline uses a mix of learned models and native
operators, so exact forward counts are not measured.

The research comparison must use the fixed original source components as
groups, retain every unavailable case and distinguish visual review from
audio/picture timing. This pilot supplies no identity-face-swap, achieved
phonetic alignment or general commercial-generator claim. It changes no
serving defaults or confidence rules.

## Consumer API

Call `scripts.grid_serving_baseline_v152.load_completed(ordinal)` with the
original v143 pilot ordinal. It validates actual successful bounded stage
receipts, whole-file journal commitments and the matching v149 roster, then
returns eleven private rows in numeric case order.

- The metadata stage is `backend/data/independent-evaluation/grid-serving-baseline-v152/clip-OO/authored`.
- Private `runtime.json`, `planned-cases.json`, `upstream.json` and
  `cases.jsonl` are under `grid-serving-baseline-v152/clip-OO` on Seagate.
- Each row carries `asset_index`, `case_number`, `source_slot`,
  `component_index`, `condition`, `status` and `reason`.
- Attempted rows add `sampled_input`, `stages`, `detector_runs`,
  `detector_inventory`, `detector_stage_outputs`, `temporal`, `decision`,
  `customer_assessment` and `evidence`. Unavailable inputs contain no verdict.
- Publish code, aggregate results and whole-artifact commitments. Keep model
  paths, per-input identities and fingerprints, private evidence and media
  in the local archive.
