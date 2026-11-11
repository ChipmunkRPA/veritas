# Initial native middle-placement transfer plan

This is a new metadata-only planning context, not generation, geometry, model
execution or accuracy evaluation. Require the actual successful v142 worker,
zero-error resource profile, bound resource/review/component journals, unchanged
code/freezes and matching published execution wrapper. Keep every original v137
row and unavailable status. No media, saved PNG/WAV/NPY/MPG or pretrained binary
weight files are opened while planning. Saved head/calibration JSON is read only
for binding and schema checks, without scoring. Prior asset qualification is bound through manifests
and actual receipts; later guarded workers must verify actual assets/media.

Start with original source slots 0–7, two clips per slot. Assign **every complete
v142 source/exact-content component touching any of those slots** to `pilot`;
assign all other components to `reserve`. Report actual counts if closure expands
the pilot. Never split a component or substitute a source based on quality,
geometry or detector responses. The observed unaliased bank gives 16 pilot clips
and 16 untouched reserve clips. Source-slot/content components do not establish
distinct people, unknown ancestry or pretraining independence.

Retain eleven native variants per clip, normally **176 pilot cases**:

| Variant | Cases per clip |
| --- | ---: |
| Original | 1 |
| MuseTalk × source-time / earlier speech conditioning | 2 |
| LivePortrait × source-time / earlier expression driver | 2 |
| Branch-specific crop/inverse/alpha resampling | 2 |
| Branch-specific resampling plus 1.10 brightness | 2 |
| Genuine-picture cut/retiming | 1 |
| Local earlier-sample audio control | 1 |

Fix the half-open output picture interval `[27,47)`. The driver is the actual
picture `i−8`, only when native PTS proves it is exactly 8/25 seconds earlier.
No nearest-picture substitution or frame-index/fps inference is permitted. Each
source retains all twenty per-frame source and driver records. Reconstruct the
unchanged v109 full-waveform conditioning plan twice per output picture: its
actual source time and that time minus 8/25 seconds. Keep the **output picture
index** in both plans; preserve separate observed audio/video origins and the
actual PCM sample count. Native clock allowance remains one native timestamp tick
plus one native audio sample. Require the source sample and all ten retained
encoder ticks; unsupported contexts stay unavailable without padding or invented
conditioning. Retain possible-anchor ambiguity and nominal convolution hulls;
whole-input attention/spectral normalization means these are not local causal
evidence.

The stateless geometry inventory is the union of actual pictures 19–46: 28 unique
pictures, plus an exact repeat of picture 27. Geometry and source/driver support
remain requirements to verify later, not inferred face eligibility from the old
qualification. Neural/resampling/brightness cases require their own branch's
actual crop, inverse transform and alpha. Use half opacity on first/last edit
pictures and full opacity inside. Genuine retiming copies the twenty actual
earlier pictures at full opacity, with original output PTS and soundtrack; it
requires those pictures, not face geometry.

The audio control changes only the exact output sample interval corresponding
to the two observed edit endpoint timestamps, copying the same-length interval
5,120 canonical samples earlier. Preserve total length, origins and all samples
outside that interval; no padding or wraparound. Its complete changed waveform
needs **new whole-input speech encoding**. A shifted slice of original hidden
states is not equivalent. The current bank's coordinates are retained privately;
the public result exposes no per-source clocks. Audio controls require audio
support, not face geometry.

Bind MuseTalk's seven MIT/Apache asset manifest, distinct acquired-assets receipt,
actual successful isolated v118 execution and its journals. The unexecuted v117
context is not a successful model run. Bind the retained MIT human animation
catalogue/license and actual full-expression v140 result/profile/four journals.
Keep its historical observed driver offset honest; it does not supply GRID
geometry or prove exact320ms achieved motion. Source-time redraw and
earlier-conditioned redraw are requested constructions. Actual generated pixels,
changed support and achieved visible/phonetic alignment require later checks.

Bind the exact three v125 heads, six own v126 point/clip study calibrations and
all settings 0/1/2, with 2 primary. No new fitting/calibration is authorized.
Study probabilities are not customer authenticity guarantees. Future evaluation
must include the current complete serving video pipeline and appearance/fallback,
alongside the dense-input correlation diagnostic; that diagnostic alone is not
the service baseline. Ordinary cut/audio-delay anomalies may be correctly flagged
as edits/timing issues; calling them AI provenance is false attribution.

After root review/publication, use one exclusive `bind-00` data stage under the
unchanged v108 owned-descendant monitor: 900 seconds, 1.5 GiB sampled descendant
RSS, 3 GiB initial RAM/project/archive disk and 1.5 GiB running reserves. Preserve
attempts; no retry, overwrite or relaxed limits. Publish aggregates and whole
artifact/code commitments only. Private plans retain all32 original rows and
their full clocks, source paths, pairings and hashes, including unexecuted reserve
specifications. No Actions, outreach, payment, deployment or serving change.

## API for the next stages

Import `scripts.grid_transfer_plan_v143`. `verify()` is metadata-only.
`load_completed()` requires the actual successful planning result/profile,
verifies `private_plan_sha256`, and reproduces the complete pure plan before
returning it. Read no plan from a failed/partial or merely frozen context.

- Trial: `backend/data/independent-evaluation/grid-transfer-plan-v143`.
- Public freeze: `docs/evaluation-panels/grid-transfer-plan-freeze-v143.json`.
- Actual private plan: `bind-00/output/private-plan.json` relative to the trial.
- Private schema: `chipmunk.grid-native-transfer-plan/143`; root `sources` has
  all32 ordered entries and `case_order` has all eleven fixed IDs.
- Source entry: `ordinal`, `source_slot`, `within_source_ordinal`,
  `component_index`, `role`, `input` (unaltered full v137 row), `availability`,
  `plan_status`, `unavailable_reason`, `geometry`, `frames`, `audio_delay`,
  `encoder_inputs` and `cases`. Only `role == "pilot"` is eligible for execution.
- `geometry_requests(row)` is pure. Its result has `status`,
  `unavailable_reason`, `frame_indices`, `repeat_frame_index`, `pictures` (exact
  original picture metadata), and declared unique/repeat task counts.
- `frame_support(row,index)` is pure. Each of twenty source `frames` entries
  includes `frame_index`, actual `source_picture` / `earlier_picture`,
  `source_time_seconds` / `earlier_time_seconds`, `source_time_conditioning` /
  `earlier_conditioning` (complete v109 dictionaries), `status` and reason.
- `source_plan(row,role,component_index,ordinal)` and
  `build_plan(rows,observations,links,dependencies)` are pure metadata helpers.
  Case readiness is per required check: branch geometry, actual copied pictures
  or audio support. Missing support is never a numerical original/AI result.
- `encoder_inputs` IDs are `original_pcm` and `delayed_audio_control`.
  `audio_delay.output_samples_half_open` and `source_samples_half_open` are
  exact integers; `shift_samples` is 5120. The original row's `pcm_mapping`,
  `pcm_file`, `wav_file` and hashes remain at `source.input`. Case `waveform_id`
  selects the changed complete waveform only for `audio_delay_320ms`.
- Actual result schema: `chipmunk.grid-transfer-plan-result/143`;
  `private_plan_sha256` and `freeze_sha256` bind the outputs. Aggregate fields
  include `pilot_clips`, `reserve_clips`, `pilot_source_slots`, `pilot_cases`,
  `pilot_ready_cases`, `pilot_unavailable_cases`, `planned_cases_all_roles`.
