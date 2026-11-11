# Pilot portrait geometry preparation

This new implementation is authored software only until its code and contract
are reviewed, frozen and manually published. Do not execute a face task before
that point. Existing frozen contexts and failed attempts remain unchanged.

Consume the actual complete v143 private transfer plan through its metadata-only
`load_completed()` API. Preserve its pilot roles, component expansion, original
clip ordinals and all unavailable inputs. No face result, image content or
detector response selects a source. Reserve clips are never admitted here.

For every ready pilot, request exactly the original native pictures 19–46 in
order, followed by an exact repeated request for picture 27: 28 distinct pictures
and 29 stateless face-task API inputs. This supplies source frames 27–46 and their
eight-frame-earlier drivers 19–38. Native integer PTS and rational time base stay
attached to every request. A planned unavailable input retains its original plan
and an explicit unavailable geometry outcome; it receives no fabricated task
input or replacement.

Before the first forward, verify the actual successful v132 CPU runtime result,
zero-exit/no-guard profile, installed-files manifest, retained original wheel
hashes and every installed source/native file against those wheels. Reuse the
same isolated MediaPipe 0.10.32 runtime qualified by v139/v140. Verify the reviewed
asset catalogue, source MIT license, exact face-task hash and the whole-file hashes
of all three retained primary bundled-model cards: BlazeFace, FaceMesh V2 and
blendshape, reviewed as Apache-2.0. The original publisher links are recorded in
the [prior asset review](mouth-sequence-v4-evaluation.md):
[BlazeFace](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf),
[Face Mesh V2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf)
and [blendshapes](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf).
Bind those records into the new public freeze. These recorded publisher
permissions do not establish training-data rights or pretraining independence.
No generator checkpoints are loaded by this stage.

Use CPU `IMAGE` mode, `num_faces=2`, all three confidence options 0.5, no blendshape
output and no transformation-matrix output, matching the successful qualification.
Before OpenCV decoding, hash each exact owned source PNG against its v137 file
commitment. After decoding, verify native dimensions and the **plain RGB byte**
hash, without the older shape/dtype-prefixed fingerprint helper.

Flush a private pre-task record before each call and a post-task record after it.
Require exactly one finite 478-point mesh for usable geometry. Run the already
qualified `crop_from_mesh` helper for the full 256×256 RGB crop, its invertible
source-to-256 matrix and source aperture. Compute the full output pixel-centre
matrix as `M512 = 2*M256`, adding 0.5 to both translations. Run the unchanged
`mouth_alpha` helper for nonempty finite native-size float32 support in [0,1].
Retain the exact PNG crop, float alpha NPY, canonical hexadecimal mesh/matrices
and aperture privately on Seagate. Verify saved crop pixels and alpha values in
the worker. Save a matching private whole-record JSON cache on Seagate and bind it
to the private stage record. No identity recognition or deepfake detector runs.

Missing faces, multiple faces, invalid meshes, invalid crop/support geometry and
unsafe picture decoding retain explicit unavailable records. Continue all other
safe planned calls, including the planned repeat, without substitution. An
integrity mismatch or task runtime exception remains a preserved stage failure.
If picture 27 and its repeat are usable, their mesh, mappings, aperture, crop and
exact float alpha must match bitwise. Persist records and repeat response before
gating this requirement. An unavailable repeat is unassessed, not a passed repeat.
Partial successful geometry stays usable for independently supported pairs;
global `geometry_qualified=false` must not erase other available pictures.

Run one exclusive `clip-OO/authored` neural stage per pilot through unchanged v108
owned-descendant monitoring: 900 seconds, 6 GiB initial available RAM and sampled
descendant RSS, 3 GiB initial project/archive disk and 1.5 GiB running reserves.
Samples are lower bounds rather than guaranteed peaks. Admission denial creates
no attempted authored stage; preserve every attempted failure without retry,
overwrite, relaxed limit, source replacement or concurrency with another guarded
stage. No generator, encoder or deepfake detector inference is included.

The stable consumer API is `completed(clip_ordinal)` and
`load_completed(clip_ordinal)`. The latter rechecks the actual completed stage,
private record/progress/repeat hashes, resource/profile observations, exact native
picture correspondence, owned cache paths, canonical metadata and ordered task
protocol. It reads no saved crop/alpha payload at admission. A consumer checks
those payload hashes and values within its own guarded stage. An available repeat
failure is rejected; a complete partial-geometry result is returned explicitly.

Private `output/geometry-records.json` uses schema
`chipmunk.grid-portrait-geometry-records/144` and contains `clip_ordinal`, full v143
`planned`, 28 ordered `pictures`, `repeat`, `repeat_bitwise` and
`geometry_qualified`. Every available picture contains `frame_index`, `pts`,
`time_base`, `source_file`, `source_file_sha256`, `source_rgb_sha256`,
`mesh_xyz_hex`, `source_to_256_matrix_hex`, `source_to_512_matrix_hex`,
`source_ratio_hex`, and `crop`/`alpha` objects with `path` and `sha256`. Unavailable
rows keep original correspondence and an explicit reason. Per-request records
are saved immediately as well as in the whole artifact. Only code, methods,
aggregate availability/call counts and whole-artifact commitments may be public;
all media, per-media fingerprints, face vectors, clocks and pairings stay private.
No Actions, outreach, serving-default, payment, deployment or confidence change.
