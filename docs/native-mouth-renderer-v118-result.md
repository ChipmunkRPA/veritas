# Native mouth renderer: executed response contract

After the owner freed memory and requested another admission check, the exact
frozen [v118 runtime context](native-mouth-runtime-v118-method.md) executed
successfully. The original preparation failure and earlier denied-admission
[snapshot](native-mouth-renderer-readiness-v118.md) remain unchanged historical
records. No model-stage directory existed before this successful start.

On the fixed fitting-source native picture and saved source speech states:

- Four renders completed using the same seed: aligned, repeated, 320-ms earlier
  and zero conditioning.
- Repeated output pixels and model proofs were bitwise identical.
- Earlier and zero conditioning changed 55,123 and 58,367 crop pixels relative
  to the aligned render.
- The aligned composite changed 2,697 native pixels. All four composites had
  nonempty exact change masks, with no changes outside the source mouth alpha.
- Forward hooks counted eight VAE encoder, four VAE decoder and four UNet
  calls, with zero speech-encoder calls. One face detection brings the actual
  total to **17 neural inputs**.

The source and aligned native result were visually inspected locally. The
mouth redraw is visible; a still image cannot establish achieved phoneme timing,
motion quality or realistic clip-level synchronization. Source images, generated
frames, learned states and individual feature fingerprints remain private on
Seagate. Only aggregates and whole-artifact commitments are published.

The stage took 19.75 seconds and exited zero with no guard or sampling error.
Initial available RAM was 7,336,427,520 bytes. Minimum sampled available RAM was
1,922,072,576 bytes, above the fixed running reserve. Sampled descendant RSS
reached 1,272,889,344 bytes; maximum reported post-render MPS driver allocation
was 4,659,937,280 bytes. These are different measurements, not additive peak
estimates. All original resource limits remained unchanged.

The [actual result and resource profile](evaluation-panels/native-mouth-renderer-result-v118.json)
have matching private receipt/journal hashes. The successful execution qualifies
this one-source native renderer response. It does **not** qualify achieved mouth
synchronization, a new detector, accuracy or customer confidence. Full native
clip construction, controls and the articulation comparison remain required.
Serving model selection, overall confidence and rollback assets are unchanged.
No GitHub Actions workflow, outside contact, real charge or deployment ran.
