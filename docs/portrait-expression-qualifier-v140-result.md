# Full-expression construction runtime passes locally

The actual frozen v140 CPU stage completed in **24.40 seconds**, with maximum
sampled descendant RSS **2,486,992,896 bytes (2.32 GiB)**. Worker exit was zero,
with no guard violations or monitor errors. The original 900-second and 6-GiB
neural limits were unchanged; samples are lower bounds rather than peak-memory
guarantees. See the [actual result and resource profile](evaluation-panels/portrait-expression-qualifier-result-v140.json)
and [pre-execution method](portrait-expression-qualifier-v140-method.md).

All five pretrained checkpoint loads were strict and tensor-exact. Three generated
512-pixel crops passed output and native mouth-containment checks. The current
expression and its repeat were byte-identical. Driving with the already selected
earlier picture changed 220,718 crop pixels and 1,856 native pixels; no native
changes escaped the declared mouth support. Cached geometry was verified, with
zero new face-task inputs. Fifteen hooked root forwards completed: three appearance,
six motion, three warping and three decoder calls. No lip-ratio head ran. The
111 tiled spatial convolutions used 711 functional tiles, with maximum estimated
tile-column workspace 133,890,048 bytes, below the fixed 128-MiB estimate budget.

The new v141 installer enforces CPU float32 inference constraints on both small
and tiled convolutions and installs/restores methods transactionally. It reuses
unchanged v139 spatial arithmetic. Local software verification passed 52 authored
tests across the expression bridge, qualification contract and installer, plus
Ruff and diff checks. These software checks and this actual neural execution are
separate evidence.

The earlier full-expression crop response was larger than the scalar-aperture
path's 127,314 pixels. **That does not establish better motion or detection.**
The two paths also produce different source-time redraws. Achieved lip trajectory,
phonetic alignment, ordinary-edit false alarms and fresh-source detector accuracy
remain unmeasured. This qualification used one previously consumed fitting source,
not the fresh transfer cohort, and changed no serving model or confidence score.

All sixteen originally selected source archives and all 32 native frame/audio
banks are now complete. The next independent saved-bank review checks all 2,400
PNGs, canonical audio and rational timestamp mappings before generated edits and
ordinary controls are constructed. Source/media identities, raw media, vectors,
parameters and individual fingerprints remain private. Only code, aggregate
results and whole-artifact commitments are published. No GitHub Actions, outreach,
payment or deployment was performed.
