# CPU renderer passes after bounded convolution execution

The new v139 context completed its actual pretrained CPU run in **25.63 seconds**,
with maximum sampled descendant RSS **2,546,991,104 bytes (2.37 GiB)**. Worker exit
was zero, with no guard or monitor errors. The unchanged limits were 6 GiB sampled
RSS, 6 GiB initial available RAM and 900 seconds, plus the existing disk/running
reserves. Samples do not guarantee the true peak.
[Actual measurement and profile](evaluation-panels/portrait-tiled-runtime-result-v139.json).

All five checkpoint state loads were strict and tensor-exact. The three face-task
API inputs passed the exact stateless-geometry repeat. All four renders passed
full-resolution output, byte-identical own repeat, counterfactual response and
native mouth-support containment checks. The first new generated crop was
**byte-identical** to the complete first crop preserved from the stopped v133
run: maximum/mean RGB-byte differences both zero. The frozen tolerance was not
changed after observing this response.

The hooked counters recorded four calls to each of the five top-level generator
modules: twenty forwards plus three face-task API inputs. These are separate
units; internal face-task graph invocations are not independently counted.
The bounded implementation executed 148 tiled spatial-convolution calls through
948 functional tiles. Its largest estimated tile-column workspace was
133,890,048 bytes, below the fixed 128-MiB estimate budget. This is not an actual
library-allocation guarantee; the real sampled resource profile is separate.
Per-render times were 3.05–3.09 seconds on one CPU thread.

The [original v133 attempt](evaluation-panels/portrait-native-resource-stop-v133.json)
remains a failed 18.48-second memory-limit stop, with 8,480,178,176 observed RSS
bytes and no completion/counter receipt. Its partial images, failure artifacts,
source, weights, precision and resolution were not overwritten or rewritten as
passing. The new [frozen method](portrait-tiled-runtime-v139-method.md) changes
only spatial execution of large convolutions, assembling complete outputs before
normalization/nonlinearities. It neither shrinks images nor removes model layers.

Nine new authored operator/context checks pass, covering full/tiled CPU numerical
agreement, halos, seven-sized kernels, grouping/stride/dilation, state preservation,
own repeats, unsupported modes and the predeclared original-crop tolerance. Ruff
and diff checks pass. The additional 27 transfer-preparation/expression arithmetic
checks remain a separate local result; those do not certify the pretrained model.

This qualifies one consumed fitting-source **construction runtime**. Scalar
aperture response does not prove achieved phonetic retiming. The richer
[expression-driving prototype](portrait-expression-drive-v138-design.md) still
needs separate neural qualification; the fresh 32-clip cohort still needs complete
acquisition, generated-edit/ordinary-control construction and frozen-head transfer
testing. At the original controller stop, eight sources/sixteen clips had complete
native banks. Remaining original source slots 8–15 are unattempted and may be
scheduled separately. Do not restart the stopped controller or retry v133.

No detector/default, customer confidence, billing, Actions workflow, outreach or
GCP deployment changed. Raw media, individual fingerprints, voice/motion vectors
and learned parameters remain private; publication is code, aggregate outcomes
and whole-artifact commitments.
