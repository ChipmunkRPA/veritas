# Corrected inventory passed; generator smoke rejected extra state

The corrected v55 preparation verified 90 packages and 25,302 installed files
and passed its resource profile. The subsequent bounded smoke **failed before
G2P or generator/detector forwards**. Do not relabel it successful generation.
Its frozen code and failed output remain unchanged.

The strict checkpoint check expected every constructed state tensor to exist
in the saved checkpoint. A separate zero-forward constructor/key inspection
found 548 saved tensors plus 140 additional normalization parameters created
by the installed upstream graph. The upstream AdaIN implementation explicitly
uses affine InstanceNorm parameters for its export workaround. These are
intended as identity scale/bias parameters; they are not additional trained
checkpoint weights. Their neutral values and every saved tensor's actual loaded
value still need explicit verification before generation.

The successor must allow only those declared AdaIN InstanceNorm identity pairs,
verify exact one scales/zero biases and their absence from the checkpoint,
require every saved key/shape/dtype/value, reject all other missing/unexpected
state, and verify state remains unchanged after inference. An independent
functional test must compare affine identity normalization with the non-affine
operation. This corrects the integrity predicate; it does not permit a partial
learned checkpoint or change a detection threshold/calibration.

The [failure receipt](evaluation-panels/audio-generator-qualification-v55-failure.json)
binds the passing preparation, failed smoke and successful static inspection.
It also retains two diagnostic launch errors before constructor/forward: a
missing project import path and a private script shadowing stdlib inspect.
The separate package-module inspection passed with no resource violation.
No research cohort was generated or scored. Defaults, customer confidence,
protected calibration, workflows, outreach and deployment are unchanged.
