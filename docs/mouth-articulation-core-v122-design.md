# Native mouth/speech reference interaction: implementation design

This introduces feature math for the next lip-sync comparison. It does not fit
a real-media head, extract new model features or establish improved detection.
The [C-MIND method](https://arxiv.org/html/2610.30872v1) motivates aligned
frame-level concatenation/products/differences instead of only coarse energy
correlation. This is an independent small reference-relation adaptation, not a
reproduction: it uses permitted frozen mouth-image features and Whisper states,
not the paper's pretrained visual/TDNN encoders or weights. Whisper is ASR-trained;
no claim of avoiding ASR-derived representations is made.

The visual input is a 768-variable mouth representation at every observed native
picture. Its temporal descriptor includes current appearance, absolute residual
against time-weighted neighboring-picture interpolation, and motion scaled to a
40-ms period. Require real neighboring pictures with gaps no larger than 80 ms;
do not bridge missing context or use known edit boundaries to choose a point.
Speech uses ten fully observed 50-Hz ticks and all five qualified 384-channel
states, summarized by per-state/channel mean and standard deviation. Explicit
out-of-prefix padding makes a point unavailable here, not original. These are
indexed feature summaries, not proof of finite temporal locality: the encoder
can depend on the whole padded input, as the v118/v110 contracts record.

Learn 16-dimensional weighted whitened projections and a fixed 0.1 ridge map from
audio coordinates into visual coordinates using **only untouched fitting-role
sources**. Equal-source weights avoid long recordings dominating the reference.
The code rejects calibration, selection, confirmation and edited reference rows.
Common coordinates precede element-wise interaction; unrelated PCA coordinates
must not be directly multiplied. Return visual-only, concatenation and explicit
interaction features as separate arms. They are not margins or probabilities.
Each later classifier/calibration must be fitted separately under frozen roles;
no representation alone contributes to customer confidence.

The complete inference path still needs native full-clip feature extraction,
per-point source/ROI bindings, model loading/repeat checks, source-balanced heads,
calibration, current-check comparison and fresh grouped results. Extraction must
not read true masks, edit positions, condition labels or source identities as
features. Evaluate original/ordinary edits, both neural conditions, missed edits,
false alarms, unassessed points and time/region overlap separately. Centered
construction alone cannot establish general placement invariance; a subsequent
boundary/placement transfer test is needed. Well-synchronized retalking remains
a particularly important test; high coarse synchronization is not originality.

Keep pretrained/checkpoint identities, source features and learned reference
parameters private. Publish method/tests and aggregate outcomes. Existing v6
mouth-image and v14 RMS/aperture results remain separate baselines; their weak
transfer and ordinary-control flags must not be erased by this new method.
No author model/code/data, noncommercial asset, serving change, new confidence,
external outreach, GitHub Actions, real payment or GCP deployment is introduced.
