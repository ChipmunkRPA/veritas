# Image residual features: development protocol

This independent implementation tests actual noise/texture features after the
failed fresh JPEG regional-calibration comparison. It does not alter serving
models, confidence, image masks in existing results, billing or reports.

## Method and comparison

Compute local luminance/chroma residual energy, variance at three scales,
adjacent-noise correlation and normalized moments from the source-resolution
decoded RGB image. Pool the 16 fields into a 64x64 grid, append robust
within-image contrasts and fit a small L2-regularized logistic ranking head.
No paired original is needed at inference. Neither raw responses nor research
comparison points are customer AI-confidence estimates.

Compare two arms on identical fit samples: all 32 features, and the same
features projected away from the top two standardized intervention-delta
covariance directions. Estimate those directions using fit-role genuine images
only. This applies a controlled-intervention idea discussed in
[EISL](https://arxiv.org/html/2608.17700v1); it is not that paper's learned
CLIP/LoRA method or a reproduction of its results. No paper implementation,
pretrained checkpoint or new dependency is imported.

## Source roles and controls

Use all JPEG inputs from the consumed 240-source v11 panel. Its specialist_cal
and fusion_fit sources become the 120-source fit set; fusion_cal remains a
source-disjoint 60-source comparison-point set; confirmation supplies 60
diagnostic pairs. Add all 48 consumed v12 pairs to diagnostics. The complete
288 source groups must be disjoint across roles. The earlier v12 holdout is
now consumed; there is **no fresh-source accuracy claim** for this development.
Known prior generator/model-training overlap remains a limitation.

Fit on original JPEG75 RGB, a fixed brightness/contrast intervention, JPEG55
recompression and an additional genuine-only local patch edit. That ordinary
patch applies blur and a mild tone change to the central quarter of image area;
it is a conventional edit, with no AI-positive truth. All source variants stay
together. Lighting/recompression keep their source truth; they are not extra
AI labels. Use exact original RGB-pair change masks, pooled as soft cell targets
for training. Sample 128 deterministic cells per fit case, with half from each
positive/negative set when both exist. Balance total soft class mass and use
L2=.01, with a fixed optimizer and convergence requirement.

Choose each arm's point using only genuine comparison-point sources. Take the
maximum response over each source's original and ordinary-patch versions; use
the descending value at floor(.05*n), with a strict-greater-than comparison.
Thus at most 5% of those source groups can exceed the point. Do not tune using
AI comparison-point labels or diagnostic responses.

Apply a 3x3 median filter, then retain 8-connected superlevel components of at
least 16 grid cells. Source pixel (x,y) maps to floor(64*x/width),
floor(64*y/height). Compute diagnostic IoU and pixel F1 at the exact original
source geometry; genuine and ordinary-edit false flags use nonempty masks.
Retain all failures and misses in denominators. Compare originals/edits with
the input-bound actual current selector measurements from v11 and v12. Current
selector ordinary-patch inference is not cached; it remains unassessed until a
candidate passes this first diagnostic, then needs a separate real comparison.

Continuation requires higher edited mean IoU, more IoU>=.3 cases and no extra
genuine flags versus current, plus ordinary-control flags<=5%. A passing
development comparison is not promotion: current-check ordinary controls,
source-disjoint fresh confirmation and the full app flow would remain required.
Do not select an arm or retune points after reading diagnostics.

## Permissions, execution and reproducibility

The [publisher TGIF/TGIF2 repository](https://github.com/IDLabMedia/tgif-dataset)
declares CC BY-SA 4.0 for the dataset; original photos retain their underlying
rights. This study retains all photos, pairings, features and fitted heads
privately and does not redistribute them or admit a new commercial checkpoint.
The code is independently written; there are no new model weights or downloads.

Freeze code, source plans, actual prior journals and runtime package versions
before feature extraction. The plan has 1,344 feature cases. Resource bounds
are 1,800 seconds, 6 GiB sampled process-tree RSS, at least 1,536 MiB available
RAM and free disk. No silent resume/replacement of a failed job. Keep exact
feature/source/hash journals and lossless prediction responses for a separate
arithmetic/coordinate replay. Public outputs are code, methods and aggregate
results only. Outside human reviewers remain zero; no customer data or external
provider uploads are used. GCP remains deferred.
