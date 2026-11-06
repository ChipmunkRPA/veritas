# Frozen-representation speech-head development, v6

Development only. The rejected v4 enhancement and v5 quality mappings are not
enabled. The running detector and calibration remain fixed. The purpose here
is to test a genuinely different decision layer, using information upstream of
the existing single raw response, while retaining the same recording and model.

Use the same consumed 120-source, 720-input panel and five connected-source folds
as v4/v5, with no source selection by response. The existing Pella v2 backbone,
layer mixture, preprocessing, original speech gate and windows remain unchanged.
Its publisher and W2V-BERT backbone declare MIT; retain the permission record in
`commercial-model-audit.md`. Unknown training-source overlap remains unknown.
The selected v3 source recordings retain their separately recorded commercial
permissions. No checkpoint is newly downloaded, fine-tuned or distributed.

## Frozen method

Capture the vector entering the existing final linear head using a read-only
forward hook. Replay every assessed input through the unchanged predictor and
require raw response parity within 1e-7. Reconstruct the original response from
the captured vector and original head within 1e-6. Preserve original source and
PCM hashes; missing original windows remain unassessed. This panel has exactly
one complete speech window for each assessed input; refuse another layout.

For each development training fold only, remove from each representation its
component parallel to the fixed pretrained head direction. Compute a
source-balanced weighted mean and PCA, without class-based PCA weighting. Keep
the first 16 residual components and scale by their training standard deviations
with minimum 1e-6. Fit a new logistic decision layer with original clipped logit,
these 16 components and an intercept. The original-logit coefficient is
nonnegative. Use fixed ridge 0.001 on that coefficient and 0.05 on residual
coefficients, no intercept penalty, source/class balanced weights, source-count
target smoothing and reference prior 0.5. No parameter sweep is permitted.

The original-only comparator fits the same original-logit mapping and smoothing,
without representation components. Each held-out fold uses only its own waveform
representation and transformations/coefficients fitted on the other four folds.
No held-out row enters PCA, centering, scaling or classifier fitting. Neither
source name, condition, true SNR, generator nor transcript enters prediction.
The current guarded service mapping remains a separately labelled reference.

Report both classes, inconclusive/unassessed counts, proper scores and paired
source-group uncertainty for all six conditions and the shared-voice panel.
Retain generator, source dataset and background partitions. A fitted residual
head can learn source or channel confounds; identity-separated CV does not prove
generator, prompt or pretraining independence. This is consumed development,
not an independently evaluated replacement.

## Fixed advancement gates

Require every v5 gate, with no changes after extraction: Brier improvement at
least 0.005, log-loss improvement at least 0.01, paired conditional Brier interval
entirely below zero, improved Brier in noise-0, music-5 and the joint-voice panel,
at most one additional false call of either kind per condition, at most two
lost clean correct calls per class, and at least five additional correct calls
in each noise-0 and music-5. Keep identical assessed inputs and zero new extraction
failures. Use the fixed inherited research operating points and 2,000 grouped
bootstrap draws, seed 461109. These are internal diagnostics, not public promises.

Freeze code, protocol, source manifest, registry, model and input hashes before
extraction. A passed development comparison permits only designing separately
frozen fresh confirmation. It does not authorize changing serving defaults or
using the head's development responses as customer confidence. A failed result
is rejected without scoring unused fresh confirmation sources to rescue it.
Any later training change requires a new method identity and evaluation.
