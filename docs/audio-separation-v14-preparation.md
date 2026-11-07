# Two-source audio inspection: implementation and preparation

6 November 2026. Implemented a different audio research method for the remaining
overlapping-voice failures: estimate two sources, inspect both with the existing
frozen voice backbone, and learn whether either estimate supports an AI response.
The method and synthetic checks are complete. **No project recording has been
scored, no project classifier fitted or calibrated, and no serving detector changed.**
The full source-bound execution freeze and consumed/fresh comparisons remain.

## Why this method

The [completed noise-aware comparison](audio-wavlm-v13-development-evaluation.md)
improved noise/music conditions but lost useful AI detection with overlapping
voices. Removing guards could not recover the required generator retention.
This preparation changes the available information by adding two source
estimates. It does not loosen the original sixteen adoption requirements.
Source separation can introduce synthetic artifacts and can fail to disentangle
speakers; a clearer-sounding estimate does not establish better AI detection.

## Pinned source, permissions and discrepancies

Acquired the author's 20,394,640-byte checkpoint
[`JorisCos/ConvTasNet_Libri2Mix_sepclean_16k`](https://huggingface.co/JorisCos/ConvTasNet_Libri2Mix_sepclean_16k/tree/e1ef95ab7a037950f3a606b9a56760cf94701d3d)
at `e1ef95ab7a037950f3a606b9a56760cf94701d3d`. Its SHA-256 matches the publisher
LFS commitment:
`8d97f012f7b2f22bb79cb0d0983a7ba27a52c1796ee3f63cbf25b4d28630adce`.
The pinned card SHA-256 is
`c34341c446d0c5abd1603ee3cb9e4c0093a7b2422389fdeceeea7625e280fce1`.

The card's metadata says CC-BY-SA-4.0 while the author's explicit notice says
CC-BY-SA-3.0. Both [3.0](https://creativecommons.org/licenses/by-sa/3.0/) and
[4.0](https://creativecommons.org/licenses/by-sa/4.0/) permit commercial reuse
subject to attribution, ShareAlike and their other conditions. Preserve both
publisher statements; this audit does not establish which version supersedes
the other or resolve all underlying rights. The checkpoint also retains its
LibriSpeech CC-BY-4.0 attribution. Do not call these weights MIT or silently
replace the author's notice. Exact training membership/voice overlap and complete
training-data rights remain unverified. The source recordings and weights are
not redistributed in this preparation package.

The small local architecture is adapted from MIT-licensed Asteroid 0.7.0 and
asteroid-filterbanks 0.4.0. Both wheels match the official PyPI SHA-256 values;
the full [MIT notice](third-party/asteroid-v14-MIT.txt) is retained. No package
was installed into the app environment. The model's own historical metadata
records Asteroid 0.4.0/Torch 1.7.0; the reference audit uses explicitly inspected
0.7.0 masker/normalization definitions and 0.4.0 native filterbanks under the
current numerical runtime. It does not recreate the historical training runtime.

The checkpoint serializes a sample-rate field of 8000 while the pinned training
card says 16000. The selected free learned-filter graph uses that field only as
metadata. Changing it from 8000 to 16000 in the upstream reference leaves all
synthetic forward outputs bit-for-bit identical; there is no resampling in this
path. Retain the serialized value unchanged, and explicitly use the original
16 kHz project samples in accordance with the author's training declaration.
This proves the tested graph's behavior, not the true historical training rate
or source-separation quality on real voices.

## Local execution evidence

Strict safe loading accepts all 345 tensors and exactly 5,066,929 parameters,
all finite/frozen on CPU float32. The checkpoint requires no unexpected pickle
globals or custom remote code. The local graph accepts the exact original
64,000 mono samples and returns two unordered 64,000-sample estimates, with no
input normalization, padding, clipping or resampling in the separator itself.
The later voice inspection retains its existing published preprocessing.

The [synthetic audit receipt](evaluation-panels/audio-v14-separation-structural-audit.json)
records three inputs: two tones, seeded noise and an impulse. The local graph
matches the inspected upstream masker/filterbank reference exactly for all
three, with maximum waveform difference zero. Repeated inference is identical,
both estimates remain complete, and caller input is unchanged. The reference
loads only the selected upstream definitions plus native filterbanks; inactive
tracing decorators are identities, and optional training/publishing/file-loading
paths are excluded. This is a restricted graph-parity reference, not a full
installed Asteroid runtime or independent human review.

Ten source/shape/normalization tests pass. Ten further tests check unordered
source invariance, stable extreme logits, finite-difference gradients for both
proposed arms, failure/omission handling and refusal of development/calibration
roles during fitting. Two include complete toy-feature fits and separate
calibration pools, confirming that an unassessed group remains unassessed.
They use synthetic numerical vectors, not project recordings. These tests are
not detection accuracy.

## Learning and calibration design

An AI label on a mixed recording does not mean both estimated sources are AI.
The new shared classifier therefore learns from whole-mixture labels using a
multiple-instance objective. Its raw bag response is the noisy-OR of the two
estimated-source responses. The stable logit expression is
`logsumexp(z1, z2, z1 + z2)`. Swapping the two sources leaves the response
unchanged; no foreground/source identity or known background condition selects
an estimate at inference time. Source estimates are not identified people or
guaranteed clean voices, and their responses are not individually calibrated
authorship probabilities.

The two fixed arms are separated estimates alone and the same bag response
combined with the original mixture raw logit. Both use the original frozen voice
features, fitting-only weighted normalization, original connected-source weights,
their own independent calibration, original stability checks and all sixteen
original requirements. The deterministic bag objective is nonconvex; optimizer
convergence will not establish global optimality or correct source assignment.
No per-source AI truth labels, known clean reference, condition, generator or
speaker identity enter the predictors. The feature arms are defined in
`backend/scripts/audio_separation_response_v14.py`; only synthetic unit-test fits
have run, with no learned project head or customer confidence update.

Next complete and publish the full execution lock before any project forward.
Retain all 4,248 original consumed inputs and fixed fitting/calibration/development
roles, including every unassessed case. Test genuine single voices, overlapping
voices, ordinary noise and music. Source-recovery diagnostics may compare with
the known constructed mixtures only for evaluation, never as inference inputs.
Any required band-filter stability probe must rerun the separator and the same
source feature extraction; applying an old calibration to altered waveforms is
not sufficient. Reserve the 1,080 unscored confirmation inputs until a fully
frozen passing development result. Adoption still needs fresh confirmation,
commercial-use conditions satisfied, UI/PDF/JSON correspondence and paid-flow
checks. Image, video, writing and outside human testing remain in scope.
