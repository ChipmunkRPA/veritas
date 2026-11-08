# Public audio watermark: authored native contract

This is a real local execution of a public detector, not a provider API stub.
It establishes implementation behavior, not real-speech detection quality or
compatibility with current Muse/Gemini/Grok products. No serving confidence,
payment requirement or customer result changes.

## Pinned public assets

Use MIT-licensed AudioSeal source revision
`1496f7662d6aa625602bb0dcd1b4997b63b47609` and public model revision
`3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba`, from the
[official source](https://github.com/facebookresearch/audioseal) and
[official model release](https://huggingface.co/facebook/audioseal).
Detector file SHA256 is
`8a78e8a83584113523e161fc599fcab10fd0e94c04d2eb9d2fa1e9ec91ab69d9`;
generator is
`7a845b5fbe9364a63a3909d8ab3fe064d13a76ae4c2e983573e08c69b7b51748`.
Verify publisher LFS byte identities; final download ETags are Xet identifiers
and are not the file SHA256. Retain the stopped initial ETag-assumption attempt.

Source and small OmegaConf2.3.0/ANTLR4.9.3 dependencies stay in a private isolated
directory; their files and archives are hash-bound. No package build/setup
script runs and the application environment is unchanged. The upstream
unrestricted pickle/network loader is disabled. Use restricted Torch
weights-only loading with a fixed data-type allowlist. Only the unused
training-metadata defaultdict factory becomes an inert ordinary dictionary
for current Torch compatibility; its items are preserved. Do not resolve or
use checkpoint training configuration. Use inspected local model cards.

Construct CPU float32 architectures from those cards, convert the old convolution
key names and weight normalization, and verify every final tensor against the
original state or the exact native weight-normalization calculation. No missing
or silently initialized parameters are allowed. Freeze code/assets/runtime
before any architecture construction or forward in this contract.

## Authored signals and readout

Four deterministic 8-second mono carriers at16kHz: harmonic pulses, band-limited
noise pulses, a chirp mixture and silence. These are **not human speech or
provider-generated samples**. Seed20261108. Fixed alternating16bit message.
For each carrier test unmarked, fully marked, partially marked, corresponding
unmarked/fully/partially marked10dB-noise variants, and unmarked/partially marked
64kbps MP3 variants. The partial mark occupies samples32000:96000 (2–6seconds);
all other source samples remain bit-identical before codec/noise transforms.
Retain all32 cases, including failures or unexpected detector responses.

MP3 files are seekable and decoded to16kHz float32. Require exact source sample
count preservation; do not hide gapless/timing problems by trimming or padding.
Use per-sample native responses strictly>0.5. A quarter-second bin is selected
only when more than half its samples exceed0.5; join adjacent selected bins and
retain exact sample/time coordinates and unmerged measurements. These are
watermark response fractions, not calibrated AI-authorship confidence.

Four convenience-API fractions/decoded messages must match the native response
readout (fraction tolerance1e-7). Repeat the first marked response bit-exactly.
Expected total:4 generator and37 detector forwards. Model-response validity,
tensor identity, source preservation, sample geometry and API parity are
implementation requirements. Perfect detection on these out-of-domain carriers
is not assumed; observed misses and false responses remain in the result.

## Limits and continuation

The public-family response does not authenticate a provider or creator, does
not enter overall AI confidence and cannot activate the trusted AI-declaration
early stop. A genuine recording can be watermarked; absence cannot clear a file
as original. Real marked/unmarked speech, noise/background, codec and partial-
interval comparisons remain required before a customer-facing default.

Bound the native job to900seconds,6GiB sampled process-tree RSS and at least
1536MiB available RAM/free disk. Retain failed profiles and never silently
restart. Keep authored arrays/private journals locally; publish code, hashes
and aggregates only. No customer input, provider upload or GCP action occurs.
