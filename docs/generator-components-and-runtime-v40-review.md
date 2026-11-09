# Generator components and isolated runtime review

This review prepares additional controlled image tests using the
[161 reviewed reference photos](image-fresh-source-cache-v39-results.md).
It does not admit a generator, add training data or measure detection accuracy.
No new generator weights were downloaded and no generator was run.

## Complete bundles, not just headline model sizes

| Generator route | Reviewed declared weight-file bytes | Licence position | Remaining qualification |
| --- | ---: | --- | --- |
| FLUX.2 Klein 4B, saved 4-bit variant | 4,608,176,698 | Apache-2.0 upstream and quantized card; MIT runtime | Exact asset verification, imports, memory and reference-edit geometry |
| SD1.5 inpainting, files labelled fp16 including safety checker | 2,740,652,750 | CreativeML Open RAIL-M, conditional uses | Component binding, intended-use conditions and hardware/geometry |
| Kandinsky 2.2 prior plus inpainting decoder, default safetensors | 15,857,315,676 | Apache-2.0 cards; encoder lineage still needs binding | Full component identity, staged loading and mask polarity |
| SDXL inpainting, files labelled fp16 | 6,938,041,649 | CreativeML Open RAIL++-M, conditional uses | Component binding, intended-use conditions and staged loading |

These are publisher-declared file inventories, not downloaded-file hashes or
measured RAM requirements. The [aggregate inventory](evaluation-panels/generator-components-v40.json)
pins all five repositories, including Kandinsky's separate prior. It labels
LFS hashes as declared rather than byte-verified. Do not count the decoder alone
as Kandinsky's complete pipeline or infer Mac compatibility from storage size.

## Licence and method distinctions

The first intended hardware trial is the Apache-licensed 4B route. Its
[upstream licence](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/e7b7dc27f91deacad38e78976d1f2b499d76a294/LICENSE.md),
[saved-variant card](https://huggingface.co/mlx-community/flux2-klein-4b-4bit/blob/860e87183ceb29e39627c0612ebd66d8ea66e68c/README.md)
and the Qwen3 encoder's full Apache licence were reviewed separately. This is
not permission to substitute the differently licensed 9B/dev family members.

The full publisher-linked [SD1.5 licence](https://huggingface.co/spaces/CompVis/stable-diffusion-license/blob/14d42d09bffd871b1666a084fc954a50cff72ac0/license.txt)
and [SDXL licence](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/462165984030d82259a11f4367a4eed129e94a7b/LICENSE.md)
contain use and downstream conditions. Their cards also describe intended
research use. They are conditional secondary candidates, not interchangeable
with an Apache/MIT grant or blanket clearance for the SaaS or expert work.

The [Kandinsky inpainting card](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint/blob/db790ad5cbcabed886f069ef2710774657621702/README.md)
documents a mask-polarity change and points to a separate prior. Its architecture
references cannot replace exact component binding. A model-card licence tag
alone does not complete that review.

FLUX's reviewed editing API takes reference images rather than an inpainting
mask. A localization benchmark must explicitly define source-coordinate
composition and matched ordinary-edit controls; it must not claim that a mask
was consumed by that model. Pin output geometry, masks, prompts, seeds and
failure denominators before project-source generation.

## Installed runtime integrity; numerical testing still pending

A separate temporary runtime contains 56 exactly pinned packages. All 1,059
installed MFlux package files were compared byte-for-byte with the reviewed
PyPI wheel. Its weight and tokenizer loader files also match the inspected
upstream blobs. The released wheel differs from current upstream code in other
places, so no whole-repository identity or untested API compatibility is assumed.

The [published lock](evaluation-panels/generator-runtime-v40-requirements.txt)
preserves the ordered package pins and hashes while omitting private command
comments. It targets this Mac ARM64/Python 3.12 trial, not production. Use a new
environment, require hashes and binary wheels, and retain the exact upstream
[MIT notice](https://github.com/mflux-community/mflux/blob/17b47e9aa0fd4b65a3ec03b46db31c911e8ac18d/LICENSE)
separately; the inspected wheel did not bundle a licence file.

The first dependency command was rejected for redundant incompatible flags;
a separate corrected resolution passed. Installation completed its package and
version checks, then failed exact notice verification because an earlier text
copy had added a newline. That installation receipt remains failed. A separate
read-only qualification verified the complete installed set and package bytes,
then retained the exact upstream notice without reinstalling packages. Both
failed attempts and the successful successor receipts are bound in the aggregate.

Numerical imports, model loading, memory and authored geometry tests remain
pending. The active backend's pinned package versions are unchanged. The current
fitting/calibration sequence takes priority; heavy trials remain sequential.
No serving default, confidence rule, workflow, outreach, billing or cloud
resource changed in this review.
