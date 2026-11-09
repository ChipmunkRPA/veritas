# Broader image testing: release metadata and controlled-source feasibility

The remaining DiffSeg30k archive has now been checked at the metadata level.
It does not add the missing ancestry fields. A separate cached-source review
identified a possible controlled-test route: **58 attributed JPEGs in 49
components with no identified link to the 637 existing study sources**. These
are candidates, not admitted test cases or evidence of detector improvement.

## Complete published archive metadata review

At publisher revision `8049819755ff430dcf4701636b14e6f1df6c4542`:

| Archive | Image/mask pairs | JSONL rows | JSONL fields |
| --- | ---: | ---: | --- |
| Validation | 6,001 | 6,001 | image filename, mask filename |
| Training | 28,998 | 28,998 | image filename, mask filename |

The training directory has 57,998 entries: those pairs, one directory and one
JSONL sidecar. Only exact ZIP64 directory ranges and the named JSONL member were
read. Its declared length and CRC32 matched; no PNG payload was read or decoded.
The bounded worker completed in 3.19 seconds with no monitor error or resource
crossing. Its largest observed worker RSS sample was 105,562,112 bytes, not a
guaranteed peak. Full-archive hashes remain unverified.

Both reviewed sidecars lack source-photo/creator links, source-type labels and
edit histories. They cannot establish the source separation, matched controls
and AI-base interpretation required by this study. Do not treat the publisher's
[Apache-2.0 card](https://huggingface.co/datasets/Chaos2629/Diffseg30k/blob/8049819755ff430dcf4701636b14e6f1df6c4542/README.md)
as a substitute for those links or individual-photo permission evidence. This
extends the [validation-only review](image-cross-generator-diffseg-metadata-review-v31.md);
its original failed monitor is still retained and not relabelled. Provenance
reconstruction through other evidence remains possible and has not been done.

## A separate cached source pool

The v20 catalog contains 68 attributed Flickr JPEGs whose pixels did not match
the old publisher PNGs. That mismatch is neither an AI finding nor proof of
camera origin. They were excluded from the v21 positive-pair study. A new
controlled study would use the current attributed JPEG itself as its reference,
not combine it with the mismatched old edit/mask pair.

We verified all 68 cached JPEG hashes, bounded dimensions and upright orientation.
We then joined them to all 637 existing-study records using the frozen v22
source/creator/exact-RGB/thumbnail connected-component rules. Excluding every
component touching any existing source leaves 58 JPEGs in 49 components. Ten
photos were excluded through identified prior-creator links. The retained
photo-license snapshots are 40 CC BY 2.0 and 18 CC BY-SA 2.0, with attribution
and source links retained privately.

This was a local, model-free feasibility check: no network request, generation,
detector inference, fitting or role assignment. It completed under the bounded
monitor with no error or resource crossing. The existing grouping rule can miss
aliases, transformed duplicates and unknown pretraining overlap; 49 components
do not prove 49 independent people or authenticated camera sources. Current
source licences do not clear a future generator's complete component chain.

## Generator candidates for known test edits

The following publisher metadata was checked without downloading any weights:

| Candidate | Publisher-declared licence | Unfinished qualification |
| --- | --- | --- |
| [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | Apache-2.0; licence file read | Complete component/runtime binding and bounded Mac memory test |
| [SD1.5 inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting) | CreativeML OpenRAIL-M | Full use-based conditions, components and hardware test |
| [Kandinsky 2.2 inpainting](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder-inpaint) | Apache-2.0 in model card | Prior/encoder/component licences, mask polarity and hardware test |

All three inspected repositories are public and ungated at their recorded
revisions. The [BFL model overview](https://github.com/black-forest-labs/flux2)
separates Apache-licensed 4B variants from non-commercial 9B/dev variants; the
licences must not be interchanged. Published memory descriptions do not establish
that a configuration fits this Mac's reserves. Kandinsky's card also documents
a mask-polarity change, which requires explicit version binding and an authored
geometry check. No candidate is qualified by a licence tag alone.

## Next action and limits

After the running training sequence finishes, complete component/asset review and
bounded authored hardware/geometry probes. Then freeze the source roles, masks,
generators, prompts, seeds, ordinary controls, formats and failure denominators
before any project-source generation or detector inference. Keep reference
coordinates and provenance for every generated edit and preserve failed cases.
Do not select examples by whether the detector succeeds or by visual quality.

The [aggregate receipt](evaluation-panels/image-fresh-source-route-v34-v35.json)
records hashes and counts only. Source IDs, creator profiles, JPEGs, sketches and
source pairings stay private. The current v26 training/held-out roles and serving
defaults are unchanged. No external person was contacted, no paid generation or
cloud resource was used, and no broader accuracy claim has been established.
