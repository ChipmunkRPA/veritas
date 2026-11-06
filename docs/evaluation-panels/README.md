# Public evaluation source references

These are public dataset references, selection metadata, file hashes, and known
edit positions for the new local comparisons. **No source media, customer uploads,
or model checkpoints are included.** They let independent reviewers identify the
same inputs and audit source separation, rather than relying on aggregate numbers
or a generic dataset link.

- `iml-vit-v4-plan.json` and `iml-vit-v4-manifest.json`: the 48 original/edit
  pairs in the [IML-ViT comparison](../iml-vit-localization-evaluation.md).
- `tiled-splice-v5-plan.json` and `tiled-splice-v5-manifest.json`: the next 48
  pairs in the [tile comparison](../tiled-splice-evaluation.md), retaining one
  explicitly unavailable secondary author mask.
- `video-v6-sources.json`: the 90 downloaded original/two-generator videos,
  their public release locations, file hashes, and matched-encoding recipe.
- `video-v6-intervals.json`: all 120 derived original, swap-insert, and ordinary
  controls, including source groups and known inserted frame/time intervals.
- `video-v6-exclusions.json`: prior source/donor identifiers excluded before
  selection. It preserves the exclusion identities, not local setup paths.

Image selection plans and manifests are exact copies of the frozen metadata.
The video source/exclusion catalogue is deliberately limited to reproducibility
fields; original ffprobe tags and local setup details are omitted. The derived
video interval manifest is exact. Native source checksums and referenced source
revisions identify the inputs. Derived encoded bytes can depend on ffmpeg and
codec versions; verify exact frames as the preparation scripts do and record any
differences. Do not silently replace an unavailable source or relabel a mismatch.

Source metadata originates from [TGIF/TGIF2 (IDLabMedia)](https://github.com/IDLabMedia/tgif-dataset/tree/fc5c557bd1a2d49a5dabc3676a54cfc44798f4ea)
and the [RTFS-10K publisher release](https://huggingface.co/datasets/godmodes/rtfs-10k/tree/bca865e675852666a426368b2f5614abf1c1926a).
These metadata extracts retain the publishers' [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) terms and
attribution; the project's MIT code license does not relicense the datasets.
Follow the linked cards for original source credits and media terms. Project
methods, curation, derived fields, unavailable-mask handling, and outcome
limitations are described in the linked experiment reports. Publisher labels
and declared rights are not independent authentication of every source.
