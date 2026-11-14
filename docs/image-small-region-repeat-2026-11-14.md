# Small JPEG edits below the image area rule

The existing regional check withholds all candidate regions when fewer
than 1% of source pixels exceed its response cutoff. Removing that rule
recovered one useful region out of 60 known edits and added eight
highlights on 126 ordinary files. Requiring the original region to
recur after a horizontal flip recovered no additional useful regions.
The serving rule stays in place. New browser and PDF evidence shows the
recorded count, its denominator and the minimum for displaying candidates.

## Local comparison

The archived panel used 18 TGIF/TGIF2 photos across Firefly, Stable
Diffusion 2 and FLUX Fill, plus 12 BG-REAL photos with LaMa removals.
Seven LaMa photos came from the primary intake and five came from the
backup intake. All 12 are distinct from the previous progressive JPEG
comparison by original file hash. The source photos had earlier project
exposure. Input images are upright quality-75 RGB JPEGs with 4:2:0
chroma sampling. Edit truth is the exact pre-encode changed-pixel mask.
Ordinary controls include untouched photos, matched local blur and tone,
and, on the LaMa set, local copy, resizing and an earlier JPEG save.
[TGIF source qualification](image-jpeg-signed-transitions-2026-11-12.md)
· [BG-REAL photo and edit terms](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)

Compare the serving area and region rules with two alternatives. First,
retain the ordinary two-component readout at the existing response point
regardless of the area rule. Second, keep the existing output when the
area rule passes. When it withholds a region, run the same checkpoint
on an exact horizontal flip. Align the repeat to the upload before
applying the original reliability grid. Require at least two-thirds
overlap of both components and retain only the strongest supported
original component. A useful location covers at least half of the known
changed pixels and reaches 0.30 intersection over union. The repeat arm
had to add useful regions on at least three photos across two editor
families with at most two additional flagged ordinary photo groups.
The work uses the existing commercially admitted checkpoint. Research
on inpainting motivates evaluating local cues with the original
background preserved. [Checkpoint and scope](native-regional-serving-v47-result.md)
· [Inpainting localization study](https://arxiv.org/abs/2608.28302)

| Regional check | Useful known edits | Misplaced highlights on edited files | Ordinary highlights |
| --- | ---: | ---: | ---: |
| Existing area rule | 0/60 | 9/60 | 20/126 |
| Area rule removed | 1/60 | 10/60 | 28/126 |
| Small region requires a matching repeat | 0/60 | 9/60 | 20/126 |

The area rule passed on 29 of 186 uploads. Ten other uploads produced
a component large enough for a repeat. Eight had no qualifying
component in the flipped response, and the other two had no overlapping
pixels. The one useful region found by ignoring the area rule was
a feathered FLUX Fill edit. Its 2,548 selected pixels covered 2,025
of 2,994 changed pixels. Both the untouched and matched blur controls
from the same photo also produced ungated regions.

These counts measure the optional regional check on one JPEG profile.
They do not measure the full app or establish authorship. The frozen
comparison ran 186 native forwards and 10 repeats.

An isolated progressive-JPEG checkout used simulated payment and the
installed regional checkpoint. The signed PDF reproduced the exact
counts from the saved JSON, and the local browser verified the source
crop and mask. No model load or inference ran before payment.

## Replay

The [published ledger](evaluation-panels/image-small-region-repeat-2026-11-14.json)
contains scalar outcomes, changed-pixel mask and input commitments, and source-group
counts. The standard-library review needs no weights or source photos.

```sh
PYTHONPATH=backend python3.12 -B -S -m scripts.image_small_region_repeat review \
  docs/evaluation-panels/image-small-region-repeat-2026-11-14.json
```

To rerun native measurement, use the retained TGIF/TGIF2 archive,
existing regional checkpoint, and primary and backup BG-REAL source
rosters. Run `freeze` and then `measure` with the same new run
directory and both roster paths supplied in order to `--bg`.
The runner verifies each supplied source by its exact hash.
