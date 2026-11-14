# JPEG fine-detail differences against an earlier copy

Two matched JPEGs can differ in fine detail even if a smoothed color
comparison shows little change. This study reuses the existing exact
same-framing check on upright JPEG90 pairs. It then reads the native
quantized luminance coefficients and compares dequantized alternating
components block by block. The comparison excludes each block's average
brightness. It selects connected groups of at least nine complete blocks
with a root-mean-square difference of 2.5 luminance levels and combines
them with the existing comparison mask. The measure identifies differences
between copies. Its AI confidence weight is zero.
[Existing JPEG comparison](image-reference-same-jpeg-confirmation-2026-11-13.md) ·
[JPEG standard](https://www.itu.int/rec/T-REC-T.81/en)

The diagnostic panel uses 154 pairs from 14 photographs. The confirmation
panel uses 476 pairs from 34 other photographs. Both panels were previously
used to evaluate earlier-copy methods. Every delivered JPEG hash and known
pre-encode edit mask matches the existing published input commitments.
A useful location covers at least half the changed pixels and reaches
0.30 intersection over union with the known mask. This study uses no
learned model or weight.

| Condition | Pairs | Current useful locations | With fine-detail check |
| --- | ---: | ---: | ---: |
| Diagnostic generated edits | 28 | 26 | 26 |
| Confirmation generated edits | 68 | 65 | 66 |
| Confirmation conventional local blur | 34 | 1 | 26 |
| Confirmation conventional local copy | 34 | 32 | 33 |
| Confirmation conventional local tone | 34 | 28 | 28 |

Both arms highlighted 32 of 170 whole-image or unchanged confirmation
controls and none of 136 incompatible pairs. Combining the masks preserved
every previously useful location. The only useful generated addition was a
sharp edit from one source. The study required additions on two confirmation
sources for AI-edit localization. The method remains an evaluation result.
Its main measured gain is locating conventional local blur. Testing ordinary
sharpening and stronger recompression on new sources would establish whether
it transfers as a separate earlier-copy evidence check.

Native scoring and replay reproduced the mask commitments and counts for all 630 pairs.
A later import-formatting change reproduced every earlier row. The
first diagnostic scoring attempt stopped because the exhibit directory
already existed, before any case measurements. The corrected freeze places
app-generated exhibits in a dedicated temporary subdirectory.

## Replay

Use the prior pair study's diagnostic source arguments or the seven
confirmation rosters for native `freeze`, `score` and `replay`. Add
`--stage diagnostic` for the diagnostic set. The offline review recomputes
the counts from published scalar rows and checks their source bindings and
pixel-count invariants. It requires no media, model weights or third-party
Python dependencies.
[Diagnostic rows](evaluation-panels/image-jpeg-ac-difference-diagnostic-2026-11-13.json) ·
[Confirmation rows](evaluation-panels/image-jpeg-ac-difference-confirmation-2026-11-13.json)

```sh
export PYTHONPATH=backend
for stage in diagnostic confirmation; do
  python3 -B -S -m scripts.image_jpeg_ac_difference_transfer review \
    "docs/evaluation-panels/image-jpeg-ac-difference-${stage}-2026-11-13.json"
done
```
