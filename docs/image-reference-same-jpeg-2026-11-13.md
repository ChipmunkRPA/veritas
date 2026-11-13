# Exact block alignment for matching JPEG90 copies

The earlier JPEG90 study compared a regional model with an earlier-copy mask.
Its earlier-copy arm stopped at integer keypoint alignment. This follow-up
first runs the complete local earlier-copy route, including the subsequent
structure check. It then compares exact decoded blocks from the two JPEGs
when the complete route cannot measure a pair. A separate experimental arm
also tries pairs for which the current route selected no region.
[Earlier JPEG90 study](image-jpeg90-paired-corroboration-2026-11-13.md) ·
[Existing JPEG structure check](image-reference-structure-jpeg-2026-11-13.md)

## Method and frozen panels

The candidate requires upright baseline 4:2:0 JPEGs with standard quality-90
tables, matching dimensions of 512–768 pixels per side, and no embedded
color profile. Complete 16 × 16 blocks at the native origin must match in
all decoded RGB values over at least 85% of the grid. The existing exact
block check also requires 24 detailed blocks that appear once in each file.
At least six of the nine fixed zones must contain two such anchors, with
every row and column represented. The check uses the existing color
difference method on both decoded files directly, avoiding another JPEG
encode. It retains each earlier and submitted crop in its original
coordinates. The comparison measures changes in the two delivered files
and carries zero AI-confidence weight.
[Existing exact block method](image-reference-exact-jpeg-2026-11-12.md) ·
[JPEG standard](https://www.itu.int/rec/T-REC-T.81/en)

The diagnostic bank replays all 154 pairs on 14 photos from the earlier
comparison. The held bank selects 11 LaMa removals, six SDXL edits and six
Kandinsky edits from the retained source qualification records. The two
banks and prior JPEG90 panels share no recorded source photograph, credited
photographer, or exact original file. The photos have prior project exposure
under other methods. Every held group supplies sharp and feathered known
edits, matched local tone, blur and copy, an unchanged file, JPEG
recompression, resizing, global color adjustment and blur, and four
incompatible variants. The incompatible pairs use unrelated photos,
fractional and integer shifts, and horizontal reflection. Final delivery
uses the same JPEG90 settings for all roles. The source photos retain
CC BY 2.0 credits. The publisher's edit contributions carry CC BY 4.0.
No generator or learned detector ran.
[LaMa source qualification](image-bg-real-backup-intake-v262-result.md) ·
[Diffusion source qualification](image-wavelet-region-transfer-2026-11-11.md) ·
[Publisher rights and attribution](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)

A useful mask covers at least half the known pre-encode changed pixels
and reaches 0.30 intersection over union. Before scoring, advancement
required useful additions on two held photographs and zero newly
highlighted whole-image controls, misplaced edit regions or accepted
incompatible references. Both arms preserve every nonempty incumbent
finding. The two versions of an edit share one source group.

## Results

On the diagnostic bank, the complete current route measures all 28
generated edits and locates 26. The narrower comparator from the earlier
study located 24. The structure check already resolves the framing on the
four pairs that the narrower method declined. The exact-anchor fallback on
unassessed pairs adds no result in this bank. Allowing the check to follow
an empty result also locates both missing FLUX edits, reaching 28 of 28.
It covers 4,035 of 4,096 changed pixels in the sharp edit and 3,198 of 3,893
in the feathered edit. It adds four conventional difference regions on
matched local blur and tone controls and none on whole-image or incompatible
controls. These observations were used to formulate the empty-result arm.

| Held check | Known edits located | Useful conventional locations | Whole-image or untouched highlights | Incompatible pairs assessed |
| --- | ---: | ---: | ---: | ---: |
| Complete current app | 44/46 | 42/69 | 21/115 | 1/92 |
| Exact fallback after unassessed | 46/46 | 42/69 | 21/115 | 1/92 |
| Exact fallback also after empty result | 46/46 | 48/69 | 21/115 | 1/92 |

Both held additions come from one LaMa photo and retain approximately 96%
of the exact changed pixels in each variant. The broader arm adds ten
conventional local highlights, each overlapping the actual local change.
Six meet the useful-location criterion. The ordinary counts in the table
describe physical differences between copies. The existing app assessed
one fractional shift with an empty mask, and neither arm admits an additional
incompatible pair or highlights any of the incompatible controls. Neither
arm meets the two-held-source requirement. The candidate stays experimental
in the evaluation scripts and needs a new multiple-photo confirmation
before app integration.
[Diagnostic results](evaluation-panels/image-reference-same-jpeg-diagnostic-2026-11-13.json) ·
[Held results](evaluation-panels/image-reference-same-jpeg-held-2026-11-13.json)

## Replay

Use the retained source directories from the linked earlier studies. The
freeze checks the exact source rosters, input bytes, truth masks and
comparison code. The private sources and rendered media stay outside the
repository. The public ledgers contain native bounds, hashes and scalar
overlap counts. A second native run reproduced all 476 result rows, masks
and exact block commitments. The standard-library reviewer matched across
Python 3.11 and 3.12 and rejected an impossible exact-block count.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for stage in diagnostic held; do
  for phase in freeze score replay; do
    backend/.venv/bin/python -B -m scripts.image_reference_same_jpeg_transfer "$phase" "$run/$stage" \
      --stage "$stage" --main "$main_plan" --external "$external_plan" \
      --confirmation "$confirmation_plan" --wavelet "$wavelet_holdout" || exit 1
  done
done
for stage in diagnostic held; do
  python3 -B -S -m scripts.image_reference_same_jpeg_transfer review \
    "docs/evaluation-panels/image-reference-same-jpeg-${stage}-2026-11-13.json" || exit 1
done
```
