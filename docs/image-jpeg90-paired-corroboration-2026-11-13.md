# Paired JPEG90 regional evidence on separate sources

The [source-context audit](image-jpeg90-reflection-source-context-2026-11-13.md)
showed that 12 of 13 ordinary regional highlights lacked support from an earlier
copy. This trial compares the original JPEG90 regional mask, the measured
earlier-copy mask and their pixel-by-pixel intersection. Both inputs are
needed for the intersection. It carries zero AI-authorship weight. Ordinary
changes and generated changes use separate truth labels in the comparison.

## Frozen design

The panel has eight object-removal photos from the qualified BG-REAL LaMa
archive and six photos with known local FLUX.2 Klein edits. All photographs
have retained CC BY 2.0 attribution. BG-REAL publisher masks and edits have
CC BY 4.0 terms, and the LaMa and FLUX.2 Klein model records specify Apache-2.0.
The test reruns no generator. It uses the locally pinned regional detector
checkpoint and its current native selection rule.
[BG-REAL data and attribution](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)
· [LaMa publisher](https://github.com/advimman/lama)
· [Exact FLUX source construction](image-flux-photo-expansion-v285-result.md)
· [FLUX.2 Klein weight terms](https://huggingface.co/mlx-community/flux2-klein-4b-4bit)
· [Regional checkpoint](native-regional-serving-v47-result.md)

Selection excludes photo IDs, credited authors and original asset digests in
the three earlier JPEG90 rosters. LaMa photos are chosen by a fixed digest
order from the qualified sources. The six FLUX photos use the established
native 600 × 550 view that preserves the complete changed area. The pairs
are new to this comparison. The photos have prior project exposure outside
those three rosters, and model pretraining exposure is unknown. Results
therefore measure transfer beyond the three JPEG90 source banks.

Each source contributes one untouched upload, a sharp and a feathered
generated change, a matched local tone change, blur and copy, three whole-image
processing controls and two incompatible earlier-copy comparisons. The
whole-image processing controls are a lower-quality prior save, a resize
round trip and a global tone change. The incompatible cases use a different
photo and a fractional shift. Exact pixel truth comes from the source before
the final encode. All final files use baseline quality-90 RGB JPEG with 4:2:0
sampling. The paired arms use the app's integer alignment and require a
zero-offset result. Literal identical pairs are assessed with an empty
difference mask. Unsupported alignments or uncertain delivery remain
unassessed. The shared arm keeps only native pixels present in both masks.

Useful locations require at least 50% recall and 30% intersection over
union. A candidate qualifies for further review if it retains every useful
standalone location, has at least four useful locations spanning both
families, and reports no region on an unchanged, global processing or
incompatible-reference control. The frozen input ledger and code
commitments precede model inference. The ordinary local controls have
known changed pixels and test the correctness of difference evidence.

## Held results

| Method | Generated edits located / 28 | Ordinary local edits located / 42 | Untouched or whole-image controls highlighted / 56 | Incompatible pairs highlighted / 28 |
| --- | ---: | ---: | ---: | ---: |
| JPEG90 regional mask | 2 | 4 | 4 | 5 |
| Earlier-copy mask | 24 | 26 | 8 | 0 |
| Pixel intersection | 0 | 3 | 0 | 0 |

The earlier-copy route assessed 24 generated deliveries, 37 local controls,
49 untouched or whole-image controls, and zero incompatible pairs. It located
all 24 assessed generated changes, including 14 of 16 LaMa and 10 of 12 FLUX
deliveries. The earlier-copy mask also highlighted two prior saves, one resize
and five tone changes, with no highlight on the 14 untouched pairs. These are
measured differences between supplied copies, and ordinary editing can
produce them. The standalone regional check sees no earlier copy, so its
incompatible-case counts describe highlights on the delivered photographs.

The shared pixels withheld every whole-image nuisance highlight and did not
produce a useful generated location. Both useful standalone locations were
lost because their earlier-copy pair did not qualify. The two remaining
regions overlapped their FLUX edits by only 199 of 4,094 pixels and 424 of
3,818 pixels. Three ordinary local changes had a useful shared region. The
candidate fails the frozen gate. The earlier-copy result remains available
on qualifying pairs without requiring regional agreement.
[Held receipt](evaluation-panels/image-jpeg90-paired-corroboration-2026-11-13.json)

The two unresolved FLUX deliveries had more than 2,200 corresponding
keypoints, zero median displacement and six agreeing zero-shift tiles.
All six were in the lower two rows of the alignment grid, so the current
requirement for support across all three rows withheld the pair. A separate
count found 1,222 and 1,229 of 1,258 same-position 16 × 16 RGB tiles
identical across the two copies. The two unresolved LaMa deliveries had
only five qualifying tiles. This inspection describes the missed cases.
It changes no settings or recorded outcomes. An exact-identity fallback
would need a separate evaluation with new sources and incompatible copies.

## Replay

Run the native replay with the private, unchanged source caches used by the
[earlier JPEG90 study](image-regional-jpeg90-quality-agreement-2026-11-13.md#reproduce)
and the [FLUX source construction](image-flux-photo-expansion-v285-result.md).
Use a new run directory and the three pinned source plans. The paired input
and region ledgers carry digests and scalar positions only.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze score replay; do
  backend/.venv/bin/python -B -m scripts.image_jpeg90_paired_corroboration "$phase" "$run" \
    --main "$main_plan" --external "$external_plan" --confirmation "$confirmation_plan" || exit 1
done
python3 -B -IS backend/scripts/image_jpeg90_paired_corroboration.py review "$run/receipt.json"
```

The second native run reproduced all 154 measurements, including their
exact pixel-mask and model-input digests. The standard-library readout
agrees on Python 3.11 and 3.12. It rejects an out-of-bounds selected region
and a changed frozen edit area.
