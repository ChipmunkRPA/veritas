# Confirm exact differences between matching areas in one PNG

The first pilot found useful exact differences with zero flags on its ordinary controls and missed its frozen coverage gate. This confirmation uses the same native comparison code and limits on separate photo sources and changed pane layouts. The two native areas are an undirected comparison. A difference proves that the aligned displayed pixels vary and contributes zero AI authorship confidence. [Original pilot and rejection](image-embedded-pair-difference-transfer-2026-11-12.md) · [Confirmation scalar ledger](evaluation-panels/image-embedded-pair-difference-confirmation-2026-11-12.json)

## Sealed confirmation

Select eight commercially qualified primary LaMa groups by recorded publisher split and four separate archived generator groups, two each for SDXL and Kandinsky. Exclude all credited source groups in the pilot intake archives. The photos have prior project exposure. The submitted horizontal and vertical 768 × 768 layouts are new. They preserve native pixels in both panes and retain at least 85% of each source edit footprint. The exact submitted RGB difference supplies the localization truth for each view. The 18 unrelated and standalone ordinary sources use distinct credited Commons photographs outside the pilot's ordinary roster. [LaMa archive qualification](image-bg-real-transport-v225-result.md) · [Other generated archive](image-background-patch-transfer-2026-11-11.md) · [Ordinary source credits](image-affine-native-scale-2026-11-11.md)

The seven no-report pane controls include exact duplicated photo content under different surrounds, global tone, unrelated photos, shifted crops, a resampling round trip, repeated photo cards with changed header text, and an ambiguous three-pane layout. The third pane carries an authored ordinary local difference. Six standalone photographs add independent ordinary controls. The separate positive ordinary adjustment applies localized blur in one matched pane and uses its exact changed pixels as truth.

The exact proof compares all RGB pixels in every witness and in both aligned crops. It applies the pilot's existing witness and interior-difference rules unchanged. The earlier boundary check is scored independently on each view. The frozen confirmation requires at least six new exact useful locations for each edit type and zero ordinary groups with a new reported difference. The intended scope is a single RGB PNG containing two matched native displays with a supported localized difference.

## Held outcomes

| Known change | Existing useful locations | Exact differences in both panes | Useful with both checks |
| --- | ---: | ---: | ---: |
| Sharp edit | 3/12 | 5/12 | 7/12 |
| Feathered edit | 0/12 | 5/12 | 5/12 |
| Ordinary local blur | 0/12 | 12/12 | 12/12 |

The same source group supplied the overlap between the incumbent sharp location and the exact proof. The candidate added four sharp and five feathered locations, while an additive comparison preserves every incumbent location. Each generated-edit type yielded one exact horizontal and four exact vertical pairs. This panel leaves orientation and source-family effects confounded because every source had a fixed assigned layout.

The new proof supplied zero difference flags across all 78 ordinary two-pane and standalone controls. It recognized all 12 duplicates and all 12 shifted pairs with zero local differences, and one repeated photo-card view also produced a comparison with zero local differences. The existing sharp boundary check returned 20 ordinary review cues in those 78 views. It measures boundaries and has a different claim scope.

All 12 three-pane controls produced a difference record, so the confirmation fails its frozen gate. In each case, the algorithm chose the first and third views, 492 pixels apart, and mapped the authored 550-pixel brightness change. The measurements describe genuine displayed differences. The admission error is treating a layout with three matching views as a unique pair. A post-run readback found that all 5,454 verified tiles supporting those comparisons also appear exactly in the middle pane. That observed third-copy support gives a concrete basis for a separately tested abstention rule. The existing app checks stay in use.

## Verify

The independently frozen confirmation digest is `58e314577bbf37bba612e3037995b00a9f387ed8141bd0ec8ae3d9a5927223f5`. The 18-second local run scored 126 views. Native readback checked all 126 original boundary scores, 59 exact comparisons, 49,089 individual texture witnesses and 22 exact two-sided masks. The portable ledger keeps witness counts and scalar results. Native tile coordinates, photo pixels, masks and individual input hashes stay in the evaluation workspace.

```sh
python3 -I -S backend/scripts/image_embedded_pair_difference_confirmation.py review \
  docs/evaluation-panels/image-embedded-pair-difference-confirmation-2026-11-12.json
```

Use the retained Commons, primary v225, and earlier generated archives. The backup v262 and earlier seven-group archives supply the source-role exclusion check. The first pilot run supplies the original code and source commitment.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for stage in freeze run audit; do
  backend/.venv/bin/python -B -m scripts.image_embedded_pair_difference_confirmation "$stage" \
    "$commons" "$primary" "$earlier" "$backup" "$pilot_generated" "$pilot_run" "$run" || exit 1
done
```
