# Same-upload JPEG quality agreement for local image edits

Saving the same photograph directly at JPEG quality 90 led the optional
regional check to highlight 30 of 154 ordinary deliveries, compared with 13
at quality 75. This comparison asks whether the highlight has enough support
after the app decodes the uploaded JPEG90 and makes a second view at JPEG75.
[Direct and tandem JPEG90 comparison](image-regional-jpeg90-carrier-2026-11-13.md)

The second view preserves orientation and native coordinates. Both model
passes use the uploaded JPEG's reliability mask and unchanged response and
area rules. Keep each of the first two native JPEG90 components if at least
half its pixels also occur in a selected component from the second view.
The native component's shape stays intact. The comparison also measures
the second view by itself to see how much the agreement rule contributes.
Both views come from the same file and the same checkpoint. Agreement
measures sensitivity to the re-encoding and carries zero AI-authorship
weight. Research on JPEG inpainting describes how recompression can
attenuate forensic traces, so losses on edited files count against this
method. No outside implementation or weights were added.
[JPEG inpainting study](https://pubmed.ncbi.nlm.nih.gov/42335053/)
· [Current regional model and terms](native-regional-serving-v47-result.md)

## Sources and decision rules

The first comparison replays all 200 direct quality-90 deliveries and the
exact changed-pixel masks from the previous trial. Two banks cover 22
source photos with the LaMa, Stable Diffusion 1.5, SDXL and Kandinsky
edit families. The trial compares the new rule against both exact
previous JPEG75 and JPEG90 responses. Advancement requires preserving
at least eight of 11 useful earlier locations and both earlier detected
families, at least one useful location in the second bank, at most five
added ordinary flags in at most two new groups and at most four extra
misplaced edited-file highlights versus the earlier quality-75 outputs.

The separately frozen confirmation uses the complete in-scope first F5
archive. The 11 photos include six LaMa removals and five Stable Diffusion
1.5 insertions, with eight deliveries per photo. One archived source falls
below the 512-pixel minimum. The runner compares photo ID, credited
photographer and original bytes against both previous rosters and between
confirmation groups. Those photos had earlier project exposure under other
methods. Ordinary controls include unchanged photos, matched local blur,
tone and copy, a resize round trip and an earlier JPEG encode. The
confirmation must locate two edits from different photos, preserve at least
two-thirds of the useful native JPEG75 files when there are three or more,
and add at most two ordinary flags and one source group. Both studies
require at least 0.30 intersection over union and 0.50 changed-pixel recall
for a useful location. Original photos carry CC BY 2.0 and the publisher's
dataset contributions carry CC BY 4.0.
[Primary roster and rights](image-regional-tile-rescue-2026-11-13.md)
· [F5 provenance and controls](image-jpeg-signed-transitions-2026-11-12.md)
· [Publisher terms](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)

## Results

| Panel | Check | Useful edited files | Ordinary files highlighted | Ordinary groups highlighted | Misplaced edited highlights |
| --- | --- | ---: | ---: | ---: | ---: |
| Main and external | Native JPEG75 | 11/46 | 13/154 | 5/22 | 4/46 |
| Main and external | Direct JPEG90 | 11/46 | 30/154 | 8/22 | 9/46 |
| Main and external | JPEG90 re-encoded to JPEG75 | 10/46 | 12/154 | 4/22 | 5/46 |
| Main and external | Native JPEG90 with repeat agreement | 10/46 | 2/154 | 2/22 | 2/46 |
| Source-disjoint confirmation | Native JPEG75 | 4/22 | 0/66 | 0/11 | 2/22 |
| Source-disjoint confirmation | Direct JPEG90 | 4/22 | 15/66 | 3/11 | 6/22 |
| Source-disjoint confirmation | JPEG90 re-encoded to JPEG75 | 2/22 | 0/66 | 0/11 | 0/22 |
| Source-disjoint confirmation | Native JPEG90 with repeat agreement | 2/22 | 0/66 | 0/11 | 0/22 |

On the first 200 deliveries, agreement removed 28 ordinary highlights
and one useful location compared with direct JPEG90. It preserved nine
SD1.5 locations in the main bank and one SDXL location in the external
bank. That passed the frozen primary rule. The external bank alone fell
from two useful direct locations to one while ordinary flags fell from
five to one. The repeat view alone found no useful external location.
[Main and external outcome ledger](evaluation-panels/image-regional-jpeg90-quality-agreement-2026-11-13.json)

On the separate confirmation, both repeat arms removed all 15 ordinary
JPEG90 flags. They also lost both variants of one of the two localized
source photos, with only the sharp and feathered variants of the other
photo remaining. No LaMa removal produced a useful location in either
study. The confirmation failed its frozen retention and distinct-source
requirements. The quality-90 agreement route stays experimental.
[Source-disjoint outcome ledger](evaluation-panels/image-regional-jpeg90-quality-confirmation-2026-11-13.json)

All three useful direct locations lost across these two comparisons had
zero selected pixels in their JPEG75 repeat. Lowering the required overlap
while keeping it above zero would recover none of them. In the confirmation,
all 15 ordinary direct highlights also had zero repeat overlap. Separating
those misses from ordinary highlights requires another measurement.

## Reproduce

Use the same local sources, model cache and complete tile journal as the
[direct JPEG90 comparison](image-regional-jpeg90-carrier-2026-11-13.md#replay).
Use an empty private directory for each run. The portable reviewers
recompute counts and decisions from overlap and area totals.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze score replay; do
  backend/.venv/bin/python -B -m scripts.image_regional_jpeg90_quality_agreement "$phase" \
    --archive "$archive" --prior-probe "$probe" --residual "$residual" \
    --early-f5 "$early_f5" --later-f5 "$later_f5" --checkpoint "$checkpoint" \
    --output-fresh "$fresh_transfer" --external "$new_editor_transfer" \
    --tile-run "$tile_run" --output "$agreement_run" || exit 1
done
for phase in freeze score replay; do
  backend/.venv/bin/python -B -m scripts.image_regional_jpeg90_quality_confirmation "$phase" \
    --main-roster "$later_f5/plan.json" --external-roster "$new_editor_transfer/plan.json" \
    --confirmation-roster "$early_f5" --earlier-freeze "$agreement_run/freeze.json" \
    --output "$confirmation_run" || exit 1
done
python3 -B -S backend/scripts/review_image_regional_jpeg90_quality_agreement.py \
  "$agreement_run/receipt.json" docs/evaluation-panels/image-regional-jpeg90-direct-2026-11-13.json
python3 -B -S backend/scripts/review_image_regional_jpeg90_quality_confirmation.py \
  "$confirmation_run/receipt.json"
```
