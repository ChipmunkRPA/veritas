# Direct and repeated JPEG quality-90 saves

The optional regional image check was evaluated on baseline JPEG quality
75. This transfer tests quality 90 on the same exact changed-pixel truth
using two delivery histories. One re-encodes the decoded quality-75 file.
The other encodes the original pre-delivery RGB directly. Both use standard
4:2:0 quantization and retain source coordinates. Comparing the two routes
tests whether a nominal output quality can predict the check's behavior.
[Regional JPEG75 baseline](image-regional-jpeg-transfer-2026-11-13.md) ·
[Matched photographic and processing controls](https://arxiv.org/html/2611.10908v1) ·
[Research on localized inpainting after JPEG compression](https://pubmed.ncbi.nlm.nih.gov/42335053/)

The runner uses the same commercially permitted local checkpoint, component
selection and geometry. It calculates the reliability mask on the actual
delivered JPEG90 bytes. Both tests reproduce the source JPEG75, its known
changed pixels and the archived baseline regional masks. The two banks cover
22 distinct photos and 200 deliveries with earlier project exposure. Some
carry matched variants from the same source. Useful locations need at least
0.30 intersection over union and half of the changed pixels. The predeclared
gate allows at most five newly flagged ordinary files from at most two new
source groups and four additional misplaced edited highlights. It also
requires preserving eight of the 11 useful earlier locations and at least
one in each previously detected family. [Exact input and source qualification](image-regional-tile-rescue-2026-11-13.md)

| Delivery | Useful locations | Ordinary file flags | Newly flagged ordinary files | Misplaced edited highlights |
| --- | ---: | ---: | ---: | ---: |
| Existing JPEG75 | 11/46 | 13/154 | — | 4/46 |
| JPEG75 decoded, then saved at JPEG90 | 11/46 | 13/154 | 0 | 4/46 |
| Pre-delivery RGB saved directly at JPEG90 | 11/46 | 30/154 | 18 | 9/46 |

The tandem run kept all 11 earlier locations and the same 13 ordinary
flags. The direct run kept ten, gained one and lost one. It flagged
four additional ordinary source groups. Neither quality-90 workflow
localized a known LaMa removal. The direct workflow failed the frozen
gate. Since a quality-90 output profile alone cannot distinguish these
histories, the broader detector remains experimental.
[Tandem ledger](evaluation-panels/image-regional-jpeg90-tandem-2026-11-13.json) ·
[Direct ledger](evaluation-panels/image-regional-jpeg90-direct-2026-11-13.json)

An isolated simulated checkout verified that the experimental route could
render an exact crop, mask and signed report from one direct quality-90
file. That integration test does not change the negative panel decision.
[App proof](evaluation-panels/image-regional-jpeg90-app-2026-11-13.json)

## Replay

The portable reader recounts all 400 variant rows and both decisions from
published pixel totals. To reproduce native inference, use the retained
sources and completed two-bank mask journal from the prior tile study.
The tandem ledger retains its original scorer digest. The runner now also
accepts `--direct` to read original pre-delivery pixels through the same
source verification path. Each study gets a fresh run directory.

```sh
python3 -B -IS backend/scripts/review_image_regional_jpeg90_carrier.py \
  docs/evaluation-panels/image-regional-jpeg90-tandem-2026-11-13.json \
  docs/evaluation-panels/image-regional-jpeg90-direct-2026-11-13.json
export PYTHONPATH=backend OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
for phase in freeze score replay; do
  backend/.venv/bin/python -B -m scripts.image_regional_jpeg90_carrier "$phase" \
    --archive "$archive" --prior-probe "$probe" --residual "$residual" \
    --early-f5 "$early_f5" --later-f5 "$later_f5" --checkpoint "$checkpoint" \
    --output-fresh "$fresh_transfer" --external "$new_editor_transfer" \
    --tile-run "$tile_run" --output "$direct_run" --direct || exit 1
done
```
