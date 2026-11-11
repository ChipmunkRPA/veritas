# BG-REAL metadata roster for a fresh JPEG comparison

Use the v2 release of BG-REAL at Hugging Face revision
`aaba6cd1ced4ce3c846be9641fed092adb4919d1`. The paper's repository
is `NoirZangetsu/bg-real` at revision
`87f209e3196df622972bf7e66de50a1d2847a733`. The two revisions
advertised matching sizes and linked object fingerprints for the five data
metadata files. The script independently verifies the retained files by SHA-256.
[Release card](https://huggingface.co/datasets/anush-123/bg-real) ·
[Author paper](https://arxiv.org/html/2607.26232v3)

| Publisher metadata file | Exact bytes |
| --- | ---: |
| `README.md` | 8,264 |
| `v2/manifest_v2.csv` | 12,922,417 |
| `v2/splits_v2.yaml` | 693,311 |
| `v2/meta/attribution.csv` | 1,078,451 |
| `v2/meta/model_revisions.csv` | 967 |
| `v2/meta/recipients.csv` | 405,123 |

## Metadata scope

Select base PNG F5 object-removal entries generated with the publisher's
recorded Apache 2.0 `big-lama` v0.1.0, paired with the base PNG A0
of the same recipient. Require both manifest rows in their declared
publisher split, matching dimensions and exact image digests, and a
semantic edit-mask path at the pinned revision. Exclude SD1.5 insertion,
JPEG originals, post-processing variants, missing or ambiguous pairings,
and unattributed inputs. The publisher distributes the Open Images V7
photos under their per-photo CC BY 2.0 grants and the authors'
contributions under CC BY 4.0. Preserve source attribution and note
changes. The publisher's F5 processing control uses an SD1.5 autoencoder
round trip. The proposed follow-up uses ordinary controls generated
locally under documented recipes. [BG-REAL licensing](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution) ·
[LaMa license](https://github.com/advimman/lama/blob/main/LICENSE) ·
[CC BY 2.0](https://creativecommons.org/licenses/by/2.0/)

Require a consistent photographer identity and Flickr photo ID in the
release's per-photo attribution. Match aliases only where the metadata
agrees. Exclude every linked component with a known Flickr photo,
photographer or encoded-byte match to the exact v39 graph, which covers
all 397 v26 sources and all 59 v189 diagnostic sources. The v19 inventory
maps the earlier COCO IDs to Flickr photo IDs. Group remaining candidates
transitively by photographer, Flickr photo ID and original encoded digest.
Take one representative per component by fixed rank, with test splits
first, validation second and publisher training last. Publisher training
status is reported separately. This is a local cross-source transfer test.

Freeze up to 24 primary candidates and the next 12 backups from that same
ordered pool before opening image or mask files. Backups fill gaps only
when the prospectively specified byte, geometry, attribution or duplicate
checks disqualify a primary during preparation. Retain every attempted
case and reason. If the bounded pool yields fewer than 24, keep the
shortfall. No backups may be used after detector scoring begins.

## Pixel admission and app evaluation

Metadata establishes candidate eligibility only. Before scoring, verify
the selected A0 and F5 image SHA-256 digests and decode all assets at their
declared upright source dimensions. Record the actual mask SHA-256. Keep
the publisher's semantic mask separate from the exact nonempty pre-JPEG
RGB difference. Require exact unchanged pixels outside the declared mask.
Run the existing exact RGB and informative thumbnail comparisons across
the older exposure graph and the complete selected candidate pool.
Retain any geometry, decoder, digest or overlap failures. The publisher's
thresholded changed-pixel mask can support review but supplies no
substitute for the exact comparison.

After admission, prepare paired Q75/Q90 images from the same lossless
source, edit and deterministic ordinary tone, blur and copy-move controls.
Keep the truth mask in precompression coordinates. Compare the fixed
current, sensitive and shared readouts under v189 criteria, including
useful locations, misses, ordinary false highlights and highlighted area.
Keep qualities separate. Then use isolated test billing to verify the
source crop, overlay, mask and finding links in the ordinary browser,
API, PDF and JSON. JPEG candidate applicability and JPEG comparison
membership require explicit isolated test configuration. Keep the
whole-image authorship estimate separate from regional editing cues.
[v189 methods](image-jpeg-diagnostic-v189-protocol.md) ·
[Existing app verification](native-regional-paid-flow-v49-protocol.md)

## Commands

Run from the repository root using the already retained metadata files.
Preflight reads metadata and prints aggregate counts only. The roster
command writes a fresh ignored directory containing `private-roster.json`
and `summary.json`. Only the aggregate is suitable for sharing. No command
opens media, downloads files or invokes a model.

```sh
metadata="/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify/image-bg-real-metadata-v218"
trial="$PWD/backend/data/independent-evaluation/image-bg-real-roster-v218"
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.image_bg_real_roster_v218 preflight "$metadata"
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.image_bg_real_roster_v218 roster "$metadata" "$trial"
```
