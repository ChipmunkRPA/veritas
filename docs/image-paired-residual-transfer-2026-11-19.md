# Matched controls for localizing edits in infographics

This test compared two ways to train a lightweight image localizer. Both used
48 high-frequency and color-residual measurements per 16-pixel cell. One
classifier learned from individual edited and ordinary cells. The other
learned from differences between the same location before and after an edit,
including ordinary blur and color changes at that location. Both located an
edit from a single uploaded image at test time. The comparison was motivated
by paired training in SEED and the separate use of local reconstruction
measurements in recent visual text forensics. It used no code, weights, or
datasets from either paper. [SEED](https://arxiv.org/html/2606.21138v1) ·
[Open-Set Visual Text Forensics](https://arxiv.org/html/2608.02258v1)

The 20 held English infographic groups came from the MIT-licensed
`infographics_en` test split of MaskEdit-10k at
`3e0c54f7a6aeca5fb3e12758cb93d0be14310f55`. All source IDs were distinct from
the 12 previously tested groups used for fitting. Eight authored layout
families set the operating points, using the second-highest family maximum
per classifier and carrier. Twelve separately authored layout families used
disjoint fonts and served as controls. The ordinary layout families were
used only for point selection or held testing. Publisher before images
were themselves generated and served as controls for local changes.
[Publisher licensing and rights](https://huggingface.co/datasets/ReyChiaro/MaskEdit-10k#licensing-and-rights)

The frozen intake summary kept a draft count of 12 held groups. Its
executable intake used 20 groups, and the held intake and evaluation plan
both record those 20. The fit summary also described ordinary layout
controls as fit inputs. The implementation reserved them for calibration;
the reported counts follow the executable procedure.

Each publisher group supplied the before image, two ordinary same-site
changes, the publisher's edited image, and a version restricted to the
edit mask. Held controls included the original, mild blur and tint of
each authored layout. Both carriers came from the delivered raster. The
PNG kept that raster's decoded pixels and JPEG quality 90 recompressed it.
A useful region covered at least half the labeled edit and had at least
0.30 intersection-over-union. The internal research gate required at least
four useful direct edits, at least two more than the unpaired fit, at most
one additional authored-control alert, and at most two misplaced direct
highlights per carrier.

| Delivery | Fit | Useful direct edits | Misplaced direct highlights | Authored control alerts | Generated before and ordinary variant alerts |
| --- | --- | ---: | ---: | ---: | ---: |
| PNG | Unpaired | 0/20 | 19/20 | 2/36 | 47/60 |
| PNG | Paired | 0/20 | 15/20 | 1/36 | 41/60 |
| JPEG quality 90 | Unpaired | 0/20 | 13/20 | 3/36 | 28/60 |
| JPEG quality 90 | Paired | 0/20 | 20/20 | 12/36 | 60/60 |

Neither fit located a mask-confined edit in either carrier. The paired
fit more often ranked the edit above matched ordinary variants **at the
known site**, but still chose the wrong region when required to find it.
Lowering the point cannot recover these deliveries because none of the
top-ranked regions passed the spatial rule even before thresholding.
The evaluation leaves both candidates outside the app. It covers one
publisher and one content family. It does not establish camera-image
performance, population error rates, or overlap with generator training
data.

The [evaluation receipt](evaluation-panels/image-paired-residual-transfer-2026-11-19.json)
records the results of recounting all 544 classifier readouts against
the pixel masks, exact region coordinates and frozen points. The companion
[per-case records](evaluation-panels/image-paired-residual-transfer-2026-11-19.jsonl.gz)
contain anonymized held family labels, geometry counts and operating
points for independently recounting the table. The records contain
no source images, crops or prompts.
