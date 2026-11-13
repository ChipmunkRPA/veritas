# Paired native face edges for short video edits

Three short swaps in the prior 720p30 comparison had a known edit at 2.100–2.467 s, while the Full face check selected 1.833–2.367 s. The dense retry recovered none of those three. This study starts from the range Full already selected and asks whether the neighboring native pictures support more precise endpoints. [Earlier retry comparison](video-face-hd-sparse-retry-2026-11-12.md)

## Frozen method

The experiment searches up to twelve native pictures on either side of each existing boundary. Each edge compares three consecutive, unambiguous face observations on each side. A supported pair lasts five to sixteen frames and must show a change at both ends and a return toward the earlier face appearance. The guard compares excess motion in a central face region with motion in four nearby sectors, using adjacent pictures as a reference. At least two sectors must be visible and the center must change at each edge. A broad change rejects the proposed interval. The temporal approach follows the research direction of using explicit, inspectable cues across a face trajectory. The detector uses the project's existing MIT-licensed face detector and Apache-2.0 appearance weights. [Temporal forensic descriptors preprint](https://arxiv.org/abs/2610.03380) · [Weight licenses and terms](video-appearance-plan-v107-method.md)

The 78 selection deliveries and 125 held deliveries are previously retained 720p30 H.264 MP4 clips with exactly 120 native pictures. They include publisher face swaps, tandem re-encodes, genuine originals, fixed-layout cuts between genuine speakers, and clips with two separate edits. The test fits a single response point from ordinary selection clips and freezes it at 0.03176017255556951 before held scoring. The value measures a visual comparison and carries no AI-confidence weight. A useful localization needs at least 0.50 temporal intersection over union with the known edit. The original gate requires at least three additional locations in at least two groups, at least 75% assessed among routed edit deliveries, and at most one additional ordinary file, one extra ordinary cut range and one incorrect range on edited files. [Source terms](video-common-source-preparation-v89.md) · [Fixed-layout cut controls](video-face-scale-context-holdout-2026-11-12.md) · [Two-edit panel](video-face-two-returns-hd-2026-11-12.md)

## H.264 results

| Held edit condition | Known ranges | Current Full located | Full plus paired edges | Files routed and assessed |
| --- | ---: | ---: | ---: | ---: |
| Single publisher swap | 7 | 6 | 6 | 7 routed, 6 assessed |
| Tandem single publisher swap | 7 | 5 | 7 | 7 routed, 6 assessed |
| Swap followed by ordinary jump | 8 | 8 | 8 | 1 routed, 1 assessed |
| Ordinary light step followed by swap | 8 | 6 | 6 | 1 routed, 1 assessed |
| Two publisher swaps | 16 | 8 | 8 | 5 routed, 1 assessed |
| Tandem two publisher swaps | 16 | 9 | 9 | 3 routed, 0 assessed |
| Total | 62 | 42 | 44 | 24 routed, 15 assessed |

Both new locations exactly match native pictures **[63, 74)**, or **2.100–2.467 s**, in distinct retained RTFS groups. The before, during and after anchors are pictures **62, 68 and 74**. The first-encode version of one of those misses remained unresolved. It had an above-point unguarded pair, which the surrounding motion check rejected. There were zero additional ranges across 79 ordinary deliveries and zero new wrong ranges on edited deliveries. The incumbent flagged 15 ordinary same-layout speaker cutaways, and an unguarded pair scored above the point on all 15. The guard withheld all 15 proposed refinements. The study leaves those existing alerts in the comparator's results.

The original advancement gate failed. It recovered two additional locations and assessed 15/24 routed edit files (62.5%) and 15/46 edit files overall (32.6%). The result supports a narrow tandem-encode finding. It leaves longer clips, natural conference recordings, other delivery profiles and lip-sync performance unresolved. An independent reader reproduced the counts, fixed point and held gate. Native replay checked all 125 held source file hashes and 42 matching source pictures and crops for above-point results. [Public H.264 case ledger](evaluation-panels/video-face-paired-edge-transfer-2026-11-12.json)

## Reproduce locally

Use a new private scratch location and the retained source media, initial study receipts and qualified weights. The receipt keeps source digests, native frame indices and bounding boxes. Original media and crops stay local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze selection held report; do
  backend/.venv/bin/python -B -m scripts.video_face_paired_edge_transfer "$phase" "$run" || exit 1
done
python3 -B -I -S backend/scripts/review_video_face_paired_edge_transfer.py "$run/receipt.json"
backend/.venv/bin/python -B -m scripts.review_video_face_paired_edge_transfer "$run/receipt.json" --native
```
