# Context around brief face-change ranges

The lower-picture qualifier identified none of eight held genuine speaker cutaways. It assessed two of eight held face-swap deliveries with an existing reported range. The candidate stays experimental and has zero AI-confidence weight. [Frozen counts](evaluation-panels/video-face-subject-context-2026-11-12.json)

## Frozen method

The comparison reuses all 61 exact 720p30 H.264 deliveries from the dense-retry study, including matched originals, publisher swaps, genuine cutaways, tandem compression, motion jumps and reframing. Four GRID pairs and three RTFS groups form the development set. Four other GRID pairs and four other RTFS groups form confirmation. Both sets were exposed in earlier app evaluation. Source groups, candidate thresholds, admissibility and scoring were fixed before this measurement. [Source construction, rights and earlier results](video-face-hd-retry-scene-challenge-2026-11-12.md)

Starting from the strongest available report interval, the check takes the median of single-face flank boxes that have sufficient spatial overlap. A fixed rectangle supplies a central ellipse, a wider head annulus and a band below the face. The method searches eight native frames on either side of each reported boundary, with entry and exit on separate sides of the range midpoint. It selects the strongest central luminance transition after subtracting adjacent-pair motion measured beyond the search window. A broad annotation needs central, head and lower-band change on both boundaries. Each band needs at least 80% visible coverage. The lower-band change must reach 25% of the central change as well as the fixed absolute floor. When geometry or transitions are unavailable, the result abstains.

The comparison requires a broad annotation on at least three held genuine cutaway groups, assessment of at least three quarters of held swap deliveries with a reported range, and at most one broad annotation among those swaps. A separate timing diagnostic repeats the same measurement around authored boundaries. It checks whether better timing alone recovers an otherwise missing measurement. It receives authored timing and therefore stays outside the deployment comparison.

## Confirmation

| Delivery | Files | Existing face range | Qualifier assessed | Broad qualifier |
| --- | ---: | ---: | ---: | ---: |
| Genuine cutaway | 4 | 4 | 1 | 0 |
| Tandem genuine cutaway | 4 | 4 | 1 | 0 |
| Publisher swap | 4 | 4 | 1 | 0 |
| Tandem publisher swap | 4 | 4 | 1 | 0 |
| Unchanged and other ordinary controls | 16 | 0 | 0 | 0 |

The lower band did not have sufficient visible coverage on six cutaways and four swaps. Two other swap deliveries lacked a qualifying central transition at both boundaries. Supplying the authored boundaries still produced zero broad qualifiers on genuine cutaways and one on a digital punch-in. The punch-in had no existing face range and would not have entered the proposed report path. The candidate missed both coverage and useful qualification requirements.

In one held speaker pair with a visible lower band, the raw and tandem cutaways had head-annulus excess of about 34–35 luminance levels on both exact edges. The lower band supplied about 2–3 levels. The native frame selection located frames 85 and 96, at 2.833 and 3.200 seconds, on both encodes. This is a useful development observation about where the ordinary cut registered. Its limited geometry and the selected example cannot establish a release threshold.

A further test could use a closer head ring with separate top, side and below-chin coverage, with conventional crop and broad illumination controls. The present panel already informed that choice. An independent source bank with genuine same-layout cuts and matched swaps would be needed to judge it.

## Replay

Use the private run from the earlier 61-delivery study and a new private output directory. The exact source clocks, existing Full result hash, face-detector asset and native RGB digest are checked locally. Both observed and authored-time arms together verified 4,064 native frame reads, including frames revisited by the other arm. Source clips, face geometry and individual measurements stay local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze score report; do
  backend/.venv/bin/python -B -m scripts.video_face_subject_context "$phase" "$prior_run" "$run" || exit 1
done
```
