# Region-aligned calibration: new consumed-source hypothesis after v8 rejection

The v8 regional patch-drift candidate remains rejected: both readouts located0/48
known edits in each format under their frozen pixel-maximum calibration. This
study changes the calibration statistic, not that historical decision. No unused
confirmation reserve, new source photo, customer upload or neural forward is used.
All312 existing input responses are consumed development material.

## Rationale and exact changed statistic

The display filters connected components smaller than0.001 of the entire
screening grid. A single pixel maximum need not produce a displayed region, yet
v8 calibrated on that pixel maximum. Align the new statistic to the actual area
requirement: for each reliable raw float32 response grid, find the highest actual
cell level C for which the reliable `response >= C` mask contains an8-connected
component of at least `ceil(0.001 * total_grid_cells)` cells. This is a monotone
finite-graph problem. Search sorted unique reliable values using OpenCV component
area. Never shrink the area denominator to the number of reliable cells.

An independent graph traversal must establish a sufficiently large component at
`>= C` and no sufficiently large component at `> C`. If no reliable connected
component can reach the minimum size, record unavailable. These two boundary
conditions certify the exact largest feasible plateau, including ties. The
display retains strict `response > point`: it has an eligible component if and
only if C>point. A large region may cross unreliable exclusions only through
reliable8-connected cells. No bounding-box filling or hole filling is allowed.

Use **only the raw median-drift readout**, the original first-priority arm. Do not
search a new layer, perturbation, resize, model, normalized arm, area, or ensemble.
Native448/block10/Haar0.2/model weights remain exactly those already verified
under the MIT/Apache-2 source and native contract in
[v8](image-patch-drift-v8-native-protocol.md). Its6 synthetic and12 real-image
native probes, original complete scalar-grid audit and source/asset commitments
are prerequisites. No new model is downloaded or admitted.

## Sources, calibration and comparison

Reuse exactly the prior60 genuine calibration sources and48 disjoint original/edit
development pairs, with native PNG and whole-image JPEG75. All original source,
input, response-array and completed native-audit hashes must match. The new
method does not use either edit labels or masks to compute C or select points.
Original calibration/development source IDs remain disjoint; generator counts
remain20 genuine calibration sources and16 development pairs per family.

Compute only the120 genuine calibration levels first. At least48/60 per format
must be eligible; all120 records must complete. Set each format's point to the
second-largest eligible genuine C, require it positive, and retain strict `>`.
Ties do not exceed. At most one calibration original can have a displayed region.
Verify this with the actual retained-area/display implementation, not the order
statistic alone. Publish both points before computing development critical levels.
These are empirical inspection operating points, not calibrated AI probabilities,
public threshold claims or bounds on future customer false alarms.

Then compute all192 development outcomes. Keep the v8 actual grid, reliability,
uint8 eligibility-preserving encoding, component filtering/ranking, two-region
per-map cap and OpenCV nearest source-pixel projection. Compare to the exact
retained current PNG/JPEG component results; do not replace that comparator with
the failed v8 candidate. Show the v8 result as historical context only. This is
still a component study; complete production packet admission remains separate.

## Frozen advancement requirements

Each format qualifies only if all96 development inputs complete; original
source/response/native commitments and independent new critical/display/pixel
audits pass; at least4 more of48 edits overlap the selected pixels than current;
mean edit IoU is not lower; no family loses an overlapping edit; and genuine
highlighted images are at most current+1. An empty/unavailable region is a miss,
not evidence of originality. The existing baseline located12/48 in both formats.
No post-result region union, alternative point or area rescue is permitted.

A passing consumed-source result requires a separately frozen fresh comparison
before integration. A failure remains a failure. No historical source reserve
from failed candidates may be repurposed. No overall confidence, serving model,
headline, customer report, prepaid billing or GCP configuration changes here.
Outside human reviewers are zero; automated replay must not be called a person's
independent review, and outreach remains unauthorized.

## Execution and retention

Freeze code, exact prior journals/panel/arrays/native receipts, package versions
and original component policy before any project C is computed. Fail on missing,
duplicated, changed or incomplete rows. Preserve all original attempts. One
bounded CPU worker runs sequentially, with6GiB sampled RSS,1.5GiB available RAM
and free disk, and30minutes per stage. Resource maxima are sampled lower bounds;
retain observation failures and terminal guard receipts. Do not silently restart.

Reuse verified cached arrays read-only. Store new scalar responses, boundary
certificates, mask/grid byte commitments and full pixel-count receipts; do not
duplicate1024-square arrays. Privately retained prior maps and source masks allow
full local reconstruction. A separately implemented plateau oracle, component
traversal and source-pixel counter check every new outcome. Publish source/hash
references, scalar/count outcomes and reproducible review instructions after
privacy/secret scanning; publish no photos, raw masks or real-image embeddings.
