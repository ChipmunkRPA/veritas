# Fresh-source transfer preparation, 10 November 2026

The complete test selection is now fixed: **32 clips across sixteen original
GRID source slots, with 32 distinct utterance stems**. Both directory batches
and the final selection stage passed with no guard/monitor failure.
[Selection receipt](evaluation-panels/grid-clip-plan-result-v136.json).
Source codes/stems do not prove independent people/ancestry or absence from
pretrained encoder data. Existing detector heads/calibrations remain unchanged.

The [explicit acquisition continuation](grid-acquisition-continuation-v135-method.md)
successfully fetched only the missing 49,600,638 bytes of the stopped second
archive, then verified the complete reconstructed archive's publisher MD5/size.
It preserved the original prefix and failed v131 stage unchanged.
[Actual first continuation result](evaluation-panels/grid-acquisition-continuation-first-result-v135.json).
Subsequent previously unattempted sources use one full archive per bounded stage;
this is a new schedule, not a claim that the failed original procedure passed.

At the attached snapshot, **four complete source archives and eight selected
clips** have full native input banks: **600 original RGB pictures**, independent
PNG round trips, and **381,184 canonical mono-16-kHz samples** with usable source
clocks. All four native stages exited zero with no guard or sampling errors.
No face eligibility or detector inference ran. The remaining twelve acquisitions/
native banks continue sequentially under the same per-stage limits; this snapshot
does not claim a complete 32-clip cohort.
[Native bank snapshot](evaluation-panels/grid-native-first-eight-result-v137.json).

The sole live controller also checks available RAM between stages for the
already published v133 CPU qualifier. At the latest checkpoint before the fifth
source acquisition, 6,033,227,776 bytes were available, below the unchanged
6-GiB neural initial minimum. That was a read-only checkpoint; the prior actual
5.32-GiB admission denial remains separate. No new model call or confidence change
is claimed by the completed source-preparation stages.

The [new expression-driving bridge](portrait-expression-drive-v138-design.md)
addresses a construction limitation: a single aperture ratio cannot represent
full lip motion. It retains source pose/identity parameters and transfers a
same-recording driver's full learned expression displacement. Its independent
NumPy equation/import checks pass, but the CPU bridge is **unqualified**: no
pretrained forward, generated-motion quality or detection gain is established.
It must receive separate frozen neural qualification after the scalar baseline,
with exact module counts, repeats, native composites and achieved-motion checks.
It is not used by the customer application or by the current live controller.

All **27 new authored local checks** for continuation, full selection, native
source correspondence/clocks and expression arithmetic passed; Ruff/diff checks
also passed. Public independent-testing instructions now link these entry points
and correct the stale statement that the earlier v118 renderer had not run.
These are software/data qualification checks, not fresh prediction results or
outside human review. No outreach is authorized or performed.

Raw footage/audio, source/utterance identities/pairings, individual fingerprints,
motion/voice vectors and learned parameters stay private. Publication contains
code, methods, aggregate receipts and whole-artifact commitments only. Local
verification and manual GitHub pushes are used; no Actions workflow, live payment,
serving/default change or GCP deployment is part of this phase.
