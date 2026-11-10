# Separate fitting-only native mouth heads

Fit three small classifiers from the completed and separately replayed
v123/v124 feature bank. This stage deliberately produces **raw margins only**.
There is no calibration, operating point, winner selection, held-out outcome,
customer confidence contribution or default-model change.

Require unchanged whole feature journals, extraction receipts/profiles and
successful replay receipts/profiles for all eight batches. Bind the published
complete aggregate and upstream frozen procedures. Verify source-connected
groups do not cross roles. Load feature arrays only for the 18 fixed fitting
sources, retaining originals and all available ordinary and neural conditions.
Other roles' metadata commitments are checked, but their vectors are not used.

## Blind support and fitting labels

For each condition, consider every interior native picture. Require its own
fully supported cached speech summary, eligible image features for the actual
previous/current/next pictures, and both native time gaps no larger than 80 ms.
Build the v122 2,304-variable visual descriptor and paired 3,840-variable speech
summary. Missing picture/audio/temporal support stays unavailable. Selection
uses no role, condition, known edit bounds or masks.

Only **after** forming these blind eligible indices, attach fitting targets:
a centre is positive when its condition is neural and its central picture is
inside the known native edit interval. All other centres are negative,
including ordinary edits and the untouched context of neural clips. Adjacent
visual descriptors can straddle an edit boundary; the target explicitly refers
to the central picture, not a proof that all inputs are local. Speech encoder
states retain their previously documented possible whole-input dependence.

Learn the frozen v122 rank-16 weighted genuine-reference projections and 0.1
ridge alignment using **only untouched fitting originals**. Then generate the
visual-only 16-variable, concatenation 32-variable and explicit-interaction
64-variable representations separately. No shared fitted classifier is reused
across the three arms.

## Fixed supervised learning

Each binary class gets half the fitting weight. Within a class, available
source groups receive equal mass, then available conditions per source/class,
then their individual eligible points. This retains an original-only short
source without inventing a positive condition or dropping genuine data. A
source with no eligible point remains in the private coverage journal.

For each arm, compute its own weighted feature mean and standard deviation,
with scale floor 0.05. Fit its own logistic linear head with C=0.1, lbfgs,
2,000 maximum iterations, tolerance 1e-8 and random seed 0. Require convergence;
do not rerun or change the solver on failure. Retain the uncalibrated signed
margin, not the classifier's sigmoid as a claimed AI probability. Separately
fitting/calibrating each arm on appropriate roles remains necessary.

Freeze code, authored tests, this procedure, package versions and upstream
whole-artifact commitments before fitting. Run one CPU/data-only stage through
the unchanged v108 owned-descendant guard outside the sandbox: 900 seconds,
1.5 GiB sampled descendant RSS, 3 GiB initial available RAM and project/archive
disk, and 1.5 GiB running reserves. Set numerical thread limits to one. No
pretrained network forward, download or device fallback is performed. Preserve
any failed attempt without retry, overwrite or relaxed limits.

Store learned reference parameters, heads, central-frame labels, coverage and
source identities privately on Seagate. Publish only method/code, aggregate
fitting counts and whole-artifact commitments. Training convergence and good
fit do not establish detection accuracy. Next work must separately freeze
calibration and measurement, compare all arms and current checks on locked
groups, test placement/generator transfer, and validate the complete app flow.
No outreach, Actions, real payment or GCP deployment is introduced.
