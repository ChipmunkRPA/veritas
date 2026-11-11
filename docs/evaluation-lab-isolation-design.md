# Preserve evaluation history while changing the app

The current evaluation chain compares historical code commitments to files in
the live checkout. A legitimate app change can therefore block a later research
stage even though the old code and its completed result still exist. The current
v143 dependency chain reaches historical decoders, face and lip-sync helpers,
sampling, swap and the detector engine: eleven shared application files. The
geometry, frame-construction and speech preparation stages inherit those checks.

A read-only audit hashed the raw Git blobs for 100 distinct code paths across the
selected historical and current freezes. All 100 match their SHA-256 commitments
in commit `c39fd05ba98470cf79fac739be65b494c930b411`. This anchors the tested
research source bytes. Git does not preserve the ignored private plans, media,
cached weights, installed runtimes or execution receipts; their existing private
bindings still matter. No result or freeze was rewritten to obtain this match.

Keep the original evaluation path and qualified runtime intact while the current
stages finish. Use a separate worktree and environment for product changes that
would affect the historical files or runtime. Do not move the active lab or
rewrite its absolute private paths. Reuse commercially cleared model assets only
through their existing read-only identities; keep evaluation and app outputs
separate. A changed product path needs its own current serving checks.

A future evaluation context should verify old evidence against the exact
historical Git blob bytes and saved freeze/receipt/profile/journal commitments.
It should separately bind the source, imports, assets and environment that its
own worker actually executes. Preserve missing private evidence as an unavailable
replay, even if a historical public aggregate is present. If the comparison uses
a new serving algorithm, freeze that algorithm and rerun the applicable fresh
comparison in its new context. Never silently import mutable app code under an
old hash or recast an old failure as a success.

Test the separation with an authored product-source change that leaves a valid
historical receipt admissible, then with wrong archived code, changed private
artifacts and unintended live imports that must fail. Keep code, aggregate
results and whole-artifact commitments public; raw inputs and learned records
stay private. This design and audit make no app, environment or deployment change.
