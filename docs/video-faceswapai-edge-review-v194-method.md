# Source-bound face-appearance end review

Compare each saved v190 appearance range with a fixed end-context rule.
Use all six existing matched cases, including the unswapped and ordinary
recompression controls. This is development on a consumed synthetic source
component. Its known edit range is already public. The candidate receives
only measured crop appearance and positions. Construction labels enter
after the end proposal has been fixed. [Existing results](video-faceswapai-full-app-v190-result.md)

Reopen the exact v190 result and saved evidence. Revalidate the appearance
packet, representative frame hashes and native presentation times against
the v186 source binding. Decode the original native pictures listed in the
packet. Reconstruct the recorded 112-pixel crops from their observed face
landmarks and require exact RGB hash agreement. Use the same pinned
commercial CPU appearance checkpoint, reproduce the existing selected
window’s arithmetic, and count every new crop forward. Keep aligned crops
and appearance vectors in process memory. [Checkpoint notice](third-party/opencv-sface-v15-Apache-2.0.txt)

The candidate pools the six crops in the two measured before triplets.
Its second reference pools the core triplets through the penultimate
center. The final core triplet and both after triplets are held out
from both references. Require all three-frame groups to be disjoint.
Require the reference distance to exceed the existing internal appearance
review point. Classify each held-out crop by which reference is closer
in cosine distance, treating differences of 0.000001 or less as ambiguous.
Each triplet must agree across its three native frames.

Extend only if the final core triplet is core-like and the second after
triplet is before-like. If the first after triplet is core-like, report its
last crop as the proposed end and the second after triplet’s first crop as
the first measured return. If both after triplets are before-like, use
the final core triplet’s last crop and the first after triplet’s first crop.
With overlapping groups, insufficient reference separation, failed replay,
mixed or missing return evidence, keep the existing endpoint. The rule
uses the same criteria on both controls and all four swap interpretations.

The existing before, during and after images remain associated with the
original measured core. Report their timestamps, the last core-like and
first before-like native PTS, each measured context crop, the unmeasured
positions in the return bracket, and the count of appended but unmeasured
positions. Interval overlap counts the inclusive displayed extent. It
does not assert measurement of every intervening frame or an exact edit
boundary. For swap cases report overlap, misses, outside extent and IoU
against the saved substitution. For controls report both interval extents
and any overlap with ordinary changed frames. A change cue on a control
remains part of the result. The experiment does not evaluate face pixels
as ground truth or change a customer assessment.

Freeze the exact v190 result, per-case bytes and checkpointed SQLite
hashes, local checkpoint, decoders, package versions, and code. Replay
only from immutable SQLite connections and existing source files. Reject
uncheckpointed database changes and block process networking and writes.
Emit JSON to standard output. No new analysis, API call or derivative
media is needed.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_faceswapai_edge_review_v194 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.video_faceswapai_edge_review_v194 readout
```
