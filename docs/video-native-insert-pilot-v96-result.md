# Known-boundary video pilot: construction and command review

The first fitting-role pilot under the [frozen native-insert procedure](video-native-inserts-v95-method.md)
completed locally. It contains six silent, six-second clips: an unchanged
original, two publisher-provenance face-swap inserts, and ordinary lighting,
blur and time-step controls. All retain the source's 1920 × 1080 RGB pictures
and exact 25-Hz clock. The planned edit occupies frames 50–99, or **2–4 seconds**.
This is test-input qualification, not detector accuracy or a serving change.

All **900 output frames** reproduced the intended RGB bytes and timestamps.
The original has no changed frames. Each of the other five clips has exactly
50 changed frames, all inside the planned interval; the original pictures are
restored before and after it. The **18 native PNGs** correspond to the actual
before, during and after frames. Ordinary changes retain their non-AI label.
Changed pixels are not automatically an AI mask or proof of a changed identity.

A separate command-interface reviewer read every frame with FFmpeg SHA256
framehash, checked dimensions and display duration through ffprobe, and checked
every PNG's decoded pixels and timestamp. It also independently recomputed
original/swap/time-step source correspondence and changed-frame support from
the saved source records. It imports neither the constructor nor the app
decoder. Both paths use the same underlying codec library: this is not an
independent decoder engine or an outside human evaluation. The ordinary blur
and lighting formulae were not independently reimplemented by this reviewer.

The [aggregate receipt](evaluation-panels/video-native-insert-pilot-v96.json)
binds construction, command review, resource profiles and reviewed code through
whole-artifact hashes. Construction took 129.33 seconds, with maximum sampled
process-tree RSS of 828,817,408 bytes; review took 20.52 seconds, with maximum
sampled combined RSS of 638,697,472 bytes. Neither stage hit a resource guard
or recorded a sampling error. Samples are not guaranteed peaks. Six authored
reviewer tests passed locally, including refusal of corrupted picture hashes,
timestamps and dimensions. No neural model was loaded.

The frozen plan retains 90 compatible groups and 540 planned cases; this pilot
materialized only one group. Six incompatible source groups remain explicit
in the parent inventory. Wider construction, actual detector comparisons,
separate calibration and customer evidence/export checks remain unfinished.
These silent clips cannot measure lip-sync performance. The audio/mouth panel
still needs sentence-aware roles and synchronized manipulation preparation.

Media, source pairings and individual fingerprints remain private on the
approved development archive. Only code and aggregate receipts are published
by direct push after local checks. No Actions workflow, outreach, payment or
GCP deployment is part of this result.
