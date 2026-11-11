# Local speech alignment results on the GRID pilot

The corrected decoder completed the known-transcript alignment for all eight
originals with a matching publisher reference. The other eight had no word
alignment matching both speaker and clip and remained unavailable. Across 96
word boundaries, the median absolute difference was 10 ms and the largest
was 310 ms. The decoder reported 14 candidate bilabial phone intervals. Phone
boundaries and releases still need independently timed references.

Blind decoding recovered zero exact six-word prompts. It made 42 word edits
across 48 reference words. This tested speech recognizer cannot supply reliable
blind transcript coverage for the timing cue on this pilot.

The first decoder process exited with SIGSEGV before writing a row. Its native
stack pointed to child iteration. Copying each word and its phones before
advancing the parent iterator resolved the crash in the corrected run. The
original failure and the same input roster are bound by the continuation.
[Runtime correction](grid-phone-alignment-v162-method.md) ·
[Actual result](evaluation-panels/grid-phone-alignment-complete-v162.json)
