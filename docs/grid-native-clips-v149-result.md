# All 176 matched native test clips are ready

The frozen constructor completed all **176 cases from 16 pilot recordings**:
16 remuxed originals, 64 observable neural mouth edits from two generators, and
96 ordinary controls. None were unavailable or replaced. Both render branches
and all conditions used the same lossless FFV1 video and mono 16-kHz PCM format.
The reserve was not constructed or scored.

The actual decode checks matched **13,200 pictures, 8,386,048 audio samples
and 2,880 native changed-pixel masks**. All 16 audio-only controls kept exactly
the same picture pixels and timestamps as their matched originals, while their
soundtrack samples differed. Every other condition preserved the matched original
soundtrack. Case labels were kept in private metadata, outside the video container.

The 16 guarded data stages completed with no monitor or resource failures and
zero neural calls. Summed worker-stage time was 290.77 seconds; maximum sampled
descendant RSS was 406,929,408 bytes (388.08 MiB), within the unchanged 1.5-GiB
data-stage limit. See the [actual receipt](evaluation-panels/grid-native-clips-complete-v149.json)
and [frozen construction method](grid-native-clips-v149-method.md).

The exact native picture timestamps and audio origins were checked. The terminal
display duration was assigned by the mux, so a window that depends on its
unverified final cell is outside the frozen scoring scope. These clips establish
known edit provenance and support for the comparison. Achieved motion, missed
edits, false alarms and serving performance are measured separately. Raw clips,
masks, clocks, individual fingerprints and source pairings remain private.
