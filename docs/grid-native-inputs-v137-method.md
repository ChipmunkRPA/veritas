# Full native inputs for the fixed fresh GRID transfer cohort

Use all sixteen original source slots and the actual v136 32-clip selection,
with two globally unique utterance stems per source. Freeze this decoder-only
contract before constructing the input banks. Run a source stage only after its
actual complete archive acquisition has a successful profile/receipt and a bound
archive journal; the original stopped second download alone is not admissible.
Acquisition is allowed to finish incrementally, without changing selected clips.

Inside each guarded worker, independently recompute complete archive size,
publisher MD5, observed SHA-256 and safe ZIP inventory. Require exact correspondence
of selected names, uncompressed sizes and CRCs between the earlier remote
planning directory and the complete acquired archive. Extract just those two
bounded members; verify ZIP CRC and source bytes. Keep CC-BY-4.0 attribution.
No changed source, alternate clip or missing-byte substitution is allowed.

Use the v134 bounds: at most 500 video frames, ten seconds and 640 pixels per
dimension, strictly increasing observed PTS, no interpolated/truncated frame table.
Decode **every** native frame, not just sample pictures. Verify actual frame
order/timestamps through the existing decoder. Save all original PNGs and require
an independent Pillow RGB round trip for every frame. Source pixels and individual
file/RGB fingerprints stay private. Probe deadline 60 seconds, RGB deadline 120.

Check the embedded first audio stream's native rational frame/sample clocks,
retaining missing audio, unsupported/noncontiguous clocks and bounded decode
failures explicitly. Decode supported audio to complete canonical mono 16 kHz
with the ten-second bound. Save private float32 PCM, WAV and the entire original
sample mapping. Preserve observed video and audio origins separately; never
silently replace either with zero or infer timestamps from frame index/fps.
No VAD, speech encoder, face task, detector, generator or fit runs here.

Run sixteen explicit sequential source stages outside the sandbox with the exact
qualified owned-descendant monitor. Per stage: 900 seconds, 1.5 GiB sampled
descendant RSS, 3 GiB initial RAM/project/archive disk and 1.5 GiB running reserves.
Samples are not guaranteed peaks. Stop on a stage failure/admission denial;
preserve partial artifacts and do not retry, overwrite or relax limits. Missing
acquisition is a pending prerequisite with no attempted decoder stage.

Publish code/methods, aggregate availability/counts and whole-artifact commitments.
Raw media, source/utterance identities/pairings, frame/audio fingerprints and
source clock tables remain private. Partial completion is reported as partial;
successful clocks are not evidence of authenticity, people/ancestry independence,
face eligibility, achieved lip synchronization, calibration or detector accuracy.
No serving/default, Actions workflow, outreach, payment or deployment changes.
