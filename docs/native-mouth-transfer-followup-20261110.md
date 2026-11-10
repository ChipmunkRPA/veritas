# Transfer follow-up: separate motion and articulation evidence

Primary-source follow-up checked 10 November 2026, alongside the existing
[research review](recent-detection-research-2026-11-09.md). This is a research
direction, not a newly executed detector or a change to customer confidence.

[LADDER, revision 5 dated 19 October 2026](https://arxiv.org/html/2604.16808v5)
refines the earlier landmark-only proposal. Its important lesson is that frame
rate can reveal a dataset label rather than manipulation. It normalizes trajectories
to a common time grid before measuring relative motion across derivative orders;
absolute mouth-motion amplitude transfers poorly. Missing detections split runs
rather than being filled across gaps. A fixed real-reference score complements
a learned detector, but some generators and noisy inputs remain difficult.
The paper also shows that better ranking does not guarantee low false alarms
under a transferred decision point, and reducing false alarms can lose detections.
Landmark trajectories remain potentially identifying data.

**Implication for our next independent implementation:** compare a separate
timing-normalized motion cue with the existing visual and audio–mouth interaction
heads. Keep native timestamps and sampled-frame support; do not interpret skipped
frames as consecutive motion. Freeze a separate motion feature/calibration plan
before scoring. Test genuine expressive speech, pose changes, noisy/blurred footage,
rate conversions and multiple generators. Report detections and false alarms
together, including coverage and losses introduced by fusion. Keep trajectories
private. Do not copy the paper's code, weights or restricted benchmark media.

This is complementary to the already tested interaction-head design motivated
by [C-MIND](https://arxiv.org/html/2610.30872v1), not a replacement chosen from an
abstract. The current v133 runtime qualification uses a scalar aperture only to
test a second generator's response and native-coordinate containment. It does
not implement LADDER, reproduce C-MIND or establish audio/phonetic retiming.
The new GRID transfer cohort must still be acquired and qualified, with heads
and their existing study calibrations kept fixed for the planned transfer test.
