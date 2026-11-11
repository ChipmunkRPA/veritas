# GRID serving readout v159

The readout accepts the validated v143 pilot plan, v149 construction records, and
v156 full-serving records. It joins privately on pilot ordinal, condition, case
number, source slot, connected component, and the constructed input hash. The
caller supplies these records through the existing validated readers and can
mark preserved guard failures. The readout imports only metadata validators and
performs no file I/O, media decoding, inference, or serving execution.
[Prospective design](grid-serving-readout-design.md)

The public output includes all 176 planned cases in its denominator. It records
missing construction and serving receipts, unavailable input, guard failure,
failed analysis, unverified source, skipped checks, missing measurement support,
and measured cases without flags. The sampling and detector counts describe
coverage observed in the supplied records. Displayed exhibits count only when
their source commitments and measurements match the endpoint that produced them.

- **Customer direction.** AI and original directions require a completed,
  committed analysis and an assessed production conclusion. Inconclusive,
  unavailable, and provenance-record outcomes remain distinct. Scope and the
  separate soundtrack outcome use the recorded projection. False AI attribution
  measures customer AI directions on verified non-neural controls.
- **Appearance review.** The existing production validator checks each measured
  window and the actual selected appearance or fallback range. The core center
  timestamps define that flag. Displayed context counts only when the three
  packet anchors match the measured frames. Unique source frame counts describe
  measured support between sparse centers.
- **Audio and picture timing.** All verified raw offset-step and correlation
  collapse spans count, including spans omitted from the displayed packets.
  Segment ticks map to exact rational times on the decoded PCM clock. The readout
  cross-checks native inputs, audio commitments, detector counts and raw exhibits.
  Measured windows without a stable offset have their own state. A raw timing
  flag is an inspection cue. Overlap on a neural redraw is descriptive because
  the construction verifies neither achieved lip synchronization nor phonetic
  timing.

The readout unions half-open flagged spans and clips them to the observed
first-to-last picture PTS. Timing also requires the decoded audio bounds. Visual
localization uses the known interval only where a changed-picture mask records
an effect. Timing localization uses the copied soundtrack output interval only
where sample values actually changed. The assessed localization rate and the
planned intervention rate use the prespecified IoU point of one fifth. Ordinary
control flags and unchanged-stream flags measure review burden independently of
customer attribution.

Publication includes planned, constructed, completed, assessed and delivered
counts and rates. Component-weighted rates average the within-component rates
with equal component weight. Exact sums and rates use rational strings.
Condition and family partitions require at least five planned components per
group. Continuous summaries require five contributing components. When a
continuous cell falls below that count, the condition and family complements for
that quantity are also withheld. Case rows, file and frame hashes, component
membership, source IDs, raw scores, paths, and frame and timing lists stay
private. This readout measures the admitted serving path. A matched comparison
with another serving configuration requires that configuration's own validated
records.
