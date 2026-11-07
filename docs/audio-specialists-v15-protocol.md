# Noisy-voice interval and context calibration

The study tests separate full/band-view and waveform-context mappings, followed
by separately fitted/calibrated file results. It extends the shared four-media
framework to controlled eight-second real/AI composites with exact edit boundaries.
This is consumed-source development, not an automatically admitted service upgrade.
GCP remains last. Video remains separate required work.

## Fixed sources and role separation

Use all 143 AI-containing connected groups in the consumed v9 source plan, across
eight generators. Choose the first AI source by a fixed salted hash. Use a human
source from the same group when available; otherwise pair it with one unused
hash-ordered genuine-only group. Do not choose using authenticity scores. Keep
the full original source/speaker/transcript ancestry together. Stratify by chosen
generator and assign a globally continuing round-robin sequence to four roles:
specialist calibration, combination fitting, combination calibration and confirmation.

Partition the 48 consumed background originals by fixed hash within each kind:
each role gets six public-domain noise recordings, three CC BY3 music recordings
and three CC BY4 genuine speech recordings. Verify original byte and production
PCM hashes, license records and known speaker/foreground separation. Select a
background and four-second offset by pair/kind hash within the pair's role.
Never share a background parent between roles or silently replace a quiet offset.

There are 35/36 foreground-pair groups per role. Shared backgrounds within a role
are repeated nuisance parents, not additional independent noise recordings.
Foreground-group calibration support does not establish background-population
confidence. Sources have already been consumed, external detector training overlap
is unknown and original provenance declarations are not independently examined.
No unused source reserve is scored to repair a failed method.

## Exact samples and local truth

Create human/human, AI/AI, human/AI and AI/human eight-second files from fixed
four-second production-decoded prefixes. Normalize each foreground to RMS0.1.
The first half is clean. The second is clean or mixed with noise, music or genuine
overlapping speech at -5 or +5 dB. One peak-safe gain is shared by every pattern
and condition of the pair. Roundtrip through the production mono16kHz s16 WAV
writer/reader. Independently verify first input per role/condition/file class
with the full production decoder before authenticity inference.

Keep the production full-profile four-second windows at 0, 2 and 4 seconds.
Run original full-clip Silero speech gating before authenticity scores. Every
planned window remains in the manifest, including rejected speech windows.
Its local AI label is any overlap with a known AI foreground interval. A mixed
file's human-only window remains genuine. Ordinary splices, loops, noise and
speaker changes alone are not AI truth. Background speech is genuine.

The band view is a whole-clip zero-phase Butterworth100–3800Hz/order4 transform,
using exactly the same original gate and source windows. It is a sensitivity
view, not a recovered original or a separately observed authenticity fact.
Score-blind routing uses normalized spectral flatness measured on the original
window: at least0.65 is diffuse; otherwise concentrated. Known generator,
recipe/SNR, label and detector response never select a customer's calibration.

## Frozen comparisons and verification

Use the existing pinned MIT Pella voice and Silero weights, CPUfloat32, four
threads and complete native four-second preprocessing. Load only local verified
assets; acquire no new model and execute no remote code. Verify native parity
on retained exact old WAV/PCM/raw responses before new authenticity inference.
The source/signal plan may run speech gating but records zero authenticity-model
forwards until the complete scoring/fitting implementation and runtime are frozen.

Compare full-view pooled calibration, full-view context calibration, full+band
pooled calibration and full+band context calibration. Full and band are one
dependence family; repeated windows do not become independent votes. Use the
unchanged 25-foreground-group/class support floor, L2=.001 and50:50 reference
prior. Retain unsupported contexts and missing evidence; never substitute zero
or an original-file verdict. Fit every arm and its combination calibration before
any confirmation authenticity-model forward.

For an operational comparison, freeze an internal AI-only point on combination
calibration genuine foreground-pair maxima among assessed files: strict greater than the nearest-rank
ceil((n+1)*0.9), capped at n. Apply this same procedure to every arm, preserving
ties and missing evidence. This is not a public threshold or guaranteed false-alarm
rate. Report both point-independent probability loss and actual confirmation
false alarms, missed AI, inconclusive/coverage rates, per-condition/local errors
and exact time/sample coordinates. Gate an improvement over the pooled full-view
research reference on more AI files found, no additional falsely flagged genuine
foreground groups, no coverage loss and lower source/class-balanced Brier loss.
Require no clean-AI detection loss, no additional falsely flagged genuine groups
in any background condition, no condition coverage loss and more AI detections
across the three -5dB conditions. These continuation gates are frozen before any
new authenticity inference. Groups with no assessed genuine file are counted
explicitly and supply no fabricated response to point selection; at least25
assessed genuine foreground groups are required for an operating point.
The current service's calibrated four-second scope must not be presented as an
existing eight-second file-confidence baseline.

Keep one direct native repeat per role/view/context, every failure and every
planned input. Resource limits per sequential role: 60 minutes,6GiB sampled tree
memory and at least1.5GiB available RAM/free disk. Do not silently restart a guarded
attempt. Raw audio, transcripts, source identities and any voice embeddings remain
private. The data has artificial boundaries and a small background-parent bank;
successful development alone cannot establish fresh independent quality, expert
review, customer probability, synthetic-background detection or court suitability.
