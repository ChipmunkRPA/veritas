# Audio v7 fresh comparison: prospective design and preparation

Status: sources and decoded-prefix checks complete; **no confirmation detector
inference**. The exact input variants, inference/review code and these rules
must be bound in a final immutable manifest before scoring. Source acquisition
and eligibility are not a passing confirmation result or outside validation.
The serving detector and overall confidence remain unchanged; GCP is deferred.

The [public source catalogue](evaluation-panels/audio-v7-fresh-sources.json)
contains126 source references (120 foreground and six background speakers),
twelve noise/music references, all110 known foreground groups and verified
archive integrity receipts. Full audio, transcripts and local paths are omitted.
The original prepared-source record SHA-256 is
`2027c36079c29cf0f5fb3c365b2d3effa50fac4fd0010712b94caefcad563e8a`.

## Fixed candidate and comparator

The [v7 development candidate](audio-guard-v7-evaluation.md) passed all twelve
development gates. Freeze its own head and independent guard artifact SHA-256
`fa1a0faec12fd6926cefb8f7ffe9c9736f517ea89b44a2ff13ab17f4d028c408`;
do not refit on any confirmation source, change its probability mapping or
relax its class/stability guards. Compare with the exact current guarded
`local-speech-v3.1` service calibration, SHA-256
`912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf`.
Use the same pinned, commercially permitted Pella V2 and Silero speech-activity
weights, native CPU float32 preprocessing and source speech gate for both arms.
No additional neural checkpoint or generator is being admitted to serving.

The internal .8/.2 tendency points, alpha .1 and original-only ln2 stability
bound remain unchanged. Candidate probes apply the candidate's own head to
the same bandpass view; the service uses its own existing raw-response probe.
These are research decision diagnostics, not public confidence claims.

## Sources chosen without detector outputs

The fixed foreground plan has sixty genuine LibriSpeech speakers, thirty
ElevenLabs voices and thirty Chatterbox prompt-speaker IDs/dialogues. Six other
LibriSpeech speakers supply genuine background speech only. LibriSpeech
`test-other` and `dev-other` provide66 known-unused speakers after excluding240
previously used IDs. All `test-clean` and `dev-clean` speakers were already
used; they are excluded. Their publisher archive checksums are retained.

Select one four-second-eligible source per known genuine speaker. Chatterbox
selection keeps distinct publisher prompt-speaker IDs and dialogue IDs and
uses actual `speech_meta` file indices, not the longer original `utterances`
list. Only publisher-declared MIT PLN scenarios at pinned revision
`7b7904b73b10ba272145606a06afb6e65bfac985` are eligible. Some isolated files are
stereo; production decoding downmixes them to mono. The initial mono-only
planning stop is retained locally; no detector was run on that attempt.

ElevenLabs selection uses all thirty public voices at pinned dataset revision
`9e41afe1a8b995d2091b602b424c11e4de8e27f6`, one distinct script per voice from
the full36-script catalogue. Each selected file matches its publisher LFS
SHA-256. Some fluent/disfluent scripts have shared openings; known full texts
and eight-word prefixes connect them for uncertainty resampling. Do not count
them as thirty independent prompts. The120 foreground files form110 known
connected groups; repeated conditions do not create additional source groups.

Known prior source IDs, speaker IDs, bytes, full decoded PCM and four-second
prefixes are excluded. Known full transcripts/eight-word prefixes and the
complete original LJ Speech text list are used for prompt exclusion. This
does not establish biometric identity disjointness across corpora, exact word
alignment within four seconds, or absence from unknown pretrained data.
Publisher permission declarations are retained; they are not an independent
audit of every underlying cloned timbre or prompt right.

## Fixed conditions and complete denominators

Prepare exactly nine variants per foreground source,1080 total:

- clean;
- noise at0,5 and15 dB SNR;
- music at5 and15 dB SNR;
- overlapping genuine background speech at0,5 and15 dB SNR.

Use the first64,000 production-decoded samples of each source, exactly four
seconds at16 kHz. This matches the existing short-window calibration scope;
it is not validation of whole recordings or finding the edited interval in
long audio. Normalize clean foreground RMS to0.1, scale backgrounds to the
declared SNR, then apply one common peak gain across all nine source variants
so no variant clips. Record realized SNR after production decoding.

Eight new MUSAN FreeSound noise recordings, four new MUSAN RFM music tracks
and six separate genuine LibriSpeech speakers form the background pools.
Their known recording IDs, source/decoded hashes and paths are disjoint from
previous fitting/calibration/development backgrounds. Noise contributor/session
IDs are unpublished; music shares a composer with older distinct tracks.
Assign backgrounds by fixed source order within each class, balanced across
genuine/synthetic foregrounds; select offsets by the fixed source seed before
inference. No source, offset, family or failure may be replaced after scores.

Background speech is always genuine. The truth label is whether AI speech is
present: genuine+genuine remains genuine; synthetic+genuine remains synthetic.
No claim about which talker or exact words caused a response follows from the
mixed-window score. Keep every planned source/condition, failure, missing
speech window and inconclusive outcome in operational denominators. Use the
same original-view speech applicability gate in both arms; failure to assess
must never produce an original-file verdict.

## Prospective analysis and advancement gates

Report all1080 planned outcomes, all assessed/unassessed/failed counts, both
false-call directions and inconclusive counts/rates by condition, true class,
generator, source dataset and background. Also report AI inputs without an AI
conclusion, including inconclusive inputs; zero false-original calls is not
zero missed AI. Show per-generator results even if pooled scores improve.

Proper scores use common assessed inputs regardless of abstention. Weight
sources equally, then balance the two truth classes. Estimate paired Brier/
log-loss differences using2,000 connected-foreground-group bootstrap draws,
seed461109, deterministic sorted strata/order. Stratify groups by known
generator/class; a group connecting several kinds must remain one joint
group. Intervals condition on fixed fit/calibration and the selected reused
background pool; they exclude model/calibration/background-population uncertainty.

Advance only if **all** these prospective requirements pass:

1. Overall Brier improves by at least.005 and log loss by at least.01; the
   paired conditional95% Brier interval lies wholly below zero.
2. Brier improves separately for noise0, music5 and each of speech0/5/15.
3. At most one additional false-AI and one false-original call overall and
   within each condition. At most one additional false-original call within
   each synthetic generator family.
4. At most two lost correct calls per class in clean audio, and at most two
   lost correct AI calls within either synthetic generator family overall.
5. At least five additional correct calls in **each** noise0 and music5, and
   at least five additional correct calls across the three background-speech
   conditions combined. In each speech condition, neither truth class may
   lose more than two correct calls.
6. Identical assessed input identities, complete1080 outcomes, unchanged
   head/calibration/checkpoint pins and zero new extraction/probe failures.

These criteria were written before fresh detector inference. Do not choose a
cutoff, subgroup or rescue arm from confirmation results. A failure preserves
the service; these sources then become consumed development material. A pass
permits only scoped local integration with production-adapter parity,
source-linked evidence, PDF/JSON, prepaid worker and rollback verification.
It does not enable the candidate automatically or establish outside review.
