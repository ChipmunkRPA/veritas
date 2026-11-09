# Historical speech identities checked before fresh selection

The metadata-only audit read 44 historical research manifests/source plans and
recorded 21,038 exclusion keys from 19,631 metadata references. These references
include mixtures and contextual records; they are **not independent recordings**.
The catalogue retains 13,321 distinct recorded byte hashes and 745 declared
voice keys: 461 LibriSpeech, 108 VCTK, 54 provider-profile hashes, 12 Kokoro stock
profiles and 110 other dataset-declared identities.

The audit joins known genuine LibriSpeech speakers to publisher-declared cloned
prompt speakers, normalizes stock voice aliases, retains source/PCM hashes and
compares full normalized transcripts and eight-word openings. Unicode width and
case variants do not create a fresh text. Unresolved known speaker metadata is
explicit; none was found in this catalogue. A separate check compared 833
explicit canonical identities with independently derived known publisher
identities and found no conflict. All input file hashes were reverified.

The pinned Kokoro metadata has 28 English stock profiles. Twelve match declared
history and sixteen have no declared history match. The latter are candidates
for source planning only. They are not proof of unseen biometric identity,
absence from pretraining or independent new generators. A different prompt
through an already tested stock voice is not a new independent speaker.

Five focused tests and lint passed. The
[summary](evaluation-panels/audio-history-v58-summary.json) and
[metadata consistency review](evaluation-panels/audio-history-v58-additive-review.json)
publish aggregate counts and whole-file commitments. Voice keys, transcripts,
source identities, pairings and the exclusion catalogue remain private.

This covers the selected available historical research manifests, including
audio, lip-sync and video context. It does not inspect customer uploads,
unmanifested runtime fixtures, biometric matches or model pretraining. No model
was run, media downloaded/decoded, benchmark role assigned, calibration/default
changed, workflow started, outreach sent or deployment performed. The next
selection must use these exclusions and preserve connected source/voice/text
groups before candidate inference.
