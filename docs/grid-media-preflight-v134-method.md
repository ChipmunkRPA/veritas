# GRID: first-source native media preflight

The v131 acquisition stopped at the unchanged 900-second limit. Its worker
journal records one complete publisher-checksummed archive and a second partial
download. Preserve that attempt and partial file. This separate data-only
preflight uses **only the complete first archive**; it neither retries nor resumes
the failed download. The remaining transfer cohort is not claimed complete.

Bind the exact original acquisition plan/freeze, stopped elapsed-only profile
and whole archive journal. Independently recompute the first archive's full
publisher MD5, observed SHA-256, size and complete safe ZIP inventory inside
the guarded worker before using any member. Retain CC-BY-4.0 attribution.

Select two real `.mpg` members by SHA-256 of `chipmunk.grid-clips/134:` plus
utterance stem, breaking ties by full member name. Exclude resource forks and
already used stems. This first source uses an empty used-stem set; later source
planning must carry it forward before model use, keeping utterances unique.
Do not choose files from face, audio or detector responses.

Freeze this implementation, source plan and installed decoder identities before
decoding. Extract only these bounded members, verify ZIP CRC and retain opaque
private filenames. Probe at most 500 native video frames, with 60-second probe/
decode deadlines, at most 640 pixels per dimension and ten seconds of native
duration. Require observed strictly increasing PTS and no interpolated/truncated
frame table. Decode/save first, middle and last pictures with their actual PTS.
Do not infer timestamps from `index / fps` or reset the source origin.

Check the embedded first audio stream and its native frame/sample clocks using
the existing v92 rational clock procedure. Retain missing audio, bad clocks or
unsupported bounds explicitly. For contiguous supported audio, decode the first
stream to canonical mono 16 kHz with a ten-second bound and preserve its entire
source-sample mapping and WAV privately. This establishes availability, not
acoustic authenticity or achieved lip alignment. No replacement source/clip.

Run the single decoder-only stage outside the sandbox with the qualified exact
owned-descendant monitor: 900 seconds, 1.5 GiB sampled descendant RSS, 3 GiB initial
RAM/project/archive disk and 1.5 GiB running reserves. No neural call, retry,
overwrite or relaxed limit. Public output contains aggregate availability and
whole-artifact commitments only; no media, source identifiers/pairings, individual
fingerprints, face/voice vectors or learned weights. No serving/default, Actions,
outreach, payment or deployment change. The complete fresh transfer cohort,
generator construction and model comparisons remain separate unfinished stages.
