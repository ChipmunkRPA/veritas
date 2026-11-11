# Paired ordinary and synthetic speech edits

This local case preparation uses the exact three-second audio admitted by
v169c. The development and holdout roles retain independent source and noise
parents. The ordinary edits reproduce the distinction that failed in the
GRID pilot.
[Source preparation](audio-short-edit-materialization-v169c.md) · [GRID result](grid-serving-completion-result.md)

Each genuine source supplies untouched, same-source hard copy, 20 ms fade,
and 50 ms fade arms. Each AI source supplies its fully synthetic source and
six matched splices. For those six splices use one genuine host and one
genuine donor from a different declared voice and source group. Compare
genuine-to-genuine with genuine-to-AI at each of the same three joins.
The host, donor and noise assignments use fixed salted metadata ordering
within their role. Reused speakers and noise make the comparison arms
dependent. The private plan retains the source, genuine host, genuine donor
and noise identities for grouped analysis.

| Frozen source availability | Development | Holdout |
| --- | ---: | ---: |
| Genuine comparisons, 12 cases each | 14 | 16 |
| AI comparisons, 21 cases each | 23 | 24 |
| Planned cases | 651 | 696 |

All cases contain exactly 48,000 signed-16-bit mono samples at 16 kHz. Copy
the half-open donor interval `[12160, 24960)` into `[17280, 30080)`. The
raised-cosine joins are contained within that interval. A synthetic splice
records its AI donor support and its separately identified blend and pure
donor intervals. Compare each realized case against its host on the same
gain and noise treatment and retain the exact changed-sample intervals.
Genuine-to-genuine edits retain genuine source truth even when samples
change. Fully synthetic truth describes the origin of the admitted source.
[Pure sample construction](../backend/scripts/audio_short_edit_controls.py)

Use clean, 0 dB and +5 dB conditions. One role-local noise recording and
one common gain serve every arm in a comparison. The genuine host sets
the noise scale for paired insertions and the genuine original sets it for
same-source copies. When the host is unavailable the fully synthetic source
can provide its own reference. The noisy sample residual at each level
is identical across arms. The actual measured SNR can differ with arm RMS.
The untouched arm preserves the source structure and applies the same
recorded gain as its matched arms.

Every case file must round-trip with exactly the written PCM and native
duration. Missing source, donor, noise, unsupported sample changes and
file-write failures remain in the private journal. Preserve comparison
denominators and join conditions even when a case is unavailable.
Outputs and role journals stay private. The aggregate readout contains
counts and whole-artifact commitments.

An authored comparison prepared 21 WAVs. The production decoder reproduced
the exact expected sample bytes and zero-origin range in every case. A
separate test checked that the decoded noise residual at each level was
identical across all seven arms. No frozen speech source was opened.

Run from the repository root after the verified v169c source preparation.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_short_edit_cases_v172 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_short_edit_cases_v172 run development
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_short_edit_cases_v172 readout development
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_short_edit_cases_v172 run holdout
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_short_edit_cases_v172 readout holdout
```

The runs use the existing v108 data-only process guard, owned local sampler
and archive write access. They can be run separately. The development
constructor opens only development audio. Both role case runs are
preparation without model scoring or fitting.
