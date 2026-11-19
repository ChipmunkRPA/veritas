# Short-window speech review on held-out readers

Requiring agreement between two views of the same audio located 21 of 32
clean inserts, up from 13 with the current two-second review. The two noisy
conditions yielded no correct locations with either method. Under dominant
competing speech, the additional check flagged five ordinary controls from
one reader group. The result supports an isolated trial of the clean-audio
evidence workflow. It provides no basis for increasing overall confidence.

## Inputs and method

The panel used 16 twelve-second recorded English hosts from eight declared
reader/donor groups. It used two stock synthesis families, each represented
by one planted insert per host. These families appeared in development.
Each host had five matched ordinary controls, including unchanged audio,
ordinary resampling, same-reader edits and genuine different-reader edits.
The replacement wording was not aligned to the original words. Raw identity
and audio records remain local. [MLS publisher and terms](https://www.openslr.org/94/)

All 368 unique deliveries were read as 16 kHz mono PCM16 WAV. The host
recordings originated in Opus. Four recorded backgrounds were reused across
the noisy deliveries at a target-to-background ratio of −10 dB, calculated
on the target speech. There were 16 background-only deliveries per noisy
condition. Repeated deliveries provide matched comparisons within these
eight groups and four background recordings.

The additional check retained the existing ranges. Each proposal needed two
adjacent 750 ms windows on a 500 ms grid. The raw arm measured the recorded
mixture. The agreement arm required matching responses at the same two
positions in both the recording and an enhancement estimate projected onto
its original time-frequency coefficients. Each view used its own frozen
decision point. For a correct location, one prediction had to cover at least
half the known edit and reach 0.30 interval intersection over union. A
range refers to the measured audio and provides no speaker attribution.
[Preceding phase-projection diagnostic](audio-phase-partition-diagnostic-2026-11-18.md)

## Held-out results

Each condition had 32 synthetic and 80 ordinary deliveries. Counts in the
last column refer to synthetic deliveries with flags exclusively outside
their planted edit. A file can have both a correct location and additional
off-target ranges.

| Condition | Review | Correct locations / 32 | Missed locations / 32 | Ordinary alerts / 80 | Background-only alerts | Synthetic files flagged only off target |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Clean | Current | 13 | 19 | 1 | — | 0 |
| Clean | Add raw short windows | 21 | 11 | 1 | — | 0 |
| Clean | Add agreeing short windows | 21 | 11 | 1 | — | 0 |
| Traffic | Current | 0 | 32 | 0 | 0 / 16 | 0 |
| Traffic | Add either short check | 0 | 32 | 0 | 0 / 16 | 0 |
| Competing speech | Current | 0 | 32 | 0 | 0 / 16 | 0 |
| Competing speech | Add either short check | 0 | 32 | 5 | 0 / 16 | 2 |

The eight added clean matches came from both synthesis families and five
reader groups. The corresponding ordinary controls were clear at each added
range. The clean agreement arm still included 13 additional off-target
ranges on 11 synthetic files. The raw arm included 17 on 15 files. Both arms
added four off-target ranges on two competing-speech files. The five ordinary
competing-speech alerts came from one reader group. For traffic, the number
of unassessed synthetic deliveries fell from 16 to 14 and ordinary deliveries
from 42 to 37. Both approaches left all 16 isolated traffic deliveries
unassessed. All 16 isolated competing-speech deliveries were assessed.

In a follow-up diagnostic on the closed records, the five ordinary alerts
and two false-only synthetic alerts all used the same two windows from one
host. The submitted PCM in those windows was identical across the five
ordinary deliveries, and both windows lay outside every matched edit span.
The clean foreground and the isolated background were each readable at
those clocks. Neither reached the raw decision point by itself. This is
evidence of a response to the mixture, with the contribution of each voice
unresolved. A future evaluation can cross new host readers with several
independently recorded backgrounds and test speaker-conditioned extraction
against both foreground-present and background-only spans.
[Speaker-conditioned extraction research](https://proceedings.neurips.cc/paper_files/paper/2025/file/67b0579a7298d9cf39c59404d867bdd7-Paper-Conference.pdf) ·
[Detector sensitivity to speaker identity](https://arxiv.org/abs/2607.21820)

## Independent verification

An independent render checker reconstructed the PCM deliveries and checked
288 byte comparisons outside known edit spans, plus source, carrier, and
background bindings. A separate standard-library recount verified 368
closed records, 8,464 short-window source clocks, preservation of the current
intervals, and the counts above. Its independent checks cover the saved
current intervals and the new window readings. Raw current-review logits
were unavailable in the score journal. The recount uses recorded detector
responses and speech activity measurements.

| Commitment | SHA-256 |
| --- | --- |
| Render manifest | `b3fe0c3755f1305dbf43f9fb92c3d0047c6634bfe83cc77095cf5d65a8bda097` |
| Scoring lock | `3e01a19aca734768e5477e6691b7b749f83f6e95a175b8bd229ead11f76dd9ab` |
| Independent outcome receipt | `943d3c6d87b22c16e68511c38346af3c2226175da137301a975369d87ecc9232` |

The two optional arms met the predeclared gate for a separate local
paid-flow trial. The agreement arm is the stronger trial candidate because
it located the same clean edits with fewer off-target ranges. The trial
needs source-bound playable excerpts, exact timestamps, preserved existing
findings, and unchanged confidence. The strong-background results motivate
testing other ways to measure the target voice and separating background
limitations from any claim about the voice itself.

## Isolated application trial

Six local worker analyses completed after simulated payment. The trial
started zero analyses before payment and made zero real charges. The four
held-panel witnesses reused the earlier test deliveries to verify
application behavior and reproduced their recorded readings. The clean
positive showed two original 750 ms windows. The matched same-reader
splice showed no selected window. The two competing-speech witnesses
retained their flags at the same unrelated windows, and an authored silent
control was unassessed.

The positive with and without the additional check had identical primary
scores, overall assessment, and existing evidence. Five experimental
appendices had signed PDF and JSON records linked to the standard signed
packages. Original-file replay checked six source excerpts and their
recorded and processed waveforms. The trial carries zero confidence weight.
The appended check remains isolated from the customer application.

The local trial receipt has SHA-256
`5fb29b195f571824072f37d1d5debc4e7b535fd07ab4eaeaeb9585cc7935da90`.

## Supplied reference under strong background noise

A development follow-up applied the existing earlier-copy checks to the
same 336 foreground deliveries. For the 12-second clean-copy check, the
earlier file was the dry host. It located 1 of 32 staged AI replacements in
the clean condition, 2 of 32 with traffic, and 0 of 32 with competing
speech. Of 48 untouched or locally roundtripped controls per condition,
the check flagged 0 clean, 3 traffic, and 9 competing-speech deliveries.
The timing-shift waveform comparison localized zero staged replacements.
These replacements preserve the original sample clock.

The result changed when both versions contained the identical background
on the same sample clock. The existing exact waveform comparison located
all 32 staged AI replacements, all 32 genuine speech splices, and all 32
local roundtrips in each condition. All 16 untouched copies per condition
had zero difference. A separate check verified 3,515,958 different samples,
300 displayed ranges, and 600 playable excerpts against the two original
sample clocks. This comparison locates processing or edits between two
supplied recordings. It carries zero weight for AI origin.

With the exact synchronized background track and its original mixing gain,
subtraction reproduced all 224 clean foregrounds sample-for-sample. The
verified identical inputs could reuse the saved clean readings. They found
13 of 32 edit locations with the current voice check and 21 of 32 with the
experimental shorter-window variant, compared with zero for both on the
noisy mixtures. Estimating the gain reproduced only 16 of 224 clean files.
None of the 96 deliberately wrong references reconstructed the clean file.
The exact subtraction trial used access to the original isolated noise
track and controlled lossless mixing. It leaves arbitrary room samples,
separate microphones, and re-encoded copies unqualified for this method.
The follow-up ran no new model inference and leaves serving checks unchanged.

The closed exact-copy receipt has SHA-256
`3f8acae971c00128de1d016aa3649ab6f86a4d991fd2c61ddc3ba5e474e11d1e`.
The broader comparison receipt has SHA-256
`eded1a2010fad63fd73368064252b025977da3c4bb91ad23aecc4d30b22f069c`.
The underlying recordings and per-case source data
remain local.
