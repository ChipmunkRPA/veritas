# Prospective in-envelope video transfer study

Under its default slice policy, the frozen production video fuser 2.1.1
admits clips with a reported frame height of at least 720 pixels and a container duration from 4 through 60
seconds, across all four face-size buckets. A fifth leaf admits clips over
60 seconds with a sampled face at least 256 pixels long. All five use pooled
video validation with AUC 0.9613. The leaf with a face from 120 to under 256
pixels and a duration from 4 through 60 seconds contains 18 real and 12 fake
examples. The corresponding larger-face leaf has 88 real and 16 fake examples.
The runtime takes the first video stream's reported height and the largest
rounded face-box side across its sampled detections. Record the actual profile,
fuser hash and envelope status for every result. See the [frozen artefact](../backend/app/engine/fusion/release/fuser-video-2.1.1.json)
and [slice rules](../backend/app/engine/fusion/abstain.py).

All 32 GRID sources have 360×288 pictures, with 75 observed native PTS across a
2.96-second first-to-last span. The pilot's approximately three-second remuxes fall
below the 480-pixel floor and outside the admitted duration and resolution
combinations. Keep the 16 GRID pilot and 16 reserve recordings in their existing
roles. The first serving batch completed with partial check coverage and no
overall customer direction. Preserve that evidence as the short-input result.
[GRID plan](grid-transfer-plan-v143-method.md) · [Serving receipt](grid-serving-baseline-v154-partial-coverage.md)

For an engineering smoke, use one previously consumed fitting group from the
audited HD visual bank, with its original, both swap variants and ordinary
controls. The bank has 90 compatible groups and 540 native 720p or 1080p
clips of approximately six seconds. The RTFS publisher declares CC BY-SA 4.0,
and its recorded portrait attribution has gaps. Admit the chosen files only
after the intended use is supported by their rights records. Preserve the
24 reserved confirmation groups for the existing comparison. The bank is
silent, so the smoke exercises visual routes and the serving disposition of
absent audio. Record cache, activation budget and other coverage failures.
[Completed HD bank](video-native-inputs-complete-v108.md) · [Rights record](video-common-source-preparation-v89.md)

A fresh visual comparison needs HD source and donor groups with known edit
boundaries. Keep original, both swap variants and ordinary controls together
by source and donor ancestry. A separate speaking-video comparison needs
genuine HD footage with speech and corresponding neural mouth redraws. The
48 previously qualified CREMA-D recordings are 480×360. Per-file licensed
Commons footage and FineVideo are leads for screening. Verify creator and
source, attribution, picture and soundtrack rights, derivative permissions and
applicable likeness permissions for each selected file. Record asset-specific
generator rights separately. See the [research and rights review](video-transfer-research-and-rights-20261110.md).

Start the speaking comparison with 32 new recording groups, 16 for the pilot
and 16 for confirmation. Keep connected speakers, sessions, source recordings
and known derivatives on one side. Select native 720p or 1080p excerpts of
6–7.2 seconds at observed 25 Hz, with contiguous speech and picture timestamps
and audio origins on exact milliseconds. These bounds fit the current mux's
200-picture limit. Match the
eleven GRID conditions, including both generators, both conditioning offsets,
branch-matched brightness and resampling, genuine retiming and the audio-only
delay. Pair lossless and H264 encodings for 704 planned cases. Keep actual
audio/video origins, edit PTS, native masks and all unavailable outcomes. The
existing GRID wrappers assume 75 pictures. The HD cohort needs its own frozen
input and construction context. [Mux contract](../backend/scripts/native_av_mux_v119.py)

Report the serving envelope, evaluated scope and actual customer direction
alongside appearance localization and audio/picture timing. The frozen fuser
lists visual, codec and legacy voice inputs. The current speech candidate is
reported separately; verify the actual contributions in each result. The
timing diagnostic and candidate mouth heads have separate outputs. Compare false flags, missed observable edits,
inconclusive cases, temporal overlap and flagged time on unchanged tracks.
Keep codec, source and component dependencies in the paired analysis. The
public result can contain aggregate outcomes and whole-artifact hashes. The
source records, media and per-case evidence stay in the private inventory.
