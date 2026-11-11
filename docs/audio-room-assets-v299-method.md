# Measured stereo room response qualification

v299 binds one development room and a candidate second-room subset for the
engineered stereo simulation designed in v296. It reads only explicitly
selected local impulse response files. It does not render speech or noise,
run a detector, modify product behavior, or download any file. Results are
written to standard output and contain metadata and hashes without samples.

## Development room

The University of Surrey measurement guide documents 48 kHz acquisition through
a Cortex MKII dummy head and 24-bit interface. The head used diffuse-field
equalization. The playback chain included loudspeaker time alignment, level
and frequency equalization, and bass management. Select `angle_0.0.mat` from
commit `d36af1bbd0f4083a9f9ee678ff8f499044e851a7`. The file is 14,282,074 bytes
with Git blob SHA-1 `9214e19a94dca7609c9e8aad4805bc277f5b33a6`. The publisher
releases the measured data under CC BY 4.0. [Publisher dataset and attribution](https://doi.org/10.15126/surreydata.00813511) [Pinned technical guide](https://github.com/IoSR-Surrey/IoSR_ListeningRoom_BRIRs/blob/d36af1bbd0f4083a9f9ee678ff8f499044e851a7/IoSR_ListeningRoom_BRIR.pdf) [Pinned MATLAB asset](https://github.com/IoSR-Surrey/IoSR_ListeningRoom_BRIRs/blob/d36af1bbd0f4083a9f9ee678ff8f499044e851a7/IoSR_ListeningRoom_BRIRs_MATLAB/angle_0.0.mat)

The guide defines `brir[samples, microphones, loudspeakers]`. Microphone
columns are left and right. Its Table 2 names source channels 3, 11 and 12 as
front centre at 0°, side left at +90° and side right at −90°. All are at 0°
elevation. The source channel numbering is one based. Channels 4 and 10 are
subwoofers and excluded. The publisher example extracts the response matrix
directly from `brir`. Preserve sample zero and all embedded offsets, with zero
additional delay in v296. This is the stored coefficient clock. The absolute
delay from original loudspeaker excitation is outside the qualification.

Verify the publisher blob before inspecting the container. For MATLAB Level 5,
require a single numeric `brir` variable with the documented axes, bounded
sample length and finite, real floating-point coefficients. Preserve each
selected channel without normalization or time alignment. Emit a distinct
coefficient hash for every mapped position. A Level 7.3 HDF5 container receives
only byte qualification until a compatible parser is verified.

## Candidate holdout room

The MAIR conventional stage microphones face a frontal group of speakers. The
separate 360° setup at St Paul's Hall has eight identical full-range speakers
spaced by 45° around a 3 m circle. It supports the intended physical front and
lateral positions with a fixed central Neumann KU100 dummy head. Its 360°
paper uses clockwise-positive azimuth. This differs from Surrey's
counterclockwise-positive convention. The MAIR guide reports 44.1 kHz, 32-bit
acquisition. The publisher v1.1 project and measured WAVs are CC BY 4.0.
The cited publications and any separately sourced renderer assets carry their
own terms. [Publisher release](https://zenodo.org/records/2639677) [Pinned technical guide](https://github.com/APL-Huddersfield/MAIR-Library-and-Renderer/blob/ef37a42552e6b17d94611856cbd3404e308e10f5/MAIR%20Technical%20Descriptions%20and%20User%20Guide.pdf) [360° study, Section 3 and Figure 2](https://pure.hud.ac.uk/ws/portalfiles/portal/14022860/AAM_AES2018_Millns_and_Lee.pdf)

Use only the six files in `360StageArrays/KU100-0deg` at commit
`ef37a42552e6b17d94611856cbd3404e308e10f5` with `S01`, `S03` or `S07` and the
corresponding `M01`/`L` and `M02`/`R` labels. Each is 529,280 bytes. The
program's `plan` mode lists exact immutable URLs and Git blob SHA-1 hashes.
Check those bytes, the original mono header, the actual 32-bit storage subtype,
the equal pair sample counts and distinct coefficient identities. Accept WAV
32-bit IEEE float coefficients directly or decode signed PCM32 to its linear
full-scale ratio. A filename proves the publisher's source ID. The available
guide and paper leave the mapping between the numbered files and physical angles
unresolved. The holdout reports the bytes and format for candidate pairs with
an empty qualified-position map. A geometry receipt requires a later review
that establishes the exact mapping and timing from source evidence without
reading an authenticity score. [Pinned WAV directory](https://github.com/APL-Huddersfield/MAIR-Library-and-Renderer/tree/ef37a42552e6b17d94611856cbd3404e308e10f5/360StageArrays/KU100-0deg)

## Read-only commands

Run from the project root. `plan` reads the exact prior authored receipt and
code hashes. `development` and `holdout` read only their role's direct files
from the supplied directory, with exact basenames and no symlink following on
the files. Record a separate SHA-256 of the verified bytes alongside the
publisher blob identity. Keep local directories outside the public repository.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_room_assets_v299 plan
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_room_assets_v299 development --asset-dir /path/to/surrey
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_room_assets_v299 holdout --asset-dir /path/to/huddersfield
```

Scope-limited intake verifies reproducible source bytes, format and documented
geometry where available. It does not establish natural stereo detection,
physical timing accuracy or the per-channel signal-to-noise ratio of a future
room-rendered speech panel.
