# Prepare original Turkish speech from RUEG

Use the first-party RUEG Corpus 1.0 Turkish WAV archive for a new local
genuine speech cohort. The deposit declares CC0 1.0. The publisher records
participants from Turkey and heritage-language communities and documents
the pseudonymous speaker codes and spoken registers. Fix 24 distinct
monolingual participants in Turkey, one original formal spoken recording
per participant. These are elicited descriptions, with age unavailable in
the checked audio directory. Group separation applies to publisher
participant codes. It establishes no individual identity or model
pretraining exclusion. [Publisher deposit](https://zenodo.org/records/11234583) ·
[Publisher rights metadata](https://api.datacite.org/dois/10.5281/zenodo.11234583) ·
[Corpus annotation and speaker-code documentation](https://korpling.german.hu-berlin.de/rueg-docs/v0.4/print.html)

Before acquisition, fetch only the exact 22-byte ZIP footer and 43,645-byte
central directory. A bounded first-party metadata check already returned
206 for both, with 392 entries and one formal WAV for each of 67 distinct
participants matching the planned scope. The directory starts at byte
3,038,331,448 of 3,038,375,115. Its SHA-256 is
`26c784410cf41501d67308166052cbb0829b614cc810bc166246e7b7ef7d3b60`.
Sort the fixed eligible participant codes by SHA-256 under the v246 study
namespace and keep the first 24. The frozen plan contains 157,768,001
compressed bytes and 200,205,096 expanded bytes, excluding ZIP headers.
The stage commits to the entire private plan and keeps member paths,
offsets, participant codes and per-member checksums private.

Request each selected local ZIP header and the corresponding name, extra
field and compressed member. Check the exact first-party `Content-Range`,
the frozen archive total, response length and identity encoding. Verify
the local header against the central directory before requesting the
compressed range. Verify the exact member name, DEFLATE end, expanded
length and publisher CRC before keeping the original WAV and its local
SHA-256. The phase uses at most 48 serial requests and 160 MB of requested
payload, spaced by at least three seconds between starts. An access limit
or unexpected partial response ends the phase and leaves later slots
unattempted. The archive-level published MD5 describes the full archive.
This intake verifies the pinned directory and selected members only.

Qualification operates locally. Verify the native WAV's complete
declared sample payload, exact production mono 16 kHz decode and
continuous source clock. Accept one or two source channels and at most
300 seconds of source PCM. Apply the existing v199 activity and pause
criterion to choose one three-to-four-second crop. An already short
source keeps its complete PCM if it passes the rule. Verify the saved
crop, source-time mapping and exact byte or PCM overlaps against the
bound v58 project catalogue and accepted members. Preserve each failed
slot. The activity proxy measures level and quiet boundaries. It does
not identify speech, classify speakers or score authenticity.
[Existing crop method](audio-voxforge-tr-genuine-v199-method.md)

Use the project environment from the repository root. The local workspace
and Seagate archive must have the existing 3 GiB disk reserve.

```sh
export PYTHONPATH=backend
backend/.venv/bin/python -B -m scripts.audio_rueg_genuine_v246 preflight
backend/.venv/bin/python -B -m scripts.audio_rueg_genuine_v246 freeze
backend/.venv/bin/python -B -m scripts.audio_rueg_genuine_v246 acquire
backend/.venv/bin/python -B -m scripts.audio_rueg_genuine_v246 qualify
backend/.venv/bin/python -B -m scripts.audio_rueg_genuine_v246 readout
```

Publish only aggregate outcomes and whole-artifact commitments. The source
recordings, group keys and per-recording timing and hashes remain private.
No inference or confidence calibration occurs in these stages.
