# Qualify independently recorded backgrounds from Wikimedia Commons

This intake fixes four kitchen and three roadside recordings for a future
matched speech experiment. The [Wikimedia Commons](https://commons.wikimedia.org/)
records name a distinct author and recording for each file. Three files list
CC BY-SA 4.0, two list CC0 1.0, and two carry author public-domain grants.
One file supplies its candidate audio through a video container. One file's
recorded author and uploading account differ, which the private source
record preserves for follow-up provenance review.

MediaWiki `imageinfo` supplied exact byte counts and hashes for the
Commons-hosted originals, along with pinned description revisions, upload
timestamps and MIME types. The owner-only local roster preserves those
commitments, exact file order, names, credits and reuse terms. Its SHA-256
is `8b8dc7da941618e5ff7ce6ac47625cc8635684bc1f9f8aabe961e57366d6fbcf`.
Published evaluation materials retain whole-roster and receipt-set
commitments and aggregate counts. Source records, media and individual
audio windows remain in the private working data.

Download the frozen Commons files once, in the listed order, with at most
seven serial requests, a three-second minimum interval between request
starts, a 125 MB file cap, a 225 MB run cap, and a ten-minute run deadline.
The process uses the exact original file routes on `upload.wikimedia.org`,
checks response type, identity encoding, exact declared length, streaming
length and publisher SHA-1 before admitting a file, and records the local
SHA-256. A 403 or 429 ends the run and leaves later slots unattempted.
The first four original routes passed a public unauthenticated HEAD check.
Three additional routes had publisher metadata but returned 429 on an
earlier HEAD batch. The method retains those source slots and any transport
failure in the eventual coverage. The trial and mounted archive each
require the existing 3 GiB disk reserve at stage start.

Run the unchanged production first-audio-stream decoder on each verified
file for a fixed maximum of 120 seconds. Bind the installed FFmpeg and
FFprobe bytes and NumPy version.
Validate the full resulting analysis timeline and read back the mono 16 kHz
signed PCM and hashes. The local receipt preserves exact source and decoded
identities, timing continuity coverage and input/output sample facts. This
is a mechanical preparation stage. It gives no clearance for intelligible
speech, music, acoustic scene fit or suitability for a detector experiment.

For a future listening review, use exact candidate PCM interval bindings
before any detector score. A reviewer checks each proposed interval for
traffic or kitchen activity, intelligible speech and music. Keep only
intervals with a supported scene and neither contaminant. Preserve rejected
or unclear intervals and bind their reason privately. Allocate all windows
from the same original recording to a single role. The intake has zero
auditory admissions and zero model calls.

From the repository root, use the project environment. Run acquisition to
completion before qualification. Each stage keeps its original receipts.

```sh
export PYTHONPATH=backend
backend/.venv/bin/python -B -m scripts.audio_commons_background_v234 preflight
backend/.venv/bin/python -B -m scripts.audio_commons_background_v234 freeze
backend/.venv/bin/python -B -m scripts.audio_commons_background_v234 acquire
backend/.venv/bin/python -B -m scripts.audio_commons_background_v234 qualify
backend/.venv/bin/python -B -m scripts.audio_commons_background_v234 readout
```
