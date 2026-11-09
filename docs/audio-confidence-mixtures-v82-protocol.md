# Fixed clean and background variants v82

Use the complete qualified v81 manifest, bound by its whole-artifact SHA-256.
Retain all 464 foregrounds and all 48 backgrounds. The corrected connected
source roles remain fitting, confidence calibration, selection and confirmation.
Every foreground is eligible for its fixed first 64,000 production samples:
four seconds at 16 kHz. No VAD, ASR, model output, quality ranking or replacement
source chooses an interval.

Within each role, sort foregrounds and each four-source background pool by the
declared SHA-256 namespace and whole-source hash. Independently within each
truth class, cycle through all four backgrounds of each kind. Thus each role's
56 genuine foregrounds use each background 14 times, and its 60 AI foregrounds
use each background 15 times. This balances background exposure between truth
classes. Only dataset construction uses labels; no classifier receives them.
Choose the background interval start by the fixed hash of foreground raw bytes
and background kind modulo the available four-second start positions. Keep the
selected interval even if difficult. Foreground/background groups must differ.

For every foreground, run the unchanged v7 mixture helper: normalize foreground
RMS to 0.1, scale each selected background to requested SNR, and use a common
peak gain across its complete nine-condition family. Conditions are clean,
noise at 0/5/15 dB, music at 5/15 dB, and overlapping genuine speech at 0/5/15 dB.
Encode through the ordinary signed-16-bit WAV helper and decode each file with
the app's production decoder. Require exactly 64,000 samples and exact equality
with the encoded PCM16 samples. Record requested and actually measured SNR,
common gain, source/background intervals, native and production hashes and full
production timeline. Never treat the normalized clean variant as unprocessed
field audio or the nine correlated variants as independent recordings.

Publish and remotely verify the code, protocol, complete allocation and source
bindings before mixing. Run four role stages sequentially, each with 116
foregrounds and 1,044 variants, under the unchanged corrected observer and
900-second/6-GiB limits. Require initial 6-GiB RAM and 3-GiB disk headroom and
1.5-GiB running RAM/disk reserves on the Mac and mounted Seagate volume.
Preserve failed profiles and partial journals without automatic retry or
overwrite. Finish only if all 4,176 planned identities, truth labels, roles,
conditions, physical bytes and PCM remain intact, with no exact cross-role byte
or PCM alias. Source groups and shared backgrounds remain explicit dependence.

This is deterministic dataset preparation, with **zero permitted detector,
VAD, ASR or confidence-fitting calls**. A separate model/calibration method must
be fixed before neural extraction on these sources. Existing source permission,
historically seen expressive profiles/music and unknown pretraining overlap
remain as documented in v81. Do not publish audio, source IDs/pairings,
transcripts, per-source fingerprints, features or learned artifacts. Publish
only code, aggregate counts and whole-artifact hashes. No serving, payment,
workflow, outreach or hosting change.
