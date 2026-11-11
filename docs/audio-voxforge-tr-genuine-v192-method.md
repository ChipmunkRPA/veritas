# Corrected per-session license check for VoxForge Turkish

The guarded [v188 intake](audio-voxforge-tr-genuine-v188-method.md)
acquired all 15 frozen session archives and declined each one before
decoding. Its parser matched a word in the bundled stock GNU GPL v3
reference and treated it as a restriction on the session. The matching
phrase appears in section 6(c), which governs a specific way of conveying
object code. [GNU GPL v3 text](https://spdx.org/licenses/GPL-3.0-or-later.html#licenseText)

A metadata-only review found a separate explicit GPL-3.0-or-later
`LICENSE` grant and a matching named contributor and Turkish language
declaration in every acquired archive. All 15 also contain a
`gpl_license.txt` reference. The two byte variants normalize to the
same text. Pin both byte digests and the normalized digest. Check the
dedicated grant and README with the original strict parser. A changed
reference, missing session grant, conflicting session terms, or invalid
source metadata still makes that session unavailable.
[Publisher submission format](https://voxforge.org/home/submitspeech/linux/step-3)

Reuse the exact acquired archives, 12 contributor slots, fixed session
and clip order, historical exclusions, and 2.5–4 second native PCM
criteria. Run a new offline data-only preparation under the v108 guard.
Preserve v188's failures and retain each unavailable slot in v192's
denominator. The guarded readout verifies the acquired-source linkage,
selected permission, unique prompt and PCM commitments, and the saved
production waveform. No model, holdout, or new acquisition enters v192.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_genuine_v192 preflight
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_genuine_v192 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_genuine_v192 prepare
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_voxforge_tr_genuine_v192 readout
```

The preflight checks archive metadata and prints aggregate counts.
Individual contributor and recording receipts stay private.
