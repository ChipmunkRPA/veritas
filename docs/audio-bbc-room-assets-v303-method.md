# BBC measured stereo room responses for holdout

The BBC Research & Development listening room in Salford is a separate physical
room from the previously qualified University of Surrey development room. The
BBC white paper documents a centered Neumann KU100 dummy head. The publisher's
README specifies its frontal orientation for the static stereo WAVs. Table 1
of the white paper identifies three
full-range source positions at 0° elevation. `M+000` is front centre at 0° and
1.99 m. `M+090` is side left at +90° and 2.28 m. `M-090` is side right at
−90° and 2.28 m. The publisher's scene configuration confirms the geometry,
and its static WAV routing names the three files. [Measurement paper](https://downloads.bbc.co.uk/rd/pubs/whp/whp-pdf-files/WHP382.pdf) [Source position scene](https://github.com/bbc/bbcrd-brirs/blob/c88a293ed81f4aa820bcbc4138f951552819e001/ssr-scenes/BBCRDLR_BRS_SystemH.asd) [Static WAV routing](https://github.com/bbc/bbcrd-brirs/blob/c88a293ed81f4aa820bcbc4138f951552819e001/ambix-configs/BBCRD_LR_FULL.config)

Use the three static WAVs at the corrected BBC repository commit
`c88a293ed81f4aa820bcbc4138f951552819e001`. The 2017 measurement paper's dataset
section used CC BY-NC-SA for the SOFA release. The BBC corrected the repository
README in 2020 to match the repository's existing CC BY-SA 4.0 license. The
original repository data commit added the license and the static WAVs together.
The qualification covers only the pinned static repository WAVs. Record
attribution to Chris Pike, Michael Romanov and BBC R&D, and apply share-alike
terms to any later publication of adapted audio. [Publisher correction](https://github.com/bbc/bbcrd-brirs/commit/c88a293ed81f4aa820bcbc4138f951552819e001) [Pinned license](https://github.com/bbc/bbcrd-brirs/blob/c88a293ed81f4aa820bcbc4138f951552819e001/LICENSE) [Initial data commit](https://github.com/bbc/bbcrd-brirs/commit/e044c45fadddb6432136f79c927b85136de85536)

| Position | Static filename | Bytes | Publisher Git blob SHA-1 |
| --- | --- | ---: | --- |
| Front | `BBCRD_LR_M+000.wav` | 131,160 | `0fc8a92eb335e372d91eeb28f00e23dd5eac5d43` |
| Left | `BBCRD_LR_M+090.wav` | 131,160 | `a5307acc7f13da3c24a85487262e59adcb58995b` |
| Right | `BBCRD_LR_M-090.wav` | 131,160 | `650f383bfac67c7bdf93c5454c4aaf1d005dbb2b` |

Verify the exact local bytes before decoding. Require a native 48 kHz stereo
WAV with 16,384 samples per ear, and verify its actual 32-bit linear storage
subtype. Use ordinary stereo left/right order. The publisher aligned the
speaker drive to a 2 m reference distance and compensated measurement-chain
latency. Preserve all response samples from zero, including the embedded ear
offsets and reflections, with zero additional delay. This preserves the
publisher's coefficient clock. It does not recover an uncompensated physical
propagation time. Report each ear's strongest sample index for inspection.
This index can differ from the first arrival.

The read-only command lists exact immutable asset URLs when called without
`--asset-dir`. Supply a private folder containing just the three exact WAVs
for qualification. The result contains file and coefficient hashes and native
clock information. It contains no rendered speech, detector scores or raw
coefficients.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_bbc_room_assets_v303
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_bbc_room_assets_v303 --asset-dir /path/to/bbc-static
```

The v299 Surrey and v296 authored receipts remain the prior records. The BBC
files qualify a second room for a future engineered stereo comparison. Neither
room study evaluates naturally recorded stereo speech on its own.
