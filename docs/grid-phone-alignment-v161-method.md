# Speech alignment check on the existing GRID pilot

Evaluate the sixteen established original clips in pilot order. Match their
verified 16 kHz mono PCM to the corpus publisher's word alignments by archive
and clip membership. The reference intervals use the 25 kHz source sample
clock. Require the word-alignment endpoint and admitted PCM duration to agree
within one 40 ms video frame. Keep reference text and individual observations
in the local journal. Record missing same-speaker word references as unavailable.
A matching utterance stem under another speaker directory supplies no timing
reference. [GRID corpus record](https://zenodo.org/records/3625687)

Verify the PocketSphinx 5.0.4 macOS wheel against the published SHA-256 and
install it in a private evaluation directory. Check the isolated package
payload against the wheel before importing. Use the bundled US English
acoustic model, dictionary and language model. The project and acoustic
notices permit binary use under their conditions. [Wheel and checksum](https://pypi.org/project/pocketsphinx/5.0.4/#files) ·
[Code licence](https://raw.githubusercontent.com/cmusphinx/pocketsphinx/v5.0.4/LICENSE) ·
[Acoustic model notice](https://raw.githubusercontent.com/cmusphinx/pocketsphinx/v5.0.4/model/en-us/en-us/README) ·
[Dictionary licence](https://raw.githubusercontent.com/cmusphinx/cmudict/master/LICENSE)

Run blind transcription using the default language model, then evaluate exact
prompt matches and word edit distance against the six known words. Separately
run word and phone alignment with the publisher transcript supplied. Compare
its word boundaries against the publisher intervals and count model-reported
bilabial phones as candidate events. Acoustic frame boundaries have 10 ms
spacing. Use separate measurements to validate phone timing, release events
and speaker attribution before using those events for an audio/visual decision.
[Publisher alignment implementation](https://github.com/cmusphinx/pocketsphinx/blob/v5.0.4/cython/_pocketsphinx.pyx)

After installing the exact wheel and retrieving the publisher-checksummed word
alignment archive, run `python -m scripts.grid_phone_alignment_v161 freeze`
and `python -m scripts.grid_phone_alignment_v161 run` with the backend
environment. The result records coverage and aggregate error counts. A
successful known-transcript run still requires blind coverage for live use.
