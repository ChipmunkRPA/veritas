# Preserve native alignment iterator lifetimes

The first run exited with SIGSEGV before recording an observation. The host
crash report located the fault in `ps_alignment_iter_children`. The caller
retained word objects after advancing their native iterator, then tried to
iterate over their children. Copy each word's scalar fields and all its phone
fields before advancing to the next word. [PocketSphinx iterator implementation](https://raw.githubusercontent.com/cmusphinx/pocketsphinx/v5.0.4/cython/_pocketsphinx.pyx)

The continuation binds the first trial and its empty journal. It uses the same
roster, verified wheel, publisher intervals, PCM, blind transcription and
scoring. Run `python -m scripts.grid_phone_alignment_v162 freeze` and then
`python -X faulthandler -m scripts.grid_phone_alignment_v162 run` with the
backend environment. [Original protocol](grid-phone-alignment-v161-method.md)
