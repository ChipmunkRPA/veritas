# Compare short voice replacements with an earlier recording under noise

The standalone two-second speech review found none of four known
replacements when competing speech was 5 dB louder. An earlier copy of the
same recording supplies a different kind of evidence. Compare waveform
continuity across the two copies to localize where the recorded speech
changes. Ordinary edits can produce the same observation. The comparison
tests a recording relationship and carries no AI-confidence weight.
[Standalone test](audio-phoneme-short-edit-score-2026-11-13.md)
· [ENFSI audio authenticity methods](https://enfsi.eu/wp-content/uploads/2026/01/FSA-BPM-002_Digital-Audio-Authenticity-Analysis-FIXED.pdf)

## Frozen procedure

The input is two six-second, already aligned 16 kHz mono recordings. Each
has different interfering speech. Band-limit both recordings to 250–3,800
Hz, remove local DC, and measure normalized waveform correlation and
least-squares reference gain independently. Use 240 ms Hann-weighted
windows every 80 ms and exclude windows with insufficient earlier-copy
energy. Require broad positive correlation across the two files before
searching for a local difference. The two cue values are normalized by
their own 70th percentiles. A proposed 720 ms range has weaker matches
than both adjacent flanks. Fit each cue's cutoff from the maximum result
on the ordinary development controls before scoring the held pairs.

The two development readers come from the earlier short-edit panel.
The three held readers are outside all four reader groups in that panel.
The source selection also excludes the earlier whole-utterance PhonemeDF
and matched ODSS readers and the exact source IDs in the historical
calibration plan. Historical reader exposure and model training overlap
remain possible. The renderer uses matched publisher word tiers to build
text-to-speech and voice-converted replacements. An ordinary different-
performance replacement uses another passage from the same genuine
recording and should also produce a change. Ordinary controls include
untouched speech, a local gain change, two versions of same-speech
resampling and a full-recording equalizer. Controls for reference
eligibility use a different reader or a separately generated reading of
the same words. [PhonemeDF data and CC BY 4.0 terms](https://zenodo.org/records/19522979)
· [Same-word preparation](audio-phoneme-short-edit-intake-2026-11-13.md)

Four AMI meetings contribute nonoverlapping six-second captures from
16 disjoint participants. The earlier copy is mixed at +5 dB and the
submitted copy at 0, −5 or −10 dB. Each pair contains at least 250 ms of
simultaneous annotated competing speech inside its edit in each recording.
All matched versions in a scene share the exact masker and gain. The
recordings have earlier project exposure. The separate backgrounds and
newly authored mixes test controlled additive overlap. The test keeps
the original equal sample clock and uses no room response, timing drift,
lossy encoding or arbitrary initial alignment. [AMI source and terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Held outcomes

| Submitted target-to-background ratio | Correlation synthetic / ordinary locations | Correlation control flags | Projection synthetic / ordinary locations | Projection control flags |
| --- | ---: | ---: | ---: | ---: |
| 0 dB | 2/6 · 1/3 | 6/15 | 2/6 · 2/3 | 9/15 |
| −5 dB | 2/6 · 1/3 | 6/15 | 2/6 · 1/3 | 9/15 |
| −10 dB | 2/6 · 1/3 | 10/15 | 2/6 · 1/3 | 9/15 |

The existing exact PCM comparison marked all 300 bins in all 90 held
pairs, so it located zero short replacements. Both new routes admitted
all 72 same-recording pairs and declined all 18 unrelated or regenerated
references. At −5 dB, normalized correlation located **4.000–4.720 s**
for both synthetic versions of a replacement at **3.825–4.535 s**.
Projection found **2.640–3.360 s** for both versions of another
replacement at **2.565–3.435 s**. Each route also reported several
misplaced ranges on actual replacements.

The high number of control flags includes untouched speech. A local
difference between independent background tracks can mimic the same
deficit even with an eligible reference. This panel supports checking
pair eligibility and supplies examples of useful location, but does
not justify enabling the localization rule. No app behavior changes.
[Anonymous pair receipt](evaluation-panels/audio-paired-speech-continuity-transfer-2026-11-13.json)

## Replay

Use the [earlier PhonemeDF source-cache instructions](audio-phoneme-short-edit-intake-2026-11-13.md#replay)
and the original 16 kHz AMI microphone WAVs. The two additional
assessment meetings are `ES2007c` and `IS1004c`, as used in the
[earlier room-transfer test](audio-wpe-fresh-room-transfer-2026-11-13.md).
Keep the recordings and source identities local. The public receipt
retains scalar measurements and waveform commitments. Native replay
reconstructs all 150 pairs. The standard-library reader checks the
decisions, relative flanks and interval overlaps.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze fit score replay; do
  backend/.venv/bin/python -B -m scripts.audio_paired_speech_continuity_transfer "$phase" \
    --cache "$new_cache" --old-cache "$old_phonemedf" \
    --record docs/evaluation-panels/audio-phonemedf-release-19522979.json \
    --odss-cache "$odss_cache" --ami "$ami_sources" --fresh-ami "$assessment_ami" \
    --intake docs/evaluation-panels/audio-phoneme-short-edit-intake-2026-11-13.json \
    --run "$run" || exit 1
done
python3 -B -IS backend/scripts/review_audio_paired_speech_continuity_transfer.py "$receipt"
```
