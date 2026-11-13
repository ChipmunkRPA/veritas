# Testing a separate background reference for noisy voice analysis

## What caused the extra flags

The earlier cleanup rule keeps the native mono result and can add an AI flag
when the same voice classifier agrees on both a dereverberated and a
frontal-cancelled view. Across the held ODSS and PhonemeDF ledgers it added
eight genuine flags. Every flagged genuine mixture's room-only counterpart
scored below the native threshold. Every exact competing-speech segment on its
own scored below all three thresholds wherever those views were admitted.
The background mixed with the foreground raised the median native score by
12.43 logit units for the five genuine additions in ODSS and 11.72 for the
three in PhonemeDF. The corresponding medians for added synthetic flags were
−0.78 and −1.07. The pattern supports a mixture interaction. These scalar
comparisons cannot isolate which acoustic features caused the score change.
[ODSS comparison](audio-odss-matched-room-2026-11-13.md)
· [PhonemeDF comparison](audio-phonemedf-matched-transfer-2026-11-13.md)

## Frozen follow-up

We added one separate comparison view. From the submitted stereo signal,
estimate a complex mid-to-side regression for each frequency bin using only
frames wholly within the first and last second. Both are background-reference
flanks in these test recordings. Apply that estimate to the whole submitted
signal and reconstruct the frontal residual on the original clock. The fit
uses the earlier frontal method's exact Hann window, floor, coefficient cap
and subtraction amount. The foreground enters the middle three seconds,
and its room response can extend beyond the dry utterance. The tested geometry
assumes usable background-reference flanks. Arbitrary continuous speech,
moving foregrounds, and unverified background-only intervals fall outside
this design.

All previous operating points were fixed. The new point is the maximum
admitted flank score over 16 genuine mixtures and 16 corresponding
background-only controls from the **ODSS development** role. The held inputs
from both previous studies enter the retrospective diagnosis only. The
new rule retains every native mono flag. It accepts a previously proposed
selective addition only when the flank view also passes the development
point.

The fresh test uses eight complete English VCTK utterance pairs from four
readers. Each pair matches a natural and publisher-generated VITS recording
by exact ODSS identifier and normalized transcript. The four declared
readers were excluded from the local voice calibration, matched-babble and
PACC source rosters, and are disjoint from the LibriSpeech IDs in the
PhonemeDF test and the Hi-Fi TTS IDs in the first ODSS test. Hash and
transcript checks exclude exact prior ODSS audio and prompt reuse. Detector
training exposure and undocumented speaker overlap are unknown. The archive
was accessed through bounded publisher byte ranges, and each selected WAV
passed ZIP CRC and native PCM verification. No audio was generated.
[ODSS release, CC BY-SA 4.0](https://zenodo.org/records/8370669)
· [VCTK source release](https://doi.org/10.7488/ds/2645)
· [Commercial classifier asset review](commercial-model-audit.md)

The assessment uses three measured paths from the RWTH Aachen AIR 1.4
meeting room. Position 4 serves as the balanced target path, with both
strongest peaks at sample 119 after resampling to 16 kHz. Positions 1 and 2
serve as the opposing background paths, with interaural peak differences
of +11 and −12 samples in the recorded channel order. The room is distinct
from Surrey and the BBC. Eight nonoverlapping AMI captures supply competing
speech across two disjoint meetings previously used in other tests. Each
exact interval excludes earlier directional-room and gain-replay intervals
and has at least 3.9 seconds of annotated speech, 350 ms of simultaneous
speech, and 300 ms of speech in each one-second flank. Every pair shares
its exact masker, direction and gain at 0 and −5 dB. Separate room-only and
background-only controls retain the same source and gain.
[Measured impulse archive and MIT terms](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/)
· [Measurement paper](https://www.iks.rwth-aachen.de/en/publications/publications-detail//jeub09a/)
· [AMI license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

## Held results

| Frozen route | VITS flags | Genuine flags | Background-only flags |
| --- | ---: | ---: | ---: |
| Native mono | 2/16 | 0/16 | 0/16 |
| Earlier selective cleanup | 2/16 | 0/16 | 0/16 |
| Selective cleanup plus flank corroboration | 2/16 | 0/16 | 0/16 |

The two flags came from the same prompt at both noise levels. All 48 noisy
or background-only deliveries passed the native speech gate. Only two of
the 16 room-only controls passed the same gate. The VITS control was flagged
and the genuine control was clear of the operating point. The earlier
selective rule proposed zero new flags in this room, so the flank requirement
had no decision to corroborate or withhold. The study therefore gives no
estimate of the flank rule's ability to reject the genuine mistakes seen in
the previous room.

The flank-only score used an operating point of 6.108575 log-odds units.
The acoustic comparison measured recovery against the known room-rendered
foreground. At −5 dB, the flank estimate improved 11 of 16 mixtures, with
median SI-SDR gain of +0.28 dB. At 0 dB it improved three of 16, with a
median loss of 0.83 dB. Whole-mixture frontal cancellation had median losses
of 0.40 and 1.57 dB, respectively. The acoustic improvements observed in the
earlier BBC room did not transfer consistently to this room. Four readers,
one synthesis family, fixed flanks, and one measured room limit the result.
The customer score stays unchanged. [Anonymous scalar receipt](evaluation-panels/audio-flank-background-transfer-2026-11-13.json)

## Replay

The runner expects the earlier pinned ODSS and AMI caches, the qualified
Surrey and BBC response paths, the AIR 1.4 archive, and the retained local
Aachen and room stress rosters. Freeze 32 development and 64 assessment
deliveries before model inference. Fit and seal the point before any
assessment inference. Public results contain hashed commitments and scalar
scores. The source audio and reader identifiers remain local. The local run made
32 development and 190 assessment classifier forwards. The waveform
replay reproduced all 96 deliveries and their comparison-view hashes.
A separate Python standard-library reader verified the decision counts and
recomputed the earlier diagnosis from the two published ledgers.

The 16 new ODSS range requests are deterministic. Use the script's
`source_plan` to obtain the entries from the prior ODSS ZIP metadata.
For each selected entry, the exact first byte is its ZIP local-header
offset. The byte count is `room.odss.HEADER.size + len(entry.filename) +
1024 + entry.compress_size`. Save each exact HTTP 206 response as
`<offset>.range` after checking its `Content-Range`. The local runner
independently checks ZIP header identity, stored length, and CRC.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VERITAS_OFFLINE=true
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze fit score replay; do
  backend/.venv/bin/python -B -m scripts.audio_flank_background_transfer "$phase" \
    --odss-cache "$odss_cache" --vctk-cache "$vctk_cache" --surrey "$surrey_room" \
    --bbc "$bbc_room" --ami "$original_ami" --fresh-ami "$fresh_ami" --air "$air_zip" \
    --prior-aachen "$prior_aachen" --prior-stress "$prior_stress" --run "$run" || exit 1
done
python3 -B -S backend/scripts/review_audio_flank_background_transfer.py "$receipt" \
  docs/evaluation-panels/audio-odss-matched-room-2026-11-13.json \
  docs/evaluation-panels/audio-phonemedf-matched-transfer-2026-11-13.json
```
