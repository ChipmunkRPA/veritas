# Speaker enrollment on same-speaker cloned edits

The same-speaker reference preserved more cloned speech than a wrong-speaker reference in both held rooms. It still failed the frozen two-room requirements for genuine speech improvement and generated edit preservation. The audio authenticity checks and customer scores were never run. [Aggregate acoustic result](evaluation-panels/audio-same-speaker-enrollment-transfer-2026-11-12.json)

## Matched comparison

The earlier study enrolled a genuine host whose generated replacement came from a different declared speaker. This comparison uses a publisher-linked clone of the enrolled reader, a separate genuine enrollment utterance, an ordinary same-host edit, and an ordinary other-speaker edit. Six point-role and eight confirmation-role speaker groups produce 42 submitted mono mixtures. All 14 source roles and their speaker ancestry were fixed before any measurement. Enrollment comes from a different chapter from the host. The corpus publisher establishes the clone's declared target speaker. The exact original clone prompt is unavailable, and the study covers one generator family. [Source qualification](audio-same-speaker-clone-sources-2026-11-12.md) · [Earlier enrollment test](audio-enrolled-target-transfer-2026-11-12.md)

Each clip contains six seconds of foreground with one 0.8-second splice. All three alternatives use the exact same edit interval and background and match sample-for-sample outside the edit. The room recording is 5 dB louder than the unedited foreground across the full clip. The 14 AMI capture intervals are new and nonoverlapping with previously used exact cuts. The meetings themselves have prior project exposure. At least two annotated background voices overlap each held edit for 650–800 ms, totaling 6.01 of the 6.40 edited seconds. [AMI terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

The earlier fixed speaker-conditioned spectral mask uses only the submitted mono samples and one clean reference. This run applies it unchanged to a true reference and a different-reader reference, against the submitted PCM baseline. The four generic references are unchanged and do not share declared identities with any target speaker. The mask never uses the exact stems to choose a signal. Separate stem replay measures foreground retention and SNR gain within the known edit. Retention is measured relative to the original foreground. SNR gain is measured relative to the unprocessed mixture.

| Held room | Edits | Genuine same-host SNR gain, true / wrong reference | Generated SNR gain, true reference | Generated retention, true / wrong reference |
| --- | ---: | ---: | ---: | ---: |
| ES2010d | 4 | +0.483 / +0.975 dB | +0.027 dB | −2.643 / −6.756 dB |
| IS1007a | 4 | +0.588 / +0.291 dB | −0.612 dB | −3.572 / −5.494 dB |

The true reference beat the wrong reference on six of eight genuine same-host controls. The frozen gate required median genuine SNR improvement of at least +0.5 dB and generated retention of at least −3 dB in each held room. One room failed each requirement. This is an acoustic measurement and provides no fixed or equal-false-alarm authenticity comparison. It leaves original PCM available and supplies no basis for replacing the current detection route.

## Replay

Native audit regenerated all 42 waveforms, 18 enrollment inputs, and the acoustic outcomes. A separate standard-library review rehashed the original PCM and stems, checked matched edit clocks and raw samples, recomputed the reported medians, and verified the exact recorded word overlap. This receipt uses the original locally retained qualification and room archives and contains aggregate results only.

```sh
run=backend/data/independent-evaluation/audio-same-speaker-enrollment-2026-11-12
export PYTHONPATH="backend${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_same_speaker_enrollment_transfer audit "$run"
```

To reconstruct a separate receipt, use a new directory and run `freeze`, then `measure`, before `audit`. [Acoustic comparison runner](../backend/scripts/audio_same_speaker_enrollment_transfer.py)
