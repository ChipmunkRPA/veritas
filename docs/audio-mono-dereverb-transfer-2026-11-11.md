# Delayed prediction on mono recordings with competing speech

## Method

This experiment asks whether a single-file, single-channel transform can reduce room-induced errors without access to an earlier copy or to physical microphone channels. A variance-weighted delayed predictor estimates a repeatable late component in the waveform. It can suppress late reverberation and can also alter direct speech. An interfering voice is not necessarily predictable from earlier frames. No speaker separation or isolated-background estimate is assumed. [Original WPE research and method](https://www.rd.ntt/cs/team_project/media/signal/wpe/) · [Selective component testing in mixed-authenticity audio](https://arxiv.org/abs/2609.03620)

Apply a 512-sample periodic Hann transform on the native 16 kHz mono grid, advancing by 128 samples. Fit ten delayed complex taps per frequency with 48 ms prediction delay and three variance-weighted least-squares passes. The variance estimate smooths residual power across nine frames and has a relative numerical floor. The subtraction applies 80% of the predicted component between 150 and 6,500 Hz. Synthesize to the exact input sample count. Reject a transformed signal if converting back to signed 16-bit would clip. The check uses no new weights and does not train a detector.

Both decision rules keep the current calibration. The conditioned view must separately pass the existing Full-profile speech gate. The suppression arm withholds an existing AI flag to inconclusive only when the conditioned response loses AI and the raw log odds fall by more than ln(2). The rescue arm changes an existing inconclusive response to AI only when the conditioned response supports AI. Evaluate both arms separately. Advancement requires a gain in both transfer rooms across at least three foreground groups, no change to any clean or native case, and at least six assessed genuine mixes in each room and interference mode. Suppression allows at most two lost synthetic flags. Rescue allows at most one additional genuine false flag and requires gains in at least two generator families. No point was fitted from the results. [Previously reviewed voice checkpoint terms](commercial-model-audit.md)

## Source roles

Replay all 101 exact inputs and frozen mono baseline directions from the earlier paired room trial. The separate transfer has 84 views from seven retained genuine foreground groups and nine retained synthetic groups. Clone, expressive, and stock voice families each contribute three groups. The two source roles share no declared foreground group. Two separate AMI rooms contribute new three-second native mono cuts, one mostly sequential and one overlapping cut per room. The cuts are outside the exact recorded intervals in their earlier harmonic study. Each foreground group supplies one clean input and four −5 dB mixtures. The four unchanged background cuts are genuine controls. The rooms and speech banks have prior project exposure, and upstream training overlap is unknown. [Retrospective roster](audio-paired-stereo-overlap-2026-11-11.md) · [Transfer voice and room source preparation](audio-harmonic-overlap-2026-11-11.md) · [AMI corpus terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

The acoustic audit applies the exact fitted linear filter separately to the known foreground and recorded residual of each constructed mixture. It checks that their transformed samples sum to the conditioned waveform and measures changes in foreground level and foreground-to-background power. These observations explain the treatment on known mixtures. At inference, the method sees only the mixed recording.

## Local results

| Source role and input | Current false AI / genuine | Suppression false AI / genuine | Rescue false AI / genuine | Current true AI / synthetic | Suppression true AI / synthetic | Rescue true AI / synthetic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Retrospective sequential mixtures | 7/16 | 6/16 | 8/16 | 12/18 | 12/18 | 12/18 |
| Retrospective simultaneous mixtures | 9/16 | 9/16 | 9/16 | 13/18 | 13/18 | 14/18 |
| Transfer sequential mixtures | 1/14 | 0/14 | 2/14 | 4/18 | 4/18 | 5/18 |
| Transfer simultaneous mixtures | 5/14 | 5/14 | 5/14 | 12/18 | 12/18 | 12/18 |
| Transfer clean | 0/7 | 0/7 | 0/7 | 9/9 | 8/9 | 9/9 |

All mixtures were assessed by the incumbent. The candidate's speech gate rejected no transformed mixture on a case eligible for a decision change. All 20 native cuts kept their incumbent directions, including the five unassessed cuts. The suppression rule corrected one genuine false flag in each role. On transfer it also withdrew a true clean synthetic flag. The rescue rule added one true flag and one false flag in each role. The gains each came from one room and one foreground group per role. Neither rule met advancement criteria, and both stay experimental.

Across the 132 mixtures, median foreground-to-background power gain by room and condition ranged from 0.07 to 0.41 dB. The median foreground level change in every room and condition stayed within 0.03 dB. The filter changed the component power ratio only slightly even under an artificial dry-foreground setup. This supports using a different separation or representation for competing speech. It does not measure perceptual speech clarity or suggest that a change in authenticity response is caused by reverberation alone.

The completed CPU run made 22,278 speech-activity frame forwards and 249 classifier forwards. It took 177.18 seconds after load and reached 3.00 GB process RSS. An initial launch stopped before inference because the runner and loader both set the process's interoperation thread count. The completed freeze removed that duplicate call and kept the source and comparison method. The separate NumPy and SciPy reviewer reconstructs the analysis and overlap-add transforms, verifies the exact 16 kHz inputs, replays the source and filter arithmetic, and checks the two decision policies. Raw voices and per-case responses stay local. [Aggregate result](evaluation-panels/audio-mono-dereverb-transfer-2026-11-11.json)

## Reproduce locally

From `backend`, use the earlier paired-room run, the harmonic-overlap run and its microphone files, the exact AMI annotation ZIP, the pinned voice and activity checkpoints, and the qualified v169c development foreground manifest. The named source readers are retained with their respective studies. Provide a new output folder.

```sh
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -B -m scripts.audio_mono_dereverb_transfer freeze "$run" --archived "$paired_run" --harmonic-run "$harmonic_run" --ami "$ami" --annotations "$annotations"
python -B -m scripts.audio_mono_dereverb_transfer score "$run"
python -B -m scripts.audio_mono_dereverb_transfer readout "$run"
python -B -m scripts.review_audio_mono_dereverb_transfer "$run" "$paired_run" "$harmonic_run" "$ami" "$annotations" "$voice_manifest" app/detectors/audio/calibrations/local-speech-v3.json
```
