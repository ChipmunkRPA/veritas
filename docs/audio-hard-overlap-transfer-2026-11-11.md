# Fresh audio routing test with similar voices

The level-normalized speaker-count check withheld 18 of 181 incorrect three-voice routes on a fresh panel. Its advantage on the earlier controlled panel did not transfer. The unscaled count check withheld 37 incorrect routes and two correct routes. Both left many overlapping recordings eligible for the experimental voice supplement. [Frozen protocol and measured results](evaluation-panels/audio-hard-overlap-transfer-2026-11-11.json)

## Source and design

The fresh panel used VoxPopuli test recordings in Estonian, Finnish, Lithuanian, Dutch and Slovenian from the official Parquet conversion at `db0a600a0776f910769e9dc1fdd7f218e8f78a45`. The dataset publisher licenses the data under CC0 and links to the European Parliament's terms for the source recordings. All mixes were made locally. The public evidence contains file commitments and aggregate counts. Individual speeches, speaker IDs and derived per-recording information remain local. Pretraining overlap is unknown. [Corpus paper](https://aclanthology.org/2021.acl-long.80/) · [Publisher data terms](https://github.com/facebookresearch/voxpopuli#license)

Qualify the center three seconds from at most five salted source utterances per speaker. Choose the first qualified utterance under the frozen salted ordering. Use exact mono 16 kHz signed PCM and the previously verified activity model. Require at least 1.75 seconds of activity per separate voice. Form disjoint triplets in the same language and published gender stratum, with pitch estimates within four semitones and at least 0.80 seconds of joint dry speech activity. This makes close-pitch mixtures a controlled stress case. Estimate pitch using normalized autocorrelation, independently of the two count methods being tested. [Earlier speech perception study](https://pubmed.ncbi.nlm.nih.gov/11757942/) · [Prior count experiment](audio-independent-overlap-count-2026-11-11.md)

The five partitions contained 122 candidate speakers, of whom 119 had qualified clips. The frozen quotas and triplet rule selected 63 different speaker IDs in 21 triplets. Each triplet produced 24 views. Rotate the three source roles. Compare two voices and three voices with the third at 0, −9, and −15 dB relative to the other interferer. Hold the foreground 5 dB below the combined interference. Repeat all roles with fixed short reflections applied to each source. Reflections are simulated. The selected clips shared at least 1.56 seconds of separate-source activity before mixing. A single clipping-safe gain applies across the 24 views of each triplet.

Replay both diarization operating points on every mixture. The comparison considers a two-voice route only when both agree. Test the earlier unscaled explicit count veto and the −24 dBFS normalized count, using the same model and fixed five-second center placement as the prior experiment. The stronger normalized rule withholds both zero counts and counts of at least three. The frozen requirement asks for evidence from at least five triplets with baseline mistakes, at least a 50% reduction in mistaken three-voice routes, at most one affected triplet remaining, and retention of at least 80% of the correct two-voice routes. It is a routing comparison. No authenticity score is measured on this all-genuine-source panel.

## Results

| Routing rule | Incorrect routes on three-voice views | Correct routes on two-voice views | Triplets with any incorrect route |
| --- | ---: | ---: | ---: |
| Diarization alone | 181/378 | 124/126 | 20/21 |
| Unscaled count veto | 144/378 | 122/126 | 19/21 |
| Normalized explicit count veto | 163/378 | 124/126 | 20/21 |
| Normalized count veto including zero | 163/378 | 124/126 | 20/21 |

The normalized rule retained all 124 correct routes admitted by the baseline. It reduced mistaken routes from 84 to 80 in dry audio and from 97 to 83 with simulated reflections. At −15 dB, the normalized rule left 76 of 80 baseline mistakes. The unscaled rule recovered more mistakes overall and still left 19 triplets with an incorrect route. This transfer has a deliberately high difficulty and repeated views within each source triplet. Its rates do not estimate the prevalence of overlap or audio detection errors in uploads. The voice supplement stays in evaluation.

## Independent reproduction

The [runner](../backend/scripts/audio_hard_overlap_transfer.py) binds all five official Parquet files, the prior commercially reviewed count and diarization assets, and each source waveform before scoring. From the project root, place the five test Parquet files in a fresh directory as `et-pinned.parquet`, `fi-pinned.parquet`, `lt-pinned.parquet`, `nl-pinned.parquet` and `sl-pinned.parquet`. Download from the pinned [VoxPopuli revision](https://huggingface.co/datasets/facebook/voxpopuli/tree/db0a600a0776f910769e9dc1fdd7f218e8f78a45). Each relative archive path is `<language>/test/0000.parquet`. Compare their SHA-256 digests with `plan.sources` in the published receipt. Place the pinned diarization archive as `runtime.tar.gz` and weights as `model.gguf` in the same run directory. Use the extracted executable, the pinned speech-activity weights and the previous CountNet assets. The count stage needs PyTorch and `h5py`. The source stage needs PyArrow, SciPy and SoundFile.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1
python -m scripts.audio_hard_overlap_transfer prepare "$run" --assets "$vad"
python -m scripts.audio_hard_overlap_transfer freeze "$run" --assets "$count_assets"
python -m scripts.audio_measured_speaker_routing freeze "$run" "$diarization_binary"
python -m scripts.audio_measured_speaker_routing run "$run" "$diarization_binary"
python -m scripts.audio_hard_overlap_transfer count "$run" --assets "$count_assets"
python -m scripts.audio_hard_overlap_transfer readout "$run" --assets "$count_assets"
```

The CPU replay completed 1,008 count and 1,008 diarization forwards. A separate replay verified the 63 source members against the original archive bytes, reconstructed all 504 mixtures exactly, and checked every diarization interval grid, score distribution, and summary total.

The [standard-library verifier](../backend/scripts/review_audio_overlap_transfer.py) checks the local PCM hashes, both inference freezes, exact 10 ms interval arithmetic, and the advancement decision. Supply the published receipt to cross-check its results against the verified local aggregate. The tool needs Python 3.11 or newer and reads the local run without writing or loading models. A fresh inference replay measures model behavior again.

```sh
python3 -I backend/scripts/review_audio_overlap_transfer.py "$run" \
  --published docs/evaluation-panels/audio-hard-overlap-transfer-2026-11-11.json
```
