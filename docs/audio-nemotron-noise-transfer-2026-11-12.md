# Compare two speech enhancers under loud restaurant and road sound

Earlier tests showed that recovering audible speech can improve coverage while
making genuine and synthetic voices harder to distinguish. This comparison
repeats the original and earlier enhancement paths on the same exact inputs
as a recently released enhancement model. It uses the same local voice
classifier for all paths and fixes a separate operating point for each
enhanced view before scoring the held recordings.
[Earlier enhancement result](audio-noise-views-transfer-2026-11-12.md) ·
[Enhancement and detection study](https://arxiv.org/abs/2603.14767) ·
[Recent enhancement model research](https://arxiv.org/abs/2606.25621)

## Model and sources

The newer enhancer uses NVIDIA's official 9.6M-parameter NemotronLabs RE-USE
weights at revision `31f82f2efad1095034492f236e8dac9239ba713b`. The exact
publisher notice grants OpenMDW-1.1 terms. It permits commercial use subject
to the license, and distribution of model materials requires retaining the
license and origin notices. This experiment uses the publisher's original
safetensors with the pinned native Apple Silicon adapter. It processes only
local files. The earlier GTCRN weights and the current voice assets retain
their earlier verified terms. No new weights are bundled with Chipmunk Verify.
[Pinned publisher model and terms](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE/tree/31f82f2efad1095034492f236e8dac9239ba713b) ·
[MLX adapter](https://github.com/Neywa-Labs/mlx-audio/pull/1101) ·
[Existing model audit](commercial-model-audit.md)

The selection role reuses 56 exact three-second inputs from the independent
operating-point portion of the earlier noise comparison. Those inputs came
from the RADAR development split and two older sound recordings. The held
role uses a 192 MiB prefix from the CC BY 4.0 LlamaPartialSpoof release. Check
all 992 complete archive members against the publisher's segment table. Use
only clips with a single label spanning a 3–14 second recording, then take its
exact center three seconds. After excluding the earlier partial-spoof study's
source-speaker IDs, take ten genuine clips and two clips from each of six
generation families by fixed hash order. All 22 retained source-speaker IDs
are distinct. They are not a count of distinct generated voices. Training
exposure and wider representativeness remain unknown.
[RADAR release](https://zenodo.org/records/22110815) ·
[LlamaPartialSpoof release and terms](https://zenodo.org/records/14214149) ·
[Earlier partial-clip exclusion roster](audio-partial-publisher-transfer-2026-11-12.md)

Each held speech clip contributes a gain-only control, a restaurant mixture
and a Vienna roadside mixture. The added recording is 5 dB louder than the
foreground by full-window RMS. All three views share the same peak-safe gain.
Independent sample arithmetic placed all 44 actual mixtures within 0.000014 dB
of that target.
Four non-overlapping excerpts from each sound recording also enter alone.
The two original background files were not used for the selection points.
The restaurant recording contains ambient human speech. The traffic recording
may include other street sounds. Results describe two backgrounds and 22
underlying source-speaker IDs. Audio and intermediate enhancer outputs remain
local. The restaurant recording is public domain by stephan. The Vienna
recording is by Matthias A. Bertsch (DrTrumpet) under CC BY-SA 4.0.
[Restaurant source](https://commons.wikimedia.org/w/index.php?title=File:Restaurant_ambience.ogg&oldid=1003321006) ·
[Vienna source](https://commons.wikimedia.org/w/index.php?title=File:2023-Mai-20_14h34min_Wien_innerer_Lerchenfelder_G%C3%BCrtel_Thaliagasse_Nachmittagsverkehr.wav&oldid=982248618) ·
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

## Frozen comparison

Both enhancers receive the same decoded 16 kHz, three-second signal. Preserve
the source length and quantize each output to 16-bit PCM. An enhanced response
can add a flag only when the usual voice decision was inconclusive or
unassessed, the enhanced output passes speech admission, and its raw voice
logit strictly exceeds the highest admitted genuine or isolated response in
the separate selection role. Preserve all earlier directions. The new
enhancement never sets an original-file direction or a customer probability.

The advancement rule requires four distinct noisy synthetic source groups
recovered across three generators, gains in both scenes, at most two
additional genuine groups flagged under noise and no additional genuine
clean or isolated flags. These fixed decision points apply only to the
experiment. The saved output hash refers to the exact local waveform used
for scoring. The local MLX repeat check allows a floating-point tolerance of
`1e-5` before PCM quantization because tiny numerical differences may change
individual samples near a rounding boundary.

## First held comparison

| Condition | Original AI flags | With earlier enhancer | With newer enhancer |
| --- | ---: | ---: | ---: |
| Restaurant, synthetic | 10/12 | 10/12 | 12/12 |
| Restaurant, genuine | 7/10 | 7/10 | 7/10 |
| Vienna roadside, synthetic | 3/12 | 5/12 | 9/12 |
| Vienna roadside, genuine | 0/10 | 0/10 | 1/10 |
| Gain-only, synthetic | 12/12 | 12/12 | 12/12 |
| Gain-only, genuine | 0/10 | 0/10 | 0/10 |
| Background-only excerpts | 0/8 | 0/8 | 0/8 |

The newer enhancer passed the first fixed gate. It added eight correct
noisy deliveries covering seven distinct source IDs and five generators,
and added one genuine flag. The original detector already flagged seven
genuine restaurant mixtures. The combined path inherited all seven. Both
paths use the same authenticity model. The enhancement can provide a
complementary view of those samples, with no independent confidence claim.

An independent replay selected the same 22 source IDs, recovered all 66
matched views and eight isolated clips byte for byte, and checked 390 raw
and enhanced PCM commitments. Four restaurant-only excerpts passed speech
admission and stayed below both flag points. None of the four Vienna-only
excerpts passed speech admission. [First held receipt](evaluation-panels/audio-nemotron-noise-transfer-2026-11-12.json)

## Replay

Use the public 192 MiB archive prefix and labels, the pinned restaurant and
Vienna files, and the completed earlier calibration preparation. The runner
verifies the files before scoring. The existing backend voice assets must be
installed locally. Use the official model revision above and the earlier
enhancer directory with its original license. The Python environment needs
MLX and MLX Metal 0.32.4, MLX Audio 0.6.5, and ONNX Runtime 1.31.1 for the
enhancement phase. The tested Metal adapter needs a locally available GPU.

Use a new directory for `held_sources`. Keep each recording's attribution
and original terms with the files. The source prefixes and model files are
verified by SHA-256 when the relevant phase starts.

```sh
publisher=https://zenodo.org/records/14214149/files
curl -fLsS --range 0-201326591 --max-filesize 201326592 \
  "$publisher/R01TTS.0.a.tgz?download=1" -o "$held_sources/part-a-192m.prefix.tar"
curl -fLsS --max-filesize 12000000 "$publisher/label_R01TTS.0.a.txt?download=1" -o "$publisher_labels"
curl -fLsS --max-filesize 2000000 \
  'https://upload.wikimedia.org/wikipedia/commons/b/b5/Restaurant_ambience.ogg' \
  -o "$held_sources/restaurant.ogg"
curl -fLsS --max-filesize 5800000 \
  'https://upload.wikimedia.org/wikipedia/commons/d/da/2023-Mai-20_14h34min_Wien_innerer_Lerchenfelder_G%C3%BCrtel_Thaliagasse_Nachmittagsverkehr.wav' \
  -o "$held_sources/vienna.wav"
```

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VERITAS_AUDIO_DETECTOR_MODE=candidate
export VERITAS_BILLING_MODE=disabled VERITAS_DATA_DIR="$scratch_state"
backend/.venv/bin/python -B -m scripts.audio_nemotron_noise_transfer prepare "$run" \
  --inputs "$held_sources" --previous "$prior_run" --labels "$publisher_labels"
backend/.venv/bin/python -B -m scripts.audio_nemotron_noise_transfer enhance "$run" \
  --model "$new_model" --old-model "$old_model"
for phase in selection point held; do
  backend/.venv/bin/python -B -m scripts.audio_nemotron_noise_transfer "$phase" "$run" || exit 1
done
python3 -I -S backend/scripts/review_audio_nemotron_noise_transfer.py export "$run" --out "$receipt"
```
