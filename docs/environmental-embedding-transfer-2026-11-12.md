# Separate checks for environmental sound under speech

Training with ordinary voices mixed into the source improved some
environmental-sound results. It also caused more false alarms. None of
the learned or combined checks qualified for the app. A sound-category
label or a voice authenticity response still supplies no determination
about the background's origin.

## Frozen design

Reuse the previously measured 40 sound groups from the EnvSDD test
release. Each group has a credited CC0 Clotho recording and two exact
publisher text-to-audio renders for its role. Twelve groups fit the heads,
ten set operating points and 18 supply the held comparison. The held
audio generators are absent from fitting and selection. All background
groups had earlier project exposure. The three source roles also use
separate sets of four, four and seven retained genuine speech sources.
Keep the foreground identical across all background origins in a group.
[Pinned source release and individual rights](environmental-sound-transfer-2026-11-11.md#frozen-inputs-and-method)

Use exact three-second native PCM, and make matched versions with the
environment 6 dB below speech. The held panel also includes balanced
speech mixtures and 32 kb/s Opus round trips of the isolated environments.
The same fitting rows supply a spectral head, a small ridge head on the
existing frozen sound encoder, and a head combining the two feature blocks.
For a separate comparison, fit the learned head on isolated recordings
alone. Standardize each feature and divide by the square root of its
block dimension. Give equal total weight to each origin, and equal
weights across its source groups and included delivery scenes.

Set separate points for isolated and mixed conditions on genuine
selection sources. Each flag must strictly exceed the second-highest
genuine selection response for its condition. Use the isolated point
for Opus and the mixed point for balanced mixtures. These are known
laboratory conditions. Inferring which condition applies to an
unknown upload would need its own test.

The held comparison is informed by research separating environmental and
speech targets and combining complementary sound representations. It
uses the already installed, commercially reviewed sound encoder. It uses
no code or weights from the cited detection studies.
[Component-specific study](https://arxiv.org/html/2607.10345v1) ·
[Separate audio branches](https://arxiv.org/html/2608.00493v1) ·
[Complementary environmental features](https://www.ijcai.org/proceedings/2026/471) ·
[Encoder weights and terms](https://zenodo.org/records/3987831)

## Held results

Each cell gives flags on ordinary backgrounds, first unseen generator,
and second unseen generator, with 18 examples per origin.

| Delivery | Spectral, fit with speech | Learned, fit with speech | Combined | Learned, isolated fit |
| --- | ---: | ---: | ---: | ---: |
| Isolated native | 4 / 4 / 11 | 0 / 1 / 2 | 1 / 3 / 8 | 0 / 1 / 2 |
| Isolated Opus | 3 / 5 / 11 | 0 / 1 / 1 | 1 / 2 / 7 | 0 / 1 / 1 |
| Speech 6 dB louder | 6 / 10 / 11 | 3 / 2 / 1 | 4 / 6 / 9 | 2 / 2 / 2 |
| Balanced speech and environment | 7 / 10 / 12 | 3 / 3 / 1 | 6 / 9 / 9 | 2 / 1 / 1 |

The earlier isolated-fit spectral check found one of 36 generated
backgrounds through loud speech with zero ordinary flags. The new
mixed-fit spectral head found 21 of 36 with six of 18 ordinary backgrounds
flagged. The combined head flagged 15 of 36 and four ordinary backgrounds.
At a retrospectively matched budget of two ordinary flags, the strongest
arm under loud speech found only five of 36 generated backgrounds.
In isolated native and Opus, the combined arm could reach 15 of 36
in each at that budget. That exploratory ranking does not establish a
new operating point.

No arm met the fixed requirement of 11 hits per generator and at most
two ordinary flags under loud speech. The comparison does not localize
edits inside the three-second inputs. It uses existing sound groups,
three-second recordings and one controlled overlap recipe. It leaves
natural mixture transfer and background-authorship confidence open.
[Frozen observations and decisions](evaluation-panels/environmental-embedding-transfer-2026-11-12.json)

## Replay

The public reader independently checks all group assignments, the four
sets of operating points, all held counts and matched false-alarm
rankings. With the local run it also checks all 348 exact PCM commitments.
Waveforms, model features and fitted heads remain local.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_environmental_embedding_transfer \
  docs/evaluation-panels/environmental-embedding-transfer-2026-11-12.json
PYTHONPATH=backend python3 -B -S -m scripts.review_environmental_embedding_transfer \
  "$run/receipt.json" --native "$run"
```

To rerun inference, use the retained environment and speech inputs
listed in the runner or pass their exact paths with `--environment` and
`--speech`. Use a new run directory and the pinned offline assets.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
backend/.venv/bin/python -B -m scripts.environmental_embedding_transfer run "$run"
```
