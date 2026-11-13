# Frozen writing position check on Qwen edits

The position check found two additional exact rewrite locations on 13 previously unscored Qwen edits, one each in Gates Open Research and Wellcome Open Research. It flagged none of the 14 genuine works. The incumbent found one different edit and flagged two genuine work groups. Combining the locations would find three of 13 edits with the same two ordinary groups flagged. [Source and generation admission](evaluation-panels/writing-position-qwen-admission-2026-11-12.json) · [Frozen response receipt](evaluation-panels/writing-position-qwen-confirmation-2026-11-12.json)

## Source freeze and test

The source search used 296 historical Python PEP work histories outside the original 200-work roster and 80 retained Wellcome and Gates publisher version pairs. It admitted three PEPs, six Gates works and five Wellcome works that had not been scored in the reviewed project sets. The two licensed source versions preserve the immediate neighbors around one materially revised sentence in the same paragraph. Both versions are pre-2020 and retain the same recorded byline. The PEP versions have explicit public-domain or CC0 terms. Each publisher XML has explicit CC BY 4.0 terms. The replay checked all 28 source versions and 91 inter-work exact 12-word fingerprint comparisons. These sources had prior intake exposure. The two journal titles share the F1000 publishing platform, and model training exposure is unknown. [PEP source history](https://peps.python.org/pep-0001/#what-is-a-pep) · [Journal source and terms](writing-journal-body-transfer-2026-11-11.md)

The pinned Qwen2.5-1.5B-Instruct checkpoint produced four attempts per source on the Mac, using a fixed prompt, seeds and four-bit affine quantization. The unchanged completion, length, copy and source-coordinate rules admitted the first qualifying rewrite from 13 works. Of 56 attempts, 45 met the rules. All 14 original and genuine revision pairs entered scoring. The eight-work minimum was frozen and passed before any detector inference. The detector and its position point came from the earlier test. No Qwen response changed that fit or point. [Pinned Qwen license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) · [Frozen initial rule](writing-two-context-transfer-2026-11-12.md)

| Source | Qwen edits | Incumbent useful | New exact locations | Ordinary works flagged, incumbent / position |
| --- | ---: | ---: | ---: | ---: |
| Python PEPs | 3 | 1 | 0 | 1/3 · 0/3 |
| Gates Open Research | 5 | 0 | 1 | 1/6 · 0/6 |
| Wellcome Open Research | 5 | 0 | 1 | 0/5 · 0/5 |
| Total | 13 | 1 | 2 | 2/14 · 0/14 |

The ordinary group counts include both source versions. The candidate alone flagged zero of 14 originals and zero of 14 genuine revisions. The incumbent flagged two originals and their two corresponding revisions. Across this Qwen test and the two earlier frozen trials, the position check added six useful locations across Granite and Qwen with zero added ordinary groups. The evidence is limited to short English passages with exactly three mapped sentences. [Earlier publisher confirmation](writing-position-confirmation-2026-11-12.md)

## Local replay

The CPU run made 21 forwards on 123 source-bound inputs. The portable review binds each of the 41 scored views to the pre-scoring source and generation receipt, verifies the frozen point and positions, and recomputes the outcomes by publisher. Original and generated text stay local.

```sh
python3 -B -I -S backend/scripts/review_writing_position_qwen.py \
  docs/evaluation-panels/writing-position-qwen-confirmation-2026-11-12.json
```
