# Longer-passage writing check on eLife

The optional longer-passage check found **zero of 21** known AI rewrite locations on 12 fresh eLife works. The current context-window check also found zero. At the existing longer-document point, the new check marked three generated files at the wrong sentence and flagged two ordinary source groups. This panel supports keeping the feature opt-in and confidence-neutral. [Frozen comparison receipt](evaluation-panels/writing-elife-long-position-transfer-2026-11-13.json)

| Held outcome | Current check | Position check at the longer-document point | Position check at the short-passage point |
| --- | ---: | ---: | ---: |
| Exact or useful locations in Granite rewrites | 0/10 | 0/10 | 0/10 |
| Exact or useful locations in Qwen rewrites | 0/11 | 0/11 | 0/11 |
| Generated files marked only at the wrong sentence | 0/21 | 3/21 | 15/21 |
| Untouched archived excerpts flagged | 0/12 | 1/12 | 7/12 |
| Constructed publisher-native substitutions flagged | 0/12 | 2/12 | 7/12 |
| Ordinary work groups flagged in either view | 0/12 | 2/12 | 7/12 |

None of the 21 edited sentences reached either existing point. The largest true-target contrast was 2.656, below the shorter point of 2.758. Only one edited sentence ranked first anywhere in its file before applying a cutoff. In work 27240 the exact same sentence and three model inputs gave the same response in the original, the constructed ordinary control, and both generated variants. The selected sentence was outside the changed passage in all three substitutions. In work 13964, changing one sentence raised the score of its unchanged neighbor enough to select that neighbor in the ordinary donor control and one AI rewrite. These results expose both a low target response and a context-sensitive false alarm. On this panel a lower cutoff would leave the ranking problem. [Previous F1000Research comparison](writing-f1000-position-transfer-2026-11-13.md)

## Archived source and comparison method

The first source intake downloaded both official pre-2020 archive versions for 36 fixed eLife IDs. Every v1 file was a Publish on Acceptance record with no article body in the archived XML. A separate v2-to-v3 intake downloaded 36 more candidate pairs. Sixteen failed the version-of-record, date or explicit CC BY 4.0 requirements, two failed the byline checks, and the other 18 did not supply a qualifying matched longer revision. Both source intakes finished before detector inference. [eLife XML archive](https://github.com/elifesciences/elife-article-xml/tree/58e688ec3d4523eb1f3b03bed6bb440bf91e4786) · [eLife terms](https://elifesciences.org/terms)

The fallback took the first 12 eligible source works in the fixed v2-to-v3 roster and used the most recent qualifying version of record among those two versions. All source terms were checked against both the official version history and article XML. Every source supplied one untouched historical excerpt and one explicitly constructed ordinary control made by moving an exact sentence from another supported prose block of the same historical article. The construction tests whether the check distinguishes synthetic rewriting from a local change made entirely of published prose. No generated or customer text enters the ordinary controls. Recorded authors and normalized 12-word fragments were screened against the nine listed retained source ledgers and previously admitted works. Model training exposure remains unknown. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Two pinned Apache-2.0 generators each made three fixed-seed attempts per work. The first structurally valid complete rewrite preserved every non-target sentence exactly. Ten Granite and eleven Qwen attempts were admitted before scoring. The 45 cases had 9–23 sentences and 162–483 words. At the original points and coefficients the comparison measured 489 distinct model inputs and all 429 eligible candidate centers. The first inference attempt stopped before result selection because three incumbent-only formula-heavy windows exceeded 512 tokens. The rerun used the existing production pooling across complete token windows for those three inputs. It kept all cases, original points, candidate token limit, location rules and comparison methods. [Granite license](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct) · [Pinned Qwen license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE)

## Replay

The model-free review replays the frozen per-case decisions, exact locations and controls from the saved measurements. It uses only Python 3.12 standard library components:

```sh
PYTHONPATH=backend python3.12 -B -S -m scripts.review_writing_elife_long_position \
  docs/evaluation-panels/writing-elife-long-position-transfer-2026-11-13.json
```

The native replay reconstructs the 12 untouched excerpts and 12 constructed controls from the publisher bytes. It checks the exact official file hashes, version histories, bylines, licenses, donor character spans and model-input hashes. Use `lxml==6.1.2`, `pysbd==0.3.4` and `psutil==7.2.2` and run from the repository root:

```sh
PYTHONPATH=backend python3.12 -B -m scripts.review_writing_elife_long_position \
  docs/evaluation-panels/writing-elife-long-position-transfer-2026-11-13.json \
  --native-cache /tmp/chipmunk-elife-xml --download-native
```

Omit `--download-native` to verify a populated cache offline. The receipt contains measurements, source hashes and offsets. Publisher excerpts and generated prose stay local. This evaluation changes no serving configuration or product estimate.
