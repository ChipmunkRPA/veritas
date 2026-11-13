# Locating one AI rewrite in a longer article excerpt

At the original short-passage operating point, the writing position check found two exact AI rewrite locations missed by the existing check. Combining them localized four of 24 known rewrites, up from two, and flagged six of 12 genuine works, up from five. A separate operating point calibrated across longer excerpts found zero additional edits and added zero genuine flags. The longer-article check stays in research. [Frozen response receipt](evaluation-panels/writing-long-position-transfer-2026-11-12.json)

| Held outcome | Existing context windows | Position check at the short-passage point | Union at the short-passage point | Union at the document-calibrated point |
| --- | ---: | ---: | ---: | ---: |
| Useful location in Granite rewrites | 1/12 | 2/12 | 3/12 | 1/12 |
| Useful location in Qwen rewrites | 1/12 | 0/12 | 1/12 | 1/12 |
| Useful location across both generators | 2/24 | 2/24 | 4/24 | 2/24 |
| Genuine article versions flagged | 7/24 | 6/24 | 9/24 | 7/24 |
| Genuine works flagged in either version | 5/12 | 3/12 | 6/12 | 5/12 |

The extra locations at the short-passage point were the exact rewritten sentence at characters `[99, 282)` in Gates Open Research work 13043 and `[470, 627)` in Wellcome Open Research work 15279. The one added genuine work was Wellcome Open Research 14073. Both versions of that work selected the same unchanged sentence at different source offsets. Four generated files also selected genuine sentences elsewhere in their excerpts at the old point.

## Source and measurement freeze

The test used three to five consecutive supported body paragraphs from CC BY 4.0 historical publisher XML. Each work contributed two genuine versions with the same recorded byline, increasing publication dates before 2020, and actual textual changes in the excerpt. The first geometry proposal required a genuine revision at the exact center and found no qualifying held works. Before generating or scoring any examples, source intake changed to require a genuine revision anywhere in the longer excerpt and two exact immediate neighbors around a separately selected synthetic target. Eleven of the 12 held works preserve that target in both genuine versions. The frozen intake preserves the result of both proposals.

The calibration set consists of 14 F1000Research works. The held set has six Gates Open Research and six Wellcome Open Research works, with no author or exact normalized 12-word overlap between the selected work groups. The 12 held works had no prior detector scoring in the checked publisher panels. All three journals share a publishing platform. Model training exposure is unknown. Source-level limitations include the number of publishers and the use of excerpts from supported prose runs rather than full articles. [Publisher source terms](writing-journal-body-transfer-2026-11-11.md) · [Earlier model calibration](writing-position-app-2026-11-12.md)

Each held work received one admitted exact sentence replacement from each of two pinned Apache 2.0 generators. Three fixed-seed attempts per generator and work were produced locally. The first complete, materially changed rewrite with the same native segmentation outside its source range was accepted. All 24 known edits were admitted before the detector ran. The final held excerpts contain 10–19 sentences and 271–593 words. [Granite publisher card](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct) · [Qwen publisher license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE)

The existing-check comparator retains the first six qualifying native three-sentence context windows. A window must cover at least half of the edited non-whitespace characters and reach 0.30 intersection over union to count as useful. The candidate evaluates each eligible middle sentence with both neighbors from the same source paragraph, subtracts the stronger edge response and retains the unique strongest focus in the entire excerpt. It reuses the existing fitted coefficients. To account for scanning more locations, the primary operating point was fixed to the larger of the previous point and the second-highest genuine work maximum across both versions in the calibration set. The point rose from 2.758 to 3.682 before held inference. This approach tests a simpler document-level control motivated by work on local score fluctuations and length-aware calibration. [Local change-point research](https://arxiv.org/html/2609.33787v1) · [Length-conditional calibration](https://arxiv.org/html/2611.10901v1)

## Replay

CPU scoring used 415 unique calibration inputs and 463 held inputs. All 326 eligible held centers had complete context readings. Local replay reconstructed 52 XML versions and 2,626 saved input occurrences, and checked all 325 source-work pairs for author and exact-fragment overlap. The public receipt carries source attribution, hashes, offsets and saved scores. Source prose and generated rewrites remain local.

```sh
python3 -B -I -S backend/scripts/review_writing_long_position.py \
  docs/evaluation-panels/writing-long-position-transfer-2026-11-12.json
```

### Independent publisher replay

From the repository root, a separate command downloads the recorded first-party article versions and reconstructs each genuine excerpt. It checks the publisher, license, byline, publication dates, original paragraph and sentence offsets, and every saved genuine input hash. The command uses no model weights. Python 3.12 with `lxml==6.1.2`, `pysbd==0.3.4`, and `psutil==7.2.2` is sufficient.

```sh
PYTHONPATH=backend python3.12 -B -m scripts.verify_writing_long_position_sources \
  docs/evaluation-panels/writing-long-position-transfer-2026-11-12.json \
  --cache /tmp/chipmunk-article-xml
```

Add `--offline` to repeat from that cache. Without `--cache`, the tool removes the XML when it exits. With `--strict-xml`, a changed XML digest returns a failure even when all measured text agrees. The fresh publisher run verified all 52 versions and all 1,816 genuine input hashes. Fifty XML versions matched their frozen byte hashes. Each of the two changed Gates v1 XML responses changed only the article acceptance day by one day. Their licensed body text and all measured inputs matched. The strict mode verified all 52 historical files when run against the frozen local witnesses. Generated rewrites and their model responses remain covered by the separate arithmetic review. [Independent source replay receipt](evaluation-panels/writing-long-position-publisher-replay-2026-11-12.json)
