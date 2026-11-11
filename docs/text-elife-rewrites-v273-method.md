# Local paragraph rewrites for the July 2022 eLife cohort

Bind the exact public and private v272 intake receipts and the SHA-256 of the admitted original journal. Recheck each of the five admitted sources, the complete original excerpt and each interior target's Unicode code point offsets. The other four roster entries remain recorded in the v272 intake. Every admitted article supplies one unchanged control and one attempt at rewriting the same interior paragraph for each of two pinned local model families. Both weights declare Apache-2.0 at the recorded revision. [Source intake](evaluation-panels/text-elife-source-intake-complete-v272.json) · [Qwen license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) · [SmolLM2 license](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/tree/31b70e2e869a7173562077fd711b654946d38674)

Use the existing offline MLX generator with the pinned tokenizer and configuration, the same quantization and sampler, and one article per call. Provide the immediately preceding and following paragraphs as context and ask for one plain English replacement preserving meaning and facts. Keep the exact output in the owner-only attempt journal. Collapse whitespace in the replacement. Reject empty, out-of-range, unchanged, wrapped or multiparagraph answers and answers containing links or email addresses. The accepted word count is 60%–150% of the original paragraph's count, bounded to 20–170 words. Reject copied text if more than a quarter of replacement tokens belong to exact eight-token phrases found in the complete original excerpt. Keep every rejected attempt and its reason. There is no resampling or detector-informed selection.

Splice the accepted replacement between the unchanged original prefix and suffix. Record the original target and generated interval as half-open Unicode code point offsets and verify that replacing the interval with the original target reconstructs the original byte-for-byte in UTF-8. An accepted label identifies the generated replacement from the recorded construction. It does not assert that the surrounding article was independently certified human, that every replacement fact was verified, or that these five science explainers represent all articles.

The two families run as separate sequential stages through the existing local process guard, with its 6 GiB initial RAM and sampled resident-memory limits, 3 GiB initial disk requirement, 1.5 GiB running reserves and 900-second stage limit. A preflight rejection creates no model attempt. An interrupted or failed stage stays recorded. Readout requires the exact qualified completion for both generators and replays all acceptance decisions and offsets. It preserves five matched controls and all ten planned rewrite dispositions in the private case journal. Public receipts contain only counts and commitments. This stage runs no detector and does not change serving.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
backend/.venv/bin/python -B -m scripts.text_elife_rewrites_v273 freeze
backend/.venv/bin/python -B -m scripts.text_elife_rewrites_v273 run 0
backend/.venv/bin/python -B -m scripts.text_elife_rewrites_v273 run 1
backend/.venv/bin/python -B -m scripts.text_elife_rewrites_v273 readout
```
