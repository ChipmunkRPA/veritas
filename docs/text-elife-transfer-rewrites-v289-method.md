# Matched rewrites on fresh eLife science explainers

Bind the exact v286 intake, its prospective roster and all 26 admitted originals. Recheck each original and its frozen interior target. Order the source groups by the hexadecimal SHA-256 of the UTF-8 string `chipmunk-elife-v289-split-v1:` followed by the recorded article group, with the group ID breaking any tie. The first 13 groups are development and the remaining 13 are confirmation. Freeze both lists before generation. The eight ineligible excerpts and one license hold from intake stay visible. [Source intake](text-elife-transfer-intake-v286-method.md)

Retain each unchanged original and attempt the same target once per family. Reuse the exact v273 Qwen and SmolLM2 revisions, offline MLX runtime, quantization, context prompt, sampler and acceptance rules. The generator receives the selected paragraph and the adjoining two paragraphs. The word bound, copied-text check and all other exclusion rules apply unchanged to both families. Each accepted result replaces only the predetermined paragraph. Recheck the UTF-8 original reconstruction and half-open Unicode offsets. Each rejected attempt retains its reason and has no scoring text. Keep all raw outputs and source text local. [Original rewrite rules and weight terms](text-elife-rewrites-v273-method.md)

Run four sequential guarded stages, each with 13 single-document calls. The order is development Qwen, development SmolLM2, confirmation Qwen, and confirmation SmolLM2. Each guarded process starts with the same v273 random seed and its assigned source order. An interrupted stage stays recorded and no stage is repeated. A complete readout binds each qualified worker and replays all 52 attempts. It retains 26 original controls and 52 rewrite dispositions with their source role. The stage runs no detector and makes no serving or calibration change. Later method selection must respect the frozen source groups.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
backend/.venv/bin/python -B -m scripts.text_elife_transfer_rewrites_v289 preflight
backend/.venv/bin/python -B -m scripts.text_elife_transfer_rewrites_v289 freeze
backend/.venv/bin/python -B -m scripts.text_elife_transfer_rewrites_v289 run 0
backend/.venv/bin/python -B -m scripts.text_elife_transfer_rewrites_v289 run 1
backend/.venv/bin/python -B -m scripts.text_elife_transfer_rewrites_v289 run 2
backend/.venv/bin/python -B -m scripts.text_elife_transfer_rewrites_v289 run 3
backend/.venv/bin/python -B -m scripts.text_elife_transfer_rewrites_v289 readout
```
