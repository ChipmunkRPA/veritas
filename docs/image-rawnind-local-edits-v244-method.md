# Test localized phase mismatch on ordinary camera edits

Reuse the four verified RAW captures, decoder and fixed native rendering from the feasibility replay. Each capture contributes a single three-by-three tile destination aligned to the image grid. Choose both source and destination by geometry, before measuring scores. Compare an even-aligned copy with the same donor shifted one pixel horizontally. Add the same localized tone adjustment and Gaussian blur as separate ordinary controls. No model is used to alter the image. [Native replay](image-rawnind-phase-v239-result.md)

Use the unchanged phase cue without a decision cutoff. Record abstention and concordance between finite tiles touching the exact changed pixels and other finite tiles. Compare the median score change in the same destination tiles when both the original and edit have coverage. The paired copy comparison tests the expected phase effect independently of edit authorship. These four already observed source groups and their conventional edits measure local cue response. AI localization sensitivity requires a separate positive panel.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.image_rawnind_local_edits_v244 /absolute/private/rawnind-v239
```
