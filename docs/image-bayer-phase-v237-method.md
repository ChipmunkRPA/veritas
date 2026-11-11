# Local Bayer-phase consistency for regional image evidence

The experimental cue compares directional second-difference magnitudes in the green channel across the four positions of a two-by-two sensor mosaic. It measures complete 64-pixel tiles and estimates a reference on the opposite checkerboard fold. Both folds need distributed agreement before the cue returns a map. The map measures local phase inconsistency. It carries no AI verdict. The method was implemented independently with NumPy, using a different residual and reference estimator from the published approach. [Related primary research](https://arxiv.org/html/2608.20558v1)

Nine local fixtures verify arithmetic, opposite-fold scoring, native coordinate mapping and abstention. Every tile inside the authored phase-breaking composite scored above every unaffected tile in its image. Cropping and sharpening preserved low scores in the authored control. JPEG 75, JPEG 90 and resizing erased distributed phase support in those fixtures.

The fixed model-free BG-REAL replay measures coverage on an already observed benchmark. Use all 24 qualified LaMa source groups and the same untouched, AI, tone, blur and copy-move views at source PNG resolution and the two frozen JPEG qualities. Verify the prior exact input bindings. Report phase coverage and, when available, the concordance of tiles touching the exact edit footprint against finite untouched tiles. The original view has no positive footprint. Use the same measure on each ordinary edit footprint. The publisher resized the images before release, which may erase original sensor phase. Keep the cue fixed throughout and reserve any future operating point for a separate genuine camera panel. [Frozen JPEG method](image-bg-real-jpeg-v229-method.md) · [Publisher paper](https://arxiv.org/html/2607.26232v3)

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.image_bayer_phase_bg_v238
```
