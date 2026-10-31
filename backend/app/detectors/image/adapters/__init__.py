"""v2 adapters for the v0.3 image detectors (P-5).

Each class implements `app.engine.detector.Detector`, and the manifest `adapter:` field names it. `loaders.py` holds the
`adapter_loader` functions the ModelManager calls. Adapters return raw scores and quality facts; the v0.3 weighting
rules live in the legacy fuser (`app.engine.stages.image.fuse`).
"""
