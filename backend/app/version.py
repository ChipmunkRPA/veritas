"""The application version (D-58): the number of the served fusers, defined here and nowhere else.

`pyproject.toml` reads it as the package's version, the settings carry it as `settings.version` (not a field: the environment
cannot set it), and the probes, the metrics, result manifests, reports, packages, webhooks and backups state that value.
`frontend/package.json` carries the same number (`tests/test_n15_version.py`). A plain literal: setuptools reads it from the
source text without importing the package.
"""
VERSION = "2.1.1"
