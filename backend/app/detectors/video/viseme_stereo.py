# SPDX-License-Identifier: MIT
"""Measure the channels behind the specific mono input used for lip timing."""

from __future__ import annotations

import hashlib

from app.detectors.audio.stereo import _metrics, _mono_dbfs
from app.evidence import video_viseme_stereo as binding
from app.evidence.video_viseme_binding import presented_samples


def capture(decoded):
    mapping, channels = decoded.mapping, decoded.stereo_pcm
    count = presented_samples(mapping)
    if (not binding.carrier(mapping) or count is None or channels is None
            or len(channels) != mapping["samples"] or len(decoded.pcm) != len(channels)):
        return None
    rows = []
    for start, stop in binding.blocks(count):
        left, right, correlation = _metrics(channels[start:stop])
        mix = _mono_dbfs(decoded.pcm[start:stop])
        rows.append([start, stop, left, right, mix, correlation])
    data = channels[:count].astype("<i2", copy=False).tobytes()
    return {"schema": binding.SCHEMA, "stereo_pcm_sha256": hashlib.sha256(data).hexdigest(),
            "mono_pcm_sha256": mapping["pcm_sha256"], "blocks": rows}
