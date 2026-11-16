# SPDX-License-Identifier: MIT
"""Source-bound suitability of a stereo downmix for local speech timing."""

from __future__ import annotations

import math
import re

SCHEMA = "chipmunk.lip-timing-stereo-scope/1"
BLOCK = 6400
MIN_TAIL = 1280
MIN_AUDIBLE = 2
MAX_GAP = 6
MAX_LOSS = 6
MIN_CORRELATION = .65
AUDIBLE = -50
HASH = re.compile(r"[a-f0-9]{64}")
CONTAINER = "mov,mp4,m4a,3gp,3g2,mj2"


def carrier(mapping):
    return (mapping.get("source_channels") == 2 and mapping.get("source_sample_rate") == 48_000
            and mapping.get("source_codec") == "aac" and mapping.get("source_profile") == "LC"
            and mapping.get("source_container") == CONTAINER and not mapping.get("audio_stream_indices"))


def blocks(samples):
    edges = list(range(0, samples, BLOCK))
    if len(edges) > 1 and samples - edges[-1] < MIN_TAIL:
        edges.pop()
    edges.append(samples)
    return list(zip(edges, edges[1:]))


def qualifies(scope, mapping, samples):
    """Check the complete placed measurements and the frozen admission rule."""
    if (not isinstance(scope, dict) or set(scope) != {"schema", "stereo_pcm_sha256", "mono_pcm_sha256", "blocks"}
            or scope["schema"] != SCHEMA or scope["mono_pcm_sha256"] != mapping.get("pcm_sha256")
            or not HASH.fullmatch(str(scope["stereo_pcm_sha256"])) or not carrier(mapping)
            or samples is None or not isinstance(scope["blocks"], list)):
        return False
    expected = blocks(samples)
    if len(scope["blocks"]) != len(expected):
        return False
    audible = 0
    for item, (start, stop) in zip(scope["blocks"], expected, strict=True):
        if not isinstance(item, list) or len(item) != 6 or item[:2] != [start, stop]:
            return False
        left, right, mono, correlation = item[2:]
        if any(type(level) not in (float, int) or not -140 <= level <= 0 for level in (left, right, mono)):
            return False
        if correlation is not None and (type(correlation) not in (float, int)
                                        or not math.isfinite(correlation) or abs(correlation) > 1):
            return False
        louder = max(left, right)
        if louder >= AUDIBLE:
            if (correlation is None or correlation < MIN_CORRELATION or abs(left - right) > MAX_GAP
                    or louder - mono > MAX_LOSS):
                return False
            audible += 1
    return audible >= MIN_AUDIBLE
