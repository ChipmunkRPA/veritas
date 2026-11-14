from copy import deepcopy

import numpy as np
import pytest

from app.detectors.video.mouth_hold_sound import measure
from app.evidence import mouth_hold_sound as binding


def test_mouth_hold_sound_binds_local_samples_and_rejects_unverified_clock():
    rate = binding.RATE
    source_origin = 800
    samples = 2 * rate
    review = {
        "schema": "chipmunk.video-mouth-local-hold/2",
        "status": "matched",
        "match": {"frames": [24, 36]},
        "pts": [i * 1000 + 3000 for i in range(60)],
        "time_base": "1/30000",
    }
    mapping = {"timeline_time_base": "1/16000", "samples": samples,
               "timeline": [{"start_sample": 0, "end_sample": samples, "source_pts": source_origin}]}
    first, last = 14400, 20800
    start, stop = first - source_origin, last - source_origin
    clock = np.arange(samples) / rate
    wave = np.zeros(samples, dtype=np.float64)
    wave[start:stop] = .03 * np.sin(2 * np.pi * 150 * clock[start:stop])
    original = wave.copy()
    record = measure(wave, mapping, review, review["time_base"])
    assert record["samples"] == [start, stop] and record["source_pts"] == [first, last]
    assert record["periodic_blocks"] == record["blocks"] == 18
    assert record["sustained_periodicity"] is True
    assert np.array_equal(wave, original)
    silent = measure(np.zeros_like(wave), mapping, review, review["time_base"])
    assert silent["sustained_periodicity"] is False
    assert silent["periodic_blocks"] == 0 and silent["samples"] == [start, stop]

    review["sound_context"] = record
    pcm = {"media_type": "audio/wav", "samples": stop - start, "sample_rate": rate, "channels": 1}
    plot = {"media_type": "image/png", "width": 960, "height": 280}
    packet = {"mouth_hold_sound": record, "limitations": [binding.LIMITATION],
              "artifacts": {binding.WAV: pcm, binding.WAVEFORM: plot}}
    binding.validate(review, packet, mapping, review["time_base"])
    changed = deepcopy(packet)
    changed["mouth_hold_sound"]["source_pts"] = [first + 1, last]
    with pytest.raises(ValueError, match="recorded measurement"):
        binding.validate(review, changed, mapping, review["time_base"])
    broken = deepcopy(mapping)
    broken["timeline"] = [
        {"start_sample": 0, "end_sample": rate, "source_pts": source_origin},
        {"start_sample": rate, "end_sample": samples, "source_pts": source_origin + rate + 1},
    ]
    assert measure(wave, broken, review, review["time_base"]) is None
    with pytest.raises(ValueError, match="continuous source coverage"):
        binding.validate(review, packet, broken, review["time_base"])
