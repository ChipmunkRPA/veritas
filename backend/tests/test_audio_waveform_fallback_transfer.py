from types import SimpleNamespace

import numpy as np
import pytest

from scripts import audio_waveform_fallback_transfer as transfer
from scripts.audio_exact_repetition import RepeatedSpan, RepetitionResult
from scripts.audio_repetition_similarity import SimilarityResult, SimilaritySpan


@pytest.mark.parametrize("existing", ("exact", "two_band"))
def test_existing_pair_never_runs_waveform_fallback(monkeypatch, existing):
    result = RepetitionResult("measured", spans=(RepeatedSpan((1600, 4000), (28000, 30400)),))
    empty = RepetitionResult("measured")
    monkeypatch.setattr(transfer, "find_repeated_spans", lambda _pcm: result if existing == "exact" else empty)
    monkeypatch.setattr(transfer, "two_band", lambda _pcm: result if existing == "two_band" else empty)
    monkeypatch.setattr(transfer, "waveform", lambda _pcm: pytest.fail("Existing pair was displaced"))
    measured = transfer.measurements(b"", 1600)
    assert measured["waveform"] is None


def test_empty_and_ambiguous_checks_run_waveform_on_the_same_pcm(monkeypatch):
    pcm = bytes(96000)
    monkeypatch.setattr(transfer, "find_repeated_spans", lambda _pcm: RepetitionResult("measured"))
    monkeypatch.setattr(transfer, "two_band", lambda _pcm: RepetitionResult("ambiguous", "span_budget"))

    def fallback(payload):
        assert payload is pcm
        span = SimilaritySpan((2000, 4880), (28400, 31280), .85, .90, 15)
        return SimilarityResult("measured", spans=(span,))

    monkeypatch.setattr(transfer, "waveform", fallback)
    assert transfer.measurements(pcm, 1600)["waveform"]["correct"]
    assert not transfer.measurements(pcm, None)["waveform"]["correct"]


def test_authored_copy_retains_blend_excluded_native_sample_geometry():
    pcm = np.arange(48000, dtype=np.float64)
    edited, donor = transfer.copy_edit(pcm)
    assert donor == max(transfer.DONORS)
    np.testing.assert_array_equal(edited[:28000], pcm[:28000])
    np.testing.assert_array_equal(edited[28320:38080], pcm[donor + 320:donor + 10080])
    assert edited[28000] == pcm[28000] and edited[38399] == pcm[38399]
    assert transfer.witness(SimpleNamespace(first=(donor + 320, donor + 2720), second=(28352, 30752)), donor, 32)
    assert not transfer.witness(SimpleNamespace(first=(donor + 320, donor + 2720), second=(28353, 30753)), donor, 32)
