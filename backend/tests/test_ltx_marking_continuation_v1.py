"""Codec display labels must not become paths or alias distinct case packets."""
import pytest

from scripts import continue_ltx_marking_audio_v1 as amended


def test_all_eighty_case_identifiers_have_unique_flat_packet_names():
    names = list(amended.old.POLICY['conditions']) + [k + '/' + c for k in amended.old.POLICY['codecs']
        for c in ('unmarked', 'matching-public-payload')]
    identifiers = [f'{r}-{i}-{n}' for r in (24000, 48000) for i in range(4) for n in names]
    actual = [amended.case_id(n) for n in identifiers]
    assert len(set(actual)) == 80 and all('/' not in n for n in actual)
    assert all(amended.case_id(n) == n for n in identifiers if '/' not in n)


@pytest.mark.parametrize('name', ['../outside', 'codec/a/b', 'codec__a', 'codec\\a', '', 'name with spaces'])
def test_unqualified_path_or_ambiguous_separator_is_refused(name):
    with pytest.raises(ValueError):
        amended.case_id(name)


def test_amended_callable_can_be_built_without_loading_models_or_audio():
    function = amended.amended_run()
    assert callable(function)
