"""Numerical comparisons keep source hashes/decisions strict and check failures."""
import pytest

from scripts.reproduce_preserved_v22 import close, normal


def test_only_floating_roundoff_is_tolerated():
    assert close({'p': .1}, {'p': .10000000001})['maximum_absolute_difference'] < 1e-6
    for actual, prior in [({'status': 'ok'}, {'status': 'inconclusive'}),
            ({'count': 2}, {'count': 3}), ({'hash': 'b'}, {'hash': 'a'}), ({'p': 0.}, {'p': None})]:
        with pytest.raises(ValueError):
            close(actual, prior)


def test_computed_artifact_hash_is_ignored_but_source_hashes_stay_bound():
    source = {'sha256': 'derived-output-digest', 'training_sha256': 'source-digest',
        'specialists': {'a': {'numerical_check': {'iterations': 5, 'independent_kkt_infinity': 1e-12}}},
        'combiners': {}}
    output = normal(source, 'artifact')
    assert output['training_sha256'] == 'source-digest'
    assert 'sha256' not in output
    assert output['specialists']['a']['numerical_check'] == {'independent_kkt_infinity': 1e-12}
    assert source['specialists']['a']['numerical_check']['iterations'] == 5
