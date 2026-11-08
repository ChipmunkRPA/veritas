"""Keep researched documentation distinct from qualified file verification."""
import copy
import json

import pytest

from scripts import provider_signal_inventory as inventory


def catalog():
    return json.loads((inventory.ROOT / 'docs/provider-signals-catalog.json').read_text())


def test_published_matrix_matches_the_full_catalog():
    document = catalog()
    assert (inventory.ROOT / 'docs/provider-signal-matrix.md').read_text() == inventory.render(document)
    assert all(set(p['modality_review']) == set(inventory.MODALITIES) for p in document['providers'])


def test_stale_counts_and_duplicate_families_are_rejected():
    document = catalog()
    document['providers'].append(copy.deepcopy(document['providers'][0]))
    with pytest.raises(ValueError, match='Duplicate provider or stale family count'):
        inventory.validate(document)


@pytest.mark.parametrize('field', ['families', 'scoped_records', 'families_with_no_scoped_specification'])
def test_latest_search_cannot_publish_stale_coverage_counts(field):
    document = catalog()
    document['latest_search'][field] += 1
    with pytest.raises(ValueError, match='Latest search counts disagree'):
        inventory.validate(document)


def test_classifier_cannot_become_a_documentary_shortcut():
    document = catalog()
    google = next(p for p in document['providers'] if p['id'] == 'google')
    classifier = next(s for s in google['signals'] if s['kind'] == 'provider_classifier')
    assert classifier['modalities'] == ['image']
    classifier['automated_shortcut'] = 'stop_statistical_detection'
    with pytest.raises(ValueError, match='unqualified provider signal'):
        inventory.validate(document)


def test_future_and_version_limited_claims_are_distinct():
    future = {'kind': 'provider_watermark', 'modalities': ['video'], 'availability': 'announced_future'}
    assert inventory.cell([future], 'video') == 'Planned'
    conditional = {**future, 'availability': 'version_conditional'}
    assert inventory.cell([conditional], 'video') == 'Watermark (version-limited)'


def test_generic_policy_does_not_fill_unestablished_modalities():
    policy = {'kind': 'structured_label', 'modalities': [], 'potential_modalities': list(inventory.MODALITIES)}
    assert all(inventory.cell([policy], m) == 'Gap' for m in inventory.MODALITIES)


def test_customer_applied_metadata_service_is_distinct_from_default_output_marking():
    service = {'kind': 'structured_label', 'modalities': ['image', 'video', 'audio', 'text'],
               'deployment_scope': 'customer_applied_service'}
    assert all(inventory.cell([service], m) == 'Metadata service' for m in inventory.MODALITIES)


def test_unspecified_export_mark_is_not_a_hidden_watermark_claim():
    mark = {'kind': 'export_mark', 'modalities': ['video']}
    assert inventory.cell([mark], 'video') == 'Export mark'
    assert inventory.cell([mark], 'audio') == 'Gap'
