"""New checkout requires the current notice; existing paid orders keep consent."""
import copy

import pytest

from app.billing import purchase_notice as notice
from app.core.canonical import canonical_hash


def old_receipt():
    return {'accepted': True, 'accepted_by': 'owner', 'accepted_at': '2026-11-01T00:00:00Z',
        'notice': copy.deepcopy(notice._V1_NOTICE), 'notice_sha256': canonical_hash(notice._V1_NOTICE)}


def test_historical_purchase_does_not_get_stranded_by_updated_warning():
    saved = old_receipt()
    assert notice.recorded(saved, 'owner')
    assert not notice.validates(True, saved['notice']['version'], saved['notice_sha256'])
    assert notice.validates(True, notice.VERSION, notice.SHA256)
    assert notice.recorded(notice.receipt('owner', '2026-11-07T00:00:00Z', stripe_notice=True), 'owner')


@pytest.mark.parametrize('damage', ['owner', 'words', 'hash', 'hash-list', 'acceptance', 'timestamp'])
def test_historical_disclosure_cannot_be_forged_or_reassigned(damage):
    saved = old_receipt()
    if damage == 'owner':
        saved['accepted_by'] = 'another-owner'
    elif damage == 'words':
        saved['notice']['points'] = []
    elif damage == 'hash':
        saved['notice_sha256'] = 'a' * 64
    elif damage == 'hash-list':
        saved['notice_sha256'] = []
    elif damage == 'acceptance':
        saved['accepted'] = False
    else:
        saved['accepted_at'] = ''
    assert not notice.recorded(saved, 'owner')
