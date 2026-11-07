"""Known provenance links must not be counted as independent source groups."""
from scripts.inventory_video_specialists_v18 import connected


def row(group, donor=None, url=None, source_hash=None, input_hash=None):
    return {'group': group, 'donor_filename': donor, 'source_video_url': url,
        'original': {'sha256': source_hash}, 'matched': {'sha256': input_hash}}


def test_donor_and_video_and_byte_links_are_transitive():
    rows = [row('a', donor='face1'), row('b', donor='face1', url='video1'),
        row('c', url='video1', source_hash='bytes1'), row('d', input_hash='bytes1'), row('e')]
    assert sorted(connected(rows)) == [['a', 'b', 'c', 'd'], ['e']]
    assert connected(rows) == connected(list(reversed(rows)))


def test_missing_donors_do_not_connect_all_genuine_clips():
    assert sorted(connected([row('a'), row('b'), row('a')])) == [['a'], ['b']]


def test_different_historical_split_names_cannot_break_known_connections():
    first = {**row('source-fit', donor='shared-photo'), 'split': 'development'}
    second = {**row('source-heldout', donor='shared-photo'), 'split': 'test'}
    assert connected([first, second]) == [['source-fit', 'source-heldout']]


def test_source_urls_and_donor_names_have_distinct_namespaces():
    assert sorted(connected([row('a', donor='same'), row('b', url='same')])) == [['a'], ['b']]


def test_both_raw_and_matched_duplicates_connect_groups():
    assert connected([row('a', source_hash='one'), row('b', input_hash='one')]) == [['a', 'b']]
