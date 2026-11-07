"""History provenance must not be repaired, reordered or inferred from a label."""
import pytest

from scripts.audit_writing_controls_v4 import near, replay, shingles


def event(identifier, kind, offset, size, text=None, order=1):
    return {'id': identifier, 'createdAt': float(identifier), 'user': 'User', 'operationType': kind,
            'offset': offset, 'span': size, 'text': text, 'order': order}


def test_replay_uses_source_order_not_repeated_within_batch_order():
    history = [event(1, 'insert', 0, 1, 'a'), event(2, 'insert', 1, 1, 'b', order=2),
               event(3, 'delete', 0, 1)]
    assert replay(history) == ('b', {'insert': 2, 'delete': 1})


def test_source_coordinates_use_publisher_codepoints_including_astral_and_combining_characters():
    history = [event(1, 'insert', 0, 4, 'A🙂e\u0301'), event(2, 'insert', 2, 1, '!'),
               event(3, 'retain', 1, 2)]
    assert replay(history)[0] == 'A🙂!e\u0301'


@pytest.mark.parametrize('fault', ['insert_bounds', 'delete_bounds', 'retain_bounds', 'bot', 'query', 'repeated_id', 'clock'])
def test_unverifiable_history_refuses_a_human_control(fault):
    history = [event(1, 'insert', 0, 1, 'a')]
    next_event = event(2, 'insert', 1, 1, 'b')
    if fault == 'insert_bounds':
        next_event['offset'] = 2
    elif fault in ('delete_bounds', 'retain_bounds'):
        next_event.update(operationType=fault.split('_')[0], span=2)
    elif fault == 'bot':
        next_event['user'] = 'Bot'
    elif fault == 'query':
        next_event = {'id': 'query', 'documentEditId': 1}
    elif fault == 'repeated_id':
        next_event['id'] = 1
    else:
        next_event['createdAt'] = float('nan')
    with pytest.raises(ValueError):
        replay(history + [next_event])


def test_near_duplicate_rule_catches_contained_passage_but_not_short_common_phrase():
    first = ' '.join('word' + str(i) for i in range(50))
    assert near(shingles(first), shingles('prefix ' + first + ' suffix'))
    assert not near(shingles(first), shingles('word0 word1 word2 word3 word4'))
