"""Data separation, exact local truth and confirmation admission boundaries."""
import copy

import pytest

from scripts import writing_specialists_v5 as method


def test_connected_questions_and_duplicate_answers_are_transitive():
    rows = [
        {'Question': 'Question one?', 'Answer': 'Unique human answer'},
        {'Question': ' QUESTION ONE ! ', 'Answer': 'Shared answer'},
        {'Question': 'Other prompt', 'Answer': 'SHARED answer!'},
        {'Question': 'Unrelated', 'Answer': 'Different answer'},
    ]
    assert sorted(method.source_components(rows)) == [[0, 1, 2], [3]]


def test_slices_preserve_unicode_positions_and_do_not_rewrite_source():
    text = '\t'.join(['café🙂'] * 60) + '\n\n' + ' '.join(['e\u0301'] * 100) + ' trailing'
    short, long = method.slices(text)
    assert short['text'] == text[short['start']:short['end']]
    assert long['text'] == text[long['start']:long['end']]
    assert len(short['text'].split()) == 60
    assert len(long['text'].split()) == 100
    assert short['end'] < long['start']


def test_short_sources_are_not_padded_or_fabricated():
    with pytest.raises(ValueError, match='160'):
        method.slices('word ' * 159)


def plan():
    return {'groups': [{'id': 'authored-group', 'role': 'specialist_cal', 'sources': [
        {'label': label, 'passages': method.slices((('human ' if label == 0 else 'synthetic ') * 160).strip())}
        for label in (0, 1)]}]}


def test_same_source_passages_are_not_counted_as_new_inputs_for_each_composition():
    document = plan()
    assert len(method.inputs(document, False)) == 4
    assert method.inputs(document, True) == {}


def test_duplicate_selected_passage_refuses_a_misleading_independent_group():
    document = plan()
    duplicate = copy.deepcopy(document['groups'][0])
    duplicate['id'] = 'different-declared-group'
    document['groups'].append(duplicate)
    with pytest.raises(ValueError, match='connected source graph'):
        method.inputs(document, False)


def test_pilot_has_no_serving_admission_even_if_quality_improves():
    assert method.METHOD['minimum_groups_per_class'] == 25
    assert method.METHOD['serving_admitted'] is False
    assert 'author IDs absent' in method.METHOD['admission']
