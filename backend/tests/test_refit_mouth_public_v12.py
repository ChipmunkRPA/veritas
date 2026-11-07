import pytest

from scripts.refit_mouth_public_v12 import reconstruct


def rows():
    conditions = ('ai-redraw', 'ai-retime', 'muse-ai-aligned', 'muse-ai-earlier')
    return [{'group': str(i % 16), 'label': (i // 16) % 2, 'role': 'fit', 'features': [0.] * 49,
             'condition': conditions[(i // 32) % 4] if (i // 16) % 2 else 'original'} for i in range(3560)]


@pytest.mark.parametrize('fault', ['omitted', 'point_role', 'boolean_truth', 'missing_class', 'missing_construction', 'nonfinite', 'incomplete_features'])
def test_invalid_or_leaking_fit_inputs_cannot_enter_the_learner(fault):
    data = rows()
    if fault == 'omitted':
        data.pop()
    elif fault == 'point_role':
        data[0]['role'] = 'point'
    elif fault == 'boolean_truth':
        data[0]['label'] = False
    elif fault == 'missing_class':
        for row in data:
            if row['group'] == '0':
                row['label'] = 0
    elif fault == 'missing_construction':
        for row in data:
            if row['condition'] == 'ai-retime':
                row['condition'] = 'ai-redraw'
    elif fault == 'nonfinite':
        data[0]['features'][0] = float('nan')
    else:
        data[0]['features'].pop()
    with pytest.raises(ValueError):
        reconstruct({'fitting_windows': data}, {}, lambda *_: 0.)
