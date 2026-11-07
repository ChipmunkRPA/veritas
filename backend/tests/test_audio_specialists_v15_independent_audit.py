"""Reject local-label, ancestry and denominator corruption without inference."""
from copy import deepcopy

import pytest

from scripts.audit_audio_specialists_v15 import CONDITIONS, PATTERNS, local_label, verify


def packet():
    foregrounds = [{'label': label, 'role': 'development', 'sha256': str(label),
        'source_interval_samples': [0, 64000], 'canonical_speaker': f'voice{label}',
        'transcript_sha256': f'text{label}'} for label in (0, 1)]
    backgrounds = [{'sha256': kind, 'kind': kind, 'role': 'specialist_cal', 'full_samples': 160000,
        'canonical_speaker': None} for kind in ('noise', 'music', 'speech')]
    plan = {'pairs': [{'id': 'pair', 'role': 'specialist_cal', 'original_groups': ['old-source'],
        'foregrounds': foregrounds, 'generator': 'published-generator'}], 'backgrounds': backgrounds}
    rows = []
    for condition, (kind, snr) in CONDITIONS.items():
        for pattern, labels in PATTERNS.items():
            windows = [{'start': start, 'end': start + 64000, 'extent': 128000, 'sample_rate': 16000,
                'label': local_label(labels, start, start + 64000), 'quality': {'flatness': .7},
                'context': 'diffuse', 'speech_eligible': False,
                'full_pcm_sha256': f'{condition}-{pattern}-{start}',
                'band_pcm_sha256': f'band-{condition}-{pattern}-{start}'} for start in (0, 32000, 64000)]
            rows.append({'id': f'{condition}-{pattern}', 'group': 'pair', 'role': 'specialist_cal',
                'generator': 'published-generator', 'pattern': pattern, 'labels': list(labels),
                'condition': condition, 'foreground_sha256': [str(v) for v in labels],
                'sample_rate': 16000, 'sample_count': 128000, 'common_peak_gain': .8,
                'background': None if kind is None else {'sha256': kind, 'kind': kind, 'snr_db': snr,
                    'start_sample': 100, 'end_sample': 64100, 'interval_pcm_sha256': kind},
                'windows': windows})
    return plan, rows


def test_unassessed_windows_and_historical_source_roles_are_retained():
    plan, rows = packet()
    result = verify(plan, rows)
    assert result['files'] == 28 and result['windows'] == 84
    assert result['counts']['specialist_cal/speech_unassessed'] == 84
    assert result['eligible_local_support_groups'] == {}


@pytest.mark.parametrize('labels,interval,expected', [
    ([0, 1], (0, 64000), 0), ([0, 1], (63999, 64001), 1),
    ([1, 0], (64000, 128000), 0), ([1, 0], (63999, 64000), 1),
    ([0, 0], (0, 128000), 0), ([1, 1], (0, 128000), 1)])
def test_half_open_local_truth(labels, interval, expected):
    assert local_label(labels, *interval) == expected


@pytest.mark.parametrize('corruption', ['file_truth', 'clock', 'missing_window', 'context', 'gain', 'background_offset', 'missing_recipe', 'reserve'])
def test_source_or_local_corruption_is_rejected(corruption):
    plan, rows = deepcopy(packet())
    if corruption == 'file_truth':
        row = next(r for r in rows if r['pattern'] == 'human-ai')
        row['windows'][0]['label'] = 1
    elif corruption == 'clock':
        rows[0]['windows'][0]['end'] += 1
    elif corruption == 'missing_window':
        rows[0]['windows'].pop()
    elif corruption == 'context':
        rows[0]['windows'][0]['context'] = 'concentrated'
    elif corruption == 'gain':
        rows[1]['common_peak_gain'] = .7
    elif corruption == 'background_offset':
        row = next(r for r in rows if r['condition'] == 'noise_-5' and r['pattern'] == 'ai')
        row['background']['start_sample'] += 1
        row['background']['end_sample'] += 1
    elif corruption == 'missing_recipe':
        rows.pop()
    else:
        plan['pairs'][0]['foregrounds'][0]['role'] = 'confirmation'
    with pytest.raises(ValueError):
        verify(plan, rows)


def test_known_speaker_cannot_cross_roles_even_with_different_bytes():
    plan, rows = packet()
    plan['backgrounds'][0].update(role='confirmation', canonical_speaker='voice0')
    with pytest.raises(ValueError, match='ancestry'):
        verify(plan, rows)
