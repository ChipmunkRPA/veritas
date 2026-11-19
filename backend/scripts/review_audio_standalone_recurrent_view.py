# SPDX-License-Identifier: Apache-2.0
"""Review the pinned aggregate standalone-audio pilot without media or models."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

PIN = '038a7fc94b89d5d9120729b779993f69c8826005d2a1ff376157ec68dfb4cd2c'
CONDITIONS = ('clean', 'washing', 'river', 'restaurant')
ARMS = ('baseline', 'modified', 'union')
TOP = {'schema', 'status', 'cohort', 'scope', 'exposure', 'method', 'conditions', 'gate',
       'retrospective_diagnostic', 'closed_error_diagnostic', 'verification', 'commitments', 'sources', 'privacy', 'public_replay_scope'}
COUNT_KEYS = {'synthetic_total', 'located', 'missed', 'synthetic_unassessed', 'synthetic_only_off_target',
              'synthetic_with_any_off_target', 'off_target_ranges', 'ordinary_total', 'ordinary_flagged',
              'ordinary_unassessed', 'background_total', 'background_flagged', 'background_unassessed',
              'matched_control_clear_locations'}
SOURCES = {
    'https://www.openslr.org/12/',
    'https://huggingface.co/datasets/openslr/librispeech_asr/tree/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1',
    'https://zenodo.org/records/1227121',
    'https://users.cs.northwestern.edu/~zra446/repet.html',
}
PRIVACY = {'raw_audio', 'transcripts', 'reader_identifiers', 'case_identifiers', 'local_paths',
           'absolute_edit_offsets', 'score_vectors', 'voice_embeddings', 'customer_material',
           'source_media_retained_privately'}
METHOD_KEYS = {'view', 'fft_samples', 'hop_samples', 'remote_neighbors', 'neighbor_exclusion_samples', 'phase',
               'amplitude_floor', 'magnitude_estimate', 'mask', 'classifier', 'review_point', 'speech_admission',
               'union', 'supplied_earlier_copy', 'isolated_background_supplied_to_method', 'edit_clock_supplied_to_method',
               'new_learned_weights', 'overall_confidence_weight'}
COMMITMENTS = {'plan.json', 'prepared.json', 'pilot.py', 'review_inputs.py', 'independent-native-review.json',
               'measurement-closure.json', 'readings.jsonl', 'result.json', 'review_outcomes.py',
               'independent-outcome-review.json', 'diagnostic-plan.json', 'diagnose_cutoffs.py', 'cutoff-diagnostic.json',
               'closed-error-review.json'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, 'Repeated JSON field')
        value[key] = item
    return value


def validate(value):
    """Check aggregate arithmetic; individual source replay stays in the private review."""
    require(set(value) == TOP and value['schema'] == 'chipmunk.standalone-recurrent-audio-public/1',
            'Unexpected public receipt fields')
    cohort = value['cohort']
    expected = {'foreground_readers': 4, 'foreground_recordings': 4, 'synthetic_events': 8,
                'generator_families': 2, 'background_recording_sessions': 3, 'background_excerpts': 12,
                'deliveries': 124, 'synthetic_deliveries': 32, 'ordinary_deliveries': 80,
                'background_deliveries': 12, 'conditions': 4,
                'ordinary_roles_per_host': ['untouched', 'same_reader_splice', 'different_reader_splice',
                                            'roundtrip_piper', 'roundtrip_kitten']}
    require(cohort == expected, 'Frozen cohort changed')
    require(value['scope'] == {'native_rate': 16000, 'channels': 1, 'samples': 192000, 'insert_samples': 15360,
                              'target_background_db': -10, 'raw_speech_minimum_fraction': .5,
                              'correct_edit_coverage_minimum': .5, 'correct_temporal_iou_minimum': .3},
            'Native clocks or matching rule changed')
    method = value['method']
    require(set(method) == METHOD_KEYS and method['review_point'] == 6.822523597593649 and method['overall_confidence_weight'] == 0
            and method['new_learned_weights'] is False
            and all(method[key] is False for key in ['supplied_earlier_copy', 'isolated_background_supplied_to_method',
                                                    'edit_clock_supplied_to_method']),
            'Method scope or frozen point changed')
    require(set(value['exposure']) == {'foreground', 'generators', 'backgrounds', 'pretraining_overlap', 'source_roles'},
            'Unexpected source exposure fields')
    require(set(value['conditions']) == set(CONDITIONS), 'A source condition is missing')
    totals = dict.fromkeys(['synthetic_total', 'ordinary_total', 'background_total'], 0)
    expected_hits = {'baseline': [2, 1, 0, 0], 'modified': [1, 1, 0, 0], 'union': [2, 1, 0, 0]}
    expected_ordinary = {'baseline': [0, 0, 0, 5], 'modified': [0, 0, 0, 1], 'union': [0, 0, 0, 5]}
    expected_unassessed = {'baseline': [0, 0, 4, 4], 'modified': [0, 0, 0, 0], 'union': [0, 0, 0, 0]}
    for index, condition in enumerate(CONDITIONS):
        group = value['conditions'][condition]
        require(set(group) == {*ARMS, 'changes'}, 'Unexpected condition fields')
        for arm in ARMS:
            cell = group[arm]
            require(set(cell) == {'counts', 'by_family'}, 'Unexpected arm fields')
            counts = cell['counts']
            require(set(counts) == COUNT_KEYS and all(type(n) is int and n >= 0 for n in counts.values()),
                    'Invalid aggregate count')
            require(counts['synthetic_total'] == 8 and counts['ordinary_total'] == 20
                    and counts['background_total'] == (0 if condition == 'clean' else 4), 'Missing case roles')
            require(counts['located'] + counts['missed'] == 8
                    and counts['synthetic_unassessed'] <= counts['missed']
                    and counts['synthetic_only_off_target'] <= counts['missed'] - counts['synthetic_unassessed']
                    and counts['synthetic_only_off_target'] <= counts['synthetic_with_any_off_target'] <= 8
                    and counts['off_target_ranges'] >= counts['synthetic_with_any_off_target']
                    and counts['matched_control_clear_locations'] <= counts['located'], 'Inconsistent location counts')
            require(counts['ordinary_flagged'] + counts['ordinary_unassessed'] <= 20
                    and counts['background_flagged'] + counts['background_unassessed'] <= counts['background_total'],
                    'An unassessed file was counted as flagged')
            require(counts['located'] == expected_hits[arm][index]
                    and counts['ordinary_flagged'] == expected_ordinary[arm][index]
                    and counts['synthetic_unassessed'] == expected_unassessed[arm][index]
                    and counts['background_flagged'] == 0, 'Counts differ from the independent closure')
            family = cell['by_family']
            require(set(family) == {'piper', 'kitten'} and all(set(item) == {'total', 'located'} for item in family.values())
                    and all(type(n) is int for item in family.values() for n in item.values())
                    and all(item['total'] == 4 and 0 <= item['located'] <= 4 for item in family.values())
                    and sum(item['located'] for item in family.values()) == counts['located']
                    and family['piper']['located'] == 0, 'Generator counts changed')
        old, new = (group[arm]['counts'] for arm in ['baseline', 'union'])
        require(new['located'] >= old['located'] and new['ordinary_flagged'] >= old['ordinary_flagged']
                and new['synthetic_unassessed'] <= old['synthetic_unassessed'], 'Union arithmetic is inconsistent')
        changes = group['changes']
        require(changes == {'new_locations': 0, 'new_ordinary_files': 0, 'new_readers': 0,
                            'new_generator_families': [], 'matched_control_collisions': 0}, 'Unexpected additions')
        for field in totals:
            totals[field] += old[field]
    require(list(totals.values()) == [32, 80, 12] and sum(totals.values()) == 124, 'Repeated conditions miscounted')
    require(value['gate']['decisions'] == {'added_noisy_locations': False, 'foreground_readers': False,
                                         'generator_families': False, 'ordinary_budget': True,
                                         'background_budget': True, 'matched_control_collisions': True}
            and value['gate']['passed'] is False, 'Qualification decision changed')
    diagnostic = value['retrospective_diagnostic']
    require(set(diagnostic) == {'status', 'candidate_score_boundaries', 'noisy_synthetic_deliveries',
                               'extra_background_budget', 'counts_at_extra_ordinary_budgets_per_condition',
                               'primary_point_unchanged', 'detector_calls'}, 'Unexpected diagnostic fields')
    require(diagnostic['primary_point_unchanged'] is True and diagnostic['detector_calls'] == 0
            and diagnostic['candidate_score_boundaries'] == 969 and diagnostic['noisy_synthetic_deliveries'] == 24
            and diagnostic['extra_background_budget'] == 0, 'Retrospective scope changed')
    require(set(diagnostic['counts_at_extra_ordinary_budgets_per_condition']) == {'0', '1', '2', '4'},
            'Retrospective budgets changed')
    for budget, entry in diagnostic['counts_at_extra_ordinary_budgets_per_condition'].items():
        require(set(entry) == {'new_clear_noisy_locations', 'new_noisy_locations',
                               'total_added_ordinary_files', 'conditions'}, 'Unexpected diagnostic budget fields')
        require(entry['new_clear_noisy_locations'] == entry['new_noisy_locations'] == 1
                and entry['total_added_ordinary_files'] == 0
                and set(entry['conditions']) == set(CONDITIONS), 'Retrospective count changed')
        for condition, counts in entry['conditions'].items():
            require(set(counts) <= {'ordinary_flagged', 'new_ordinary', 'located', 'only_wrong', 'background_flagged',
                                    'new_noisy_locations', 'new_clear_noisy_locations'}
                    and all(type(number) is int and number >= 0 for number in counts.values())
                    and counts.get('new_ordinary', 0) <= int(budget)
                    and counts.get('background_flagged', 0) == 0, 'Retrospective error budget exceeded')
            require(counts['located'] == {'clean': 4, 'washing': 2, 'river': 0, 'restaurant': 0}[condition],
                    'Retrospective locations changed')
    proof = value['verification']
    require(proof == {'native_deliveries_checked': 124, 'original_and_modified_sample_clocks_checked': 248,
                      'independent_modified_waveforms_byte_exact': 124, 'outside_edit_pcm_samples_checked': 16957440,
                      'outcome_rows_recounted': 124, 'admitted_window_clock_bindings_checked': 3145,
                      'actual_classifier_forwards': 1269, 'cpu_threads': 2,
                      'elapsed_seconds': 593.6475601670099, 'weight_changes': 0}, 'Independent proof changed')
    require(value['closed_error_diagnostic'] == {'passed': True, 'ordinary_false_flagged_files': 5,
                                               'synthetic_only_wrong_files': 2, 'connected_foreground_records': 1,
                                               'shared_false_ranges': 1, 'byte_identical_same_clock_excerpts': 7,
                                               'all_shared_false_ranges_outside_every_planted_edit': True,
                                               'new_detector_calls': 0}, 'Closed error custody changed')
    require(set(value['privacy']) == PRIVACY and value['privacy']['source_media_retained_privately'] is True
            and all(flag is False for key, flag in value['privacy'].items() if key != 'source_media_retained_privately'),
            'Unexpected disclosure policy')
    require(set(value['commitments']) == COMMITMENTS
            and all(re.fullmatch('[0-9a-f]{64}', pin) for pin in value['commitments'].values()), 'Invalid whole-study commitment')
    require(len(value['sources']) == 4 and {source['url'] for source in value['sources']} == SOURCES
            and all(set(source) == {'purpose', 'url', 'terms'} for source in value['sources']), 'Unexpected source link')
    return {'passed': True, 'deliveries': 124, 'synthetic_events': 8, 'foreground_readers': 4,
            'added_locations': 0, 'gate_passed': False, 'original_source_replay': 'completed privately; public aggregate recount only'}


def review(path):
    body = path.read_bytes()
    require(hashlib.sha256(body).hexdigest() == PIN, 'The pinned reviewed public capsule changed')
    return validate(json.loads(body, object_pairs_hook=unique))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.receipt), sort_keys=True))


if __name__ == '__main__':
    main()
