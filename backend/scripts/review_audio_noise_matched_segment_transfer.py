# SPDX-License-Identifier: Apache-2.0
"""Review the pinned aggregate noise/segment transfer without models or source media."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

PIN = '787ed81383d3de834f6d405780b5e4e734e6b5c39be2029214cbfff8ba0da2d4'
CONDITIONS = ('clean', 'washing', 'river', 'restaurant')
BASE_ARMS = {'current', 'raw_short_only', 'score_only_strict_only', 'score_only_wider_only',
             'noise_aware_strict_only', 'noise_aware_wider_only'}
ARMS = BASE_ARMS | {name + '_plus_current' for name in BASE_ARMS - {'current'}}
TOP = {'schema', 'status', 'cohort', 'exposure', 'method', 'conditions', 'primary_gate', 'verification',
       'whole_study_commitment', 'sources', 'privacy', 'public_replay_scope'}
METRICS = {'synthetic_total', 'synthetic_flagged', 'synthetic_unassessed', 'located', 'missed', 'only_wrong',
           'located_and_control_clear', 'ordinary_total', 'ordinary_flagged', 'ordinary_unassessed',
           'background_total', 'background_flagged', 'background_unassessed', 'new_locations',
           'new_clear_locations', 'new_matched_collisions', 'new_ordinary_files', 'new_background_files'}
PRIVACY = {'raw_media', 'transcripts', 'individual_rows', 'reader_identifiers', 'per_recording_hashes',
           'exact_edit_offsets', 'voice_response_vectors', 'voice_embeddings', 'learned_head_weights',
           'local_paths', 'customer_material', 'individual_source_replay_retained_privately'}
SOURCES = {'https://www.openslr.org/94/', 'https://www.openslr.org/12/', 'https://zenodo.org/records/1227121',
           'https://groups.inf.ed.ac.uk/ami/corpus/license.shtml'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Repeated JSON field')
        result[key] = value
    return result


def validate(value):
    require(set(value) == TOP and value['schema'] == 'chipmunk.noise-matched-segment-public/1', 'Unexpected receipt fields')
    cohort = value['cohort']
    require(cohort == {'fit_foreground_readers': 4, 'calibration_foreground_readers': 4,
                       'fit_readers_including_genuine_donors': 8, 'calibration_readers_including_genuine_donors': 8,
                       'shared_declared_fit_calibration_readers': 0, 'old_source_groups': 8,
                       'fit_deliveries': 184, 'calibration_deliveries': 184,
                       'transfer_foreground_readers': 8, 'transfer_recordings': 16, 'transfer_connected_components': 4,
                       'transfer_synthetic_events': 32, 'transfer_conditions': 4, 'transfer_deliveries': 496,
                       'transfer_synthetic_deliveries': 128, 'transfer_ordinary_deliveries': 320,
                       'transfer_background_deliveries': 48, 'transfer_background_sessions': 3,
                       'generator_families': 2, 'generator_voices_per_family': 1,
                       'prior_standalone_deliveries': 124, 'additional_standalone_deliveries': 372}, 'Frozen cohort changed')
    require(cohort['transfer_synthetic_events'] * cohort['transfer_conditions'] == cohort['transfer_synthetic_deliveries']
            and 128 + 320 + 48 == 496 and 124 + 372 == 496, 'Dependent deliveries were miscounted')
    method = value['method']
    require(method['samples'] == 192000 and method['rate'] == 16000 and method['channels'] == 1
            and method['window_samples'] == 12000 and method['step_samples'] == 8000
            and method['windows_per_recording'] == 23 and method['insert_samples'] == 15360, 'Native clock changed')
    require(method['raw_short_point'] == 6.642364579351768 and method['current_two_second_point'] == 6.822523597593649
            and method['new_pretrained_checkpoint'] is False and method['product_weight_changes'] == 0
            and method['overall_confidence_weight'] == 0 and method['supplied_reference_at_inference'] is False
            and method['edit_clocks_supplied_at_inference'] is False, 'Frozen method or scope changed')
    require(method['learned_heads'] == {'score_only_parameters': 2, 'noise_aware_parameters': 19}
            and len(method['feature_names']) == 18 and len(set(method['feature_names'])) == 18
            and method['fit_windows'] == 2530 and method['fit_positive_windows'] == 47
            and method['paired_fit_comparisons'] == 216, 'Fitting metadata changed')
    require(set(value['conditions']) == set(CONDITIONS), 'A noise condition is missing')
    expected = {'current': ([16, 5, 0, 0], [0, 0, 0, 5]),
                'raw_short_only_plus_current': ([28, 8, 0, 0], [0, 0, 0, 5]),
                'score_only_strict_only_plus_current': ([27, 7, 0, 0], [0, 0, 0, 5]),
                'score_only_wider_only_plus_current': ([30, 13, 0, 0], [0, 0, 0, 10]),
                'noise_aware_strict_only_plus_current': ([26, 5, 0, 0], [2, 0, 0, 20]),
                'noise_aware_wider_only_plus_current': ([28, 5, 0, 0], [2, 0, 5, 20])}
    for index, condition in enumerate(CONDITIONS):
        arms = value['conditions'][condition]
        require(set(arms) == ARMS, 'A frozen comparison arm is missing')
        for name, arm in arms.items():
            require(set(arm) == {'counts', 'by_generator', 'new_ordinary_components', 'new_clear_components',
                                 'new_clear_generators'}, 'Unexpected arm fields')
            count = arm['counts']
            require(set(count) == METRICS and all(type(number) is int and number >= 0 for number in count.values()),
                    'Invalid scalar count')
            require(count['synthetic_total'] == 32 and count['ordinary_total'] == 80
                    and count['background_total'] == (0 if condition == 'clean' else 16), 'Missing source roles')
            require(count['located'] + count['missed'] == 32
                    and count['synthetic_flagged'] == count['located'] + count['only_wrong']
                    and count['synthetic_flagged'] + count['synthetic_unassessed'] <= 32
                    and count['ordinary_flagged'] + count['ordinary_unassessed'] <= 80
                    and count['background_flagged'] + count['background_unassessed'] <= count['background_total'],
                    'Location, missed or unassessed arithmetic differs')
            require(count['new_clear_locations'] + count['new_matched_collisions'] == count['new_locations']
                    and count['located_and_control_clear'] <= count['located'], 'Matched-control counts differ')
            family = arm['by_generator']
            require(set(family) == {'piper', 'kitten'}
                    and all(set(item) == {'total', 'located', 'located_and_control_clear'} for item in family.values())
                    and all(type(number) is int and number >= 0 for item in family.values() for number in item.values())
                    and all(item['total'] == 16 for item in family.values())
                    and sum(item['located'] for item in family.values()) == count['located']
                    and sum(item['located_and_control_clear'] for item in family.values()) == count['located_and_control_clear'],
                    'Generator denominator or outcomes changed')
            require(type(arm['new_ordinary_components']) is int and 0 <= arm['new_ordinary_components'] <= 4
                    and type(arm['new_clear_components']) is int and 0 <= arm['new_clear_components'] <= 4
                    and isinstance(arm['new_clear_generators'], list)
                    and set(arm['new_clear_generators']) <= {'piper', 'kitten'}, 'Invalid source-group aggregate')
            if name in expected:
                hits, alarms = expected[name]
                require(count['located'] == hits[index] and count['ordinary_flagged'] == alarms[index],
                        'Counts differ from the independent closure')
            if name.endswith('_plus_current'):
                baseline = arms['current']['counts']
                require(count['located'] >= baseline['located'] and count['ordinary_flagged'] >= baseline['ordinary_flagged']
                        and count['synthetic_unassessed'] <= baseline['synthetic_unassessed'], 'Current findings were displaced')
    gate = value['primary_gate']
    require(gate['candidate'] == 'noise_aware_wider_only_plus_current' and gate['passed'] is False
            and gate['decisions'] == {'new_clear_noisy_locations': False, 'connected_components': False,
                                     'families': False, 'ordinary_files': False,
                                     'ordinary_components': False, 'background': True}, 'Qualification decision changed')
    require(re.fullmatch('[0-9a-f]{64}', value['whole_study_commitment']) is not None, 'Invalid whole-study commitment')
    require(set(value['privacy']) == PRIVACY and value['privacy']['individual_source_replay_retained_privately'] is True
            and all(flag is False for key, flag in value['privacy'].items() if key != 'individual_source_replay_retained_privately'),
            'Individual information would be disclosed')
    require(len(value['sources']) == 4 and {item['url'] for item in value['sources']} == SOURCES
            and all(set(item) == {'scope', 'url', 'terms'} for item in value['sources']), 'Unexpected publisher link')
    encoded = json.dumps(value)
    require(not any(path in encoded for path in ['/private/', '/Users/', '/Volumes/']), 'Local path in public receipt')
    proof = value['verification']
    require(proof['native_input_deliveries_checked'] == 864 and proof['outcome_rows_recounted'] == 496
            and proof['transfer_short_window_clocks_checked'] == 11408
            and proof['transfer_admitted_current_window_bindings_checked'] == 5903
            and proof['all_feature_vectors_reconstructed'] == 11683 and proof['features_per_vector'] == 18
            and proof['prior_current_readings_exact'] == 124 and proof['current_model_forwards'] == 3028
            and proof['authored_arithmetic_checks'] == 9 and proof['product_writes'] == 0, 'Independent proof changed')
    return {'passed': True, 'transfer_deliveries': 496, 'synthetic_events': 32, 'transfer_components': 4,
            'primary_gate_passed': False, 'source_replay_scope': 'Private native replay completed; public aggregate recount only'}


def review(path):
    body = path.read_bytes()
    require(hashlib.sha256(body).hexdigest() == PIN, 'The pinned aggregate receipt changed')
    return validate(json.loads(body, object_pairs_hook=unique))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.receipt), sort_keys=True))


if __name__ == '__main__':
    main()
