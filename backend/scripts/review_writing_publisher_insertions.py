"""Recount the anonymous numbers from the matched publisher insertion pilot."""

import argparse
import json
import re
from pathlib import Path

FORMATS = ('docx_to_docx', 'docx_to_txt', 'txt_to_docx', 'txt_to_txt')
FAMILIES = ('qwen', 'smol2')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def count_comparison(comparison, groups, positives):
    check(set(comparison) == set(FORMATS), 'Incomplete carrier direction coverage')
    results = {}
    for pair in FORMATS:
        counts = comparison[pair]
        for role, total in (('ordinary_revision', groups), ('generated_insertion', positives)):
            row = counts[role]
            check(row['exact_span'] + row['period_choice_differs'] == total, 'Bad insertion bounds tally')
            check(row['exact_span'] <= row['all_wording_found'] <= total, 'Bad wording tally')
            check(0 <= row['earlier_changed_hunks'] <= total, 'Bad earlier change tally')
            check(row['docx_native_locations_checked'] == (total if pair.endswith('to_docx') else 0),
                  'Bad native Word count')
        results[pair] = {key: sum(row[key] for row in counts.values()) for key in (
            'exact_span', 'period_choice_differs', 'all_wording_found', 'earlier_changed_hunks', 'docx_native_locations_checked')}
    return results


def recount(receipt):
    check(receipt['schema'] == 'chipmunk.held-publisher-insertion-public/1', 'Unknown receipt')
    sources = receipt['source_cohorts']
    groups = sources['programming_historian_articles'] + sources['data_carpentry_courses']
    check(groups == sum(sources['insertions_by_sentence_count'].values()), 'Inconsistent source or sentence counts')
    ordinary = receipt['ordinary_controls']
    for stream in ('standalone_before', 'standalone_after', 'paired_after'):
        cohort = ordinary[stream]
        check(0 <= cohort['flags'] <= cohort['assessed'] == groups, f'Bad assessment for {stream}')
    control_files = ordinary['standalone_before']['assessed'] + ordinary['standalone_after']['assessed']
    check(set(receipt['families']) == set(FAMILIES), 'Expected two generator families')
    positive_files = 0
    generation = {}
    for family in FAMILIES:
        row = receipt['families'][family]
        review, rejected = row['review_decisions'], row['rejected_by_shape']
        check(0 <= row['variants'] <= review['equivalent'] <= groups, f'Invalid factual admission for {family}')
        check(sum(review.values()) + sum(rejected.values()) == row['requests'], f'Invalid request ledger for {family}')
        check(row['generated_wins'] + row['ordinary_wins'] + row['tie'] == row['variants'], 'Incomplete paired ranking')
        check(0 <= row['incumbent_exact_alert'] <= row['incumbent_any_alert'] <= row['variants'], 'Invalid standalone counts')
        check(0 <= row['paired_directional'] <= row['variants'], 'Invalid paired counts')
        check(0 <= row['ordinary_directional_on_same_groups'] <= row['variants'], 'Invalid matched control counts')
        positive_files += row['variants']
        generation[family] = {'requests': row['requests'], 'shape_pass': sum(review.values()),
                              'factual_pass': review['equivalent'], 'retained': row['variants'],
                              'standalone_exact': row['incumbent_exact_alert'], 'paired_directional': row['paired_directional'],
                              'rank_generated_before_ordinary': row['generated_wins']}
    check(max(row['retained'] for row in generation.values()) <= sources['positive_source_groups']
          <= min(groups, positive_files), 'Insufficient whole-work accounting')
    carrier = receipt['carrier_geometry']
    check(carrier['archive_files_txt_and_docx'] == control_files * 2, 'Unbalanced archive carrier count')
    check(carrier['generated_files_txt_and_docx'] == positive_files * 2, 'Unbalanced generated carrier count')
    comparators = receipt['product_draft_comparison']
    check(set(comparators) == {'prior', 'current'}, 'Incomplete comparison versions')
    production = {mode: count_comparison(results, groups, positive_files) for mode, results in comparators.items()}
    for field, mark in receipt['commitments'].items():
        length = 40 if field == 'application_revision' else 64
        check(re.fullmatch(f'[0-9a-f]{{{length}}}', mark) is not None, 'Malformed commitment')
    return {'work_groups': groups, 'positive_work_groups': sources['positive_source_groups'],
            'standalone_assessed': control_files + positive_files, 'ordinary': ordinary,
            'carrier_replays': sum(carrier.values()), 'generation': generation, 'production': production}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    print(json.dumps(recount(json.loads(args.receipt.read_bytes())), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
