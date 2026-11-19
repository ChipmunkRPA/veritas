"""Validate and recount the public tallies for the balanced compression pilot."""

import argparse
import json
import re
from fractions import Fraction
from pathlib import Path

METHODS = ('incumbent', 'masked', 'lexical')
DOMAINS = ('technical', 'fiction')


def require(test, reason):
    if not test:
        raise ValueError(reason)


def tallies(receipt):
    require(receipt['schema'] == 'chipmunk.writing-balanced-code-length-anonymous/1', 'Unsupported receipt')
    intake = receipt['fiction_intake']
    review = intake['review']
    require(set(review) == {'equivalent', 'changed', 'unclear'}, 'Incomplete semantic review tally')
    require(sum(review.values()) == intake['shape_qualified'] <= intake['generation_requests'], 'Inconsistent generation counts')
    fiction_works = intake['qualified_whole_works']
    require(0 < fiction_works <= intake['candidate_whole_works'], 'Inconsistent fiction intake counts')
    technical_positives = sum(receipt['technical_generator_positives'].values())
    metrics = receipt['metrics']
    totals = {}
    for method in METHODS:
        totals[method] = {}
        for domain in DOMAINS:
            row = metrics[method][domain]
            controls, positives = row['ordinary_files'], row['positive_files']
            groups = row['ordinary_groups']
            require(row['assessed'] == controls + positives, 'Incomplete assessed count')
            require(0 <= row['positive_found'] <= positives, 'Invalid positive count')
            require(0 <= row['ordinary_groups_flagged'] <= min(row['ordinary_files_flagged'], groups) <= controls,
                    'Invalid ordinary alert count')
            require(controls == groups * (3 if domain == 'fiction' else 2), 'Unbalanced ordinary controls')
            require(positives == (review['equivalent'] if domain == 'fiction' else technical_positives), 'Inconsistent positive labels')
            if domain == 'fiction':
                require(groups == fiction_works, 'Fiction denominator differs from intake')
            if method != 'incumbent':
                baseline = metrics['incumbent'][domain]
                require(all(row[key] == baseline[key] for key in ('assessed', 'ordinary_files', 'positive_files', 'ordinary_groups')),
                        'Methods use different denominators')
                require(row['positive_found'] <= row['positive_rank_one_unique'] <= positives, 'Incorrect preselection count')
            totals[method][domain] = {
                'exact_locations': [row['positive_found'], positives],
                'ordinary_file_alerts': [row['ordinary_files_flagged'], controls],
                'ordinary_work_alerts': [row['ordinary_groups_flagged'], groups],
            }
    replay = metrics['fiction_native_replay']
    require(replay['files'] == metrics['incumbent']['fiction']['assessed'] * 2, 'Missing carrier replay')
    require(replay['source_anchors'] == 2 * (fiction_works * 2 + review['equivalent']), 'Missing fiction source positions')
    for detail in receipt['points'].values():
        threshold = Fraction(*detail['threshold_fraction'])
        require(0 <= threshold <= 1 and 0 < detail['dictionary_bytes'] <= 32768, 'Invalid operating point')
        require(0 <= detail['calibration_groups_flagged'] <= 1 < receipt['calibration_works'], 'Invalid ordinary calibration counts')
    require(len(receipt['commitments']) == 6 and all(re.fullmatch(r'[0-9a-f]{64}', value) for value in receipt['commitments'].values()),
            'Invalid private commitment')
    return {'corpora':totals,'fiction_native_replay':replay,'fiction_admitted_out_of_requests':[review['equivalent'],intake['generation_requests']]}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('receipt', type=Path)
    args = cli.parse_args()
    print(json.dumps(tallies(json.loads(args.receipt.read_bytes())), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
