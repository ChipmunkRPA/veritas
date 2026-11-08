"""Render the research coverage matrix; never imported by the serving gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODALITIES = ('image', 'video', 'audio', 'text')
ROOT = Path(__file__).resolve().parents[2]


def review_state(signals, modality):
    applicable = [s for s in signals if modality in s['modalities']]
    if any(s.get('availability') not in ('announced_future', 'historical') for s in applicable):
        return 'scoped_signal_documented'
    if any(s.get('availability') == 'announced_future' for s in applicable):
        return 'announced_future_only'
    if applicable:
        return 'historical_documentation_only'
    return 'not_established_by_this_review'


def cell(signals, modality):
    state = review_state(signals, modality)
    if state != 'scoped_signal_documented':
        return {'announced_future_only': 'Planned', 'historical_documentation_only': 'Historical',
                'not_established_by_this_review': 'Gap'}[state]
    names = {'signed_credentials': 'Credentials', 'provider_watermark': 'Watermark',
             'public_watermark': 'Research', 'keyed_text': 'Keyed text', 'provider_metadata': 'Metadata',
             'structured_label': 'Label policy', 'visible_label': 'Visible label',
             'audible_label': 'Audible label', 'provider_classifier': 'Classifier',
             'publicly_documented_custom_mark': 'Custom mark'}
    applicable = [s for s in signals if modality in s['modalities']
                  and s.get('availability') not in ('announced_future', 'historical')]
    return ', '.join(dict.fromkeys(names[s['kind']] for s in applicable))


def validate(catalog):
    providers = catalog['providers']
    ids = [p['id'] for p in providers]
    if len(set(ids)) != len(ids) or catalog['provider_families'] != len(providers):
        raise ValueError('Duplicate provider or stale family count')
    for provider in providers:
        if set(provider['modality_review']) != set(MODALITIES):
            raise ValueError('Every provider needs all four explicit modality states')
        for modality in MODALITIES:
            if provider['modality_review'][modality] != review_state(provider['signals'], modality):
                raise ValueError('Modality state disagrees with its scoped/announced signals')
        for signal in provider['signals']:
            if not set(signal['modalities']) <= set(MODALITIES):
                raise ValueError('Unsupported modality')
            if not signal['source_urls'] or any(not u.startswith('https://') for u in signal['source_urls']):
                raise ValueError('Every signal needs primary-source HTTPS references')
            if signal['automated_shortcut'] != 'disabled' and not (
                signal['kind'] == 'signed_credentials'
                and signal['verification_route'] == 'local_c2pa'
                and signal['automated_shortcut'] == 'only_after_all_local_credential_requirements'
            ):
                raise ValueError('An unqualified provider signal cannot promise an automated shortcut')
            if not signal['limitations'] or not signal['runtime_status']:
                raise ValueError('Signal scope and runtime qualification must remain explicit')


def render(catalog):
    validate(catalog)
    providers = catalog['providers']
    signals = sum(len(p['signals']) for p in providers)
    gaps = sum(not p['signals'] for p in providers)
    rows = [
        '# Provider marking coverage matrix',
        '',
        f"Inventory updated {catalog['reviewed_utc']}: **{len(providers)} families, {signals} scoped records**, "
        f'with **{gaps} families remaining explicit review gaps**.',
        'Records include planned features, public research and visible notices; these counts are not working detectors.',
        '',
        'Each cell records documentation scope, not measured accuracy or verified production compatibility. '
        '“Gap” means not established by this review, including where the provider does not offer that modality. '
        '“Planned” means an announcement only; “Historical” is not current export coverage. '
        '“Research” identifies a released scheme, without claiming production-provider compatibility. '
        'A generic policy with unresolved export scope does not fill the four cells.',
        '',
        'All external verification routes are disabled for customer files. Only the existing trusted, valid, '
        'current-file-bound, explicitly AI-declaring Content Credentials gate can skip statistical checks. '
        'A video-container audio check covers sound only. A provider classifier score is not a verified watermark hit.',
        '',
        'See the [source catalog](provider-signals-catalog.json), [coverage and gate](provider-provenance-coverage.md) '
        'and [supplemental findings](provider-signal-review-2026-11-08.md) for exact products, controls, dates, '
        'limits, contradictions and qualification requirements.',
        '',
        '| Provider family | Image | Video | Audio | Writing | Reviewed |',
        '|---|---|---|---|---|---|',
    ]
    for provider in providers:
        name = provider['name']
        if provider['signals']:
            name = f"[{name}]({provider['signals'][0]['source_urls'][0]})"
        cells = [cell(provider['signals'], m) for m in MODALITIES]
        rows.append('| ' + ' | '.join([name, *cells, provider['reviewed_utc']]) + ' |')
    rows += ['', 'Generated from the catalog with `python backend/scripts/provider_signal_inventory.py --write`. '
             'Use `--check` to detect stale counts or a matrix that no longer matches the research records.', '']
    return '\n'.join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--check', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'docs/provider-signal-matrix.md'
    text = render(json.loads((ROOT / 'docs/provider-signals-catalog.json').read_text()))
    if args.write:
        output.write_text(text)
    elif args.check:
        if not output.exists() or output.read_text() != text:
            raise SystemExit('Provider matrix is stale; render and review it before publishing.')
        print('Provider catalog and matrix agree.')
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
