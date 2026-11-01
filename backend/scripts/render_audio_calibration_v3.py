"""Render the completed offline study as an English local report and CSV."""
from __future__ import annotations

import csv
import html
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/independent-evaluation/audio-v3'
OUT = ROOT.parent / 'output/evaluation'


def load(name):
    return json.loads((DATA / name).read_bytes())


def esc(value):
    return html.escape(str(value))


def pct(value):
    return 'Not established' if value is None else f'{100 * value:.1f}%'


def number(value):
    return 'Not established' if value is None else f'{value:.4f}'


def interval(value):
    bounds = value.get('percentile_95')
    return 'Not established' if bounds is None else f'[{bounds[0]:+.4f}, {bounds[1]:+.4f}]'


def plot_outcomes(rows, conditions):
    os.environ.setdefault('MPLCONFIGDIR', str(DATA / 'plot-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = ['#388376', '#c45243', '#d7dcd6']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    fig.patch.set_facecolor('white')
    labels = list(conditions.values())
    for axis, truth, title in zip(axes, (1, 0), ('Known AI speech', 'Known authentic speech'), strict=True):
        totals = []
        correct_decision, wrong_decision = ('ai', 'original') if truth else ('original', 'ai')
        for condition in conditions:
            selected = [r for r in rows if r['label'] == truth and r['condition'] == condition]
            counts = Counter(r['new_guarded'] for r in selected)
            totals.append([counts[correct_decision], counts[wrong_decision], counts['inconclusive']])
        left = [0] * len(totals)
        for column, color in enumerate(colors):
            values = [r[column] for r in totals]
            axis.barh(range(len(labels)), values, left=left, height=.63, color=color, edgecolor='white', linewidth=.7)
            for y, (value, start) in enumerate(zip(values, left, strict=True)):
                if value >= 8:
                    axis.text(start + value / 2, y, str(value), ha='center', va='center', fontsize=9,
                              color='white' if column < 2 else '#172f34')
            left = [start + value for start, value in zip(left, values, strict=True)]
        axis.set_title(title, loc='left', fontsize=12, fontweight='bold', color='#172f34', pad=15)
        axis.set_yticks(range(len(labels)), labels, fontsize=10)
        axis.set_xlim(0, max(sum(r) for r in totals))
        axis.set_xlabel('Test inputs in each sound condition', fontsize=10, labelpad=10)
        axis.spines[['top', 'right', 'left']].set_visible(False)
        axis.spines['bottom'].set_color('#aebeb8')
        axis.tick_params(axis='y', length=0)
        axis.tick_params(axis='x', labelsize=9, colors='#587074')
    axes[0].invert_yaxis()
    fig.legend([Patch(facecolor=c) for c in colors], ['Correct conclusion', 'Incorrect conclusion', 'Inconclusive'],
               loc='lower center', bbox_to_anchor=(.5, .025), ncol=3, frameon=False, fontsize=10)
    fig.subplots_adjust(left=.12, right=.98, top=.87, bottom=.22, wspace=.13)
    fig.savefig(OUT / 'audio-calibration-v3-outcomes.png', dpi=170, facecolor='white')
    plt.close(fig)


def main():
    report, rows, manifest = load('evaluation.json'), load('predictions.json'), load('manifest.json')
    audit = load('audits/independent-postrun-audit.json')
    if audit.get('status') != 'point_metrics_calibration_and_bootstrap_intervals_verified':
        raise ValueError('The completed study requires an independent post-run audit')
    for name, field in [('evaluation.json', 'evaluation_sha256'), ('predictions.json', 'predictions_sha256')]:
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != audit.get(field):
            raise ValueError('Report input differs from its independent audit: ' + name)
    if audit['manifest_sha256'] != manifest['sha256']:
        raise ValueError('Report and audited cohort differ')
    OUT.mkdir(exist_ok=True, parents=True)
    csv_path = OUT / 'audio-calibration-v3.csv'
    with csv_path.open('w', newline='') as stream:
        fields = ['source_id', 'voice_group', 'truth', 'generator', 'condition', 'background_id', 'seconds',
            'existing_status', 'existing_probability_ai', 'existing_decision', 'candidate_status', 'candidate_probability_ai', 'candidate_before_stability',
            'candidate_decision', 'stability_veto', 'probe_status', 'probe_skip_reason', 'probe_error',
            'filter_log_odds_change', 'input_sha256', 'pcm_sha256']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(source_id=r['base_id'], voice_group=r['cluster_id'], truth='AI' if r['label'] else 'Authentic',
                generator=r.get('generator'), condition=r['condition'], background_id=r.get('background_id'), seconds=r['seconds'],
                existing_status=r['old_status'], candidate_status=r['new_status'],
                existing_probability_ai=r['old_p'], existing_decision=r['old_guarded'], candidate_probability_ai=r['new_p'],
                candidate_before_stability=r['new_unguarded'], candidate_decision=r['new_guarded'],
                stability_veto=r['new_decision_detail']['original_stability_veto'],
                probe_status=r['new_views'].get('probe_status', 'original_scoring_failed'),
                probe_skip_reason=r['new_views'].get('probe_skip_reason'), probe_error=r['new_views'].get('probe_error'),
                filter_log_odds_change=r['new_decision_detail']['sensitivity']['absolute_log_odds_delta'],
                input_sha256=r['sha256'], pcm_sha256=r['pcm_sha256']))
    names = {'old_guarded': 'Existing detector, refitted', 'new_unguarded': 'Candidate before stability check',
             'new_guarded': 'Candidate with stability check'}
    conditions = {'clean': 'Clean', 'noise_0': 'Noise · 0 dB', 'noise_5': 'Noise · 5 dB', 'noise_15': 'Noise · 15 dB',
                  'music_5': 'Music · 5 dB', 'music_15': 'Music · 15 dB'}
    plot_outcomes(rows, conditions)
    table = []
    for condition, label in conditions.items():
        for system, name in names.items():
            m = report['groups'][condition][system]
            table.append(f'''<tr data-system="{system}"><td>{label}</td><td>{name}</td>
                <td>{m['false_ai_count']} / {m['real_inputs']}<small>{pct(m['false_ai_rate'])}</small></td>
                <td>{m['false_original_count']} / {m['ai_inputs']}<small>{pct(m['false_original_rate'])}</small></td>
                <td>{pct(m['real_coverage'])}</td><td>{pct(m['ai_coverage'])}</td>
                <td>{pct(m['original_call_risk'])}</td></tr>''')
    charts = []
    for condition, label in conditions.items():
        selected = [r for r in rows if r['condition'] == condition and r['label'] == 1]
        counts = Counter(r['new_guarded'] for r in selected)
        n = len(selected)
        charts.append(f'''<div class="bar-row"><span>{label}</span><div class="bar" aria-label="{label}: {counts['ai']} correctly flagged, {counts['original']} wrongly cleared, {counts['inconclusive']} inconclusive">
            <b class="correct" style="width:{100*counts['ai']/n}%"></b><b class="wrong" style="width:{100*counts['original']/n}%"></b>
            <b class="uncertain" style="width:{100*counts['inconclusive']/n}%"></b></div><span>{counts['ai']} / {counts['original']} / {counts['inconclusive']}</span></div>''')
    sources = manifest['base_files']
    cohort = []
    for split in ('fit', 'cal', 'test'):
        selected = [r for r in sources if r['split'] == split]
        for label in (0, 1):
            items = [r for r in selected if r['label'] == label]
            cohort.append(f'<tr><td>{split.title()}</td><td>{"AI" if label else "Authentic"}</td><td>{len(items)}</td><td>{len({r["cluster_id"] for r in items})}</td><td>{len(items)*6}</td></tr>')
    overall = report['groups']['all']['new_guarded']
    probability_only = report['groups']['all']['new_probability_only']
    calibration_guard = report['groups']['all']['new_unguarded']
    vetoed = [r for r in rows if r['new_decision_detail']['original_stability_veto']]
    avoided_wrong_original = sum(r['label'] == 1 for r in vetoed)
    withheld_correct_original = sum(r['label'] == 0 for r in vetoed)
    loud = report['groups']['noise_0']['new_guarded']
    strong = report['groups']['noise_5']['new_guarded']
    music = report['groups']['music_5']['new_guarded']
    original_bound = overall['false_original_identity_event_bound']
    failures = report['admission']['failed_gates']
    failed_calls = [r for r in rows if (r['label'] == 1 and r['new_guarded'] == 'original') or (r['label'] == 0 and r['new_guarded'] == 'ai')]
    error_rows = ''.join(f'''<tr><td class="code">{esc(r['base_id'])}</td><td>{esc(r['condition'])}</td><td>{'AI' if r['label'] else 'Authentic'}</td>
        <td>{esc(r['new_guarded'])}</td><td>{pct(r['new_p'])}</td><td>{esc(r.get('generator'))}</td></tr>''' for r in failed_calls[:30])
    if not error_rows:
        error_rows = '<tr><td colspan="6">No incorrect candidate conclusions were observed in this cohort. Check coverage and finite-sample bounds before interpreting this.</td></tr>'
    probability = []
    for system, name in names.items():
        values = report['common_assessed'][system]
        probability.append(f'<tr><td>{name}</td><td>{number(values["brier"])}</td><td>{number(values["log_loss"])}</td></tr>')
    common = [r for r in rows if r['old_p'] is not None and r['new_p'] is not None]
    comparison_intervals = []
    for metric, label in [('brier', 'Brier'), ('log_loss', 'Log loss')]:
        voice = report['uncertainty']['intervals']['voice']['all']['new_minus_old:' + metric]
        crossed = report['uncertainty']['intervals']['voice_background']['all']['new_minus_old:' + metric]
        comparison_intervals.append(f'<tr><td>{label}</td><td>{interval(voice)}</td><td>{interval(crossed)}</td></tr>')
    family_rows = []
    for group, systems in report['groups'].items():
        if not group.startswith('generator:'):
            continue
        m = systems['new_guarded']
        family_rows.append(f'<tr><td>{esc(group.split(":",1)[1])}</td><td>{m["ai_inputs"]}</td><td>{m["false_original_count"]}</td><td>{pct(m["ai_coverage"])}</td><td>{pct(m["correct_ai_recall"])}</td></tr>')
    family_noise_rows = []
    for family in sorted({r['generator'] for r in rows if r['label'] == 1}):
        for condition in ('noise_0', 'noise_5'):
            selected = [r for r in rows if r['generator'] == family and r['condition'] == condition]
            counts = Counter(r['new_guarded'] for r in selected)
            family_noise_rows.append(f'<tr><td>{esc(family)}</td><td>{conditions[condition]}</td><td>{len(selected)}</td><td>{counts["ai"]}</td><td>{counts["original"]}</td><td>{counts["inconclusive"]}</td></tr>')
    ci_rows = []
    operating_rows = []
    for condition, label in conditions.items():
        m = report['groups'][condition]['new_guarded']
        a, b = m['false_ai_identity_event_bound'], m['false_original_identity_event_bound']
        ci_rows.append(f'<tr><td>{label}</td><td>{a["events"]} / {a["identities"]}</td><td>{pct(a["one_sided_95_upper"])}</td><td>{b["events"]} / {b["identities"]}</td><td>{pct(b["one_sided_95_upper"])}</td></tr>')
        old = report['groups'][condition]['old_guarded']
        operating_rows.append(f'<tr><td>{label}</td><td>{pct(m["threshold_fpr"])}</td><td>{pct(m["threshold_recall"])}</td><td>{pct(m["threshold_false_ai_identity_event_bound"]["one_sided_95_upper"])}</td><td>{pct(old["threshold_fpr"])}</td><td>{pct(old["threshold_recall"])}</td></tr>')
    decision = 'Not admitted to overall confidence' if not report['admission']['deployment_eligible'] else 'Admission recorded'
    text = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Chipmunk Verify — Audio validation v3</title><style>
    :root{{color-scheme:light;--ink:#172f34;--muted:#587074;--orange:#c45f2d;--line:#dfe7e5}}*{{box-sizing:border-box}}
    body{{margin:0;background:#f4f6f3;color:var(--ink);font:15px/1.55 system-ui,-apple-system,sans-serif}}main{{max-width:1250px;margin:auto;padding:36px 32px 70px}}
    header{{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding-bottom:20px}}.brand{{font-weight:800;font-size:21px;letter-spacing:-.5px}}.brand i{{color:var(--orange);font-style:normal}}
    .tag{{padding:5px 11px;border:1px solid #cbd8d4;border-radius:30px;font-size:12px;font-weight:650}}h1{{font-size:42px;line-height:1.12;max-width:850px;letter-spacing:-1.3px;margin:34px 0 18px}}h2{{font-size:23px;letter-spacing:-.4px;margin:0 0 10px}}p{{max-width:950px;color:var(--muted)}}.lede{{font-size:18px;max-width:970px}}.status{{background:#fff0e5;border-left:4px solid var(--orange);padding:18px 22px;margin:25px 0;border-radius:0 12px 12px 0}}.status strong{{font-size:20px}}.status p{{margin:8px 0 0}}
    .cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:26px 0}}.card{{background:white;border:1px solid var(--line);border-radius:12px;padding:20px}}.card b{{font-size:29px;display:block;line-height:1.1;margin-bottom:10px}}.card span{{font-size:13px;color:var(--muted)}}section{{background:#fff;border:1px solid var(--line);border-radius:15px;padding:26px;margin:22px 0}}a{{color:#975025;font-weight:650}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;text-align:left}}th{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px;background:#f5f8f6}}th,td{{padding:12px 10px;border-bottom:1px solid var(--line);vertical-align:top}}td small{{display:block;color:var(--muted)}}.code{{font:11px ui-monospace,monospace;overflow-wrap:anywhere;max-width:360px}}.bar-row{{display:grid;grid-template-columns:140px 1fr 155px;align-items:center;gap:16px;margin:15px 0;font-size:13px}}.bar{{height:25px;display:flex;overflow:hidden;border-radius:5px;background:#e4e8e5}}.correct{{background:#388376}}.wrong{{background:#c45243}}.uncertain{{background:#d7dcd6}}.legend{{display:flex;gap:20px;font-size:12px;margin:20px 0}}.dot{{display:inline-block;width:10px;height:10px;margin-right:6px;border-radius:50%}}select{{padding:8px;border:1px solid var(--line);border-radius:7px;background:#fff;color:var(--ink);margin:10px 0 18px}}details{{border-top:1px solid var(--line);padding-top:14px;margin-top:18px}}summary{{cursor:pointer;font-weight:700}}li{{margin:7px 0;color:var(--muted)}}footer{{font-size:12px;color:var(--muted);margin-top:30px}}@media(max-width:750px){{main{{padding:20px 15px}}h1{{font-size:32px}}.cards{{grid-template-columns:repeat(2,1fr)}}section{{padding:18px}}.bar-row{{grid-template-columns:100px 1fr}}.bar-row>span:last-child{{grid-column:2}}}}
    </style></head><body><main><header><div class="brand"><i>Chipmunk</i> Verify</div><span class="tag">LOCAL RESEARCH · ENGLISH AUDIO</span></header>
    <h1>Does the audio check hold up under loud background sound?</h1><p class="lede">A frozen test of {len({r['base_id'] for r in rows})} new source recordings, three AI voice generators excluded from our fitting and calibration data, and six sound conditions. Multiple passages or noise versions of a voice are not independent people.</p>
    <div class="status"><strong>{decision}</strong><p>{esc(report['admission']['decision'])} Candidate contribution to the overall confidence score remains <b>0</b>. Existing model files and serving settings were {'unchanged' if report['serving_unchanged'] else 'changed; inspect audit'}.</p></div>
    <p class="lede">The 80% probability rule alone wrongly cleared <b>{probability_only['false_original_count']} AI inputs as original</b> and incorrectly flagged {probability_only['false_ai_count']} authentic inputs as AI. Requiring the separate calibration check reduced those counts to <b>{calibration_guard['false_original_count']} and {calibration_guard['false_ai_count']}</b>, while conclusion coverage fell from {pct(probability_only['coverage'])} to {pct(calibration_guard['coverage'])}. The improvement comes from leaving uncertain cases inconclusive.</p>
    <p>The additional stability check withheld {withheld_correct_original} correct original conclusions and prevented {avoided_wrong_original} additional wrong clearances on this fresh test. It does not create additional AI detections or improve the underlying probability. Zero observed wrong clearances is not proof of zero risk.</p>
    <div class="cards"><div class="card"><b>{len({r['base_id'] for r in rows})}</b><span>held-out source recordings</span></div><div class="card"><b>{len(rows):,}</b><span>test inputs including sound variants</span></div><div class="card"><b>{overall['false_original_count']} / {overall['ai_inputs']}</b><span>AI inputs incorrectly cleared as original</span></div><div class="card"><b>{pct(overall['coverage'])}</b><span>candidate conclusion coverage</span></div></div>
    <section><h2>How did the candidate classify the test audio?</h2><p>Green means the conclusion matches the known source class. Red means an incorrect conclusion. Grey means inconclusive, including files without enough detected speech. At 0 dB, the added noise has the same whole-clip RMS level as the clean source, including pauses.</p><img src="audio-calibration-v3-outcomes.png" alt="Correct, incorrect and inconclusive conclusions for known AI and authentic speech under each sound condition" style="width:100%;height:auto"><p>Each bar represents {overall['ai_inputs']//6} source files under one sound condition. The same sources appear in each condition; these bars are not independent samples. <a href="audio-calibration-v3-outcomes.png">Open the chart</a>.</p></section>
    <section><h2>Errors and useful conclusions</h2><p>The candidate keeps one probability from the original audio. A fixed speech-band check can withhold an original conclusion when the score is unstable; it cannot turn a recording into an AI verdict. All three systems require at least 80% reference probability for a class and a calibration check that supports only that class. If the check also leaves the other class plausible, the result stays inconclusive. The guarded candidate additionally requires its stability check before clearing a recording as original. The band probe runs only when that check can affect an initial original conclusion; other probes are intentionally skipped.</p><label for="system">Show </label><select id="system"><option value="all">All three systems</option><option value="new_guarded">Candidate with stability check</option><option value="new_unguarded">Candidate before stability check</option><option value="old_guarded">Existing detector, refitted</option></select><div class="scroll"><table><thead><tr><th>Condition</th><th>System</th><th>Authentic wrongly flagged AI</th><th>AI wrongly cleared original</th><th>Authentic coverage</th><th>AI coverage</th><th>Error among original calls</th></tr></thead><tbody>{''.join(table)}</tbody></table></div><p>Coverage is the share receiving a conclusion, not accuracy. An inconclusive result is never counted as correct. <a href="audio-calibration-v3.csv">Download all per-input results (CSV)</a>.</p></section>
    <section><h2>Calibration on the same assessed inputs</h2><p>Lower scores are better. These comparisons use {len(common)} identical assessed inputs ({sum(r["label"] == 0 for r in common)} authentic and {sum(r["label"] == 1 for r in common)} AI), with a balanced reference class prior. Withholding a conclusion does not remove a hard example from these probability scores.</p><table><thead><tr><th>System</th><th>Brier score</th><th>Log loss</th></tr></thead><tbody>{''.join(probability)}</tbody></table><p>The stability check leaves the candidate's probability unchanged, so it cannot improve these two scores by itself.</p><h3>95% intervals for candidate minus existing detector</h3><table><thead><tr><th>Metric</th><th>Paired voice clusters</th><th>Voice and background clusters</th></tr></thead><tbody>{''.join(comparison_intervals)}</tbody></table><p>Negative differences favor the candidate. Intervals condition on the frozen calibration and tested source populations.</p></section>
    <section><h2>Results across the held-out generators</h2><p>Generator names below identify the labelled test sources, not the detection technology used in customer reports. Counts include six sound variants of each original.</p><table><thead><tr><th>Test source generator</th><th>AI inputs</th><th>Wrongly original</th><th>AI coverage</th><th>Correct AI recall</th></tr></thead><tbody>{''.join(family_rows)}</tbody></table><h3>Strong-noise detail by generator</h3><p>These are descriptive results for every held-out generator, not separately selected release scopes.</p><table><thead><tr><th>Test generator</th><th>Condition</th><th>AI inputs</th><th>Correct AI</th><th>Wrongly original</th><th>Inconclusive</th></tr></thead><tbody>{''.join(family_noise_rows)}</tbody></table></section>
    <section><h2>How much uncertainty remains?</h2><p>Each voice is counted once if any of its relevant files makes an incorrect call. The upper bounds are one-sided 95% identity-event bounds, not customer confidence intervals. Shared background sounds and unpublished exact prompt mappings further limit generalization.</p><div class="scroll"><table><thead><tr><th>Condition</th><th>Authentic voices with an error</th><th>Upper false-AI bound</th><th>AI voices with an error</th><th>Upper false-original bound</th></tr></thead><tbody>{''.join(ci_rows)}</tbody></table></div><p>Even zero observed errors has a positive upper risk bound. The study also uses 2,000 paired voice-cluster resamples and 2,000 crossed voice/background resamples; shared authentic and generated identities stay together.</p></section>
    <section><h2>The separate false-positive benchmark</h2><p>For this benchmark, an AI-flag threshold is selected using only authentic calibration voices, targeting a nominal 5% false-AI rate. That target is not a guarantee and is separate from the 80% conclusion rule. The table shows what actually happened on the held-out test. An operating point that could not be established stays unknown; it is not treated as zero error. Missing authentic calibration scores prevented the existing detector from establishing this threshold. Its separate conclusion calibration also left every test result inconclusive. These are results for the refitted offline comparator, not a description of the current live app.</p><div class="scroll"><table><thead><tr><th>Condition</th><th>Candidate false AI</th><th>Candidate AI recall</th><th>95% upper false-AI bound</th><th>Existing false AI</th><th>Existing AI recall</th></tr></thead><tbody>{''.join(operating_rows)}</tbody></table></div></section><section><h2>Recorded incorrect conclusions</h2><p>Up to 30 failures are shown here. The CSV contains every input, including failures and inconclusive results. A probability below 20% AI is not proof that the file is authentic.</p><div class="scroll"><table><thead><tr><th>Source</th><th>Condition</th><th>Known label</th><th>Conclusion</th><th>Reference p(AI)</th><th>Source generator</th></tr></thead><tbody>{error_rows}</tbody></table></div></section>
    <section><h2>Study design and release decision</h2><table><thead><tr><th>Split</th><th>Class</th><th>Source files</th><th>Voice groups</th><th>Derived inputs</th></tr></thead><tbody>{''.join(cohort)}</tbody></table><p>Fitting, rejection calibration and test inputs were selected separately before scoring. Original bytes, production-decoded PCM, source intervals, transformations, code, weights and calibration are hashed. Prior study voices and recordings are excluded; exact DFADD prompt mappings are unpublished and text independence is not fully established.</p><h3>Why this is not ready for overall confidence</h3><p>Only {pct(loud['real_coverage'])} of authentic recordings received a conclusion under 0 dB noise, and {pct(music['real_coverage'])} under 5 dB music. Both fall below the required 50% coverage. Some noisy conditions also lack enough evidence that false-AI errors stay within the allowed increase over the existing detector.</p><p>The {original_bound['identities']} AI voice groups produced no wrong original calls, but the one-sided 95% upper bound is still {pct(original_bound['one_sided_95_upper'])}. This is not precise enough to establish the required two-percentage-point error margin. The existing detector also lacks an established operating point for the separate 5% false-AI benchmark, so a comparable gain in AI recall remains unknown.</p><p>The independent post-run audit passed. Statistical admission still failed, so the candidate remains research-only and no existing model is removed.</p><details><summary>{len(failures)} unmet statistical gates — technical detail</summary><ul>{''.join('<li>'+esc(f)+'</li>' for f in failures) or '<li>Statistical gates passed; scoped release and independent audit remain separate requirements.</li>'}</ul></details><details><summary>Limits of this test</summary><ul>{''.join('<li>'+esc(v)+'</li>' for v in report['limitations'])}</ul></details></section>
    <footer>Chipmunk Verify · Offline development report · The separate 5% false-AI benchmark uses a threshold selected only from authentic calibration voices; it is not the 80% conclusion rule or a promised error rate.<br>No customer charge or production calibration change.<br>Sources: <a href="https://huggingface.co/datasets/realnetworks-kontxt/fleurs-hs">FLEURS-HS</a> · <a href="https://huggingface.co/datasets/isjwdu/DFADD">DFADD</a> · <a href="https://www.openslr.org/12/">LibriSpeech</a> · <a href="https://datashare.ed.ac.uk/handle/10283/3443">VCTK</a> · <a href="https://www.openslr.org/17/">MUSAN</a>. Individual rights, attribution and source receipts are retained in the study manifest.<br>Result digest: <span class="code">{esc(report['scores_sha256'])}</span></footer></main><script>document.getElementById('system').addEventListener('change',e=>{{document.querySelectorAll('tr[data-system]').forEach(r=>r.hidden=e.target.value!=='all'&&r.dataset.system!==e.target.value)}})</script></body></html>'''
    (OUT / 'audio-calibration-v3.html').write_text(text)
    print(OUT / 'audio-calibration-v3.html')


if __name__ == '__main__':
    main()
