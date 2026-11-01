"""Render already-frozen audio evaluation data without fitting or selecting a system."""
from __future__ import annotations

import csv
import hashlib
import json
from html import escape
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
DATA = BACKEND / 'data/independent-evaluation/audio'
OUTPUT = BACKEND.parent / 'output/evaluation'
LABELS = {'S0_old_service_score':'S0 · Existing service score', 'S1_old_recalibrated':'S1 · Existing check, recalibrated',
          'S2_new_calibrated':'S2 · New check, calibrated alone', 'S3_nonnegative_combination':'S3 · Existing + new checks, calibrated'}
CONDITIONS = {'all':'All three conditions combined', 'clean':'Clean speech', 'noise':'Background noise · 10 dB SNR', 'music':'Background music · 10 dB SNR'}


def number(value, kind='decimal'):
    if value is None:
        return 'Not established'
    return f'{100*value:.1f}%' if kind=='percent' else f'{value:.4f}'


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in r)+'</tr>' for r in rows)+'</tbody></table></div>'


def main():
    data = json.loads((DATA/'evaluation.json').read_text())
    manifest = json.loads((DATA/'manifest.json').read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    export = []
    sections = []
    for condition, cohort in data['cohorts'].items():
        metrics = cohort['metrics']
        rows = []
        for system, m in metrics.items():
            counts=cohort['counts'][system]
            interval=cohort['source_rate_wilson_ci95'].get(system, {}).get('false_ai_80')
            rows.append([escape(LABELS[system]),number(m['brier']),number(m['log_loss']),number(m['ece']),
                         number(m['false_ai_80'],'percent'),number(m['false_original_80'],'percent'),
                         number(m['decision_coverage_80'],'percent'),
                         f"{counts['inconclusive_80']} / {counts['missing']}",
                         'Reported per condition only' if interval is None else number(interval['high'],'percent')])
            for metric, value in m.items():
                ci=cohort['ci95'][system].get(metric)
                export.append({'condition':condition,'system':system,'metric':metric,'value':value,
                               'ci95_low':None if ci is None else ci['low'], 'ci95_high':None if ci is None else ci['high'],
                               'test_original_files':80,'condition_inputs':counts['all_condition_rows'],
                               'assessed_inputs':counts['assessed'],'deployment_eligible':False})
        sections.append(f'<section><h2>{CONDITIONS[condition]}</h2>'+table(
            ['System','Brier ↓','Log loss ↓','ECE ↓','Real called AI','AI called original','Decision coverage','Inconclusive / Unassessable','Real called AI: Wilson upper bound'],rows))
        raw=cohort['raw_ranking']
        sections.append('<div class="twocol"><div><h3>Raw discrimination · Before calibration</h3>'+table(['Check','ROC AUC ↑','EER ↓','Assessment coverage'],[
            ['Existing check',number(raw['old']['auc']),number(raw['old']['eer'],'percent'),number(raw['old']['assessed_fraction'],'percent')],
            ['New check',number(raw['new']['auc']),number(raw['new']['eer'],'percent'),number(raw['new']['assessed_fraction'],'percent')]])+'</div><div><h3>Frozen calibration threshold · Nominal 5% false-positive target</h3>'+table(['System','Frozen threshold','Observed real-file false positives','AI detected'],[
            [escape(LABELS[s]),number(data['thresholds'][s]['threshold']),number(m['fpr_at_cal_5'],'percent'),number(m['tpr_at_cal_5'],'percent')] for s,m in metrics.items()])+'</div></div>')
        differences=[]
        for pair, values in cohort['paired_difference'].items():
            ci=cohort['paired_ci95'][pair]
            names=pair.split('_minus_')
            for key, label in [('brier','Brier difference'),('log_loss','Log loss difference'),('false_original_80','Difference in AI called original')]:
                bound=ci.get(key)
                format_value = (lambda v: f'{100*v:+.1f} percentage points') if key=='false_original_80' else number
                differences.append([escape(LABELS[names[0]]+' − '+LABELS[names[1]]),label,format_value(values[key]),
                    'Not established' if bound is None else f"[{format_value(bound['low'])}, {format_value(bound['high'])}]"])
        sections.append('<details><summary>Paired differences and 95% grouped intervals</summary><p>A negative difference means the first system has lower error. The 2,000 bootstrap replicates resample speaker/voice groups, carrying all background conditions from a source together. Intervals are conditional on the fixed models and backgrounds in this experiment.</p>'+table(['Comparison','Metric','Difference','95% interval'],differences)+'</details></section>')
    with (OUTPUT/'audio-comparison.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(export[0]))
        writer.writeheader()
        writer.writerows(export)
    sections.append('<section><h2>Scope and reproducible records</h2><ul>'
        '<li>The final test contains 40 real speakers and 40 AI voice sources: 80 original files. Three conditions per source produce 240 test inputs, not 240 independent samples.</li>'
        '<li>A separate 32 original files fit the combined weights, and 48 fit calibration. Test sources, voices and transcripts are separated from fitting and calibration. The test generator family was withheld from calibration.</li>'
        '<li>Only English speech at normalized levels and the first four seconds of each source were tested. Results do not establish performance on long recordings, all languages, arbitrary noise or video soundtracks.</li>'
        '<li>The existing encoder pretraining shares a source family with LibriSpeech; the new check does not disclose its complete training set. This study establishes split separation, not complete independence from model training data.</li>'
        '<li>Noise and music sources have explicit permissions. Background recordings are reused within each split, so intervals do not represent all environments. Mixtures use a full-window RMS SNR of 10 dB. Original files, time bounds, hashes and the shared gain used to avoid clipping are retained.</li>'
        '<li>S0 uses the existing frozen fusion score and detector applicability gates. The full API also applies eligibility and conflict rules; this table does not replace the actual customer conclusion.</li>'
        '<li>The original offline records, frozen code, thresholds and calibration files are preserved. Production PCM replays of the six API inputs are diagnostics only; calibration was not refitted.</li><li>All calibration uses a 50:50 reference prior. An 80% estimate does not establish 80% accuracy in every real user population.</li>'
        '<li>A zero-error bootstrap interval can collapse to zero. For absolute risk, also consider the Wilson upper bound over independent sources: even with no false positives among 40 real samples, the upper bound is approximately 8.76%.</li>'
        '</ul><p>Research candidates remain offline. Production fusion weights, confidence intervals and the existing model were not replaced. Test labels were not used to tune thresholds.</p>'
        '<p class="hash">Manifest SHA-256: '+escape(manifest['sha256'])+'<br>Evaluation SHA-256: '+hashlib.sha256((DATA/'evaluation.json').read_bytes()).hexdigest()+'</p></section>')
    header='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Chipmunk Verify · Independent Audio Evaluation</title><style>
    :root{color-scheme:light;font-family:Inter,-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;color:#25231f;background:#f6f4ef}*{box-sizing:border-box}body{margin:0}main{max-width:1450px;margin:0 auto;padding:40px 32px 72px}.brand{letter-spacing:.12em;font-size:13px;font-weight:750;color:#8a592b}h1{font-size:38px;line-height:1.3;margin:18px 0 12px}h2{font-size:23px;margin:0 0 20px}h3{font-size:16px;margin:16px 0 12px}p,li{line-height:1.75}p{max-width:1050px}.muted{color:#6c665d}.badge{display:inline-block;background:#f5e4c6;color:#664411;padding:7px 12px;border-radius:7px;font-size:13px;font-weight:700}.intro{border-left:4px solid #bb7e40;padding:12px 22px;background:#fff9ef;margin:24px 0 32px}section{background:white;border:1px solid #e6e0d6;border-radius:12px;padding:26px;margin:24px 0;box-shadow:0 3px 10px #342a1505}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums;white-space:nowrap}th,td{text-align:left;border-bottom:1px solid #ece6dc;padding:13px 12px}th{font-weight:650;color:#6b6257;background:#f8f6f1}td:first-child{font-weight:600}tr:last-child td{border-bottom:none}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin:20px 0}details{border-top:1px solid #eee4d6;margin-top:16px;padding-top:18px}summary{cursor:pointer;font-weight:650}details p{font-size:13px;color:#746a5a}.hash{overflow-wrap:anywhere;font-family:monospace;font-size:11px;color:#7d766c}footer{font-size:12px;color:#7d766c;margin-top:32px}@media(max-width:1000px){.twocol{grid-template-columns:1fr}main{padding:26px 16px}h1{font-size:28px}section{padding:18px}}@media print{body{background:white}main{padding:0}section{break-inside:avoid;box-shadow:none}.scroll{overflow:visible}table{font-size:10px}details{display:block}details>*{display:block}}
    </style><main><div class="brand">CHIPMUNK VERIFY · INTERNAL VALIDATION</div><h1>Voice checks: independent evaluation and calibration</h1><p class="muted">A paired comparison of real and AI speech, with noise and music added to the same sources. Data, fitting methods and thresholds were frozen before scoring.</p><span class="badge">Offline candidate experiment · Not current product performance</span><div class="intro"><strong>Decision: do not admit this calibration into combined confidence; retain the existing check.</strong><p>The new check improves discrimination and overall probability prediction error: combined AUC is 0.9999 (existing check: 0.5789), and Brier score is 0.0299 (existing check after recalibration: 0.2457). However, it incorrectly calls 7 of 40 AI sources with background noise original at the fixed confidence threshold (17.5%). This fails the prespecified requirement to control errors in both directions. AUC measures ranking; it does not establish the reliability of the fixed 80% rule.</p><p>Input processing is a separate limitation: offline scoring read FLOAT WAV directly, while the actual API quantizes to signed 16-bit PCM. The existing check also uses a different default device in the API. Six preselected samples were replayed as engineering diagnostics; this experiment does not establish full production-path eligibility. Sample size is also limited, but the decision is not based on sample size alone: the background-noise calibration produced actual errors.</p><p>The combined fit assigned zero weight to the existing check, yielding the same results as the new check alone. That does not provide enough evidence across sources to remove the existing check. The next version needs better calibration under background conditions and matching production decoding, followed by validation on new sources. Adjusting thresholds on this already-inspected test set would not constitute an independent pass.</p></div><p><a href="audio-comparison.csv">Download complete CSV</a> · <a href="web-sample-results.html">View image, video and text results</a></p><p class="muted">Fixed rule: ≥ 80% leans AI; ≤ 20% leans original; the middle is inconclusive. Unassessable inputs are counted separately, not treated as original. Lower Brier, log loss and ECE are better; higher AUC is better. False-positive and false-original rates use all inputs of the corresponding class as their denominators.</p>'''
    predictions=json.loads((DATA/'predictions.json').read_text())
    simple=[]
    for condition in ('clean','noise','music'):
        test=[r for r in predictions if r['split']=='test' and r['condition']==condition]
        real=[r for r in test if r['label']==0]
        fake=[r for r in test if r['label']==1]
        def count(subset, predicate):
            return sum(1 for r in subset if (value:=r['probabilities']['S2_new_calibrated']) is not None and predicate(value))
        simple.append([CONDITIONS[condition], f"{count(real,lambda p:p<=.2)} / {len(real)}", f"{count(fake,lambda p:p>=.8)} / {len(fake)}",
                       str(count(fake,lambda p:p<=.2)),str(count(fake,lambda p:.2<p<.8))])
    plain='<section><h2>File counts: the new check with offline calibration</h2><p>Each row covers the same 40 real sources and 40 AI sources under a different background condition. The three rows must not be added together as 240 independent sources. These candidate probabilities are not used by the current website.</p>'+table(['Condition','Real correctly leaning original','AI correctly leaning AI','AI incorrectly leaning original','AI inconclusive'],simple)+'<p><strong>The main failure is background noise:</strong>7 AI sources were incorrectly cleared as original. Retaining the existing check preserves the current production path; it does not establish that the existing check is better overall. Its discrimination was also weak in this experiment, and most results were inconclusive.</p></section>'
    api_path=DATA/'api-parity-audit.json'
    if api_path.exists():
        api=json.loads(api_path.read_text())
        api_rows=[]
        for item in api['items']:
            label='Real' if '-real-' in item['sample_id'] else 'AI'
            condition=item['sample_id'].split('-')[-1]
            api_rows.append([label+' · '+CONDITIONS[condition], 'Inconclusive: neither tendency reaches 80%',
                f"{item['customer_assessment']['probability_ai_pct']:.1f}%", '<a href="http://127.0.0.1:5230/analyses/'+escape(item['analysis_id'])+'?view=plain">Open local analysis</a>'])
        plain+='<section><h2>What the current website actually returned</h2><p>One real source and one AI source were selected in advance, each with three background conditions, for six actual API analyses. All six customer conclusions are inconclusive. The new check remains a separate inspection signal with zero fusion weight. These are engineering replays, not additional independent accuracy samples.</p>'+table(['Input','Actual customer conclusion','Current combined AI estimate','View'],api_rows)+'<p>51 evidence files were downloaded and their hashes verified. Selected audio clips, waveforms and sample intervals match the JSON records; public responses contain no model names. After replaying production PCM on the CPU, the new check matches all six API scores to six decimal places. The largest difference for the existing check is 0.000001. The larger original offline differences were primarily caused by 16-bit quantization rather than CPU/MPS differences.</p></section>'
    (OUTPUT/'audio-comparison.html').write_text(header+plain+''.join(sections)+'<footer>Chipmunk Verify · Research record. Original materials and intermediate results remain in the local independent-evaluation/audio folder. Export: <a href="audio-comparison.csv">audio-comparison.csv</a></footer></main></html>',encoding='utf-8')
    print(OUTPUT/'audio-comparison.html')
    print(OUTPUT/'audio-comparison.csv')


if __name__=='__main__':
    main()
