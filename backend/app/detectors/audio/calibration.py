"""Frozen candidate calibration for the explicitly selected local audio rollout.

Inference only. No fitting, old-detector mapping or research-script imports.
The original experimental admission result is preserved in the source artifact.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

VERSION = 'local-speech-v3.1'
SHA256 = '912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf'
PATH = Path(__file__).with_name('calibrations') / 'local-speech-v3.json'
NOTE = ('Local screening estimate calibrated on a balanced reference set of authentic and AI speech. '
        'It is not a measured accuracy rate for this recording. Validation used short English speech; '
        'heavy background sound can cause errors or an inconclusive result. No per-file confidence interval is established.')


def load() -> dict:
    raw = PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError('The audio confidence calibration failed its integrity check.')
    return json.loads(raw)


def valid(p):
    return isinstance(p, (int, float, np.number)) and not isinstance(p, bool) and math.isfinite(p) and 0 <= p <= 1


def logit(p):
    if not valid(p):
        raise ValueError('The voice check returned an invalid response.')
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return math.log(p) - math.log1p(-p)


def probability(raw, bundle=None):
    bundle = load() if bundle is None else bundle
    if not valid(raw):
        return None
    fit = bundle['parameters']
    z = fit['slope'] * logit(raw) + fit['intercept']
    return 1 / (1 + math.exp(-z))


def initial_decision(p, bundle):
    if not valid(p):
        return 'inconclusive'
    included = []
    for label in (0, 1):
        maxima = bundle['guards'][str(label)]
        if len(maxima) < 9:
            return 'inconclusive'
        nc = p if label == 0 else 1 - p
        rank = (1 + sum(m is None or m >= nc for m in maxima)) / (len(maxima) + 1)
        if rank > bundle['alpha']:
            included.append(label)
    if p >= .8 and included == [1]:
        return 'ai'
    if p <= .2 + 1e-12 and included == [0]:
        return 'original'
    return 'inconclusive'


def evaluate(raw, pcm, windows, classifier, deadline) -> dict:
    bundle = load()
    p = probability(raw, bundle)
    direction = initial_decision(p, bundle)
    seconds = len(pcm) / 16000
    supported = (bundle['min_seconds'] <= seconds <= bundle['max_seconds'] and len(windows) == 1
                 and windows[0][0] == 0 and len(windows[0][1]) == len(pcm))
    probe_status, delta, veto = 'not_required_for_decision', None, False
    if direction == 'original' and supported:
        try:
            deadline.check('voice stability check')
            filtered = sosfiltfilt(butter(4, [100., 3800.], btype='bandpass', fs=16000, output='sos'),
                                  np.asarray(pcm, np.float32)).astype(np.float32)
            scores = classifier.predict([filtered[round(start * 16000):round(start * 16000) + len(chunk)]
                                         for start, chunk in windows])
            if len(scores) != len(windows) or not all(valid(x) for x in scores):
                raise ValueError('Missing stability response')
            delta = abs(logit(round(float(raw), 6)) - logit(round(float(np.mean(scores)), 6)))
            veto = delta > math.log(2)
            probe_status = 'computed'
        except Exception:
            # A failed probe cannot clear a recording as original. Cancellation
            # and deadline state are checked again before returning the result.
            deadline.check('voice stability check')
            veto, probe_status = True, 'unavailable'
    if not supported:
        direction = 'inconclusive'
        reason = ('The recording is outside the 2.5–4 second calibration scope. Individual speech windows were checked, '
                  'but no calibrated whole-recording confidence is assigned.')
    elif veto:
        direction = 'inconclusive'
        reason = 'The original-file tendency was withheld because the voice response was unstable or the stability check was unavailable.'
    elif direction == 'inconclusive':
        reason = 'The calibrated voice estimate and uncertainty check do not jointly support a reliable tendency.'
    else:
        reason = ('Leaning towards AI-generated speech.' if direction == 'ai' else 'Leaning towards original speech.')
    return {'version': VERSION, 'calibration_sha256': SHA256, 'probability_ai': p if supported else None,
            'reference_estimate_ai': p, 'decision': direction, 'scope_supported': supported,
            'seconds': seconds, 'probe_status': probe_status, 'stability_log_odds_change': delta,
            'original_stability_veto': veto, 'reason': reason, 'limitation': NOTE}
