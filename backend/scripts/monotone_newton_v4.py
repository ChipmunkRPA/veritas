"""Research-only, numerically KKT-checked small monotone logistic fits.

Source/class balance, target smoothing and standardization match legacy v2.
The optimizer is new: exact small bound-constrained Newton subproblems, Armijo
backtracking, then a separate scalar-summed terminal gradient check. No retry,
library-status override, dependency install or serving import.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from scipy.special import expit

METHOD = 'bounded-newton-independent-kkt/4'
POLICY = {'maximum_dimensions': 8, 'maximum_iterations': 100, 'maximum_backtracks': 60,
    'vector_kkt_stop': 1e-10, 'independent_kkt_limit': 1e-9, 'bound_snap': 1e-12,
    'armijo': 1e-4, 'objective_roundoff_units': 8, 'reference_prior': .5}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def objective(theta, z, target, weight, penalties):
    eta = np.einsum('nj,j->n', z, theta[:-1]) + theta[-1]
    probability = expit(eta)
    residual = weight * (probability - target)
    value = float(np.dot(weight, np.logaddexp(0., eta) - target * eta)
        + np.dot(penalties, theta[:-1] ** 2))
    gradient = np.r_[np.einsum('n,nj->j', residual, z) + 2 * penalties * theta[:-1], residual.sum()]
    design = np.column_stack((z, np.ones(len(z))))
    curvature = weight * probability * (1 - probability)
    hessian = np.einsum('ni,n,nj->ij', design, curvature, design)
    hessian[np.diag_indices(len(theta))] += np.r_[2 * penalties, 0.]
    require(math.isfinite(value) and np.isfinite(gradient).all() and np.isfinite(hessian).all(),
        'Nonfinite logistic objective, derivative or curvature')
    return value, gradient, hessian


def residual(theta, gradient):
    stationarity = np.asarray(gradient).copy()
    for i, coefficient in enumerate(theta[:-1]):
        if coefficient == 0:
            stationarity[i] = min(0., stationarity[i])
    return float(np.max(np.abs(stationarity)))


def newton_direction(theta, gradient, hessian):
    """Enumerate at most256 active sets; solve each strictly convex Newton QP."""
    dimension = len(theta) - 1
    tolerance = 1e-10
    for mask in range(1 << dimension):
        active = [i for i in range(dimension) if mask & (1 << i)]
        free = [i for i in range(dimension + 1) if i not in active]
        step = np.zeros_like(theta)
        step[active] = -theta[active]
        right = -gradient[free] - hessian[np.ix_(free, active)] @ step[active]
        step[free] = np.linalg.solve(hessian[np.ix_(free, free)], right)
        proposed = theta + step
        if any(proposed[i] < -POLICY['bound_snap'] for i in free if i < dimension):
            continue
        derivative = gradient + hessian @ step
        if active and min(derivative[active]) < -tolerance:
            continue
        proposed[:-1] = np.maximum(proposed[:-1], 0.)
        return proposed - theta
    raise ValueError('No numerically feasible Newton active set; fit refused')


def independent_check(theta, z, target, weight, penalties):
    """Recalculate terminal objective/gradient with scalar math and fsum."""
    terms, per_coordinate = [], [[] for _ in theta]
    for row, truth, mass in zip(z, target, weight, strict=True):
        eta = math.fsum([float(theta[-1]), *[float(a) * float(b) for a, b in zip(row, theta[:-1], strict=True)]])
        if eta >= 0:
            p = 1 / (1 + math.exp(-eta))
        else:
            value = math.exp(eta)
            p = value / (1 + value)
        terms.append(float(mass) * (max(eta, 0.) + math.log1p(math.exp(-abs(eta))) - float(truth) * eta))
        error = float(mass) * (p - float(truth))
        for i, value in enumerate(row):
            per_coordinate[i].append(error * float(value))
        per_coordinate[-1].append(error)
    gradient = np.asarray([math.fsum(values) for values in per_coordinate])
    gradient[:-1] += 2 * penalties * theta[:-1]
    value = math.fsum([*terms, *[float(p) * float(v) ** 2 for p, v in zip(penalties, theta[:-1], strict=True)]])
    require(np.isfinite(theta).all() and np.isfinite(gradient).all() and math.isfinite(value), 'Nonfinite independent fit check')
    feasible = bool((theta[:-1] >= 0).all())
    kkt = residual(theta, gradient)
    require(feasible and kkt <= POLICY['independent_kkt_limit'], 'Independent numerical KKT check failed; fit refused')
    return {'feasible': feasible, 'independent_kkt_infinity': kkt, 'limit': POLICY['independent_kkt_limit'],
        'independent_objective': value, 'check': 'scalar sigmoid/logloss and math.fsum gradient, separate from Newton vector derivatives'}


def solve(z, target, weight, penalties, *, maximum_iterations=100):
    require(type(maximum_iterations) is int and 0 <= maximum_iterations <= POLICY['maximum_iterations'], 'Bounded iteration count required')
    theta = np.zeros(z.shape[1] + 1, np.float64)
    for iteration in range(maximum_iterations + 1):
        value, gradient, hessian = objective(theta, z, target, weight, penalties)
        if residual(theta, gradient) <= POLICY['vector_kkt_stop']:
            return theta, {**independent_check(theta, z, target, weight, penalties),
                'iterations': iteration, 'vector_kkt_infinity': residual(theta, gradient)}
        require(iteration < maximum_iterations, 'Bounded Newton iteration limit; no fitted artifact')
        step = newton_direction(theta, gradient, hessian)
        derivative = float(np.dot(gradient, step))
        require(np.isfinite(step).all() and derivative < 0, 'Newton direction is not finite/descent; fit refused')
        slack = POLICY['objective_roundoff_units'] * np.finfo(np.float64).eps * max(1., abs(value))
        accepted = False
        for backtrack in range(POLICY['maximum_backtracks']):
            alpha = 2. ** -backtrack
            proposed = theta + alpha * step
            proposed[:-1][proposed[:-1] < POLICY['bound_snap']] = 0.
            trial, _, _ = objective(proposed, z, target, weight, penalties)
            if trial <= value + POLICY['armijo'] * alpha * derivative + slack:
                theta, accepted = proposed, True
                break
        require(accepted, 'Bounded Newton line search failed; no fitted artifact')
    raise AssertionError('Unreachable bounded solver state')


def fit(x, y, groups, *, minimum=25, l2=.001, penalty_multipliers=None):
    require(type(minimum) is int and minimum >= 2 and math.isfinite(l2) and l2 > 0, 'Explicit support and positive regularization required')
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.ndim == 1:
        x = x[:, None]
    require(x.ndim == 2 and 1 <= x.shape[1] <= POLICY['maximum_dimensions'] and len(x) == len(y) == len(groups)
        and y.ndim == 1 and np.isfinite(x).all() and np.isfinite(y).all() and np.isin(y, [0., 1.]).all()
        and all(isinstance(g, str) and g for g in groups), 'Finite aligned binary source-group fit data required')
    count = {label: len({g for g, truth in zip(groups, y, strict=True) if truth == label}) for label in (0, 1)}
    if min(count.values()) < minimum:
        return {'status': 'insufficient_support', 'source_groups_by_class': {str(k): v for k, v in count.items()}}
    cells = Counter(zip(groups, y, strict=True))
    weight = np.asarray([1 / (2 * count[int(t)] * cells[g, t]) for g, t in zip(groups, y, strict=True)], np.float64)
    center = np.einsum('n,nj->j', weight, x)
    scale = np.maximum(np.sqrt(np.einsum('n,nj->j', weight, (x - center) ** 2)), 1e-6)
    require(np.isfinite(center).all() and np.isfinite(scale).all(), 'Numerical raw response range exceeded')
    z = np.clip((x - center) / scale, -8., 8.)
    target = np.where(y == 1, (count[1] + 1) / (count[1] + 2), 1 / (count[0] + 2))
    factors = np.ones(x.shape[1]) if penalty_multipliers is None else np.asarray(penalty_multipliers, np.float64)
    require(factors.shape == (x.shape[1],) and np.isfinite(factors).all() and (factors > 0).all(), 'Positive penalty for each feature required')
    theta, certificate = solve(z, target, weight, l2 * factors)
    return {'status': 'fitted', 'source_groups_by_class': {str(k): v for k, v in count.items()},
        'center': center.tolist(), 'scale': scale.tolist(), 'coefficients': theta[:-1].tolist(),
        'intercept': float(theta[-1]), 'reference_prior': .5, 'source_count_target_smoothing': True,
        'fit_method': METHOD, 'penalty_multipliers': factors.tolist(), 'numerical_check': certificate}
