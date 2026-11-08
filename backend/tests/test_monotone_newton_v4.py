"""Independent derivatives, constrained reference optima and failure refusal."""
import numpy as np
import pytest
from scipy.optimize import minimize

from app.bench.specialists import _logit
from scripts import monotone_newton_v4 as method


def problem(seed=17):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(80, 3))
    target = np.where(z[:, 0] - z[:, 1] > 0, .96, .04)
    return z, target, np.full(80, 1 / 80), np.full(3, .001)


def test_gradient_and_hessian_match_independent_finite_differences():
    z, target, weight, penalty = problem()
    theta = np.array([.3, .1, .2, -.1])
    _, gradient, hessian = method.objective(theta, z, target, weight, penalty)
    for i in range(4):
        step = np.eye(4)[i] * 1e-5
        plus, gp, _ = method.objective(theta + step, z, target, weight, penalty)
        minus, gm, _ = method.objective(theta - step, z, target, weight, penalty)
        assert gradient[i] == pytest.approx((plus - minus) / 2e-5, abs=1e-9)
        assert hessian[:, i] == pytest.approx((gp - gm) / 2e-5, abs=1e-9)


@pytest.mark.parametrize('seed', [17, 49, 101])
def test_checked_newton_matches_separate_slsqp_objective_with_active_bound(seed):
    z, target, weight, penalty = problem(seed)
    theta, certificate = method.solve(z, target, weight, penalty)
    reference = minimize(lambda t: method.objective(t, z, target, weight, penalty)[:2],
        np.zeros(4), jac=True, method='SLSQP', bounds=[(0, None)] * 3 + [(None, None)],
        options={'ftol': 1e-13, 'maxiter': 1000})
    assert reference.success
    assert method.objective(theta, z, target, weight, penalty)[0] == pytest.approx(reference.fun, abs=1e-11)
    assert theta[1] == 0
    assert certificate['independent_kkt_infinity'] <= 1e-9


def test_scalar_check_rejects_a_feasible_but_unoptimized_fit():
    with pytest.raises(ValueError, match='KKT'):
        method.independent_check(np.zeros(4), *problem())


def test_iteration_limit_is_failure_not_a_success_flag_override():
    with pytest.raises(ValueError, match='iteration limit'):
        method.solve(*problem(), maximum_iterations=0)


def test_duplicate_features_with_matched_penalty_preserve_predictions():
    x = np.linspace(-2., 2., 100)
    y = [int(v > .3) for v in x]
    groups = [str(i) for i in range(100)]
    single = method.fit(x, y, groups)
    repeated = method.fit(np.column_stack([x] * 3), y, groups, penalty_multipliers=[3] * 3)
    for value in (-3., -1., 0., 1., 3.):
        assert _logit(single, [value]) == pytest.approx(_logit(repeated, [value] * 3), abs=1e-8)


def test_source_replicates_do_not_manufacture_more_support_or_weight():
    x = np.linspace(-2., 2., 60)
    y, groups = [int(v > 0) for v in x], [str(i) for i in range(60)]
    first = method.fit(x, y, groups)
    repeated = method.fit(np.repeat(x, 3), np.repeat(y, 3), np.repeat(groups, 3))
    assert first['source_groups_by_class'] == repeated['source_groups_by_class']
    assert first['coefficients'] == pytest.approx(repeated['coefficients'], abs=1e-9)
    assert method.fit(x[:10], y[:10], groups[:10])['status'] == 'insufficient_support'


def test_constant_and_inverse_responses_can_fit_only_the_bias():
    y = np.tile([0, 1], 30)
    fitted = method.fit(np.column_stack([np.ones(60), 1 - y]), y, [str(i) for i in range(60)])
    assert fitted['coefficients'] == [0., 0.]
    assert abs(fitted['intercept']) < 1e-10
