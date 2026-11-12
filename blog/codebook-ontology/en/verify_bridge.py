#!/usr/bin/env python3
"""Deterministic numerical checks for the local writer/reader algebra bridge.

These checks validate only finite-dimensional algebra and calculus identities;
they are not evidence about LLMs or Goodfire's empirical claims.
"""
import json
from pathlib import Path
import numpy as np

SEED = 20261112
TOL = 1e-10


def silu(z):
    return z / (1.0 + np.exp(-z))


def neuron_acts(x, wg, wu):
    return silu(wg.T @ x) * (wu.T @ x)


def check_rank_one(rng):
    x1, x2 = rng.normal(size=(2, 5))
    wg, wu = rng.normal(size=(5, 4)), rng.normal(size=(5, 4))
    W, v, k = rng.normal(size=(6, 4)), rng.normal(size=6), rng.normal(size=4)
    a1, a2 = neuron_acts(x1, wg, wu), neuron_acts(x2, wg, wu)
    dW = np.outer(v, k)
    d1 = (W + dW) @ a1 - W @ a1
    d2 = (W + dW) @ a2 - W @ a2
    rhs1, rhs2 = v * (k @ a1), v * (k @ a2)
    assert np.allclose(d1, rhs1, atol=TOL, rtol=TOL)
    assert np.allclose(d2, rhs2, atol=TOL, rtol=TOL)
    assert not np.isclose(k @ a1, k @ a2)
    return {"max_identity_error": float(max(np.max(np.abs(d1-rhs1)), np.max(np.abs(d2-rhs2)))),
            "activation_dependent_strengths": [float(k @ a1), float(k @ a2)]}


def two_layer_forward(x, W1, wg1, wu1, W2, wg2, wu2):
    a1 = neuron_acts(x, wg1, wu1)
    h = W1 @ a1
    a2 = neuron_acts(h, wg2, wu2)
    return W2 @ a2, a1, h, a2


def check_cached_input(rng):
    # Chosen signs ensure at least one second-layer activation is negative.
    x = np.array([0.7, -0.4, 0.2])
    W1 = np.array([[0.8, -0.3], [-0.5, 0.4], [0.2, 0.6]])
    wg1 = np.array([[0.2, -0.6], [0.7, 0.3], [-0.4, 0.5]])
    wu1 = np.array([[0.4, 0.1], [-0.2, 0.8], [0.5, -0.7]])
    W2 = np.array([[0.5, -0.9], [-0.2, 0.6]])
    wg2 = np.array([[0.9, -0.4], [-0.7, 0.3], [0.2, 0.5]])
    wu2 = np.array([[-0.8, 0.6], [0.5, -0.9], [0.4, 0.2]])
    v1, k1 = np.array([0.35, 0.1, -0.2]), np.array([1.0, -0.4])
    v2, k2 = np.array([0.2, -0.15]), np.array([0.7, -0.6])
    dW1, dW2 = np.outer(v1, k1), np.outer(v2, k2)
    _, _, h0, a20 = two_layer_forward(x, W1, wg1, wu1, W2, wg2, wu2)
    # Independent weight-edited forward: both edits are applied and the
    # second-layer activation is recomputed from the edited first-layer state.
    a1 = neuron_acts(x, wg1, wu1)
    h1_weight = (W1+dW1) @ a1
    a21_weight = neuron_acts(h1_weight, wg2, wu2)
    weight_edited = (W2+dW2) @ a21_weight
    # Activation-hook realization of the same edits, with current activations.
    h1_hook = W1 @ a1 + v1 * (k1 @ a1)
    a21_hook = neuron_acts(h1_hook, wg2, wu2)
    hook_edited = W2 @ a21_hook + v2 * (k2 @ a21_hook)
    # Wrong cached hook: only the second hook uses the original activation a20.
    stale_hook = W2 @ a21_hook + v2 * (k2 @ a20)
    assert np.allclose(weight_edited, hook_edited, atol=TOL, rtol=TOL)
    assert np.any(a20 < 0.0) or np.any(a21_hook < 0.0)
    assert np.linalg.norm(stale_hook-hook_edited) > 1e-8
    return {"weight_vs_hook_max_error": float(np.max(np.abs(weight_edited-hook_edited))),
            "stale_second_hook_vs_correct_norm": float(np.linalg.norm(stale_hook-hook_edited)),
            "negative_activation_values": [float(v) for v in np.r_[a20, a21_hook] if v < 0]}


def check_nonlinear_derivative(rng):
    A, r, b, direction = rng.normal(size=(4, 3)), rng.normal(size=3), rng.normal(size=4), rng.normal(size=3)
    def F(z): return np.tanh(A @ z + b)
    y = F(r)
    J = (1.0-y*y)[:, None] * A
    eps = 1e-6
    fd = (F(r+eps*direction)-F(r-eps*direction))/(2*eps)
    analytic = J @ direction
    assert np.allclose(fd, analytic, atol=1e-9, rtol=1e-9)
    return {"max_central_difference_error": float(np.max(np.abs(fd-analytic)))}


def check_finite_integral(rng):
    B, r, delta = rng.normal(size=(3, 3)), rng.normal(size=3), rng.normal(size=3)
    def F(z): return np.sin(B @ z) + 0.1*z*z
    exact = F(r+delta)-F(r)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    integ = np.zeros(3)
    for s, w in zip(nodes, weights):
        t = (s+1)/2
        z = r+t*delta
        J = np.cos(B @ z)[:, None]*B + np.diag(0.2*z)
        integ += (w/2)*(J @ delta)
    assert np.allclose(integ, exact, atol=1e-12, rtol=1e-12)
    return {"max_integral_error": float(np.max(np.abs(integ-exact)))}


def check_constraints():
    P = np.array([[1., 0., 0.]])
    h = np.array([0., 1., 0.])
    q = np.array([0., 1., 0.])
    tau = 0.75
    beta = (tau/(h@q))*q
    assert np.allclose(P@beta, 0.) and np.isclose(h@beta, tau)
    # Nontrivial nullspace, but the margin gradient lies entirely in the
    # row-space of P, so every reference-preserving direction has zero margin.
    P_bad = np.array([[1., 0., 0.]])
    assert np.linalg.matrix_rank(P_bad) == 1
    h_bad = np.array([1., 0., 0.])
    _, _, vh = np.linalg.svd(P_bad)
    null = vh[np.linalg.matrix_rank(P_bad):]
    assert null.shape[0] == 2 and np.allclose(null @ h_bad, 0.)
    r0 = np.zeros(2)
    def R(z): return np.array([z[0] + 0.5*z[1]**2])
    J = np.array([[1., 0.]])
    d = np.array([0., 0.4])
    assert np.allclose(J@d, 0.)
    drift = float((R(r0+d)-R(r0))[0])
    assert drift > 0.
    return {"feasible_beta": [float(v) for v in beta], "tau": tau,
            "infeasible_nullspace_dimension": int(null.shape[0]),
            "infeasible_margin_on_nullspace_max": float(np.max(np.abs(null @ h_bad))),
            "finite_probe_drift_with_zero_linear_drift": drift}


def main():
    rng = np.random.default_rng(SEED)
    result = {"seed": SEED, "tolerance": TOL,
              "scope": "finite-dimensional numerical checks only; not LLM/Goodfire evidence",
              "checks": {"rank_one_downprojection": check_rank_one(rng),
                         "cached_input_recompute": check_cached_input(rng),
                         "nonlinear_reader_derivative": check_nonlinear_derivative(rng),
                         "finite_path_integral": check_finite_integral(rng),
                         "nullspace_constraints": check_constraints()}}
    out = Path(__file__).parent / "output" / "goodfire_rewrite" / "algebra_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
