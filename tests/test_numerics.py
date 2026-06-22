"""The NumPy-only fallbacks (browser/WASM path) must match SciPy bit-for-bit enough.

These guard the browser build: the WASM bundle ships NumPy but not SciPy, so
control._numerics replaces scipy.linalg.solve_continuous_are and scipy.optimize.root.
If these drift from SciPy, the in-browser advisor would compute a different control law.
"""
import math

import numpy as np
import pytest

from config.params import DEFAULT_CONTROLLER, DEFAULT_VEHICLE
from control import equilibria as eqmod
from control._numerics import root_newton, solve_care
from control.equilibria import solve_drift_equilibrium
from control.stability import linearize

MU = 0.95
CASES = [(12.0, math.radians(-30.0)), (10.0, math.radians(-25.0)), (14.0, math.radians(-35.0))]


@pytest.mark.parametrize(("V", "beta"), CASES)
def test_solve_care_matches_scipy(V, beta):
    scipy_care = pytest.importorskip("scipy.linalg").solve_continuous_are
    eq = solve_drift_equilibrium(V, beta, DEFAULT_VEHICLE, MU, MU)
    assert eq.feasible
    A, B = linearize(eq, DEFAULT_VEHICLE, MU, MU)
    Bs = B[:, [0]]
    Q = np.diag(DEFAULT_CONTROLLER.Q)
    R = np.array([[DEFAULT_CONTROLLER.r_delta]])
    P_ref = scipy_care(A, Bs, Q, R)
    P_np = solve_care(A, Bs, Q, R)
    assert np.allclose(P_ref, P_np, rtol=1e-6, atol=1e-6)
    K_ref = np.linalg.solve(R, Bs.T @ P_ref)
    K_np = np.linalg.solve(R, Bs.T @ P_np)
    assert np.allclose(K_ref, K_np, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(("V", "beta"), CASES)
def test_equilibrium_newton_matches_scipy(V, beta, monkeypatch):
    pytest.importorskip("scipy.optimize")
    eq_ref = solve_drift_equilibrium(V, beta, DEFAULT_VEHICLE, MU, MU)
    assert eq_ref.feasible

    # force the NumPy-only root finder (the WASM path) and re-solve
    monkeypatch.setattr(eqmod, "_root",
                        lambda func, x0, args: root_newton(func, x0, args=args))
    eq_np = solve_drift_equilibrium(V, beta, DEFAULT_VEHICLE, MU, MU)
    assert eq_np.feasible
    assert eq_np.delta == pytest.approx(eq_ref.delta, abs=1e-4)
    assert eq_np.Fxr == pytest.approx(eq_ref.Fxr, rel=1e-3, abs=1.0)
    assert eq_np.r == pytest.approx(eq_ref.r, abs=1e-4)
