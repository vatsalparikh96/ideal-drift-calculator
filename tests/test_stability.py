"""Open-loop instability, controllability, and well-posed Jacobian (no fake-marginal)."""
import math

import numpy as np
import pytest

from config.params import DEFAULT_VEHICLE as P
from control.equilibria import solve_drift_equilibrium
from control.stability import controllability, linearize, unstable_mode

MU = 0.95


def _lin():
    eq = solve_drift_equilibrium(12.0, math.radians(-30), P, MU, MU)
    A, B = linearize(eq, P, MU, MU)
    return eq, A, B


def test_open_loop_unstable():
    _eq, A, _B = _lin()
    eig = np.linalg.eigvals(A)
    assert np.max(eig.real) > 1e-3            # at least one RHP eigenvalue
    um = unstable_mode(A)
    assert um.n_unstable >= 1
    assert um.lambda_u > 0


def test_jacobian_not_fake_marginal():
    # C1 tire model -> Jacobian has a genuine (non-zero) spectrum, not all ~0
    _eq, A, _B = _lin()
    eig = np.linalg.eigvals(A)
    assert np.max(np.abs(eig)) > 0.1          # not a degenerate zero-gradient matrix


def test_steering_controllable():
    _eq, A, B = _lin()
    rank, cond = controllability(A, B[:, [0]])  # steering-only
    assert rank == 3
    assert cond < 1e4


def test_throttle_has_no_lateral_authority():
    # B throttle column should be ~ pure speed (1/m on v_x, ~0 on v_y, r)
    _eq, A, B = _lin()
    assert abs(B[0, 1]) == pytest.approx(1.0 / P.m, rel=0.2)
    assert abs(B[1, 1]) < 1e-2
    assert abs(B[2, 1]) < 1e-2
