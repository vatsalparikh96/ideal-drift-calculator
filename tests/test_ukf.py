"""UKF sideslip estimation: convergence from a wrong init, and accelerometer-bias rejection."""
import math

import numpy as np

from config.params import DEFAULT_VEHICLE as P
from control.equilibria import solve_drift_equilibrium
from estimation.ukf import DriftStateEstimator
from sim.vehicle_model import compute_forces, rk4_step, sideslip

MU = 0.95


def _run(bias_ay=0.0, seed=0, T=4.0):
    rng = np.random.default_rng(seed)
    eq = solve_drift_equilibrium(12.0, math.radians(-30), P, MU, MU)
    est = DriftStateEstimator(P, MU, x0=[eq.vx, 0.0, eq.r])     # wrong initial v_y = 0
    # hold the equilibrium open-loop-ish with the true inputs (state stays ~constant)
    x = [eq.vx, eq.vy, eq.r, 0.0, 0.0, 0.0]
    dt = 0.01
    errs = []
    for k in range(int(T / dt)):
        f = compute_forces(x[0], x[1], x[2], eq.delta, eq.Fxr, P, MU, MU, Fxf=eq.Fxf)
        z = [x[2] + rng.normal(0, 0.01),
             x[0] + rng.normal(0, 0.10),
             f.ax_body + rng.normal(0, 0.20),
             f.ay_body + bias_ay + rng.normal(0, 0.20)]
        est.update(z, eq.delta, eq.Fxr, dt, Fxf=eq.Fxf)
        x = rk4_step(x, eq.delta, eq.Fxr, P, MU, MU, dt, Fxf=eq.Fxf)
        if k > 150:                                            # after convergence
            errs.append(math.degrees(est.beta) - math.degrees(sideslip(x[0], x[1])))
    return est, float(np.sqrt(np.mean(np.square(errs)))), eq


def test_beta_converges_from_wrong_init():
    est, rmse, eq = _run(bias_ay=0.0)
    assert rmse < 6.0                                          # tracks true beta within a few deg
    assert abs(math.degrees(est.beta) - math.degrees(eq.beta)) < 8.0


def test_accelerometer_bias_is_estimated_and_rejected():
    true_bias = 2.0
    est, rmse, _ = _run(bias_ay=true_bias)
    # the bias state should converge near the true bias, keeping beta accurate
    assert abs(est.bias_ay - true_bias) < 1.0
    assert rmse < 7.0


def test_filter_is_stable():
    est, rmse, _ = _run(bias_ay=0.0)
    assert np.all(np.isfinite(est.x))
    assert math.isfinite(rmse)
