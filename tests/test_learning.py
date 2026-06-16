"""Online estimation + learned residual: convergence, gating, error reduction."""
import math

import numpy as np

from config.params import DEFAULT_LEARNING, DEFAULT_VEHICLE
from estimation.rls import AxleStiffnessEstimator, ScalarRLS
from learning.tire_residual import TireResidualModel


def test_scalar_rls_converges():
    true_theta = 3.5
    rls = ScalarRLS(theta0=0.0, P0=1e3, lam=1.0, eps_pe=1e-9, P_max=1e9)
    rng = np.random.default_rng(0)
    for _ in range(200):
        phi = rng.uniform(-1, 1)
        rls.update(phi, true_theta * phi)
    assert abs(rls.theta - true_theta) < 1e-3


def test_rls_pe_gate_holds_without_excitation():
    rls = ScalarRLS(theta0=1.0, P0=1.0, lam=0.99, eps_pe=1e6, P_max=1e9)
    for _ in range(50):
        rls.update(0.0, 5.0)          # phi=0 -> no information
    assert rls.theta == 1.0           # never updated
    assert rls.n_updates == 0


def test_axle_stiffness_learns_true_value():
    true_Ca_f, true_Ca_r = 95_000.0, 210_000.0
    est = AxleStiffnessEstimator(DEFAULT_VEHICLE.Ca_f, DEFAULT_VEHICLE.Ca_r, DEFAULT_LEARNING)
    rng = np.random.default_rng(1)
    for _ in range(400):
        af = rng.uniform(-0.05, 0.05)   # linear regime, below gates
        ar = rng.uniform(-0.05, 0.05)
        est.update(af, -true_Ca_f * af, ar, -true_Ca_r * ar, beta=0.0, U_f=0.1, U_r=0.1)
    assert abs(est.front.theta - true_Ca_f) / true_Ca_f < 0.05
    assert abs(est.rear.theta - true_Ca_r) / true_Ca_r < 0.05


def test_axle_stiffness_frozen_in_saturation():
    est = AxleStiffnessEstimator(60_000.0, 160_000.0, DEFAULT_LEARNING)
    for _ in range(100):
        # beta beyond freeze gate -> no learning even with data present
        est.update(0.02, -9999 * 0.02, 0.02, -9999 * 0.02, beta=0.5, U_f=0.1, U_r=0.1)
    assert est.front.theta == 60_000.0
    assert est.rear.theta == 160_000.0


def test_residual_reduces_prediction_error():
    from dataclasses import replace
    cfg = replace(DEFAULT_LEARNING, beta_freeze=10.0, U_freeze=10.0, rls_lambda=1.0,
                  resid_bound_frac=0.5)
    res = TireResidualModel(cfg, n_centers=9)
    mu, Fz = 0.95, 8000.0
    # an arbitrary smooth "true minus brush" gap to learn
    def gap(a):
        return 1500.0 * math.sin(a)
    alphas = np.radians(np.linspace(-30, 30, 80))
    for _ in range(40):
        for a in alphas:
            res.update(a, mu, Fz, gap(a), beta=0.0, U=0.0, dt=0.01)
    err = np.mean([abs(res.predict(a, mu, Fz) - gap(a)) for a in alphas])
    assert err < 150.0     # learned the gap to within ~10%


def test_residual_output_bounded():
    res = TireResidualModel(DEFAULT_LEARNING, n_centers=5)
    mu, Fz = 0.9, 8000.0
    bound = DEFAULT_LEARNING.resid_bound_frac * mu * Fz
    # force huge targets; output must stay within the hard bound
    for _ in range(50):
        res.update(0.0, mu, Fz, 1e6, beta=0.0, U=0.0, dt=0.01)
    assert abs(res.predict(0.0, mu, Fz)) <= bound + 1e-6
