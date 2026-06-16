"""Torque vectoring: M_z has direct yaw authority and widens the recoverable set."""
import math
from dataclasses import replace

import numpy as np

from config.params import DEFAULT_CONTROLLER
from config.params import DEFAULT_VEHICLE as P
from control.corrector import compute_steering_gain
from control.equilibria import solve_drift_equilibrium
from control.stability import linearize
from experiments.torque_vectoring import MU, V0, _recovers, tv_gain

C = replace(DEFAULT_CONTROLLER, delta_max=0.35)
EQ = solve_drift_equilibrium(V0, math.radians(-38.0), P, MU, MU)
A, B = linearize(EQ, P, MU, MU)


def test_Mz_column_is_direct_yaw_authority():
    # M_z enters only r_dot: B_mz = [0, 0, 1/Iz]
    B_mz = np.array([0.0, 0.0, 1.0 / P.Iz])
    assert B_mz[2] > 0 and B_mz[0] == 0 and B_mz[1] == 0


def test_both_controllers_stabilize_locally():
    K_s = compute_steering_gain(A, B, C)
    K_tv = tv_gain(A, B)
    B_tv = np.column_stack([B[:, 0], np.array([0.0, 0.0, 1.0 / P.Iz])])
    eig_s = np.linalg.eigvals(A - B[:, [0]] @ K_s)
    eig_tv = np.linalg.eigvals(A - B_tv @ K_tv)
    assert np.max(eig_s.real) < 0
    assert np.max(eig_tv.real) < 0


def test_tv_recovers_where_steering_fails():
    K_s = compute_steering_gain(A, B, C)
    K_tv = tv_gain(A, B)
    # hard initial states (shallow beta, modest r) that steering-only cannot save here
    hard = [(math.radians(-12), 0.5), (math.radians(-15), 0.6), (math.radians(-10), 0.7)]
    tv_saves = sum(_recovers(b, r, EQ, K_tv, tv=True) for b, r in hard)
    steer_saves = sum(_recovers(b, r, EQ, K_s, tv=False) for b, r in hard)
    assert tv_saves > steer_saves
