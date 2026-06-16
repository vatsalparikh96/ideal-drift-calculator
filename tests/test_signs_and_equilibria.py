"""Sign conventions, equilibrium branch selection, feasibility gate."""
import math

import pytest

from config.params import DEFAULT_VEHICLE as P
from control.equilibria import solve_drift_equilibrium
from sim.vehicle_model import compute_forces, reduced_derivative

MU = 0.95


def test_left_drift_signs():
    eq = solve_drift_equilibrium(12.0, math.radians(-30), P, MU, MU)
    assert eq.feasible
    # ISO 8855: left-hand drift -> r>0, beta<0, countersteer delta<0
    assert eq.beta < 0
    assert eq.r > 0
    assert eq.delta < 0
    # residual really is an equilibrium
    res = reduced_derivative(eq.x3, eq.delta, eq.Fxr, P, MU, MU, Fxf=eq.Fxf)
    assert max(abs(v) for v in res) < 1e-6


def test_rear_restoring_yaw_and_force_to_center():
    eq = solve_drift_equilibrium(12.0, math.radians(-30), P, MU, MU)
    f = compute_forces(eq.vx, eq.vy, eq.r, eq.delta, eq.Fxr, P, MU, MU, Fxf=eq.Fxf)
    # left turn: corner center is to the left (+y); rear lateral force points +y
    assert f.Fyr > 0
    # rear yaw term -b*Fyr is restoring: opposite sign to the front-driven yaw term
    yaw_front = P.a * (f.Fyf * math.cos(eq.delta))
    yaw_rear = -P.b * f.Fyr
    assert yaw_front > 0 and yaw_rear < 0
    assert yaw_front + yaw_rear == pytest.approx(0.0, abs=1.0)   # balanced at equilibrium


def test_right_drift_mirrors():
    eq = solve_drift_equilibrium(12.0, math.radians(+30), P, MU, MU)
    assert eq.feasible
    assert eq.beta > 0 and eq.r < 0 and eq.delta > 0


def test_branch_is_rear_saturated_front_authority():
    eq = solve_drift_equilibrium(12.0, math.radians(-30), P, MU, MU)
    assert eq.rear_saturated
    assert eq.front_authority      # steering must retain authority


def test_radius_set_by_speed():
    # R ~ V^2/(mu g): two betas at same V give nearly the same radius
    e1 = solve_drift_equilibrium(12.0, math.radians(-25), P, MU, MU)
    e2 = solve_drift_equilibrium(12.0, math.radians(-40), P, MU, MU)
    assert abs(e1.R - e2.R) / e1.R < 0.1


def test_feasibility_gate():
    # Degenerate: beta ~ 0 is not a drift
    eq = solve_drift_equilibrium(12.0, math.radians(1.0), P, MU, MU)
    assert not eq.feasible
    # A very deep, fast drift exceeds the motor/friction limit -> no valid root
    too_deep = solve_drift_equilibrium(18.0, math.radians(-48), P, MU, MU)
    assert not too_deep.feasible
