"""Nonlinear single-track (bicycle) vehicle dynamics -- the SHARED model.

This module is the single source of truth for the dynamics: the simulator uses it as
the "truth" plant, and the advisor (equilibrium solver, Jacobians, LQR) imports the
same functions so the controller's model matches the plant.  (A model-mismatch toggle
and the learned residual layer live elsewhere; here the model is exact.)

State (reduced, for control/equilibria):  x3 = [v_x, v_y, r]
State (full, for the plant/trajectory):    x6 = [v_x, v_y, r, X, Y, psi]
Inputs:  delta (front road-wheel angle), F_xf (front axle long. force),
         F_xr (rear axle long. force)

Body-frame accelerations (transport theorem, omega = r * z_hat):
    a_x_body = v_x_dot - r*v_y = net_longitudinal_force / m   (drives load transfer)
    a_y_body = v_y_dot + r*v_x = net_lateral_force / m

See config/params.py for the ISO 8855 sign convention.
"""
from __future__ import annotations

import math
from typing import NamedTuple

from config.params import VehicleParams
from control.tire import fiala_lateral


class AxleForces(NamedTuple):
    alpha_f: float
    alpha_r: float
    Fzf: float
    Fzr: float
    Fyf: float
    Fyr: float
    Fxf: float
    Fxr: float
    ax_body: float        # body-frame longitudinal accel (= a_x measured on level ground)
    ay_body: float        # body-frame lateral accel
    F_aero: float


def slip_angles(vx: float, vy: float, r: float, delta: float,
                p: VehicleParams) -> tuple[float, float]:
    """Front/rear slip angles with a low-speed denominator guard (atan2)."""
    vx_safe = vx if vx > p.v_eps else p.v_eps
    alpha_f = math.atan2(vy + p.a * r, vx_safe) - delta
    alpha_r = math.atan2(vy - p.b * r, vx_safe)
    return alpha_f, alpha_r


def _aero(vx: float, p: VehicleParams) -> float:
    """Longitudinal aerodynamic drag (opposes forward motion)."""
    return p.k_aero * vx * abs(vx)


def compute_forces(
    vx: float, vy: float, r: float, delta: float, Fxr: float,
    p: VehicleParams, mu_f: float, mu_r: float, Fxf: float = 0.0, n_iter: int = 2,
) -> AxleForces:
    """Axle forces with longitudinal load transfer resolved by a short fixed-point.

    Load transfer uses the body-frame longitudinal acceleration a_x_body =
    net_longitudinal_force / m (what a level-ground accelerometer reads).  Because
    a_x_body depends on the forces and the forces depend on Fz, we iterate a couple
    of times; load transfer is a modest correction so this converges immediately.
    """
    alpha_f, alpha_r = slip_angles(vx, vy, r, delta, p)
    F_aero = _aero(vx, p)
    cd, sd = math.cos(delta), math.sin(delta)

    Fzf, Fzr = p.Fzf_static, p.Fzr_static
    Fyf = Fyr = 0.0
    ax_body = 0.0
    for _ in range(max(1, n_iter)):
        Fyf = fiala_lateral(alpha_f, p.Ca_f, mu_f, Fzf, Fxf)
        Fyr = fiala_lateral(alpha_r, p.Ca_r, mu_r, Fzr, Fxr)
        net_long = Fxf * cd - Fyf * sd + Fxr - F_aero
        ax_body = net_long / p.m
        Fzf = max(p.Fz_min, (p.m * p.g * p.b - p.m * ax_body * p.h) / p.L)
        Fzr = max(p.Fz_min, (p.m * p.g * p.a + p.m * ax_body * p.h) / p.L)

    net_lat = Fyf * cd + Fxf * sd + Fyr
    ay_body = net_lat / p.m
    return AxleForces(alpha_f, alpha_r, Fzf, Fzr, Fyf, Fyr, Fxf, Fxr, ax_body, ay_body, F_aero)


def reduced_derivative(
    x3, delta: float, Fxr: float, p: VehicleParams, mu_f: float, mu_r: float, Fxf: float = 0.0
):
    """xdot for x3 = [v_x, v_y, r].  The dynamics the controller linearizes.

    Fxr (rear drive force) is the primary input; Fxf (front) defaults to 0 for a
    rear-biased drift.  Order chosen so the rear force can't be silently swapped."""
    vx, vy, r = x3
    f = compute_forces(vx, vy, r, delta, Fxr, p, mu_f, mu_r, Fxf=Fxf)
    cd, sd = math.cos(delta), math.sin(delta)

    vx_dot = (f.Fxf * cd - f.Fyf * sd + f.Fxr - f.F_aero) / p.m + vy * r
    vy_dot = (f.Fyf * cd + f.Fxf * sd + f.Fyr) / p.m - vx * r
    r_dot = (p.a * (f.Fyf * cd + f.Fxf * sd) - p.b * f.Fyr) / p.Iz
    return [vx_dot, vy_dot, r_dot]


def full_derivative(
    x6, delta: float, Fxr: float, p: VehicleParams, mu_f: float, mu_r: float, Fxf: float = 0.0
):
    """xdot for x6 = [v_x, v_y, r, X, Y, psi] (adds global pose for trajectory)."""
    vx, vy, r, _X, _Y, psi = x6
    vx_dot, vy_dot, r_dot = reduced_derivative((vx, vy, r), delta, Fxr, p, mu_f, mu_r, Fxf=Fxf)
    cpsi, spsi = math.cos(psi), math.sin(psi)
    X_dot = vx * cpsi - vy * spsi
    Y_dot = vx * spsi + vy * cpsi
    return [vx_dot, vy_dot, r_dot, X_dot, Y_dot, r]


def rk4_step(x6, delta, Fxr, p, mu_f, mu_r, dt: float, Fxf: float = 0.0):
    """One fixed-step RK4 integration of the full 6-state plant."""
    def f(state):
        return full_derivative(state, delta, Fxr, p, mu_f, mu_r, Fxf=Fxf)

    k1 = f(x6)
    k2 = f([xi + 0.5 * dt * ki for xi, ki in zip(x6, k1, strict=False)])
    k3 = f([xi + 0.5 * dt * ki for xi, ki in zip(x6, k2, strict=False)])
    k4 = f([xi + dt * ki for xi, ki in zip(x6, k3, strict=False)])
    return [
        xi + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d)
        for xi, a, b, c, d in zip(x6, k1, k2, k3, k4, strict=False)
    ]


def sideslip(vx: float, vy: float) -> float:
    """beta = atan2(v_y, v_x)."""
    return math.atan2(vy, vx)


def speed(vx: float, vy: float) -> float:
    return math.hypot(vx, vy)
