"""Torque vectoring: the 4-motor EV's marquee capability.

A left/right rear torque split produces a yaw moment M_z that enters the yaw equation
DIRECTLY (B column = [0, 0, 1/I_z]) — unlike rear drive force, whose lateral/yaw authority
collapses at a saturated rear. Adding M_z as a second control input (alongside steering)
gives the stabilizer real authority and widens the basin of attraction.

This compares the recoverable region for a steering-only LQR vs a steering + torque-
vectoring LQR, and renders a side-by-side basin map.

    python -m experiments.torque_vectoring     # writes fig_torque_vectoring.png + prints
"""
from __future__ import annotations

import math
from dataclasses import replace

import matplotlib
import numpy as np
from scipy.linalg import solve_continuous_are

from config.params import DEFAULT_CONTROLLER, DEFAULT_VEHICLE
from control.corrector import compute_steering_gain
from control.equilibria import solve_drift_equilibrium
from control.stability import linearize
from sim.vehicle_model import rk4_step, sideslip, speed

P = DEFAULT_VEHICLE
# A DEMANDING regime where torque vectoring matters most: lower grip, a deep drift, and a
# realistic limited steering range (~20 deg road-wheel).  In the easy nominal case
# (mu=0.95, beta=-30, full lock) steering alone already recovers ~93%, so TV adds little;
# the point of TV is to restore authority when steering is weak/saturated.
MU = 0.8
V0 = 12.0
TARGET_BETA = math.radians(-38.0)
C = replace(DEFAULT_CONTROLLER, delta_max=0.35)


def tv_gain(A, B):
    """2-input LQR for u = [delta, M_z].  M_z enters only r_dot (column [0,0,1/Iz])."""
    B_tv = np.column_stack([B[:, 0], np.array([0.0, 0.0, 1.0 / P.Iz])])
    Q = np.diag(C.Q)
    R = np.diag([C.r_delta, C.r_mz])
    Pmat = solve_continuous_are(A, B_tv, Q, R)
    return np.linalg.solve(R, B_tv.T @ Pmat)        # 2x3


def _recovers(beta0, r0, eq, K, tv: bool, T=3.0):
    dt = 0.01
    x = [V0 * math.cos(beta0), V0 * math.sin(beta0), r0, 0.0, 0.0, 0.0]
    delta, Fxr = eq.delta, eq.Fxr
    xstar = np.array(eq.x3)
    for _ in range(int(T / dt)):
        e = np.array([x[0], x[1], x[2]]) - xstar
        du = -K @ e
        Mz = 0.0
        if tv:
            delta = eq.delta + float(du[0])
            Mz = float(np.clip(du[1], -P.Mz_max, P.Mz_max))
        else:
            delta = eq.delta + float(du[0])
        delta = float(np.clip(delta, -C.delta_max, C.delta_max))
        x = rk4_step(x, delta, Fxr, P, MU, MU, dt, Fxf=eq.Fxf, Mz=Mz)
        if speed(x[0], x[1]) < 1.0 or abs(math.degrees(sideslip(x[0], x[1]))) > 80:
            return False
    return (abs(math.degrees(sideslip(x[0], x[1])) - math.degrees(eq.beta)) < 10.0
            and abs(x[2] - eq.r) < 0.25)


def basin(K, tv, eq, n=27):
    betas = np.linspace(-60, -5, n)
    rs = np.linspace(0.0, 1.5, n)
    grid = np.zeros((n, n))
    for i, b in enumerate(betas):
        for j, r0 in enumerate(rs):
            grid[i, j] = 1.0 if _recovers(math.radians(b), r0, eq, K, tv) else 0.0
    return grid, betas, rs


def main():
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eq = solve_drift_equilibrium(V0, TARGET_BETA, P, MU, MU)
    A, B = linearize(eq, P, MU, MU)
    K_s = compute_steering_gain(A, B, C)            # 1x3 steering-only
    K_tv = tv_gain(A, B)                            # 2x3 steering + M_z

    g_s, betas, rs = basin(K_s, tv=False, eq=eq)
    g_tv, _, _ = basin(K_tv, tv=True, eq=eq)
    f_s, f_tv = g_s.mean() * 100, g_tv.mean() * 100
    print(f"recoverable region: steering-only {f_s:.0f}%  ->  + torque vectoring {f_tv:.0f}%")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, grid, title, frac in (
        (axes[0], g_s, "Steering only", f_s),
        (axes[1], g_tv, "Steering + torque vectoring", f_tv)):
        ax.imshow(grid, origin="lower", aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                  extent=[rs[0], rs[-1], betas[0], betas[-1]])
        ax.plot(eq.r, math.degrees(eq.beta), "*", color="gold", ms=20, markeredgecolor="k")
        ax.set_xlabel("initial yaw rate r0 [rad/s]")
        ax.set_title(f"{title}\nrecoverable: {frac:.0f}%")
    axes[0].set_ylabel("initial sideslip beta0 [deg]")
    fig.suptitle("Torque vectoring widens the stabilizable region (4-motor yaw control)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig("fig_torque_vectoring.png", dpi=120)
    print("saved fig_torque_vectoring.png")


if __name__ == "__main__":
    main()
