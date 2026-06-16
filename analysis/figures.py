"""Vehicle-dynamics figures: the (beta, r) phase portrait and the advisor's
basin of attraction.

  python -m analysis.figures             # writes both PNGs
  python -m analysis.figures phase       # just the phase portrait
  python -m analysis.figures basin       # just the basin map
"""
from __future__ import annotations

import math
import sys

import matplotlib
import numpy as np

from config.params import DEFAULT_CONTROLLER, DEFAULT_VEHICLE, VehicleParams
from control.corrector import DriftController
from control.equilibria import solve_drift_equilibrium
from sim.vehicle_model import reduced_derivative, rk4_step, sideslip, speed

MU = 0.95
V0 = 12.0
TARGET_BETA = math.radians(-30.0)


def _run_closed_loop(V, beta0, r0, target_beta=TARGET_BETA, T=4.0,
                     advisor_on=True, plant_p: VehicleParams = DEFAULT_VEHICLE,
                     model_p: VehicleParams | None = None):
    """Run from initial (beta0, r0) at speed V; return (recovered, beta_traj, r_traj).

    Uses the DriftController directly against a FIXED target equilibrium (no intent
    latching), which is what the basin/phase analysis wants."""
    model_p = model_p or plant_p
    eq = solve_drift_equilibrium(V, target_beta, model_p, MU, MU)
    if not eq.feasible:
        return False, [], []
    ctrl = DriftController(DEFAULT_CONTROLLER)
    ctrl.refresh(eq, model_p, MU, MU)

    dt = 0.01
    x = [V * math.cos(beta0), V * math.sin(beta0), r0, 0.0, 0.0, 0.0]
    delta, Fxr = eq.delta, eq.Fxr
    betas, rs = [], []
    for _ in range(int(T / dt)):
        if advisor_on:
            adv = ctrl.advise(x[:3], delta, Fxr, model_p, MU, MU)
            delta, Fxr = adv.delta_target, adv.Fxr_target
        else:
            delta, Fxr = eq.delta, eq.Fxr
        x = rk4_step(x, delta, Fxr, plant_p, MU, MU, dt, Fxf=eq.Fxf)
        betas.append(math.degrees(sideslip(x[0], x[1])))
        rs.append(x[2])
        if speed(x[0], x[1]) < 1.0 or abs(betas[-1]) > 80.0:
            return False, betas, rs
    recovered = (abs(betas[-1] - math.degrees(target_beta)) < 10.0
                 and abs(rs[-1] - eq.r) < 0.25)
    return recovered, betas, rs


def phase_portrait(save="fig_phase_portrait.png"):
    import matplotlib.pyplot as plt
    eq = solve_drift_equilibrium(V0, TARGET_BETA, DEFAULT_VEHICLE, MU, MU)
    p = DEFAULT_VEHICLE

    # vector field of the (beta, r) dynamics at FIXED equilibrium inputs
    bgrid = np.radians(np.linspace(-60, 10, 28))
    rgrid = np.linspace(-0.6, 1.6, 28)
    R, Bm = np.meshgrid(rgrid, bgrid)
    dR = np.zeros_like(R); dB = np.zeros_like(Bm)
    for i in range(Bm.shape[0]):
        for j in range(Bm.shape[1]):
            beta, r = Bm[i, j], R[i, j]
            vx, vy = V0 * math.cos(beta), V0 * math.sin(beta)
            vxd, vyd, rd = reduced_derivative([vx, vy, r], eq.delta, eq.Fxr, p, MU, MU,
                                              Fxf=eq.Fxf)
            dB[i, j] = (vx * vyd - vy * vxd) / (V0 ** 2)
            dR[i, j] = rd

    fig, ax = plt.subplots(figsize=(8, 6))
    speed_field = np.hypot(dR, dB)
    ax.streamplot(rgrid, np.degrees(bgrid), dR, np.degrees(dB), color=speed_field,
                  cmap="viridis", density=1.3, linewidth=0.8, arrowsize=0.8)

    # open-loop (diverges) vs closed-loop (recovers) trajectory from a perturbed start
    b0, r0 = math.radians(-38), eq.r * 1.1
    _, bo, ro = _run_closed_loop(V0, b0, r0, advisor_on=False, T=2.5)
    _, bc, rc = _run_closed_loop(V0, b0, r0, advisor_on=True, T=4.0)
    if bo:
        ax.plot(ro, bo, color="red", lw=2.2, label="open-loop (no advice) → spins")
    if bc:
        ax.plot(rc, bc, color="lime", lw=2.2, label="advisor on → recovers")
    ax.plot(eq.r, math.degrees(eq.beta), "*", color="gold", ms=22,
            markeredgecolor="k", label="drift sweet spot (unstable)", zorder=6)
    ax.plot(r0, math.degrees(b0), "o", color="white", ms=8, markeredgecolor="k",
            label="perturbed start", zorder=6)

    ax.set_xlabel("yaw rate  r  [rad/s]")
    ax.set_ylabel("sideslip  β  [deg]")
    ax.set_title(f"β–r phase portrait at V={V0:.0f} m/s — the drift is an unstable equilibrium")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save, dpi=120)
    print(f"saved {save}")
    return fig


def basin_map(save="fig_basin.png", n=29):
    import matplotlib.pyplot as plt
    eq = solve_drift_equilibrium(V0, TARGET_BETA, DEFAULT_VEHICLE, MU, MU)
    betas = np.linspace(-60, -5, n)
    rs = np.linspace(0.0, 1.5, n)

    grid_on = np.zeros((n, n))
    grid_off = np.zeros((n, n))
    for i, bdeg in enumerate(betas):
        for j, r0 in enumerate(rs):
            rec_on, _, _ = _run_closed_loop(V0, math.radians(bdeg), r0, advisor_on=True, T=3.0)
            rec_off, _, _ = _run_closed_loop(V0, math.radians(bdeg), r0, advisor_on=False, T=3.0)
            grid_on[i, j] = 1.0 if rec_on else 0.0
            grid_off[i, j] = 1.0 if rec_off else 0.0

    frac_on = grid_on.mean() * 100
    frac_off = grid_off.mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, grid, title, frac in (
        (axes[0], grid_off, "No advisor (open-loop)", frac_off),
        (axes[1], grid_on, "Advisor ON", frac_on)):
        ax.imshow(grid, origin="lower", aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                  extent=[rs[0], rs[-1], betas[0], betas[-1]])
        ax.plot(eq.r, math.degrees(eq.beta), "*", color="gold", ms=20, markeredgecolor="k")
        ax.set_xlabel("initial yaw rate r₀ [rad/s]")
        ax.set_title(f"{title}\nrecoverable region: {frac:.0f}%")
    axes[0].set_ylabel("initial sideslip β₀ [deg]")
    fig.suptitle("Basin of attraction — fraction of drift states the advisor can save", fontsize=13)
    fig.tight_layout()
    fig.savefig(save, dpi=120)
    print(f"saved {save}  (open-loop {frac_off:.0f}% -> advisor {frac_on:.0f}%)")
    return fig, frac_off, frac_on


def main():
    matplotlib.use("Agg")
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "phase"):
        phase_portrait()
    if which in ("all", "basin"):
        basin_map()


if __name__ == "__main__":
    main()
