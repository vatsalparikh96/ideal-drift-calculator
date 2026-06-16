"""Sideslip-estimation showpiece.

A real car has no cheap sideslip sensor, so the advisor must run on an ESTIMATED state.
This compares two estimators feeding the same advisor while it holds a drift under noisy
IMU + wheel-speed measurements:

  * UKF        — the unscented Kalman filter (estimation/ukf.py), fusing the single-track
                 model with [r, v_x, a_x, a_y].
  * kinematic  — the naive baseline: integrate v_y from a_y - v_x*r (no model). It drifts
                 with sensor bias/noise, the advisor's β error grows, and the drift is lost.

Result: the UKF tracks β to a few degrees and the advisor holds the drift; the naive
estimator's β diverges and the car spins.

    python -m experiments.estimation_eval     # writes fig_estimation.png + prints RMSE
"""
from __future__ import annotations

import math

import matplotlib
import numpy as np

from config.params import DEFAULT_CONTROLLER as C
from config.params import DEFAULT_VEHICLE as P
from control.corrector import DriftController
from control.equilibria import solve_drift_equilibrium
from estimation.ukf import DriftStateEstimator
from sim.vehicle_model import compute_forces, rk4_step, sideslip, speed

MU = 0.95
SIG = {"r": 0.01, "vx": 0.10, "a": 0.20}
BIAS_AY = 2.0   # m/s^2 accelerometer bias - fatal to dead-reckoning, fused away by the UKF


def _run(mode: str, seed: int = 0, T: float = 6.0):
    rng = np.random.default_rng(seed)
    eq = solve_drift_equilibrium(12.0, math.radians(-30), P, MU, MU)
    ctrl = DriftController(C)
    ctrl.refresh(eq, P, MU, MU)
    ukf = DriftStateEstimator(P, MU, x0=[eq.vx, 0.0, eq.r])      # wrong initial v_y
    vy_kin = 0.0                                                 # naive integrator state

    x = [eq.vx, eq.vy, eq.r * 1.05, 0.0, 0.0, 0.0]
    delta, Fxr = eq.delta, eq.Fxr
    dt = 0.01
    hist = {k: [] for k in ("t", "beta_true", "beta_est", "spun")}
    spun = False
    for k in range(int(T / dt)):
        f = compute_forces(x[0], x[1], x[2], delta, Fxr, P, MU, MU, Fxf=eq.Fxf)
        zr = x[2] + rng.normal(0, SIG["r"])
        zvx = x[0] + rng.normal(0, SIG["vx"])
        zay = f.ay_body + BIAS_AY + rng.normal(0, SIG["a"])     # biased accelerometer
        zax = f.ax_body + rng.normal(0, SIG["a"])

        if mode == "ukf":
            ukf.update([zr, zvx, zax, zay], delta, Fxr, dt, Fxf=eq.Fxf)
            xhat = [ukf.vx, ukf.vy, ukf.r]
        else:  # naive kinematic integration of v_y (drifts)
            vy_kin += (zay - zvx * zr) * dt
            xhat = [zvx, vy_kin, zr]

        adv = ctrl.advise(xhat, delta, Fxr, P, MU, MU)
        delta, Fxr = adv.delta_target, adv.Fxr_target
        x = rk4_step(x, delta, Fxr, P, MU, MU, dt, Fxf=eq.Fxf)

        hist["t"].append(k * dt)
        hist["beta_true"].append(math.degrees(sideslip(x[0], x[1])))
        hist["beta_est"].append(math.degrees(math.atan2(xhat[1], xhat[0])))
        if speed(x[0], x[1]) < 1.0 or abs(hist["beta_true"][-1]) > 80:
            spun = True
            break
    hist["spun"] = spun
    hist["target"] = math.degrees(eq.beta)
    return hist


def main():
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ukf = _run("ukf")
    naive = _run("kinematic")

    # beta RMSE for the UKF after a 1 s convergence window
    err = [e - tr for e, tr, t in zip(ukf["beta_est"], ukf["beta_true"], ukf["t"], strict=False)
           if t > 1.0]
    rmse = float(np.sqrt(np.mean(np.square(err))))
    ukf_status = "SPUN" if ukf["spun"] else "held"
    naive_status = "SPUN" if naive["spun"] else "held"
    print(f"UKF sideslip RMSE (after 1 s) = {rmse:.2f} deg; drift {ukf_status}")
    print(f"naive kinematic: drift {naive_status} (t_end={naive['t'][-1]:.2f}s)")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(ukf["t"], ukf["beta_true"], "k", lw=2, label="true β")
    ax[0].plot(ukf["t"], ukf["beta_est"], "tab:green", lw=1.8, label="UKF estimate")
    ax[0].axhline(ukf["target"], ls="--", color="gray", label="target β*")
    ax[0].set_title(f"UKF: advisor holds drift on ESTIMATED state\nβ RMSE = {rmse:.1f}°")
    ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("sideslip β [deg]")
    ax[0].legend(loc="lower right"); ax[0].grid(alpha=0.3); ax[0].set_ylim(-85, 10)

    ax[1].plot(naive["t"], naive["beta_true"], "k", lw=2, label="true β")
    ax[1].plot(naive["t"], naive["beta_est"], "tab:red", lw=1.8, label="naive (kinematic) estimate")
    ax[1].axhline(naive["target"], ls="--", color="gray", label="target β*")
    ax[1].set_title("Naive integration: β drifts → car spins")
    ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("sideslip β [deg]")
    ax[1].legend(loc="lower right"); ax[1].grid(alpha=0.3); ax[1].set_ylim(-85, 10)

    fig.suptitle("Sideslip estimation: the advisor runs on estimated state (no β sensor)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig("fig_estimation.png", dpi=120)
    print("saved fig_estimation.png")


if __name__ == "__main__":
    main()
