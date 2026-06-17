"""Robustness & real-time budget.

Sweeps the recoverable-state fraction (the advisor's basin) against the things that
degrade a real deployment, and measures the control-loop compute time:

  * control LATENCY  — a pure transport delay on the command (answers "does it survive
    real-world actuation/processing lag at 100 Hz?").
  * sensor NOISE     — Gaussian noise on the state fed to the advisor.
  * friction MU      — works across surfaces (the drift radius adapts with grip).
  * loop-time budget — wall-clock per advise() call vs the 10 ms (100 Hz) budget.

    python -m experiments.robustness     # writes fig_robustness.png + prints the budget
"""
from __future__ import annotations

import math
import time

import matplotlib
import numpy as np

from config.params import DEFAULT_CONTROLLER as C
from config.params import DEFAULT_VEHICLE as P
from control.corrector import DriftController
from control.equilibria import solve_drift_equilibrium
from sim.vehicle_model import rk4_step, sideslip, speed

V0 = 12.0
TARGET_BETA = math.radians(-30.0)
DT = 0.01


def _controller(mu):
    eq = solve_drift_equilibrium(V0, TARGET_BETA, P, mu, mu)
    ctrl = DriftController(C)
    ctrl.refresh(eq, P, mu, mu)
    return ctrl, eq


def _recovers(beta0, r0, ctrl, eq, mu, latency=0, noise=0.0, rng=None, T=3.0):
    x = [V0 * math.cos(beta0), V0 * math.sin(beta0), r0, 0.0, 0.0, 0.0]
    delta, Fxr = eq.delta, eq.Fxr
    buf = [(eq.delta, eq.Fxr)] * (latency + 1)        # transport-delay buffer
    for _ in range(int(T / DT)):
        xm = list(x[:3])
        if noise and rng is not None:
            xm[0] += rng.normal(0, noise)
            xm[1] += rng.normal(0, noise)
            xm[2] += rng.normal(0, noise * 0.05)
        adv = ctrl.advise(xm, delta, Fxr, P, mu, mu)
        buf.append((adv.delta_target, adv.Fxr_target))
        delta, Fxr = buf.pop(0)                       # apply delayed command
        x = rk4_step(x, delta, Fxr, P, mu, mu, DT, Fxf=eq.Fxf)
        if speed(x[0], x[1]) < 1.0 or abs(math.degrees(sideslip(x[0], x[1]))) > 80:
            return False
    return (abs(math.degrees(sideslip(x[0], x[1])) - math.degrees(eq.beta)) < 10.0
            and abs(x[2] - eq.r) < 0.25)


def _fraction(ctrl, eq, mu, n=11, **kw):
    rng = np.random.default_rng(0)
    betas = np.linspace(-55, -8, n)
    rs = np.linspace(0.1, 1.4, n)
    rec = sum(_recovers(math.radians(b), r, ctrl, eq, mu, rng=rng, **kw)
              for b in betas for r in rs)
    return 100.0 * rec / (n * n)


def loop_budget(n=2000):
    ctrl, eq = _controller(0.95)
    x = list(eq.x3)
    delta, Fxr = eq.delta, eq.Fxr
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        ctrl.advise(x, delta, Fxr, P, 0.95, 0.95)
        times.append((time.perf_counter() - t0) * 1e3)        # ms
    return float(np.mean(times)), float(np.percentile(times, 95))


def main():
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ctrl95, eq95 = _controller(0.95)

    latencies = [0, 10, 20, 30, 40, 60]                       # ms (steps * 10 ms)
    lat_rec = [_fraction(ctrl95, eq95, 0.95, latency=int(round(ms / 10))) for ms in latencies]

    noises = [0.0, 0.1, 0.25, 0.5, 1.0]                       # m/s on v_x, v_y
    noise_rec = [_fraction(ctrl95, eq95, 0.95, noise=s) for s in noises]

    mus = [0.6, 0.7, 0.8, 0.95, 1.1]
    mu_rec = []
    for mu in mus:
        c, e = _controller(mu)
        mu_rec.append(_fraction(c, e, mu) if e.feasible else 0.0)

    mean_ms, p95_ms = loop_budget()
    print(f"loop budget: advise() mean {mean_ms:.3f} ms, p95 {p95_ms:.3f} ms "
          f"(budget 10 ms @ 100 Hz -> {'OK' if p95_ms < 10 else 'OVER'})")
    def _pairs(xs, ys):
        return dict(zip(xs, [round(v) for v in ys], strict=False))
    print(f"recovery vs latency [ms]: {_pairs(latencies, lat_rec)}")
    print(f"recovery vs noise [m/s]:  {_pairs(noises, noise_rec)}")
    print(f"recovery vs mu:           {_pairs(mus, mu_rec)}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].plot(latencies, lat_rec, "o-", color="tab:red")
    ax[0].set_xlabel("control latency [ms]"); ax[0].set_ylabel("recoverable states [%]")
    ax[0].set_title(f"Latency (loop {mean_ms:.2f} ms, p95 {p95_ms:.2f} ms)")
    ax[1].plot(noises, noise_rec, "o-", color="tab:blue")
    ax[1].set_xlabel("sensor noise sigma [m/s]"); ax[1].set_title("Sensor noise")
    ax[2].plot(mus, mu_rec, "o-", color="tab:green")
    ax[2].set_xlabel("friction coefficient mu"); ax[2].set_title("Across surfaces")
    for a in ax:
        a.set_ylim(0, 100); a.grid(alpha=0.3)
    fig.suptitle("Robustness: recoverable region vs latency, noise, and grip", fontsize=13)
    fig.tight_layout()
    fig.savefig("fig_robustness.png", dpi=120)
    print("saved fig_robustness.png")


if __name__ == "__main__":
    main()
