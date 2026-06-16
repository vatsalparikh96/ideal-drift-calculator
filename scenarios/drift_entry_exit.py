"""Full drift maneuver: automatically initiate a drift, hold it, and exit cleanly.

    python -m scenarios.drift_entry_exit       # writes fig_entry_exit.png + prints phases
"""
from __future__ import annotations

import math
from typing import Any

from config.params import DEFAULT_CONTROLLER, DEFAULT_VEHICLE
from control.corrector import DriftController
from control.drift_sequence import DriftSequencer
from control.equilibria import solve_drift_equilibrium
from sim.vehicle_model import rk4_step, sideslip, speed

MU = 0.95
V0 = 12.0
BETA0 = math.radians(-30.0)
PHASE_COLOR = {"GRIP": "#9aa7b2", "ENTER": "tab:orange", "DRIFT": "tab:green", "EXIT": "tab:blue"}


def simulate(T: float = 7.0):
    p = DEFAULT_VEHICLE
    eq = solve_drift_equilibrium(V0, BETA0, p, MU, MU)
    ctrl = DriftController(DEFAULT_CONTROLLER)
    ctrl.refresh(eq, p, MU, MU)
    seq = DriftSequencer(ctrl, eq, p, MU)

    x = [V0, 0.0, 0.0, 0.0, 0.0, 0.0]
    dt = 0.01
    hist: dict[str, Any] = {k: [] for k in
                            ("t", "beta", "r", "delta", "Fxr", "V", "X", "Y", "phase")}
    for k in range(int(T / dt)):
        t = k * dt
        delta, Fxr, phase = seq.command(x[:3], t, dt)
        hist["t"].append(t)
        hist["beta"].append(math.degrees(sideslip(x[0], x[1])))
        hist["r"].append(x[2])
        hist["delta"].append(math.degrees(delta))
        hist["Fxr"].append(Fxr)
        hist["V"].append(speed(x[0], x[1]))
        hist["X"].append(x[3]); hist["Y"].append(x[4])
        hist["phase"].append(phase)
        x = rk4_step(x, delta, Fxr, p, MU, MU, dt, Fxf=eq.Fxf)
    hist["eq"] = eq
    return hist


def _phase_spans(hist):
    spans, start, cur = [], 0.0, hist["phase"][0]
    for t, ph in zip(hist["t"], hist["phase"], strict=False):
        if ph != cur:
            spans.append((cur, start, t)); start, cur = t, ph
    spans.append((cur, start, hist["t"][-1]))
    return spans


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = simulate()
    spans = _phase_spans(h)
    print("phase timeline:")
    for ph, t0, t1 in spans:
        print(f"  {ph:5s} {t0:.2f}-{t1:.2f}s")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for ph, t0, t1 in spans:
        ax[0].axvspan(t0, t1, color=PHASE_COLOR.get(ph, "white"), alpha=0.15)
    ax[0].plot(h["t"], h["beta"], "k", lw=2, label="β")
    ax[0].plot(h["t"], h["delta"], "tab:purple", lw=1.3, label="steering δ")
    ax[0].axhline(math.degrees(h["eq"].beta), ls="--", color="gray", lw=1, label="drift β*")
    ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("deg")
    ax[0].set_title("GRIP → ENTER → DRIFT → EXIT → GRIP")
    ax[0].legend(loc="lower right", fontsize=8); ax[0].grid(alpha=0.3)
    # phase labels
    for ph, t0, t1 in spans:
        ax[0].text((t0 + t1) / 2, 28, ph, ha="center", fontsize=8,
                   color=PHASE_COLOR.get(ph, "k"))

    ax[1].plot(h["X"], h["Y"], "k", lw=1, alpha=0.4)
    for ph, t0, t1 in spans:
        i0 = int(t0 / 0.01); i1 = int(t1 / 0.01)
        ax[1].plot(h["X"][i0:i1 + 1], h["Y"][i0:i1 + 1], color=PHASE_COLOR.get(ph, "k"),
                   lw=2.5, label=ph)
    ax[1].plot(h["X"][0], h["Y"][0], "ko", ms=6)
    ax[1].set_xlabel("X [m]"); ax[1].set_ylabel("Y [m]"); ax[1].axis("equal")
    ax[1].set_title("Path (colored by phase)")
    ax[1].legend(fontsize=8)

    fig.suptitle("Automatic drift initiation & exit", fontsize=13)
    fig.tight_layout()
    fig.savefig("fig_entry_exit.png", dpi=120)
    print("saved fig_entry_exit.png")


if __name__ == "__main__":
    main()
