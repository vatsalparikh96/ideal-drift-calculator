"""Sample scenario: car is drifting, the driver gives TOO MUCH acceleration.

We start the car in a steady left-hand drift with the advisor running, then at t_excess
the driver stabs the throttle (excess rear drive force).  Three driver behaviours are
compared:

  * "ignore"  : keeps the excess throttle and original steering, ignoring the advisor.
                Expected: the rear's lateral budget collapses (friction circle), yaw runs
                away, the car SPINS.
  * "rescue"  : makes the same mistake briefly, then OBEYS the advisor (lift throttle +
                add countersteer).  Expected: recovers and holds the drift.
  * "assist"  : always follows the advisor.  Expected: holds the drift throughout.

This is exactly the user's scenario: the corrector indicates "away from acceleration"
and "turn the wheel the opposite way" to keep the car in the drift.
"""
from __future__ import annotations

import math
from typing import Any

from config.params import (
    DEFAULT_CONTROLLER,
    DEFAULT_LEARNING,
    DEFAULT_MONITOR,
    DEFAULT_RATES,
    DEFAULT_VEHICLE,
)
from control.equilibria import solve_drift_equilibrium
from realtime.loop import Advisor
from sim.sensors import SensorBus
from sim.vehicle_model import rk4_step, sideslip, speed

MU = 0.95
V0 = 12.0
BETA0 = math.radians(-30.0)


def simulate(mode: str, T: float = 6.0, t_excess: float = 1.0,
             excess: float = 3000.0, mistake_dur: float = 0.3, noise: bool = False,
             plant_p=None, model_p=None, adapt: bool = False):
    """Run one scenario; return a history dict of time series.

    plant_p is the TRUE vehicle the simulator integrates; model_p is the advisor's
    (possibly wrong) model.  Defaults keep them matched.  `adapt` turns on the online
    RLS/residual learning in the advisor."""
    p_plant = plant_p or DEFAULT_VEHICLE
    p_model = model_p or p_plant
    dt = DEFAULT_RATES.dt_sim
    eq0 = solve_drift_equilibrium(V0, BETA0, p_plant, MU, MU)   # true initial drift
    assert eq0.feasible, eq0.reason

    advisor = Advisor(p_model, DEFAULT_CONTROLLER, DEFAULT_MONITOR, DEFAULT_LEARNING,
                      DEFAULT_RATES, adapt=adapt)
    bus = SensorBus(noise=noise)

    x = [eq0.vx, eq0.vy, eq0.r, 0.0, 0.0, 0.0]
    delta, Fxr = eq0.delta, eq0.Fxr

    hist: dict[str, Any] = {k: [] for k in
            ("t", "beta", "r", "V", "delta", "Fxr", "gas", "label", "severity",
             "tau", "margin", "steer_text", "pedal_text", "X", "Y", "spun",
             "delta_target", "gas_target", "beta_star", "r_star",
             "Ca_f_hat", "Ca_r_hat")}
    spun = False

    n = int(T / dt)
    for k in range(n):
        t = k * dt
        s = bus.read(x, delta, Fxr, MU, MU, p_plant)
        tel = advisor.update(s, dt)
        adv = tel.advice

        # --- driver behaviour ---
        in_mistake = t_excess <= t < t_excess + mistake_dur
        if mode == "ignore":
            if t >= t_excess:
                Fxr = eq0.Fxr + excess
                delta = eq0.delta
            # else hold equilibrium
        elif mode == "rescue":
            if in_mistake:
                Fxr = eq0.Fxr + excess
                delta = eq0.delta
            elif adv is not None and adv.feasible:
                delta, Fxr = adv.delta_target, adv.Fxr_target
        elif mode == "assist" and adv is not None and adv.feasible:
            delta, Fxr = adv.delta_target, adv.Fxr_target

        # --- record ---
        b = math.degrees(sideslip(x[0], x[1]))
        hist["t"].append(t)
        hist["beta"].append(b)
        hist["r"].append(x[2])
        hist["V"].append(speed(x[0], x[1]))
        hist["delta"].append(math.degrees(delta))
        hist["Fxr"].append(Fxr)
        hist["gas"].append(adv.gas_current if adv else 0.0)
        hist["label"].append(tel.monitor.label if tel.monitor else "—")
        hist["severity"].append(tel.monitor.severity if tel.monitor else "—")
        hist["tau"].append(tel.monitor.tau if tel.monitor else float("nan"))
        hist["margin"].append(tel.monitor.margin if tel.monitor else float("nan"))
        hist["steer_text"].append(adv.steer_text if adv else "—")
        hist["pedal_text"].append(adv.pedal_text if adv else "—")
        hist["delta_target"].append(math.degrees(adv.delta_target) if adv else float("nan"))
        hist["gas_target"].append(adv.gas_target if adv else float("nan"))
        hist["beta_star"].append(math.degrees(tel.eq.beta) if tel.eq else float("nan"))
        hist["r_star"].append(tel.eq.r if tel.eq else float("nan"))
        hist["Ca_f_hat"].append(tel.Ca_f_hat)
        hist["Ca_r_hat"].append(tel.Ca_r_hat)
        hist["X"].append(x[3])
        hist["Y"].append(x[4])
        hist["spun"].append(spun)

        # --- integrate plant ---
        x = rk4_step(x, delta, Fxr, p_plant, MU, MU, dt, Fxf=eq0.Fxf)
        if speed(x[0], x[1]) < 1.0 or abs(math.degrees(sideslip(x[0], x[1]))) > 80.0:
            spun = True
            hist["spun"][-1] = True
            break

    hist["eq0"] = eq0
    hist["spun_final"] = spun
    return hist


def summarize(mode: str, **kw) -> str:
    h = simulate(mode, **kw)
    eq0 = h["eq0"]
    end_t = h["t"][-1]
    end_beta = h["beta"][-1]
    maxbeta = max(abs(b) for b in h["beta"])
    status = "SPUN" if h["spun_final"] else "held drift"
    return (f"[{mode:7s}] {status:10s} t_end={end_t:.2f}s  beta_end={end_beta:6.1f}deg "
            f"(target {math.degrees(eq0.beta):.0f}) max|beta|={maxbeta:.0f}")


if __name__ == "__main__":
    print(f"Drift target: V={V0} m/s, beta={math.degrees(BETA0):.0f} deg, "
          f"R={solve_drift_equilibrium(V0, BETA0, DEFAULT_VEHICLE, MU, MU).R:.1f} m\n")
    for mode in ("ignore", "rescue", "assist"):
        print(summarize(mode))
    # Show the advice at the moment of the throttle mistake
    h = simulate("ignore")
    idx = next(i for i, t in enumerate(h["t"]) if t >= 1.05)
    print(f"\nAt the throttle stab (t=1.05s, mode=ignore): "
          f"beta={h['beta'][idx]:.1f}deg label={h['label'][idx]} -> "
          f"ADVICE: '{h['steer_text'][idx]}'  +  '{h['pedal_text'][idx]}'")
