"""Learning / KI experiment.

Three honest results (the headline advisor-vs-none number lives in analysis/figures.py):

  1. ROBUSTNESS  — the model-based advisor is insensitive to large cornering-stiffness
     error (recovery rate ~unchanged under +/-40% Ca error).  A strength, and the reason
     learning is a refinement, not a stability crutch.
  2. RLS SELF-CALIBRATION — online recursive least squares converges to the TRUE
     cornering stiffness during sub-limit driving (the "adapts" claim, made concrete).
  3. LEARNED RESIDUAL — against a Pacejka reference tire whose SHAPE the brush model
     cannot match, a small learned residual reduces lateral-force RMSE markedly.

  python -m experiments.learning_gain        # writes fig_learning.png + prints numbers
"""
from __future__ import annotations

import math
from dataclasses import replace

import matplotlib
import numpy as np

from config.params import DEFAULT_LEARNING, DEFAULT_VEHICLE
from control.tire import fiala_lateral
from estimation.rls import AxleStiffnessEstimator
from learning.tire_residual import TireResidualModel
from sim.vehicle_model import compute_forces, rk4_step, sideslip

MU = 0.95
V0 = 12.0
TGT = math.radians(-30.0)


# ----------------------------------------------------------------------------- 1
def recovery_fraction(plant_p, model_p, n=11) -> float:
    from analysis.figures import _run_closed_loop
    betas = np.linspace(-55, -8, n)
    rs = np.linspace(0.1, 1.4, n)
    rec = tot = 0
    for b in betas:
        for r in rs:
            ok, _, _ = _run_closed_loop(V0, math.radians(b), r, advisor_on=True, T=3.0,
                                        plant_p=plant_p, model_p=model_p)
            rec += int(ok); tot += 1
    return 100.0 * rec / tot


def robustness_table():
    nominal = DEFAULT_VEHICLE
    soft = replace(DEFAULT_VEHICLE, Ca_r=0.6 * DEFAULT_VEHICLE.Ca_r,
                   Ca_f=0.6 * DEFAULT_VEHICLE.Ca_f)
    stiff = replace(DEFAULT_VEHICLE, Ca_r=1.4 * DEFAULT_VEHICLE.Ca_r,
                    Ca_f=1.4 * DEFAULT_VEHICLE.Ca_f)
    out = {}
    for tag, plant in (("matched", nominal), ("plant -40% Ca", soft), ("plant +40% Ca", stiff)):
        out[tag] = recovery_fraction(plant, DEFAULT_VEHICLE)   # advisor always believes nominal
    return out


# ----------------------------------------------------------------------------- 2
def rls_warmup(true_Ca_f=95_000.0, true_Ca_r=210_000.0, T=8.0):
    """Drive a gentle sub-limit slalom on a plant with unknown (true) stiffness and let
    RLS estimate it.  Returns (t, Ca_f_hat[], Ca_r_hat[])."""
    plant = replace(DEFAULT_VEHICLE, Ca_f=true_Ca_f, Ca_r=true_Ca_r)
    est = AxleStiffnessEstimator(DEFAULT_VEHICLE.Ca_f, DEFAULT_VEHICLE.Ca_r, DEFAULT_LEARNING)
    dt = 0.01
    x = [V0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ts, cf, cr = [], [], []
    for k in range(int(T / dt)):
        t = k * dt
        delta = math.radians(3.0) * math.sin(2 * math.pi * 0.5 * t)   # gentle slalom
        f = compute_forces(x[0], x[1], x[2], delta, 0.0, plant, MU, MU)
        beta = sideslip(x[0], x[1])
        U_f = math.hypot(f.Fxf, f.Fyf) / max(1.0, MU * f.Fzf)
        U_r = math.hypot(f.Fxr, f.Fyr) / max(1.0, MU * f.Fzr)
        est.update(f.alpha_f, f.Fyf, f.alpha_r, f.Fyr, beta, U_f, U_r)
        ts.append(t); cf.append(est.front.theta); cr.append(est.rear.theta)
        x = rk4_step(x, delta, 0.0, plant, MU, MU, dt)
    return np.array(ts), np.array(cf), np.array(cr), true_Ca_f, true_Ca_r


# ----------------------------------------------------------------------------- 3
def _pacejka(alpha, mu, Fz, B=8.0, C=1.5, E=0.97):
    """Simplified Magic Formula lateral force (the 'true' tire the brush can't match)."""
    D = mu * Fz
    x = -alpha   # force opposes slip
    return D * math.sin(C * math.atan(B * x - E * (B * x - math.atan(B * x))))


def residual_fit(Fz=8000.0):
    """Fit the learned residual to the gap between the brush model and a Pacejka tire;
    report lateral-force RMSE brush-only vs brush+residual."""
    Ca = DEFAULT_VEHICLE.Ca_r
    # Offline calibration on logged data: no forgetting (lambda=1), more basis functions,
    # and a generous bound so the residual can cover the post-peak shape gap.
    cfg = replace(DEFAULT_LEARNING, beta_freeze=10.0, U_freeze=10.0, rls_lambda=1.0,
                  resid_bound_frac=0.4)
    res = TireResidualModel(cfg, n_centers=11)
    alphas = np.radians(np.linspace(-35, 35, 140))

    # train (offline calibration on logged samples)
    for _ in range(40):
        for a in alphas:
            true = _pacejka(a, MU, Fz)
            brush = fiala_lateral(a, Ca, MU, Fz, 0.0)
            res.update(a, MU, Fz, true - brush, beta=0.0, U=0.0, dt=0.01)

    brush_pred, learned_pred, truth = [], [], []
    for a in alphas:
        true = _pacejka(a, MU, Fz)
        brush = fiala_lateral(a, Ca, MU, Fz, 0.0)
        truth.append(true); brush_pred.append(brush)
        learned_pred.append(brush + res.predict(a, MU, Fz))
    truth = np.array(truth)
    brush_pred = np.array(brush_pred)
    learned_pred = np.array(learned_pred)
    rmse_brush = float(np.sqrt(np.mean((brush_pred - truth) ** 2)))
    rmse_learn = float(np.sqrt(np.mean((learned_pred - truth) ** 2)))
    return np.degrees(alphas), truth, brush_pred, learned_pred, rmse_brush, rmse_learn


def main():
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("1) ROBUSTNESS - recovery rate when the advisor's model is WRONG:")
    rob = robustness_table()
    for k, v in rob.items():
        print(f"     {k:14s}: {v:.0f}% recoverable")

    ts, cf, cr, tcf, tcr = rls_warmup()
    print("\n2) RLS SELF-CALIBRATION (sub-limit warmup):")
    print(f"     Ca_f: start {DEFAULT_VEHICLE.Ca_f:,.0f} -> est {cf[-1]:,.0f} (true {tcf:,.0f})")
    print(f"     Ca_r: start {DEFAULT_VEHICLE.Ca_r:,.0f} -> est {cr[-1]:,.0f} (true {tcr:,.0f})")

    adeg, truth, brush, learned, rb, rl = residual_fit()
    print("\n3) LEARNED RESIDUAL vs Pacejka reference tire:")
    print(f"     lateral-force RMSE: brush {rb:,.0f} N -> brush+residual {rl:,.0f} N "
          f"({100 * (1 - rl / rb):.0f}% lower)")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].bar(list(rob.keys()), list(rob.values()), color=["tab:green", "tab:blue", "tab:blue"])
    ax[0].set_ylabel("recoverable states [%]"); ax[0].set_ylim(0, 100)
    ax[0].set_title("Robust to model error\n(advisor believes nominal Ca)")
    ax[0].tick_params(axis="x", rotation=15)

    ax[1].axhline(tcf, ls="--", color="tab:blue", alpha=0.6, label="true Ca_f")
    ax[1].axhline(tcr, ls="--", color="tab:orange", alpha=0.6, label="true Ca_r")
    ax[1].plot(ts, cf, color="tab:blue", label="Ca_f estimate")
    ax[1].plot(ts, cr, color="tab:orange", label="Ca_r estimate")
    ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("cornering stiffness [N/rad]")
    ax[1].set_title("RLS self-calibration (sub-limit)"); ax[1].legend(fontsize=8)

    ax[2].plot(adeg, np.array(truth) / 1e3, "k", lw=2, label="true tire (Pacejka)")
    ax[2].plot(adeg, np.array(brush) / 1e3, "r--", label=f"brush (RMSE {rb:.0f} N)")
    ax[2].plot(adeg, np.array(learned) / 1e3, "g-.", label=f"brush+residual (RMSE {rl:.0f} N)")
    ax[2].set_xlabel("slip angle [deg]"); ax[2].set_ylabel("lateral force [kN]")
    ax[2].set_title("Learned tire residual"); ax[2].legend(fontsize=8)

    fig.suptitle("Adaptation: robust model-based control + online self-calibration", fontsize=13)
    fig.tight_layout()
    fig.savefig("fig_learning.png", dpi=120)
    print("\nsaved fig_learning.png")


if __name__ == "__main__":
    main()
