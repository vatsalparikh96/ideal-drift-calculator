"""Over/understeer monitor + stability margin.

Primary directional signal: the SIGNED unstable-mode coordinate z_u = w_u . S(x - x*),
where w_u is the LEFT eigenvector of the (scaled) open-loop Jacobian for the dominant
unstable mode.  This disambiguates the near-neutral 4-wheel-drift case (U_f ~ U_r ~ 1)
that friction utilization alone cannot.

Corroborating signal: per-axle friction utilization U = sqrt(Fx^2+Fy^2)/(mu*Fz) in
[0,1].  Rear saturated + diverging => oversteer; front saturated + running wide =>
understeer.  Labels carry hysteresis to stop flip-flop.

Time-to-loss-of-control is computed from the OPEN-LOOP A (the closed loop A-BK is stable
by design) as tau = (1/lambda_u) ln(z_thresh/|z_u|), clamped, and we trigger advice
early (predictively) because 1/lambda_u can be comparable to human reaction time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from config.params import MonitorConfig, VehicleParams
from control.equilibria import DriftEquilibrium
from control.stability import unstable_mode
from sim.vehicle_model import compute_forces


@dataclass
class MonitorState:
    label: str                 # "grip" | "oversteer" | "understeer" | "limit"
    severity: str              # "ok" | "watch" | "act"
    z_u: float                 # signed unstable-mode coordinate
    tau: float                 # time-to-loss-of-control [s]
    margin: float              # 1 (on target) .. 0 (at threshold)
    U_f: float
    U_r: float
    lambda_u: float
    complex_unstable: bool


class StabilityMonitor:
    def __init__(self, cfg: MonitorConfig):
        self.cfg = cfg
        self._label = "grip"   # for hysteresis

    def update(self, x3, eq: DriftEquilibrium, A: np.ndarray | None,
               delta_current: float, Fxr_current: float,
               p: VehicleParams, mu_f: float, mu_r: float) -> MonitorState:
        cfg = self.cfg
        vx, vy, r = x3

        # --- friction utilization per axle (corroborating) ---
        f = compute_forces(vx, vy, r, delta_current, Fxr_current, p, mu_f, mu_r, Fxf=eq.Fxf)
        U_f = math.hypot(f.Fxf, f.Fyf) / max(1.0, mu_f * f.Fzf)
        U_r = math.hypot(f.Fxr, f.Fyr) / max(1.0, mu_r * f.Fzr)

        # --- signed unstable-mode coordinate (primary) ---
        if A is not None:
            um = unstable_mode(A, scale=(cfg.scale_vx, cfg.scale_vy, cfg.scale_r))
            S = np.array([cfg.scale_vx, cfg.scale_vy, cfg.scale_r])
            e_scaled = S * (np.array(x3) - np.array(eq.x3))
            z_u = float(um.w_u @ e_scaled)
            lambda_u = um.lambda_u
            complex_unstable = um.complex_unstable
        else:
            z_u, lambda_u, complex_unstable = 0.0, 0.0, False

        # --- time-to-loss (open-loop divergence) ---
        if lambda_u > 1e-3 and abs(z_u) < cfg.z_thresh and abs(z_u) > 1e-9:
            tau = (1.0 / lambda_u) * math.log(cfg.z_thresh / abs(z_u))
            tau = max(cfg.tau_floor, min(cfg.tau_safe, tau))
        elif abs(z_u) >= cfg.z_thresh:
            tau = cfg.tau_floor            # already past threshold -> act now
        else:
            tau = cfg.tau_safe             # negligible divergence -> safe

        margin = float(np.clip(1.0 - abs(z_u) / cfg.z_thresh, 0.0, 1.0))

        # --- label with hysteresis ---
        on = cfg.U_thresh
        off = cfg.U_thresh - cfg.U_hysteresis
        rear_sat = U_r >= (off if self._label == "oversteer" else on)
        front_sat = U_f >= (off if self._label == "understeer" else on)
        if rear_sat and front_sat:
            label = "limit"
        elif rear_sat:
            label = "oversteer"
        elif front_sat:
            label = "understeer"
        else:
            label = "grip"
        self._label = label

        # --- severity from margin/tau ---
        if tau <= cfg.react_horizon or margin < 0.15:
            severity = "act"
        elif margin < 0.5:
            severity = "watch"
        else:
            severity = "ok"

        return MonitorState(label, severity, z_u, tau, margin, U_f, U_r,
                            lambda_u, complex_unstable)
