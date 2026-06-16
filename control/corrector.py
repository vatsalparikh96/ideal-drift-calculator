"""Drift sweet-spot corrector: steering-only LQR about the target drift equilibrium
plus a throttle speed-trim, turned into concrete steering + pedal ADVICE.

Why steering-only LQR (from the design review): at the saturated rear the throttle
column of B has ~zero lateral/yaw authority, so letting an LQR use throttle to
stabilize sideslip/yaw produces explosive, useless gains.  Steering is the only fast
lateral/yaw actuator, and the unstable mode is v_x-involved, so we keep the full state
x = [v_x, v_y, r] and stabilize with steering:

    delta = delta* - K (x - x*)          # K is 1x3 (steering)

Throttle is handled separately and physically:
  * speed trim:  F_xr = F_xr* - k_speed * (v_x - v_x*)   (hold the drift speed)
  * anti-spin:   the target is anchored at F_xr* (the equilibrium drive force).  If the
    driver is on too much throttle (excess F_xr eats the rear's lateral budget via the
    friction circle), the target sits below current -> the cue is "LIFT", exactly the
    friction-circle recovery.  This is a finite-amplitude effect, not in the linear K.

The two cues are presented as TARGETS to move toward (delta_target, F_xr_target ->
pedal); they retract on their own as x returns to x*, which is what prevents the
lift+countersteer pendulum into the opposite spin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_continuous_are

from config.params import ControllerConfig, VehicleParams
from control.equilibria import DriftEquilibrium
from control.stability import controllability, linearize
from control.tire import friction_budget
from sim.vehicle_model import compute_forces


def compute_steering_gain(A: np.ndarray, B: np.ndarray, cfg: ControllerConfig) -> np.ndarray:
    """Steering-only LQR gain K (1x3) from the CARE, using the steering column of B."""
    Bs = B[:, [0]]                      # 3x1 steering input
    Q = np.diag(cfg.Q)
    R = np.array([[cfg.r_delta]])
    P = solve_continuous_are(A, Bs, Q, R)
    K = np.linalg.solve(R, Bs.T @ P)    # 1x3
    return K


@dataclass
class Advice:
    feasible: bool
    delta_target: float
    delta_current: float
    steer_text: str
    Fxr_target: float
    Fxr_current: float
    gas_target: float          # fraction [-1(brake)..+1(full throttle)]
    gas_current: float
    pedal_text: str
    escalate: bool
    level: str                 # "hold" | "correct" | "saturated" | "unrecoverable"
    notes: str = ""

    @property
    def ddelta(self) -> float:
        return self.delta_target - self.delta_current


class DriftController:
    """Holds the target equilibrium + steering LQR gain (refreshed slowly) and produces
    advice (fast)."""

    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg
        self.eq: DriftEquilibrium | None = None
        self.A: np.ndarray | None = None
        self.B: np.ndarray | None = None
        self.K: np.ndarray | None = None
        self.ctrb_rank = 0
        self.ctrb_cond = float("inf")

    def refresh(self, eq: DriftEquilibrium, p: VehicleParams, mu_f: float, mu_r: float):
        """Recompute Jacobians + steering LQR gain at the target equilibrium (slow loop)."""
        self.eq = eq
        if not eq.feasible:
            self.A = self.B = self.K = None
            return
        A, B = linearize(eq, p, mu_f, mu_r)
        self.A, self.B = A, B
        self.ctrb_rank, self.ctrb_cond = controllability(A, B[:, [0]])
        try:
            self.K = compute_steering_gain(A, B, self.cfg)
        except Exception:
            self.K = None

    def advise(self, x3, delta_current: float, Fxr_current: float,
               p: VehicleParams, mu_f: float, mu_r: float) -> Advice:
        """Fast-loop advice given the current measured state and driver inputs."""
        cfg = self.cfg
        eq = self.eq
        if eq is None or not eq.feasible or self.K is None:
            g = _gas(Fxr_current, p)
            return Advice(False, delta_current, delta_current, "—",
                          Fxr_current, Fxr_current, g, g, "—",
                          True, "unrecoverable", "no feasible drift target")

        vx, vy, r = x3
        x = np.array(x3, dtype=float)
        xstar = np.array(eq.x3, dtype=float)
        err = x - xstar

        # --- steering: LQR ---
        delta_t = eq.delta - float((self.K @ err)[0])

        # --- throttle: speed trim around the equilibrium drive force ---
        Fxr_t = eq.Fxr - cfg.k_speed * (vx - eq.vx)

        # --- saturation limits ---
        f = compute_forces(vx, vy, r, delta_current, Fxr_current, p, mu_f, mu_r, Fxf=eq.Fxf)
        Fxr_circle = friction_budget(mu_r, f.Fzr, 0.0)        # = mu_r*Fzr (max usable)
        Fxr_max = min(p.Fx_motor_max, Fxr_circle)
        Fxr_min = -p.Fx_brake_max
        delta_clip = float(np.clip(delta_t, -cfg.delta_max, cfg.delta_max))
        Fxr_clip = float(np.clip(Fxr_t, Fxr_min, Fxr_max))
        steer_saturated = abs(delta_t - delta_clip) > 1e-3

        # --- text cues (ISO 8855: delta>0 steer left) ---
        ddelta = delta_clip - delta_current
        if abs(ddelta) < np.radians(1.0):
            steer_text = "hold steering"
        elif ddelta > 0:
            steer_text = f"steer LEFT +{np.degrees(abs(ddelta)):.0f} deg"
        else:
            steer_text = f"steer RIGHT +{np.degrees(abs(ddelta)):.0f} deg"

        dFxr = Fxr_clip - Fxr_current
        if Fxr_clip < -50.0:
            pedal_text = "BRAKE"
        elif dFxr < -150.0:
            pedal_text = "LIFT throttle"
        elif dFxr > 150.0:
            pedal_text = "MORE throttle"
        else:
            pedal_text = "hold throttle"

        # --- escalation / level ---
        big_error = abs(err[1]) > 2.0 or abs(err[2]) > 0.2     # vy [m/s], r [rad/s]
        if not eq.front_authority or self.ctrb_cond > 1e4:
            level, notes = "saturated", "front near saturation -> limited steering authority"
        elif steer_saturated and big_error:
            level, notes = "unrecoverable", "steering correction exceeds limit -> reduce speed"
        elif big_error:
            level, notes = "correct", ""
        else:
            level, notes = "hold", ""
        escalate = level in ("saturated", "unrecoverable")

        return Advice(
            True, delta_clip, delta_current, steer_text,
            Fxr_clip, Fxr_current, _gas(Fxr_clip, p), _gas(Fxr_current, p),
            pedal_text, escalate, level, notes,
        )


def _gas(Fxr: float, p: VehicleParams) -> float:
    """Map a rear drive/brake force to a pedal fraction in [-1, +1]."""
    if Fxr >= 0:
        return min(1.0, Fxr / p.Fx_motor_max)
    return max(-1.0, Fxr / p.Fx_brake_max)
