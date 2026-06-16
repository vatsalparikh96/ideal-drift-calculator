"""Robust, branch-aware drift-equilibrium solver.

DESIGN NOTE (from exploring the equilibrium map): for a given vehicle and friction, a
steady drift uses essentially all of the lateral grip, so the path radius is fixed by
speed, R ~= V^2 / (mu*g).  Sideslip beta is then the (near-free) drift-angle / style
parameter that barely changes R.  Parameterizing by (V, R) and solving the square
system is therefore ILL-CONDITIONED (R is insensitive to beta) and fsolve latches onto
normal-cornering or degenerate roots.  So we parameterize the drift equilibrium by
(V, beta) -- which solves cleanly -- and report R = V / r* as an output.

For a target (V, beta) we solve [v_x_dot, v_y_dot, r_dot] = 0 for (delta*, F_xr*, r*),
then VALIDATE: v_x > 0, rear axle saturated, F_xr inside the friction circle and within
motor capability, and (preferably) the FRONT axle below full slide so steering retains
authority.  If no valid root exists, the equilibrium is reported infeasible so the
advisor can tell the driver to back off rather than chase an impossible drift.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import root

from config.params import VehicleParams
from control.tire import slide_slip_angle
from sim.vehicle_model import compute_forces, reduced_derivative


def _wrap(angle: float) -> float:
    """Wrap to (-pi, pi] to kill atan2 branch aliases."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class DriftEquilibrium:
    feasible: bool
    V: float
    beta: float = float("nan")
    delta: float = float("nan")
    Fxr: float = float("nan")
    Fxf: float = 0.0
    r: float = float("nan")
    R: float = float("nan")          # path radius V/r (output, signed)
    vx: float = float("nan")
    vy: float = float("nan")
    rear_saturated: bool = False
    front_authority: bool = False    # front below full slide -> steering still works
    reason: str = ""

    @property
    def x3(self):
        return [self.vx, self.vy, self.r]

    @property
    def u(self):
        """Control input [delta, F_xr] at the equilibrium."""
        return [self.delta, self.Fxr]


def _residual(z, V, beta, Fxf, p, mu_f, mu_r):
    """z = [delta, Fxr, r]; fixed (V, beta)."""
    delta, Fxr, r = z
    vx = V * math.cos(beta)
    vy = V * math.sin(beta)
    return reduced_derivative([vx, vy, r], delta, Fxr, p, mu_f, mu_r, Fxf=Fxf)


def solve_drift_equilibrium(
    V: float, beta_target: float, p: VehicleParams, mu_f: float, mu_r: float,
    Fxf: float = 0.0,
) -> DriftEquilibrium:
    """Solve the drift equilibrium at target speed V and sideslip beta_target.

    Fxf is the (policy) front drive force; default 0 = rear-biased drift on the
    4-motor EV.  Returns a DriftEquilibrium (R is an output = V / r*).
    """
    if V <= 0.1 or abs(beta_target) < math.radians(2.0):
        return DriftEquilibrium(False, V, beta=beta_target,
                                reason="degenerate target (V<=0 or |beta| too small)")

    vx = V * math.cos(beta_target)
    vy = V * math.sin(beta_target)
    # Left drift (beta<0) turns left (r>0); right drift (beta>0) turns right (r<0).
    r_sign = -math.copysign(1.0, beta_target)

    best: DriftEquilibrium | None = None
    best_score = (-1, 1e18)
    for dmag in (5.0, 12.0, 20.0):
        for ffrac in (0.3, 0.5, 0.7):
            for rmag in (V / 12.0, V / 18.0, V / 25.0):
                delta0 = math.copysign(math.radians(dmag), beta_target)  # countersteer
                Fxr0 = ffrac * mu_r * p.Fzr_static
                r0 = r_sign * rmag
                sol = root(_residual, [delta0, Fxr0, r0],
                           args=(V, beta_target, Fxf, p, mu_f, mu_r), method="hybr")
                if not sol.success:
                    continue
                delta, Fxr, r = _wrap(sol.x[0]), float(sol.x[1]), float(sol.x[2])
                if abs(r) < 1e-3 or math.copysign(1.0, r) != r_sign:
                    continue
                if vx <= 0.0:
                    continue
                f = compute_forces(vx, vy, r, delta, Fxr, p, mu_f, mu_r, Fxf=Fxf)
                asl_r = slide_slip_angle(p.Ca_r, mu_r, f.Fzr, Fxr)
                asl_f = slide_slip_angle(p.Ca_f, mu_f, f.Fzf, Fxf)
                rear_saturated = asl_r > 0.0 and abs(f.alpha_r) >= 0.9 * asl_r
                front_authority = asl_f > 0.0 and abs(f.alpha_f) < asl_f
                budget_ok = (Fxr * Fxr <= (mu_r * f.Fzr) ** 2 + 1.0
                             and abs(Fxr) <= p.Fx_motor_max + 1.0)
                if not (rear_saturated and budget_ok):
                    continue
                # Prefer front authority, then the smallest-effort (|delta|) root.
                score = (1 if front_authority else 0, abs(delta))
                if (score[0] > best_score[0]
                        or (score[0] == best_score[0] and score[1] < best_score[1])):
                    best_score = score
                    best = DriftEquilibrium(
                        True, V, beta_target, delta, Fxr, Fxf, r, V / r, vx, vy,
                        rear_saturated=rear_saturated, front_authority=front_authority,
                        reason="ok" if front_authority else "front near/at saturation",
                    )

    if best is None:
        return DriftEquilibrium(
            False, V, beta=beta_target,
            reason="no rear-saturated drift root within friction circle / motor limit")
    return best


def required_lateral_accel(V: float, R: float) -> float:
    """Centripetal demand V^2/R; a sustainable drift needs this <= ~mu*g."""
    return V * V / abs(R)
