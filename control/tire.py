"""C1-continuous, fully-derated Fiala (brush) lateral tire model with a friction
CIRCLE longitudinal coupling.

Why this exact form (from the adversarial design review):
  * It is a friction CIRCLE, not an ellipse -- a single scalar mu per axle.
  * The longitudinal force F_x must derate the WHOLE lateral curve (peak AND the
    cubic interior), not merely cap the peak.  We do this by replacing every
    occurrence of the friction limit (mu*Fz) with the derated peak
        eta = xi * mu * Fz = sqrt(max(0, (mu*Fz)^2 - F_x^2))
    i.e. eta is exactly the remaining lateral force budget F_y_max.
  * The model is C0 AND C1 continuous at the full-slide slip angle alpha_sl
    (F_y -> -eta and dF_y/dalpha -> 0 there).  This is mandatory because the
    controller takes Jacobians A, B through this function; a hard clamp would have
    a zero gradient that FAKES marginal stability at the saturated drift point.

Sign convention: lateral force OPPOSES slip angle (restoring), C_alpha > 0, leading
minus sign.  See config/params.py.
"""
from __future__ import annotations

import math


def friction_budget(mu: float, Fz: float, Fx: float) -> float:
    """Remaining lateral force budget eta = F_y_max = sqrt(max(0,(mu*Fz)^2 - Fx^2)).

    Returns 0.0 when the axle is longitudinally saturated (|Fx| >= mu*Fz) -- the
    physical spin-onset condition the advisor must catch.
    """
    cap = mu * Fz
    rem2 = cap * cap - Fx * Fx
    if rem2 <= 0.0:
        return 0.0
    return math.sqrt(rem2)


def derate_factor(mu: float, Fz: float, Fx: float) -> float:
    """xi = eta / (mu*Fz) in [0, 1]; the fraction of grip left for lateral use."""
    cap = mu * Fz
    if cap <= 0.0:
        return 0.0
    return friction_budget(mu, Fz, Fx) / cap


def fiala_lateral(alpha: float, Ca: float, mu: float, Fz: float, Fx: float) -> float:
    """Lateral tire force F_y(alpha) for one axle, derated by longitudinal use.

    Parameters
    ----------
    alpha : slip angle [rad]
    Ca    : cornering stiffness [N/rad] (> 0)
    mu    : friction coefficient for this axle
    Fz    : vertical load [N] (>= 0)
    Fx    : longitudinal force currently used by this axle [N]
    """
    eta = friction_budget(mu, Fz, Fx)
    # Guard FIRST: if longitudinally saturated there is no lateral capacity.
    if eta <= 0.0:
        return 0.0

    t = math.tan(alpha)
    tan_sl = 3.0 * eta / Ca          # tan(alpha_sl)
    if abs(t) < tan_sl:
        return (
            -Ca * t
            + (Ca * Ca / (3.0 * eta)) * abs(t) * t
            - (Ca ** 3 / (27.0 * eta * eta)) * t ** 3
        )
    # Full slide: flat sliding force at the derated friction limit.
    return -math.copysign(eta, alpha)


def fiala_lateral_and_dalpha(
    alpha: float, Ca: float, mu: float, Fz: float, Fx: float
) -> tuple[float, float]:
    """Return (F_y, dF_y/dalpha).  Used to verify C1 continuity and (optionally) to
    build analytic Jacobians."""
    eta = friction_budget(mu, Fz, Fx)
    if eta <= 0.0:
        return 0.0, 0.0

    t = math.tan(alpha)
    sec2 = 1.0 + t * t               # d(tan)/dalpha = sec^2(alpha)
    tan_sl = 3.0 * eta / Ca
    if abs(t) < tan_sl:
        fy = (
            -Ca * t
            + (Ca * Ca / (3.0 * eta)) * abs(t) * t
            - (Ca ** 3 / (27.0 * eta * eta)) * t ** 3
        )
        dfy_dt = (
            -Ca
            + (2.0 * Ca * Ca / (3.0 * eta)) * abs(t)
            - (Ca ** 3 / (9.0 * eta * eta)) * t * t
        )
        return fy, dfy_dt * sec2
    return -math.copysign(eta, alpha), 0.0


def slide_slip_angle(Ca: float, mu: float, Fz: float, Fx: float) -> float:
    """alpha_sl = atan(3*eta/Ca): slip angle at which the tire is fully sliding."""
    eta = friction_budget(mu, Fz, Fx)
    if eta <= 0.0:
        return 0.0
    return math.atan(3.0 * eta / Ca)
