"""Online parameter estimation via recursive least squares (RLS).

With sideslip beta and per-axle mu MEASURED, state estimation is unnecessary; the only
things still worth estimating online are the tire-curve slope (cornering stiffness
C_alpha, in the linear regime) and the drivetrain effective radius.  Both are
bounded, cheap, and SAFE -- and gated so a held (non-exciting) drift cannot cause
covariance windup, and so adaptation FREEZES in deep saturation where C_alpha is not
identifiable (dF_y/dalpha -> 0).  Estimates here are advisory refinements; they are
never required for the stability guarantee.
"""
from __future__ import annotations

from config.params import LearningConfig


class ScalarRLS:
    """RLS for a single parameter theta in the model  y = phi * theta."""

    def __init__(self, theta0: float, P0: float, lam: float, eps_pe: float, P_max: float):
        self.theta = theta0
        self.P = P0
        self.lam = lam
        self.eps_pe = eps_pe
        self.P_max = P_max
        self.n_updates = 0

    def update(self, phi: float, y: float) -> float:
        info = phi * self.P * phi                      # persistent-excitation measure
        if info < self.eps_pe:
            return self.theta                          # not enough excitation -> hold
        K = self.P * phi / (self.lam + info)
        e = y - phi * self.theta
        self.theta += K * e
        self.P = (self.P - K * phi * self.P) / self.lam
        if self.P_max < self.P:
            self.P = self.P_max
        self.n_updates += 1
        return self.theta


class AxleStiffnessEstimator:
    """Per-axle cornering stiffness C_alpha (front, rear) from (alpha, F_y) pairs in the
    linear regime, with a saturation freeze gate."""

    def __init__(self, Ca_f0: float, Ca_r0: float, cfg: LearningConfig,
                 alpha_linear: float = 0.10):
        self.cfg = cfg
        self.alpha_linear = alpha_linear        # rad; only learn slope where ~linear
        self.front = ScalarRLS(Ca_f0, 1e8, cfg.rls_lambda, cfg.rls_eps_pe, cfg.rls_P_max)
        self.rear = ScalarRLS(Ca_r0, 1e8, cfg.rls_lambda, cfg.rls_eps_pe, cfg.rls_P_max)

    def update(self, alpha_f, Fyf, alpha_r, Fyr, beta, U_f, U_r):
        """Update each axle's C_alpha when in the linear regime and not saturated.
        Model F_y = -C_alpha * alpha  ->  phi = -alpha, y = F_y."""
        frozen = abs(beta) > self.cfg.beta_freeze
        if not frozen and abs(alpha_f) < self.alpha_linear and U_f < self.cfg.U_freeze:
            self.front.update(-alpha_f, Fyf)
        if not frozen and abs(alpha_r) < self.alpha_linear and U_r < self.cfg.U_freeze:
            self.rear.update(-alpha_r, Fyr)
        return self.front.theta, self.rear.theta


class MotorRadiusEstimator:
    """Effective wheel radius / driveline trim from  m*a_x + F_drag = (1/r_eff)*sum(T).

    Estimates 1/r_eff (well-excited whenever throttle varies).  A safe slow trim; small
    value-add since motor torque and a_x are already clean inputs."""

    def __init__(self, inv_reff0: float, cfg: LearningConfig):
        self.rls = ScalarRLS(inv_reff0, 1e3, cfg.rls_lambda, 1e-3, 1e6)

    def update(self, sum_torque: float, m_ax_plus_drag: float) -> float:
        # y = m*a_x + F_drag, phi = sum_torque, theta = 1/r_eff
        return self.rls.update(sum_torque, m_ax_plus_drag)

    @property
    def r_eff(self) -> float:
        return 1.0 / self.rls.theta if self.rls.theta > 1e-6 else float("inf")
