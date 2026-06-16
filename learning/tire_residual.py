"""Learned tire-force residual.

Learns the part of the lateral force the Fiala/brush model gets wrong:
    dF_y = F_y_measured - F_y_Fiala(alpha, Fz, Fx)
as a function of slip angle, per axle, using a small fixed set of RBF features fit by
recursive least squares.  This keeps inference BOUNDED (a handful of features) so it is
real-time safe -- unlike an exact GP whose cost grows with data.

Safety rails (this is a confidence-gated REFINEMENT, never load-bearing):
  * output is hard-bounded to +/- resid_bound_frac * mu*Fz,
  * the mean is rate-limited before it may shift the equilibrium/Jacobians,
  * adaptation FREEZES above the saturation/|beta| gate (no excitation, least forgiving).
The variance/uncertainty (here, feature coverage) should drive driver-facing margins,
not the nominal controller.
"""
from __future__ import annotations

import numpy as np

from config.params import LearningConfig


class TireResidualModel:
    """Per-axle RBF-in-alpha residual model, RLS-fit, bounded output."""

    def __init__(self, cfg: LearningConfig, n_centers: int = 5,
                 alpha_range: float = np.radians(40.0)):
        self.cfg = cfg
        self.centers = np.linspace(-alpha_range, alpha_range, n_centers)
        self.width = (self.centers[1] - self.centers[0])
        self.w = np.zeros(n_centers)
        self.P = np.eye(n_centers) * 1e2
        self.lam = cfg.rls_lambda
        self._last_output = 0.0

    def _phi(self, alpha: float) -> np.ndarray:
        return np.exp(-((alpha - self.centers) / self.width) ** 2)

    def predict(self, alpha: float, mu: float, Fz: float) -> float:
        bound = self.cfg.resid_bound_frac * mu * Fz
        dFy = float(self.w @ self._phi(alpha))
        return float(np.clip(dFy, -bound, bound))

    def update(self, alpha: float, mu: float, Fz: float, dFy_meas: float,
               beta: float, U: float, dt: float):
        """RLS update, gated and rate-limited.  No update when saturated/diverging."""
        if abs(beta) > self.cfg.beta_freeze or self.cfg.U_freeze < U:
            return                                   # freeze adaptation in saturation
        bound = self.cfg.resid_bound_frac * mu * Fz
        dFy_meas = float(np.clip(dFy_meas, -bound, bound))
        phi = self._phi(alpha)
        Pphi = self.P @ phi
        denom = self.lam + float(phi @ Pphi)
        K = Pphi / denom
        e = dFy_meas - float(self.w @ phi)
        self.w = self.w + K * e
        self.P = (self.P - np.outer(K, Pphi)) / self.lam

    def coverage(self, alpha: float) -> float:
        """Crude confidence proxy in [0,1]: how well alpha is covered by the RBF basis."""
        return float(np.clip(np.max(self._phi(alpha)), 0.0, 1.0))
