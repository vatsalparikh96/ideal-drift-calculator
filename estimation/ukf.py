"""Unscented Kalman Filter for drift state estimation.

Drops the "sideslip is measured" simplification: a real car does NOT have a cheap
sideslip sensor, so beta must be reconstructed.  This UKF estimates the planar state
x = [v_x, v_y, r] from realistic, noisy measurements

    z = [r_gyro, v_x_wheel, a_x, a_y]

using the single-track dynamics (sim.vehicle_model) as the process model.  The trick
is that the lateral acceleration a_y couples to the unmeasured lateral velocity v_y
through the tire forces, so a_y + the model make v_y (hence beta = atan2(v_y, v_x))
observable.

Implementation is the standard Wan & Van der Merwe scaled unscented transform; no
external dependency beyond numpy/scipy.
"""
from __future__ import annotations

import math

import numpy as np

from config.params import VehicleParams
from sim.vehicle_model import compute_forces, reduced_derivative


class UKF:
    """Additive-noise scaled UKF for a generic (f, h)."""

    def __init__(self, n: int, m: int, Q: np.ndarray, R: np.ndarray,
                 alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        self.n = n
        self.m = m
        self.Q = Q
        self.R = R
        lam = alpha ** 2 * (n + kappa) - n
        self.lam = lam
        self.gamma = math.sqrt(n + lam)
        self.Wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = lam / (n + lam)
        self.Wc[0] = lam / (n + lam) + (1.0 - alpha ** 2 + beta)

    def _sigma_points(self, x, P):
        P = 0.5 * (P + P.T) + 1e-9 * np.eye(self.n)     # symmetrize + jitter
        S = np.linalg.cholesky(P)
        pts = np.zeros((2 * self.n + 1, self.n))
        pts[0] = x
        for i in range(self.n):
            pts[1 + i] = x + self.gamma * S[:, i]
            pts[1 + self.n + i] = x - self.gamma * S[:, i]
        return pts

    def step(self, x, P, f, h, z):
        """One predict+update.  f: state->state, h: state->measurement."""
        # --- predict ---
        pts = self._sigma_points(x, P)
        fpts = np.array([f(p) for p in pts])
        x_pred = self.Wm @ fpts
        P_pred = self.Q.copy()
        for i in range(2 * self.n + 1):
            d = fpts[i] - x_pred
            P_pred += self.Wc[i] * np.outer(d, d)

        # --- update ---
        pts2 = self._sigma_points(x_pred, P_pred)
        zpts = np.array([h(p) for p in pts2])
        z_pred = self.Wm @ zpts
        S = self.R.copy()
        Pxz = np.zeros((self.n, self.m))
        for i in range(2 * self.n + 1):
            dz = zpts[i] - z_pred
            dx = pts2[i] - x_pred
            S += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)
        K = Pxz @ np.linalg.inv(S)
        x_new = x_pred + K @ (np.asarray(z) - z_pred)
        P_new = P_pred - K @ S @ K.T
        return x_new, P_new


class DriftStateEstimator:
    """UKF specialised to the single-track drift model, with accelerometer-bias estimation.

    State x = [v_x, v_y, r, b_ay] (the 4th entry is the lateral-accelerometer bias).
    Measurements z = [r, v_x (wheel), a_x, a_y].  Estimating the bias lets the filter
    reject a constant accelerometer offset that would otherwise corrupt the sideslip
    estimate (and which destroys naive dead-reckoning).
    """

    def __init__(self, p: VehicleParams, mu: float, x0=None,
                 sigma_r=0.01, sigma_vx=0.10, sigma_a=0.20,
                 q=(0.15, 0.8, 0.05, 0.05)):
        self.p = p
        self.mu = mu
        # v_y gets a larger Q (weakly observable at a saturated rear, dF_yr/dalpha_r -> 0);
        # b_ay is a slowly-varying bias (small Q, random walk).
        Q = np.diag(q) ** 2
        R = np.diag([sigma_r, sigma_vx, sigma_a, sigma_a]) ** 2
        self.ukf = UKF(n=4, m=4, Q=Q, R=R)
        x0 = list(x0) if x0 is not None else [10.0, 0.0, 0.0]
        if len(x0) == 3:
            x0 = [*x0, 0.0]                           # append accelerometer-bias state
        self.x = np.array(x0, dtype=float)
        self.P = np.diag([1.0, 9.0, 0.5, 1.0])        # v_y and bias initially unknown

    def update(self, z, delta, Fxr, dt, Fxf=0.0):
        p, mu = self.p, self.mu

        def f(x):
            d = reduced_derivative(x[:3], delta, Fxr, p, mu, mu, Fxf=Fxf)
            return np.array([x[0] + dt * d[0], x[1] + dt * d[1], x[2] + dt * d[2], x[3]])

        def h(x):
            fr = compute_forces(x[0], x[1], x[2], delta, Fxr, p, mu, mu, Fxf=Fxf)
            return np.array([x[2], x[0], fr.ax_body, fr.ay_body + x[3]])   # +b_ay

        self.x, self.P = self.ukf.step(self.x, self.P, f, h, z)
        return self.x

    @property
    def bias_ay(self) -> float:
        return float(self.x[3])

    @property
    def beta(self) -> float:
        return math.atan2(self.x[1], self.x[0])

    @property
    def vx(self) -> float:
        return float(self.x[0])

    @property
    def vy(self) -> float:
        return float(self.x[1])

    @property
    def r(self) -> float:
        return float(self.x[2])
