"""Linearization and eigen-analysis of the drift equilibrium.

Two jobs:
  1. Build the Jacobians A = df/dx, B = df/du of the reduced dynamics
     (x = [v_x, v_y, r], u = [delta, F_xr]) about an equilibrium, by central finite
     differences of the C1 tire model (the C1-continuity is what makes these
     gradients meaningful at the saturated drift point -- a hard clamp would give a
     fake zero-gradient / marginal spectrum).
  2. Classify the open-loop instability and extract the dominant unstable mode's
     LEFT eigenvector w_u (A^T w_u = lambda_u w_u, normalized w_u . v_u = 1).  The
     left eigenvector is the correct quantity to project a state error onto to read
     off "how far along the divergent direction" we are (z_u = w_u . e) -- the right
     eigenvector or a naive dot product would only be correct if A were normal, which
     the vehicle Jacobian is not.

We work the eigen-analysis in SCALED coordinates y = S x (S = diag(scale_vx,
scale_vy, scale_r)) so that m/s and rad/s are comparable in z_u and a single scalar
threshold is meaningful (eigenvalues are scale-invariant; eigenvectors transform).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.params import VehicleParams
from control.equilibria import DriftEquilibrium
from sim.vehicle_model import reduced_derivative


def linearize(eq: DriftEquilibrium, p: VehicleParams, mu_f: float, mu_r: float):
    """Return (A 3x3, B 3x2) about the equilibrium via central differences."""
    x0 = np.array(eq.x3, dtype=float)
    u0 = np.array([eq.delta, eq.Fxr], dtype=float)
    Fxf = eq.Fxf

    def f(x, u):
        return np.array(reduced_derivative(x, u[0], u[1], p, mu_f, mu_r, Fxf=Fxf))

    dx = np.array([1e-4, 1e-4, 1e-5])      # v_x, v_y [m/s], r [rad/s]
    du = np.array([1e-5, 1e-1])            # delta [rad], F_xr [N]

    A = np.zeros((3, 3))
    for j in range(3):
        e = np.zeros(3); e[j] = dx[j]
        A[:, j] = (f(x0 + e, u0) - f(x0 - e, u0)) / (2.0 * dx[j])

    B = np.zeros((3, 2))
    for j in range(2):
        e = np.zeros(2); e[j] = du[j]
        B[:, j] = (f(x0, u0 + e) - f(x0, u0 - e)) / (2.0 * du[j])

    return A, B


@dataclass
class UnstableMode:
    n_unstable: int
    lambda_u: float            # real part of the dominant unstable eigenvalue [1/s]
    complex_unstable: bool
    v_u: np.ndarray            # right eigenvector (scaled coords), real
    w_u: np.ndarray            # left eigenvector (scaled coords), real, w_u . v_u = 1
    eigvals: np.ndarray        # all eigenvalues (raw)


def unstable_mode(A: np.ndarray, scale=(1.0, 1.0, 10.0)) -> UnstableMode:
    """Dominant unstable mode of A, analysed in scaled coordinates y = S x."""
    S = np.diag(scale)
    Sinv = np.diag([1.0 / s for s in scale])
    Ay = S @ A @ Sinv

    vals, R = np.linalg.eig(Ay)
    valsL, L = np.linalg.eig(Ay.T)

    reals = vals.real
    n_unstable = int(np.sum(reals > 1e-6))
    iu = int(np.argmax(reals))                      # dominant unstable (or least stable)
    lam = vals[iu]

    v = R[:, iu]
    # Match the corresponding left eigenvector by closest eigenvalue.
    jl = int(np.argmin(np.abs(valsL - lam)))
    w = L[:, jl]

    complex_unstable = abs(lam.imag) > 1e-6
    v_r = np.real(v)
    w_r = np.real(w)
    denom = float(w_r @ v_r)
    if abs(denom) < 1e-12:
        denom = 1.0
    w_r = w_r / denom

    return UnstableMode(
        n_unstable=n_unstable,
        lambda_u=float(lam.real),
        complex_unstable=complex_unstable,
        v_u=v_r,
        w_u=w_r,
        eigvals=vals,
    )


def controllability(A: np.ndarray, B: np.ndarray):
    """Return (rank, condition_number) of the controllability matrix [B, AB, A^2B]."""
    n = A.shape[0]
    blocks = [B]
    M = B.copy()
    for _ in range(1, n):
        M = A @ M
        blocks.append(M)
    C = np.hstack(blocks)
    s = np.linalg.svd(C, compute_uv=False)
    rank = int(np.sum(s > 1e-9 * s[0]))
    cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    return rank, cond
