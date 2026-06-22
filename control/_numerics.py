"""NumPy-only fallbacks for the two SciPy calls on the advisor's runtime path.

The control loop normally uses ``scipy.linalg.solve_continuous_are`` (LQR gain) and
``scipy.optimize.root`` (drift-equilibrium solve).  SciPy is heavy and fragile to ship to
the browser (WebAssembly / pygbag), whereas NumPy is rock-solid there.  So
``corrector.py`` and ``equilibria.py`` use SciPy when it is importable and these
drop-in replacements otherwise.  ``tests/test_numerics.py`` pins them to SciPy's output.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


def solve_care(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Continuous-time algebraic Riccati solution via the Hamiltonian eigenvectors.

    Solves ``AᵀP + PA - PBR⁻¹BᵀP + Q = 0`` for the stabilizing (symmetric PSD) ``P``.
    For the small (n=3, m=1) systems here the eigenvector method is accurate and needs
    only ``numpy.linalg``.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    n = A.shape[0]
    Rinv = np.linalg.inv(R)
    H = np.block([[A, -B @ Rinv @ B.T],
                  [-Q, -A.T]])
    w, V = np.linalg.eig(H)
    # stabilizing subspace = the n eigenvectors with negative-real-part eigenvalues
    idx = np.argsort(w.real)[:n]
    U = V[:, idx]
    U1 = U[:n, :]
    U2 = U[n:, :]
    P = U2 @ np.linalg.inv(U1)
    P = np.real(P)
    return (P + P.T) / 2.0


@dataclass
class RootResult:
    """Mimics the subset of ``scipy.optimize.OptimizeResult`` the solver reads."""

    x: np.ndarray
    success: bool


def root_newton(func: Callable[..., Any], x0, args: tuple = (),
                tol: float = 1e-10, maxiter: int = 80) -> RootResult:
    """Damped Newton root-find with a finite-difference Jacobian.

    Drop-in for ``scipy.optimize.root(func, x0, args=args, method="hybr")`` on the
    small square systems in ``equilibria.py``; returns ``.x`` and ``.success``.
    """
    x = np.array(x0, dtype=float)
    f = np.asarray(func(x, *args), dtype=float)
    for _ in range(maxiter):
        fn = float(np.linalg.norm(f))
        if fn < tol:
            return RootResult(x, True)
        J = _fd_jacobian(func, x, f, args)
        try:
            step = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            step, *_ = np.linalg.lstsq(J, -f, rcond=None)
        # backtracking line search so a bad Newton step cannot diverge
        scale = 1.0
        x_new, f_new = x + step, None
        for _ in range(25):
            x_try = x + scale * step
            f_try = np.asarray(func(x_try, *args), dtype=float)
            if np.linalg.norm(f_try) < fn:
                x_new, f_new = x_try, f_try
                break
            scale *= 0.5
        if f_new is None:                       # no decrease found -> stuck
            return RootResult(x, False)
        x, f = x_new, f_new
    return RootResult(x, bool(np.linalg.norm(f) < 1e-6))


def _fd_jacobian(func: Callable[..., Any], x: np.ndarray, f0: np.ndarray,
                 args: tuple, eps: float = 1e-7) -> np.ndarray:
    n = x.size
    J = np.empty((f0.size, n))
    for i in range(n):
        h = eps * max(1.0, abs(x[i]))
        dx = np.zeros(n)
        dx[i] = h
        fi = np.asarray(func(x + dx, *args), dtype=float)
        J[:, i] = (fi - f0) / h
    return J
