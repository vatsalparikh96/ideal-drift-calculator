"""Signal bus: package the simulator's truth state into the clean signal set the
advisor consumes.

Per the user's simplifying assumption, all signals (incl. sideslip beta and per-axle mu)
are already available -- so this is mostly a repackaging layer, not an estimator.  An
optional noise toggle adds realistic sensor noise so the advisor's robustness can be
exercised; the design review recommends a light complementary filter on beta/mu even
when "measured", which can be layered on here later.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config.params import VehicleParams
from sim.vehicle_model import compute_forces, sideslip, speed


@dataclass
class SignalSet:
    vx: float
    vy: float
    r: float
    V: float
    beta: float
    a_x: float            # body-frame longitudinal accel (accelerometer)
    a_y: float            # body-frame lateral accel
    delta: float          # current road-wheel angle (driver)
    Fxr: float            # current rear axle drive force (from throttle)
    Fxf: float            # current front axle drive force
    mu_f: float
    mu_r: float

    @property
    def x3(self):
        return [self.vx, self.vy, self.r]


class SensorBus:
    def __init__(self, noise: bool = False, seed: int = 0):
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        # 1-sigma noise levels (only used if noise=True)
        self.sigma_V = 0.05        # m/s
        self.sigma_beta = np.radians(0.3)
        self.sigma_r = 0.005       # rad/s
        self.sigma_a = 0.05        # m/s^2

    def read(self, x6, delta: float, Fxr: float, mu_f: float, mu_r: float,
             p: VehicleParams, Fxf: float = 0.0) -> SignalSet:
        vx, vy, r = x6[0], x6[1], x6[2]
        f = compute_forces(vx, vy, r, delta, Fxr, p, mu_f, mu_r, Fxf=Fxf)
        V = speed(vx, vy)
        beta = sideslip(vx, vy)
        a_x, a_y = f.ax_body, f.ay_body

        if self.noise:
            V += self.rng.normal(0.0, self.sigma_V)
            beta += self.rng.normal(0.0, self.sigma_beta)
            r += self.rng.normal(0.0, self.sigma_r)
            a_x += self.rng.normal(0.0, self.sigma_a)
            a_y += self.rng.normal(0.0, self.sigma_a)
            vx, vy = V * np.cos(beta), V * np.sin(beta)

        return SignalSet(vx, vy, r, V, beta, a_x, a_y, delta, Fxr, Fxf, mu_f, mu_r)
