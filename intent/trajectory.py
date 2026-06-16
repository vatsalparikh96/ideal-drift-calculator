"""Driver-intent estimator.

The advisor's target is "what line the driver is trying to hold", not the instantaneous
(possibly diverging) state, and not the raw steering wheel (during a drift the wheel is
countersteer/correction, not a curvature command).  So we:

  * track speed V and sideslip beta with a low-pass filter (the velocity-vector heading
    rate r/V gives the intended path curvature; R_path = V/r),
  * detect drift entry/exit with hysteresis,
  * once in a drift, LATCH the established (V, beta) as the target and hold it; adapt the
    target only while the drift is STABLE, and FREEZE it while diverging so the advisor
    guides the driver back to the line he had -- not toward wherever the spin is going,
  * project the target onto the feasible drift band.

Output is a target (V_target, beta_target) that the equilibrium solver turns into the
sweet spot.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class IntentConfig:
    lp_tau: float = 0.4            # s, low-pass time constant for V, beta
    beta_enter: float = math.radians(12.0)   # |beta| to declare "in drift"
    beta_exit: float = math.radians(7.0)     # |beta| to declare "out of drift"
    beta_min: float = math.radians(10.0)     # feasible drift band (magnitude)
    beta_max: float = math.radians(48.0)
    adapt_tau: float = 1.5        # s, slow adaptation of the latched target when stable


@dataclass
class IntentTarget:
    in_drift: bool
    V_target: float
    beta_target: float            # 0 when not drifting


class DriverIntent:
    def __init__(self, cfg: IntentConfig | None = None):
        self.cfg = cfg or IntentConfig()
        self._started = False
        self.V_filt = 0.0
        self.beta_filt = 0.0
        self.in_drift = False
        self.V_target = 0.0
        self.beta_target = 0.0

    def update(self, V: float, beta: float, stable: bool, dt: float) -> IntentTarget:
        cfg = self.cfg
        if not self._started:
            self.V_filt, self.beta_filt = V, beta
            self._started = True

        a = dt / max(cfg.lp_tau, dt)
        self.V_filt += a * (V - self.V_filt)
        self.beta_filt += a * (beta - self.beta_filt)

        # drift entry/exit hysteresis
        if not self.in_drift and abs(self.beta_filt) > cfg.beta_enter:
            self.in_drift = True
            # latch the established drift as the intended target
            self.V_target = self.V_filt
            self.beta_target = _clamp_beta(self.beta_filt, cfg)
        elif self.in_drift and abs(self.beta_filt) < cfg.beta_exit:
            self.in_drift = False

        if self.in_drift:
            if stable:
                # slowly track the driver's intended drift while it is steady
                aa = dt / max(cfg.adapt_tau, dt)
                self.V_target += aa * (self.V_filt - self.V_target)
                self.beta_target += aa * (_clamp_beta(self.beta_filt, cfg) - self.beta_target)
            # else: FREEZE target (do not chase a divergence)
            return IntentTarget(True, self.V_target, self.beta_target)

        # not drifting: target is grip driving (no commanded drift angle)
        self.V_target = self.V_filt
        self.beta_target = 0.0
        return IntentTarget(False, self.V_filt, 0.0)


def _clamp_beta(beta: float, cfg: IntentConfig) -> float:
    s = math.copysign(1.0, beta) if beta != 0 else 1.0
    return s * min(cfg.beta_max, max(cfg.beta_min, abs(beta)))
