"""Automatic drift INITIATION and EXIT — a state machine over the LQR core.

Holding a drift (the corrector) is only half the story; a complete maneuver also has to
*enter* and *leave* one.  This sequences four phases:

    GRIP  -> drive straight on grip
    ENTER -> steer in + stab the throttle to break the rear (open-loop kick) until the
             sideslip crosses into the drift region and the rear saturates
    DRIFT -> hand off to the drift-equilibrium LQR, which captures and holds the sweet spot
    EXIT  -> lift toward a coast force and unwind steering; restored rear grip pulls the
             sideslip back to zero, then return to GRIP

The kick only has to land the state inside the LQR's (large) basin of attraction; the
stabilizer does the rest.
"""
from __future__ import annotations

import math

from config.params import VehicleParams
from control.corrector import DriftController
from control.equilibria import DriftEquilibrium
from control.tire import slide_slip_angle
from sim.vehicle_model import compute_forces, sideslip

_ENTER_STEER_RAD = math.radians(20.0)    # turn-in angle for the entry kick


class DriftSequencer:
    def __init__(self, controller: DriftController, eq: DriftEquilibrium,
                 p: VehicleParams, mu: float, t_enter: float = 0.5, t_exit: float = 3.5,
                 enter_steer: float = _ENTER_STEER_RAD, enter_throttle: float = 8500.0,
                 coast: float = 1200.0):
        self.ctrl = controller
        self.eq = eq
        self.p = p
        self.mu = mu
        self.t_enter = t_enter
        self.t_exit = t_exit
        self.enter_steer = math.copysign(enter_steer, -eq.beta)   # steer into the turn
        self.enter_throttle = enter_throttle
        self.coast = coast
        self.phase = "GRIP"
        self.delta = 0.0
        self.Fxr = coast
        self._entered = False        # one-shot: only initiate the drift once

    def command(self, x3, t: float, dt: float):
        p, mu, eq = self.p, self.mu, self.eq
        vx, vy, r = x3
        beta = sideslip(vx, vy)
        f = compute_forces(vx, vy, r, self.delta, self.Fxr, p, mu, mu, Fxf=eq.Fxf)
        asl_r = slide_slip_angle(p.Ca_r, mu, f.Fzr, self.Fxr)
        rear_sat = asl_r > 0 and abs(f.alpha_r) >= 0.9 * asl_r

        if self.phase == "GRIP" and t >= self.t_enter and not self._entered:
            self.phase = "ENTER"
            self._entered = True

        if self.phase == "ENTER":
            self.delta, self.Fxr = self.enter_steer, self.enter_throttle
            sign_ok = math.copysign(1, beta) == math.copysign(1, eq.beta)
            if abs(beta) > math.radians(18.0) and rear_sat and sign_ok:
                self.phase = "DRIFT"

        if self.phase == "DRIFT":
            adv = self.ctrl.advise(x3, self.delta, self.Fxr, p, mu, mu)
            if adv.feasible:
                self.delta, self.Fxr = adv.delta_target, adv.Fxr_target
            if t >= self.t_exit:
                self.phase = "EXIT"

        if self.phase == "EXIT":
            a = min(1.0, 3.0 * dt)
            self.Fxr += (self.coast - self.Fxr) * a
            self.delta += (0.0 - self.delta) * a
            if abs(beta) < math.radians(5.0) and abs(r) < 0.2:
                self.phase = "GRIP"

        if self.phase == "GRIP":
            self.delta, self.Fxr = 0.0, self.coast

        return self.delta, self.Fxr, self.phase
