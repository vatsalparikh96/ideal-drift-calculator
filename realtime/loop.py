"""Real-time advisor orchestrator.

Wires the pipeline together with timescale separation:
  * fast loop (100 Hz): read signals -> update intent -> monitor -> advise
  * slow loop (~20 Hz): re-solve the drift equilibrium for the current intent and
    refresh the LQR gain / Jacobian (decimated to save compute)

It does NOT integrate the plant -- the simulation/driver does that and calls update()
each tick with the latest signals and the driver's current inputs.  The optional
adaptation layer (RLS + learned residual) runs here but, by default, is observe-only
(not fed back into the stability-critical controller).
"""
from __future__ import annotations

from dataclasses import dataclass

from config.params import ControllerConfig, LearningConfig, LoopRates, MonitorConfig, VehicleParams
from control.corrector import Advice, DriftController
from control.equilibria import DriftEquilibrium, solve_drift_equilibrium
from control.stability_monitor import MonitorState, StabilityMonitor
from estimation.rls import AxleStiffnessEstimator
from intent.trajectory import DriverIntent, IntentTarget
from learning.tire_residual import TireResidualModel
from sim.sensors import SignalSet
from sim.vehicle_model import compute_forces


@dataclass
class Telemetry:
    signals: SignalSet
    intent: IntentTarget
    eq: DriftEquilibrium | None
    advice: Advice | None
    monitor: MonitorState | None
    Ca_f_hat: float
    Ca_r_hat: float


class Advisor:
    def __init__(self, p: VehicleParams,
                 ctrl_cfg: ControllerConfig, mon_cfg: MonitorConfig,
                 learn_cfg: LearningConfig, rates: LoopRates, adapt: bool = False):
        self.p = p
        self.rates = rates
        self.adapt = adapt
        self.controller = DriftController(ctrl_cfg)
        self.monitor = StabilityMonitor(mon_cfg)
        self.intent = DriverIntent()
        self.stiffness = AxleStiffnessEstimator(p.Ca_f, p.Ca_r, learn_cfg)
        self.residual_f = TireResidualModel(learn_cfg)
        self.residual_r = TireResidualModel(learn_cfg)
        self._k = 0
        self._eq_every = max(1, round(rates.control_hz / rates.equilibrium_hz))
        self._last_severity = "ok"
        self.eq: DriftEquilibrium | None = None
        self._eq_key: tuple | None = None

    def update(self, s: SignalSet, dt: float) -> Telemetry:
        p = self.p
        # 1) driver intent (freeze target while diverging)
        stable = self._last_severity != "act"
        target = self.intent.update(s.V, s.beta, stable, dt)

        # 2) slow loop: re-solve equilibrium + refresh controller.  Decimated to the
        # equilibrium rate AND gated on the (rounded) target/grip actually changing --
        # re-solving an unchanged target yields the same equilibrium and just burns
        # compute (costly without SciPy, e.g. the browser/WASM build), so a held drift
        # solves once and the loop stays cheap.
        if target.in_drift:
            due = self._k % self._eq_every == 0 or self.eq is None or not self.eq.feasible
            key = (round(target.V_target, 1), round(target.beta_target, 3),
                   round(s.mu_f, 3), round(s.mu_r, 3))
            if due and (key != self._eq_key or self.eq is None or not self.eq.feasible):
                self.eq = solve_drift_equilibrium(
                    target.V_target, target.beta_target, p, s.mu_f, s.mu_r)
                self.controller.refresh(self.eq, p, s.mu_f, s.mu_r)
                self._eq_key = key
        else:
            self.eq = None
            self._eq_key = None

        # 3) monitor + 4) advise
        mon = None
        adv = None
        if self.eq is not None and self.eq.feasible:
            mon = self.monitor.update(s.x3, self.eq, self.controller.A,
                                      s.delta, s.Fxr, p, s.mu_f, s.mu_r)
            self._last_severity = mon.severity
            adv = self.controller.advise(s.x3, s.delta, s.Fxr, p, s.mu_f, s.mu_r)
        else:
            self._last_severity = "ok"

        # 5) optional adaptation (observe-only by default).  Derive axle slip/forces
        # from the model at the current state and feed the gated RLS.  (In this sim the
        # plant and model match, so RLS confirms the nominal C_alpha; under a deliberate
        # plant/controller mismatch it would track the true value.)
        if self.adapt and mon is not None:
            f = compute_forces(s.vx, s.vy, s.r, s.delta, s.Fxr, p, s.mu_f, s.mu_r, Fxf=s.Fxf)
            self.stiffness.update(f.alpha_f, f.Fyf, f.alpha_r, f.Fyr,
                                  s.beta, mon.U_f, mon.U_r)
        Ca_f_hat, Ca_r_hat = self.stiffness.front.theta, self.stiffness.rear.theta

        self._k += 1
        return Telemetry(s, target, self.eq, adv, mon, Ca_f_hat, Ca_r_hat)
