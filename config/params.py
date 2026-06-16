"""Configuration: vehicle parameters, sign conventions, controller/monitor/learning
tuning, and real-time loop rates for the drift sweet-spot advisor.

SIGN CONVENTION (ISO 8855) -- pinned here and assumed everywhere in the codebase:
    x : forward (body longitudinal)
    y : left    (body lateral)
    z : up
    r > 0  : yaw rate counter-clockwise  == LEFT turn
    beta = atan2(v_y, v_x)  > 0 : velocity vector points LEFT of heading
    delta > 0 : steer LEFT (front road-wheel angle)

Consequences (unit-tested in tests/test_signs.py):
    * LEFT-hand drift  ->  r > 0, beta < 0, countersteer delta < 0 (wheels point right)
    * Countersteer = delta with the SAME sign as beta and OPPOSITE sign to r
    * Tire lateral force OPPOSES slip angle (C_alpha > 0, leading minus sign in F_y)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleParams:
    """Physical parameters of a sporty ~1.8 t, 4-motor EV (near 50/50 weight split)."""

    # --- mass / inertia / geometry ---
    m: float = 1800.0       # kg, total mass
    Iz: float = 2800.0      # kg*m^2, yaw moment of inertia
    a: float = 1.40         # m, CG -> front axle
    b: float = 1.40         # m, CG -> rear axle
    h: float = 0.50         # m, CG height (drives longitudinal load transfer)
    g: float = 9.81         # m/s^2

    # --- tires: per-axle (lumped L+R) cornering stiffness, N/rad ---
    # Front is intentionally softer than rear so that at a steady drift the FRONT
    # axle stays below its full-slide angle (alpha_sl_f ~ 20 deg) while the rear is
    # saturated.  This is what preserves STEERING AUTHORITY for stabilization; with a
    # stiff front the front saturates too and the pure-Fiala flat post-peak leaves the
    # controller with no steering gain.  (A tuning choice for the lumped single-track
    # model; documented in the plan's "known modeling gaps".)
    Ca_f: float = 60_000.0
    Ca_r: float = 160_000.0

    # --- drivetrain ---
    r_eff: float = 0.33     # m, effective wheel radius (motor torque -> wheel force)
    # Total per-axle longitudinal force limits are governed by the friction circle
    # (mu * Fz); we additionally cap commanded force by motor capability:
    Fx_motor_max: float = 9000.0   # N, max drive force per axle (2 motors)
    Fx_brake_max: float = 12000.0  # N, max brake force per axle

    # --- aerodynamic drag: F_aero = k_aero * v^2 (0.5 * rho * Cd * A) ---
    k_aero: float = 0.40    # N/(m/s)^2

    # --- numerical guards ---
    v_eps: float = 2.0      # m/s, low-speed slip-angle denominator guard
    Fz_min: float = 200.0   # N, minimum axle vertical load (avoid div-by-zero)

    @property
    def L(self) -> float:
        return self.a + self.b

    # Static axle loads (no load transfer)
    @property
    def Fzf_static(self) -> float:
        return self.m * self.g * self.b / self.L

    @property
    def Fzr_static(self) -> float:
        return self.m * self.g * self.a / self.L


@dataclass(frozen=True)
class ControllerConfig:
    """LQR weights and gain-scheduling for the drift stabilizer.

    State error x - x* is ordered [v_x, v_y, r]; input u is [delta, F_xr].

    Q penalizes v_y (sideslip) and r (yaw) heavily -- those are the drift state we
    must hold; v_x (speed) is regulated more gently.  R penalizes input *effort*.
    We do NOT need to hand-force "no throttle for yaw": at a saturated rear the
    B[:, F_xr] lateral/yaw entries are ~1e-3, so the LQR naturally uses steering for
    [v_y, r] and throttle only for v_x.  R is scaled to each input's magnitude.
    """

    # Steering-only LQR weights on full-state error [v_x, v_y, r].  Steering is the
    # ONLY fast lateral/yaw actuator (throttle has ~0 lateral authority at a saturated
    # rear), and the unstable mode is v_x-involved, so we keep all 3 states and
    # stabilize with steering.  v_y (sideslip) and r (yaw) are penalized most.
    # q_vx = 0: steering does NOT chase speed error (throttle owns v_x).  Weights tuned
    # for well-damped closed-loop poles (~-1.4 +/- 1.4j) and humanly-trackable cues
    # (~10 deg steering correction for a 5 deg sideslip error).
    q_vx: float = 0.0
    q_vy: float = 6.0
    q_r: float = 8.0
    r_delta: float = 200.0         # steering effort penalty

    # Throttle is NOT used by the LQR.  It (a) trims speed toward V_target via a
    # simple proportional law and (b) is advised back toward F_xr* (the equilibrium
    # drive force) -- the friction-circle anti-spin logic: excess F_xr -> "lift".
    k_speed: float = 1500.0        # N per (m/s) speed-error trim on F_xr

    # Actuator limits (advisory clipping + escalation triggers)
    delta_max: float = 0.6         # rad, road-wheel angle limit (~34 deg)
    # F_xr limit is min(motor cap, friction circle) computed at runtime.

    @property
    def Q(self):
        return (self.q_vx, self.q_vy, self.q_r)


@dataclass(frozen=True)
class MonitorConfig:
    """Over/understeer classifier + stability-margin thresholds."""

    U_thresh: float = 0.92         # axle friction-utilization "saturated" threshold
    U_hysteresis: float = 0.06     # dead-band to stop label flip-flop
    z_thresh: float = 1.0          # |z_u| (scaled, projected coord) => "act now"
    # State scaling applied BEFORE projecting onto the left unstable eigenvector,
    # so v_x, v_y (m/s) and r (rad/s) are comparable in z_u.
    scale_vx: float = 1.0
    scale_vy: float = 1.0
    scale_r: float = 10.0          # rad/s -> comparable to m/s magnitudes
    tau_floor: float = 0.05        # s, clamp on time-to-loss
    tau_safe: float = 5.0          # s, margins above this are reported as "safe"
    react_horizon: float = 0.6     # s, human reaction latency -> trigger early


@dataclass(frozen=True)
class LearningConfig:
    """RLS + learned-residual gating.  Learning is a confidence-gated REFINEMENT and
    must never be load-bearing for stability."""

    rls_lambda: float = 0.99       # forgetting factor (~100-sample window @100 Hz)
    rls_eps_pe: float = 1.0e3      # persistent-excitation gate: update only if phi'P phi > eps
    rls_P_max: float = 1.0e9       # trace(P) clamp to prevent covariance windup
    # Freeze ALL adaptation above these saturation gates (no excitation, least forgiving):
    beta_freeze: float = 0.30      # rad (~17 deg) sideslip gate
    U_freeze: float = 0.85         # axle-utilization gate
    # Learned residual hard bounds (fraction of mu*Fz) before it may touch A,B:
    resid_bound_frac: float = 0.15
    resid_rate_limit: float = 2000.0   # N/s, slew on residual mean


@dataclass(frozen=True)
class LoopRates:
    """Real-time loop rates.  Timescale separation: cheap fast loop + slow re-solve."""

    control_hz: float = 100.0      # LQR + saturation + monitor + HMI mapping
    equilibrium_hz: float = 20.0   # equilibrium re-solve + Jacobian/LQR-gain refresh
    hmi_hz: float = 30.0           # display render
    sim_hz: float = 100.0          # plant integration step (RK4)

    @property
    def dt_control(self) -> float:
        return 1.0 / self.control_hz

    @property
    def dt_sim(self) -> float:
        return 1.0 / self.sim_hz


# Convenient module-level defaults
DEFAULT_VEHICLE = VehicleParams()
DEFAULT_CONTROLLER = ControllerConfig()
DEFAULT_MONITOR = MonitorConfig()
DEFAULT_LEARNING = LearningConfig()
DEFAULT_RATES = LoopRates()

# Nominal friction (per axle) when not otherwise supplied; in the sim mu is a signal.
MU_NOMINAL = 0.95
