# 🏎️ Drift Sweet-Spot Advisor

**A real-time co-pilot that keeps a car in a drift.** It watches a 4-motor EV mid-slide and
tells the driver — every 10 ms — exactly **how to move the steering wheel and accelerator** to
hold the drift and not spin, blending vehicle-dynamics control with online machine learning.

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![tests](https://img.shields.io/badge/tests-27%20passing-brightgreen)
![coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)
![lint](https://img.shields.io/badge/lint-ruff-purple)
![types](https://img.shields.io/badge/types-mypy-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

![drive demo](media/drive_demo.gif)

*Drive it yourself — the HUD shows the advisor's steering and pedal **targets**, a stability
margin, and a live β–r phase dot. Here the driver over-throttles, the advisor flashes
**LIFT + countersteer**, and the drift is saved.*

---

## Why it's interesting

A steady-state drift is an **open-loop-unstable equilibrium** of the vehicle dynamics — left
alone it diverges into a spin or washes out. Holding it requires active stabilizing feedback.
This project computes that feedback law in real time and **shows it to the driver** instead of
taking over.

The headline result, measured by sweeping thousands of initial drift states:

> **The advisor expands the recoverable region from 7% of drift states to 88%.**

![basin of attraction](media/fig_basin.png)

The user's exact scenario — *too much throttle mid-drift* — is **friction-circle coupling**:
extra rear drive force eats the rear tire's lateral grip (`F_yr_max = √((μ·F_zr)² − F_xr²)`),
yaw runs away, the car spins. The physics-derived fix is exactly **lift throttle + countersteer**:

![scenario](media/out_comparison.png)

*Ignore the advice → spin (1.9 s). Brief mistake then obey → recover. Always assist → hold.*

---

## The two sides of the project

**Vehicle dynamics / control** — the drift is an unstable equilibrium stabilized by steering:

![phase portrait](media/fig_phase_portrait.png)

**Machine learning / adaptation** — the advisor's tire model self-calibrates online. Recursive
least squares recovers the true cornering stiffness from sub-limit driving, and a small learned
residual captures tire-curve shape the physics model misses (lateral-force RMSE **−56%** vs a
Pacejka reference). Crucially, the controller is *robust* to model error (recovery unchanged
under ±40% stiffness error), so learning is a **refinement, never a stability crutch**:

![learning](media/fig_learning.png)

---

## Run it

```bash
pip install -e ".[interactive]"      # or: pip install -r requirements.txt

drift-drive                          # 🎮 drive it yourself (keyboard)
drift-demo                           # scenario comparison + animated HUD
drift-figures                        # regenerate the phase portrait + basin map
drift-learn                          # regenerate the learning experiment
pytest                               # run the test suite (27 tests)
```

Controls: **←/→** steer · **↑** throttle · **↓** brake · **SPACE** autopilot · **R** reset.

---

## How it works

```mermaid
flowchart LR
    S["signals<br/>V, β, r, δ, F_xr, μ"] --> I["driver intent<br/>(infer the line)"]
    I --> E["drift equilibrium<br/>(V, β) → β*, δ*, F_xr*"]
    E --> C["steering LQR<br/>+ throttle logic"]
    C --> A["advice<br/>steering & pedal targets"]
    E --> M["stability monitor<br/>z_u, time-to-loss"]
    S --> M
    M --> A
    L["online learning<br/>RLS + tire residual"] -. refines .-> E
```

| Module | Role |
|---|---|
| `sim/vehicle_model.py`, `control/tire.py` | Single-track plant + C1 fully-derated **Fiala** tire (friction circle) |
| `control/equilibria.py` | Robust **(V, β)** drift-equilibrium solver + feasibility gate |
| `control/stability.py` | Linearization, eigen-analysis, left/right unstable eigenvectors |
| `control/corrector.py` | **Steering-only LQR** + friction-circle throttle advice |
| `control/stability_monitor.py` | Signed unstable-mode coordinate `z_u`, over/understeer, time-to-loss |
| `intent/trajectory.py` | Infer the driver's intended drift (latched, hysteresis, frozen while diverging) |
| `estimation/rls.py`, `learning/tire_residual.py` | Gated online stiffness RLS + learned tire residual |
| `realtime/loop.py` | 100 Hz orchestrator (equilibrium re-solve decimated to 20 Hz) |
| `hmi/display.py`, `interactive/drive.py` | HUD + the drive-it-yourself sim |

**Why steering-only LQR?** At a saturated rear, the throttle's *linear* yaw authority is ~0
(`B[:,F_xr] ≈ 1/m`, pure speed) — so steering is the only fast lateral actuator, and throttle is
the *spin trigger* handled by the nonlinear friction-circle logic. This emerged from numerically
verifying the model, not from assumption.

### Sign convention (ISO 8855)
`x` forward, `y` left, `r>0` = left turn, `β = atan2(v_y, v_x)`, `δ>0` = steer left. A **left
drift** has `r>0, β<0`, countersteer `δ<0`. Pinned in `config/params.py` and unit-tested — a
wrong sign inverts every cue.

---

## Engineering

Typed (`mypy` clean), linted (`ruff`), **95% test coverage** across 27 tests (tire C0/C1
continuity, sign conventions, equilibrium branch selection, open-loop instability, RLS
convergence, end-to-end scenario), CI on Python 3.10–3.12 (`.github/workflows/ci.yml`),
`pip`-installable with console entry points.

## Limitations & the path to a real car

* Single-track lumps left/right, so it omits the 4-motor torque-vectoring yaw moment (correct
  for analysis/advice, conservative for the real plant).
* The front cornering stiffness is tuned soft so the front keeps steering authority at the drift
  point (pure Fiala has a flat post-peak; real tires don't).
* Python proves the physics/control/learning; a car needs an RTOS port and, in reality, a
  dual-antenna RTK-GNSS+IMU to *measure* sideslip (here it is given). LQR holds one equilibrium;
  transitions and general paths would use NMPC / nonlinear model inversion.

**References:** Hindiyeh & Gerdes 2014; Goh, Goel & Gerdes 2020; Velenis et al.; learned-tire
drift work (Djeumou et al. 2023; Broadbent et al. 2024).
