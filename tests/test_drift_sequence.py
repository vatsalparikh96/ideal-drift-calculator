"""Automatic drift initiation & exit: reaches the drift, then returns to grip."""
import math

from scenarios.drift_entry_exit import simulate


def test_full_cycle_phases_occur():
    h = simulate()
    phases = set(h["phase"])
    assert {"GRIP", "ENTER", "DRIFT", "EXIT"} <= phases


def test_drift_is_reached():
    h = simulate()
    target = math.degrees(h["eq"].beta)
    drift_betas = [b for b, ph in zip(h["beta"], h["phase"], strict=False) if ph == "DRIFT"]
    assert drift_betas, "never entered DRIFT"
    # at some point during the drift, sideslip is within a few degrees of the target
    assert min(abs(b - target) for b in drift_betas) < 5.0


def test_returns_to_grip_upright():
    h = simulate()
    assert h["phase"][-1] == "GRIP"
    assert abs(h["beta"][-1]) < 5.0          # straightened out
    assert abs(h["r"][-1]) < 0.2
    assert min(h["V"]) > 1.0                 # never spun/stalled
