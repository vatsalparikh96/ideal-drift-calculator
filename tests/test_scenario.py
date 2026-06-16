"""End-to-end: the too-much-throttle scenario and the advice it produces."""
import math

from scenarios.too_much_throttle import simulate


def test_ignore_spins():
    h = simulate("ignore")
    assert h["spun_final"] is True
    assert max(abs(b) for b in h["beta"]) > 60


def test_assist_holds_drift():
    h = simulate("assist")
    assert h["spun_final"] is False
    assert abs(h["beta"][-1] - math.degrees(h["eq0"].beta)) < 5   # near target beta*


def test_rescue_recovers():
    h = simulate("rescue")
    assert h["spun_final"] is False
    assert abs(h["beta"][-1] - math.degrees(h["eq0"].beta)) < 5


def test_advice_at_throttle_stab_is_lift_and_countersteer():
    h = simulate("ignore")
    # find the first sample after the throttle stab (t_excess = 1.0s)
    i = next(k for k, t in enumerate(h["t"]) if t >= 1.05)
    # countersteer: target steering more negative than current (more to the right)
    assert h["delta_target"][i] < h["delta"][i] + 1e-6
    # lift: target throttle below current
    assert h["gas_target"][i] < h["gas"][i]
    assert "RIGHT" in h["steer_text"][i]
    assert "LIFT" in h["pedal_text"][i] or "BRAKE" in h["pedal_text"][i]


def test_runs_with_sensor_noise():
    h = simulate("assist", noise=True)
    assert h["spun_final"] is False
