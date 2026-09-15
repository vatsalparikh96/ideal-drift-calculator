"""Driver-intent latching: entry hysteresis, freeze while diverging, slow re-adapt,
and recovery after a rapid grip-change / instability burst."""
import math

from intent.trajectory import DriverIntent


def _spin(intent, V, beta, stable, dt, n):
    t = None
    for _ in range(n):
        t = intent.update(V, beta, stable, dt)
    return t


def test_latches_target_on_entry():
    intent = DriverIntent()
    dt = 0.01
    t = _spin(intent, V=15.0, beta=math.radians(3.0), stable=True, dt=dt, n=50)
    assert not t.in_drift          # below beta_enter: still grip driving, no target

    t = _spin(intent, V=15.0, beta=math.radians(20.0), stable=True, dt=dt, n=300)
    cfg = intent.cfg
    assert t.in_drift
    assert cfg.beta_min <= abs(t.beta_target) <= cfg.beta_max


def test_adapts_target_while_stable():
    intent = DriverIntent()
    dt = 0.01
    # start below the entry threshold so the low-pass filter has a transient to
    # chase once beta steps up (an instant first call would latch at full value).
    _spin(intent, V=15.0, beta=math.radians(3.0), stable=True, dt=dt, n=50)
    t_mid = _spin(intent, V=15.0, beta=math.radians(20.0), stable=True, dt=dt, n=300)
    t_far = _spin(intent, V=15.0, beta=math.radians(20.0), stable=True, dt=dt, n=500)
    # slow adaptation should keep closing the gap to the driver's held line
    assert abs(math.degrees(t_far.beta_target) - 20.0) < abs(math.degrees(t_mid.beta_target) - 20.0)
    assert math.degrees(t_far.beta_target) > 19.0


def test_freezes_target_while_diverging():
    intent = DriverIntent()
    dt = 0.01
    _spin(intent, V=15.0, beta=math.radians(20.0), stable=True, dt=dt, n=700)
    held_beta, held_V = intent.beta_target, intent.V_target

    # instantaneous signals swing wildly while UNSTABLE -> target must not chase them
    t = _spin(intent, V=8.0, beta=math.radians(45.0), stable=False, dt=dt, n=30)
    assert t.beta_target == held_beta
    assert t.V_target == held_V


def test_recovers_original_line_after_rapid_grip_change():
    """A brief instability burst (e.g. a rapid mu drop) must not permanently corrupt
    the latched target: it holds exactly during the burst, and once the driver's
    original line is stably re-established, the target converges back to it rather
    than getting stuck on the transient."""
    intent = DriverIntent()
    dt = 0.01
    _spin(intent, V=15.0, beta=math.radians(20.0), stable=True, dt=dt, n=700)
    held_beta, held_V = intent.beta_target, intent.V_target

    _spin(intent, V=8.0, beta=math.radians(45.0), stable=False, dt=dt, n=30)
    # frozen through the burst
    assert intent.beta_target == held_beta and intent.V_target == held_V

    t = _spin(intent, V=15.0, beta=math.radians(20.0), stable=True, dt=dt, n=400)
    assert abs(math.degrees(t.beta_target) - math.degrees(held_beta)) < 1.0
    assert abs(t.V_target - held_V) < 0.5
