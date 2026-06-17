"""Robustness sanity: nominal recovery is high, loop budget is real-time, latency tolerated."""
import math

from experiments.robustness import _controller, _fraction, loop_budget


def test_nominal_recovery_high():
    ctrl, eq = _controller(0.95)
    assert _fraction(ctrl, eq, 0.95, n=9) > 70.0


def test_tolerates_modest_latency():
    ctrl, eq = _controller(0.95)
    frac = _fraction(ctrl, eq, 0.95, n=9, latency=2)   # 20 ms transport delay
    assert frac > 60.0


def test_loop_budget_real_time():
    mean_ms, p95_ms = loop_budget(n=300)
    assert math.isfinite(mean_ms) and mean_ms > 0
    assert p95_ms < 50.0          # generous ceiling (typical ~0.05 ms); just a sanity bound
