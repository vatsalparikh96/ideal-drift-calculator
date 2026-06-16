"""Tire model: monotone-to-peak, saturation, C0+C1 continuity, friction-circle derate."""
import math

import pytest

from control.tire import fiala_lateral, fiala_lateral_and_dalpha, friction_budget, slide_slip_angle

Ca, mu, Fz = 60_000.0, 0.95, 8000.0


def test_friction_budget_shrinks_with_Fx():
    full = friction_budget(mu, Fz, 0.0)
    half = friction_budget(mu, Fz, 0.6 * mu * Fz)
    assert full == pytest.approx(mu * Fz)
    assert 0.0 < half < full
    # longitudinally saturated -> no lateral budget
    assert friction_budget(mu, Fz, 1.2 * mu * Fz) == 0.0


def test_no_nan_when_over_circle():
    # |Fx| > mu*Fz must not produce NaN; lateral force collapses to 0
    fy = fiala_lateral(math.radians(20), Ca, mu, Fz, 2.0 * mu * Fz)
    assert fy == 0.0


def test_monotone_to_peak_and_saturates():
    asl = slide_slip_angle(Ca, mu, Fz, 0.0)
    # increasing magnitude up to slide -> increasing |Fy|
    prev = 0.0
    for d in range(1, int(math.degrees(asl))):
        fy = abs(fiala_lateral(math.radians(d), Ca, mu, Fz, 0.0))
        assert fy >= prev - 1e-6
        prev = fy
    # beyond slide -> flat at the friction budget eta
    eta = friction_budget(mu, Fz, 0.0)
    fy_sat = fiala_lateral(asl + math.radians(15), Ca, mu, Fz, 0.0)
    assert abs(fy_sat) == pytest.approx(eta, rel=1e-6)


def test_C0_and_C1_continuity_at_slide():
    eta = friction_budget(mu, Fz, 0.0)
    asl = slide_slip_angle(Ca, mu, Fz, 0.0)
    # value at slide equals -eta (for positive alpha)
    fy_at = fiala_lateral(asl, Ca, mu, Fz, 0.0)
    assert fy_at == pytest.approx(-eta, rel=1e-6)
    # left/right limits of value and slope match (C0 + C1)
    eps = 1e-5
    fy_lo, d_lo = fiala_lateral_and_dalpha(asl - eps, Ca, mu, Fz, 0.0)
    fy_hi, d_hi = fiala_lateral_and_dalpha(asl + eps, Ca, mu, Fz, 0.0)
    assert fy_lo == pytest.approx(fy_hi, abs=1e-2)
    assert d_lo == pytest.approx(0.0, abs=1e-2)   # slope -> 0 at slide
    assert d_hi == pytest.approx(0.0, abs=1e-2)


def test_no_overshoot_past_friction_circle():
    # The derated cubic must never exceed the friction budget anywhere.
    eta = friction_budget(mu, Fz, 0.5 * mu * Fz)
    for d in range(0, 60):
        fy = fiala_lateral(math.radians(d), Ca, mu, Fz, 0.5 * mu * Fz)
        assert abs(fy) <= eta + 1e-6


def test_sign_opposes_slip():
    assert fiala_lateral(math.radians(5), Ca, mu, Fz, 0.0) < 0   # +alpha -> -Fy
    assert fiala_lateral(math.radians(-5), Ca, mu, Fz, 0.0) > 0
