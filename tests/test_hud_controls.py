from config.params import DEFAULT_VEHICLE
from interactive.drive import pedal_signal, steering_fraction


def test_steering_fraction_is_always_available():
    assert steering_fraction(0.0, 1.0) == 0.0
    assert steering_fraction(0.5, 1.0) == 0.5
    assert steering_fraction(-0.5, 1.0) == -0.5
    assert steering_fraction(2.0, 1.0) == 1.0
    assert steering_fraction(-2.0, 1.0) == -1.0


def test_pedal_signal_maps_force_without_waiting_for_loss_state():
    p = DEFAULT_VEHICLE
    assert pedal_signal(0.0, p) == 0.0
    assert pedal_signal(p.Fx_motor_max * 0.5, p) == 0.5
    assert pedal_signal(-p.Fx_brake_max * 0.25, p) == -0.25
    assert pedal_signal(p.Fx_motor_max * 2.0, p) == 1.0
    assert pedal_signal(-p.Fx_brake_max * 2.0, p) == -1.0
