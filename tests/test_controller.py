import numpy as np
from fmsim.models import VehicleParams, BicycleKinematic, pure_pursuit_control


def test_pure_pursuit_straight_zero_steer():
    path = np.stack([np.linspace(0.0, 20.0, 50), np.zeros(50)], axis=1)
    state = np.array([0.0, 0.0, 0.0, 5.0], dtype=float)
    steer, idx = pure_pursuit_control(state, path, lookahead_base=2.0, lookahead_gain=0.1)
    assert abs(steer) < 0.2
    assert 0 <= idx < len(path)


def test_bicycle_forward_integration():
    veh = BicycleKinematic(VehicleParams(wheelbase=1.6))
    state = np.array([0.0, 0.0, 0.0, 5.0], dtype=float)
    dt = 0.1
    s1 = veh.step(state, (0.0, 0.0), dt)
    assert s1[0] > state[0]
    assert abs(s1[1]) < 1e-6
