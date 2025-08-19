import numpy as np


class VehicleParams:
    """Container for simple kinematic bicycle parameters."""

    def __init__(self, wheelbase=1.6, max_steer=np.deg2rad(35), mass=230.0):
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.mass = mass


class BicycleKinematic:
    """Kinematic bicycle model forward integrator."""

    def __init__(self, params: VehicleParams):
        self.p = params

    def step(self, state, u, dt):
        X, Y, yaw, v = state
        delta, a = u
        delta = np.clip(delta, -self.p.max_steer, self.p.max_steer)
        L = self.p.wheelbase
        Xn = X + v * np.cos(yaw) * dt
        Yn = Y + v * np.sin(yaw) * dt
        yawn = yaw + (v / L) * np.tan(delta) * dt
        vn = max(0.0, v + a * dt)
        return np.array([Xn, Yn, yawn, vn], dtype=float)


def pure_pursuit_control(state, path_xy, lookahead_base=2.0,
                         lookahead_gain=0.1):
    """Return (steer, target_index) using a simple Pure Pursuit geometry."""
    X, Y, yaw, v = state
    if len(path_xy) < 2:
        return 0.0, 0
    dists = np.linalg.norm(path_xy - np.array([X, Y]), axis=1)
    i_near = int(np.argmin(dists))
    Ld = max(0.5, lookahead_base + lookahead_gain * max(0.0, v))
    i_target = i_near
    accum = 0.0
    while i_target < len(path_xy) - 1 and accum < Ld:
        step = np.linalg.norm(path_xy[i_target + 1] - path_xy[i_target])
        accum += step
        i_target += 1
    target = path_xy[i_target]
    dx = target[0] - X
    dy = target[1] - Y
    alpha = np.arctan2(dy, dx) - yaw
    steer = np.arctan2(2.0 * np.sin(alpha), Ld)
    return steer, i_target
