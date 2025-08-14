import argparse
import numpy as np

from fmsim.utils import load_cones_json
from fmsim.planner import pair_cones_to_midline, laplacian_smooth, curvature_discrete
from fmsim.models import VehicleParams, BicycleKinematic, pure_pursuit_control
from fmsim.ui import run_animation


def car_triangle(x, y, yaw, scale=0.8):
    """Return a simple triangle representing the car pose."""
    pts = np.array([[1.0, 0.0], [-0.6, 0.4], [-0.6, -0.4], [1.0, 0.0]]) * scale
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return (R @ pts.T).T + np.array([x, y])


def simulate(cones_path):
    """Generator that yields frames for the UI."""
    left, right = load_cones_json(cones_path)
    mid_raw = pair_cones_to_midline(left, right)
    if len(mid_raw) < 2:
        raise RuntimeError("Not enough cones to build a path.")

    # Build a corridor-aligned smoothed path
    N = len(mid_raw)
    idxL = np.linspace(0, len(left) - 1, N).astype(int) if len(left) > 0 else np.zeros(N, int)
    idxR = np.linspace(0, len(right) - 1, N).astype(int) if len(right) > 0 else np.zeros(N, int)
    left_s = left[idxL] if len(left) > 0 else np.zeros((N, 2))
    right_s = right[idxR] if len(right) > 0 else np.zeros((N, 2))
    path = laplacian_smooth(mid_raw, alpha=0.25, iters=200, corridor=(left_s, right_s))

    veh = BicycleKinematic(VehicleParams(wheelbase=1.6))
    p0, p1 = path[0], path[1]
    yaw0 = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    state = np.array([p0[0], p0[1], yaw0, 0.0], dtype=float)

    i_track, t, dt = 0, 0.0, 0.03
    while i_track < len(path) - 2:
        steer, i_track = pure_pursuit_control(state, path, lookahead_base=2.5, lookahead_gain=0.25)
        v = state[3]
        a = (10.0 - v) * 0.8
        state = veh.step(state, (steer, a), dt)
        t += dt
        yield {
            "cones_left": left,
            "cones_right": right,
            "centerline": path,
            "car_tri": car_triangle(state[0], state[1], state[2]),
            "metrics": {
                "t": t,
                "speed": float(state[3]),
                "curvature_mean": float(np.mean(curvature_discrete(path))),
                "ax": 0.0,
                "ay": 0.0,
                "gz": 0.0,
            },
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cones", type=str, default="data/sample_cones.json", help="Path to cones JSON")
    args = ap.parse_args()
    run_animation(simulate(args.cones))


if __name__ == "__main__":
    main()
