#!/usr/bin/env python3
import os, argparse, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from fmsim.utils import load_cones_json, car_triangle
from fmsim.planner import laplacian_smooth, curvature_discrete, pair_cones_to_midline
from fmsim.models import VehicleParams, BicycleKinematic, pure_pursuit_control
def run_once(cones_path):
    left, right = load_cones_json(cones_path)
    mid_raw = pair_cones_to_midline(left, right)
    N = len(mid_raw)
    idxL = np.linspace(0, len(left) - 1, N).astype(int) if len(left) > 0 else np.zeros(N, dtype=int)
    idxR = np.linspace(0, len(right) - 1, N).astype(int) if len(right) > 0 else np.zeros(N, dtype=int)
    left_s = left[idxL] if len(left) > 0 else np.zeros((N, 2))
    right_s = right[idxR] if len(right) > 0 else np.zeros((N, 2))
    path = laplacian_smooth(mid_raw, alpha=0.25, iters=200, corridor=(left_s, right_s))
    veh = BicycleKinematic(VehicleParams(wheelbase=1.6))
    p0, p1 = path[0], path[1]; yaw0 = float(np.arctan2(p1[1]-p0[1], p1[0]-p0[0]))
    state = np.array([p0[0], p0[1], yaw0, 0.0], dtype=float)
    i_track, dt = 0, 0.03; last_frame = None
    while i_track < len(path) - 2:
        steer, i_track = pure_pursuit_control(state, path, lookahead_base=2.5, lookahead_gain=0.25)
        v = state[3]; a = (10.0 - v) * 0.8
        state = veh.step(state, (steer, a), dt)
        last_frame = {"L": left, "R": right, "C": path,
                      "car": car_triangle(state[0], state[1], state[2]),
                      "curv_mean": float(np.mean(curvature_discrete(path))),
                      "speed": float(state[3])}
    return last_frame
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cones", type=str, default="data/sample_cones.json")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    frame = run_once(args.cones)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frame["L"][:, 0], frame["L"][:, 1], "^", label="left")
    ax.plot(frame["R"][:, 0], frame["R"][:, 1], "o", label="right")
    ax.plot(frame["C"][:, 0], frame["C"][:, 1], "-", label="center")
    ax.plot(frame["car"][:, 0], frame["car"][:, 1], "-", label="car")
    ax.set_aspect("equal"); ax.legend(loc="best"); ax.set_title("Track snapshot")
    out_png = os.path.join(args.out, "track.png"); fig.tight_layout(); fig.savefig(out_png, dpi=150)
    print("Wrote", out_png)
if __name__ == "__main__": main()
