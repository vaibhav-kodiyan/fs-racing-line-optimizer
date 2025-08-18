import json
import numpy as np


def load_cones_json(path):
    """Load cones JSON with keys 'left' and 'right'."""
    with open(path, "r") as f:
        d = json.load(f)
    left = d.get("left", [])
    right = d.get("right", [])
    return np.array(left, dtype=float), np.array(right, dtype=float)


def car_triangle(x, y, yaw, scale=0.8):
    """Return a simple triangle representing the car pose."""
    pts = np.array([[1.0, 0.0], [-0.6, 0.4], [-0.6, -0.4], [1.0, 0.0]]) * scale
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return (R @ pts.T).T + np.array([x, y])
