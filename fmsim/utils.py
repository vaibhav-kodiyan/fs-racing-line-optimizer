import json
import numpy as np


def load_cones_json(path):
    """Load cones JSON with keys 'left' and 'right'."""
    with open(path, "r") as f:
        d = json.load(f)
    left = d.get("left", [])
    right = d.get("right", [])
    return np.array(left, dtype=float), np.array(right, dtype=float)
