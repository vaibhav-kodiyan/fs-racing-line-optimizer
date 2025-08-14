import numpy as np
from fmsim.planner import pair_cones_to_midline, laplacian_smooth, curvature_discrete


def test_pair_midline_straight():
    xs = np.linspace(0.0, 10.0, 6)
    left = np.stack([xs, np.ones_like(xs)], axis=1)
    right = np.stack([xs, -np.ones_like(xs)], axis=1)
    mid = pair_cones_to_midline(left, right)
    assert np.allclose(mid[:, 1], 0.0, atol=1e-6)
    assert np.all(np.diff(mid[:, 0]) >= 0.0)


def test_smooth_and_curvature_proxy():
    xs = np.linspace(0.0, 10.0, 20)
    left = np.stack([xs, np.ones_like(xs)], axis=1)
    right = np.stack([xs, -np.ones_like(xs)], axis=1)
    mid = pair_cones_to_midline(left, right)
    smooth = laplacian_smooth(mid, alpha=0.25, iters=50, corridor=(left, right))
    assert smooth.shape == mid.shape
    curv = curvature_discrete(smooth)
    assert curv.shape[0] == smooth.shape[0]
    assert np.all(curv >= 0.0)
