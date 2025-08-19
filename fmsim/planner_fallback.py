"""Fallback planner implementation without scipy dependencies."""

import numpy as np


def pair_cones_to_midline(cones_left, cones_right, method='greedy'):
    """Pair left/right cones to produce a raw midline.
    
    Args:
        cones_left: Array of left cone positions
        cones_right: Array of right cone positions  
        method: 'greedy' for nearest-neighbor (hungarian fallback to greedy)
    """
    L = np.array(cones_left, dtype=float)
    R = np.array(cones_right, dtype=float)
    if len(L) == 0 or len(R) == 0:
        all_pts = (np.vstack([L, R]) if len(L) + len(R) > 0
                   else np.zeros((0, 2)))
        return order_polyline(all_pts)

    # Use greedy pairing (hungarian fallback)
    return _greedy_pair_cones(L, R)


def _greedy_pair_cones(L, R):
    """Greedy nearest-neighbor pairing."""
    used_R = np.zeros(len(R), dtype=bool)
    mids = []
    for cone_left in L:
        d = np.linalg.norm(R - cone_left, axis=1)
        idx = int(np.argmin(d + used_R * 1e6))
        used_R[idx] = True
        mids.append(0.5 * (cone_left + R[idx]))
    mids = np.array(mids)
    return order_polyline(mids)


def order_polyline(P):
    """Nearest-neighbor polyline ordering, starting from smallest x."""
    if len(P) <= 1:
        return P.copy()
    start = int(np.argmin(P[:, 0]))
    order = [start]
    used = np.zeros(len(P), dtype=bool)
    used[start] = True
    for _ in range(len(P) - 1):
        last = P[order[-1]]
        d = np.linalg.norm(P - last, axis=1)
        d[used] = 1e9
        nxt = int(np.argmin(d))
        order.append(nxt)
        used[nxt] = True
    return P[order]


def laplacian_smooth(path_xy, alpha=0.25, iters=200, corridor=None):
    """Simple Laplacian smoothing with optional soft pull to corridor mid."""
    P = path_xy.copy().astype(float)
    N = len(P)
    if N < 3:
        return P
    for _ in range(iters):
        Pn = P.copy()
        for i in range(1, N - 1):
            target = 0.5 * (P[i - 1] + P[i + 1])
            Pn[i] = P[i] + alpha * (target - P[i])
            if corridor is not None:
                left, right = corridor
                i_corr = min(i, len(left) - 1, len(right) - 1)
                mid = 0.5 * (left[i_corr] + right[i_corr])
                Pn[i] = 0.9 * Pn[i] + 0.1 * mid
        P = Pn
    return P


def spline_smooth(path_xy, corridor=None, smoothing_factor=0.1, num_points=None):
    """Fallback to Laplacian smoothing when scipy not available."""
    print("Warning: Spline smoothing requires scipy. Falling back to Laplacian smoothing.")
    return laplacian_smooth(path_xy, corridor=corridor)


def optimization_based_racing_line(left_cones, right_cones, num_points=50, 
                                  curvature_weight=1.0, smoothness_weight=0.1):
    """Fallback to standard pairing when cvxpy not available."""
    print("Warning: Optimization-based racing line requires cvxpy. Falling back to standard pairing.")
    return pair_cones_to_midline(left_cones, right_cones)


def curvature_discrete(path_xy):
    """Second-difference norm as a curvature proxy (>=0)."""
    if len(path_xy) < 3:
        return np.zeros(len(path_xy))
    d2 = path_xy[2:] - 2 * path_xy[1:-1] + path_xy[:-2]
    kappa = np.zeros(len(path_xy))
    kappa[1:-1] = np.linalg.norm(d2, axis=1)
    return kappa


def curvature_geometric(path_xy):
    """Compute geometric curvature using derivatives."""
    if len(path_xy) < 3:
        return np.zeros(len(path_xy))
    
    # Compute first and second derivatives using finite differences
    dx = np.gradient(path_xy[:, 0])
    dy = np.gradient(path_xy[:, 1])
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    
    # Geometric curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    numerator = np.abs(dx * d2y - dy * d2x)
    denominator = np.power(dx**2 + dy**2, 1.5)
    
    # Avoid division by zero
    denominator = np.maximum(denominator, 1e-8)
    
    return numerator / denominator
