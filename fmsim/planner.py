import numpy as np
try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - optional dependency
    linear_sum_assignment = None
try:
    from scipy.interpolate import UnivariateSpline
except Exception:  # pragma: no cover - optional dependency
    UnivariateSpline = None
try:
    from scipy.spatial.distance import cdist
except Exception:  # pragma: no cover - optional dependency
    cdist = None
try:
    import cvxpy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None


def pair_cones_to_midline(cones_left, cones_right, method='greedy'):
    """Pair left/right cones to produce a raw midline.

    Args:
        cones_left: Array of left cone positions
        cones_right: Array of right cone positions
        method: 'greedy' for nearest-neighbor or 'hungarian' for optimal assignment
    """
    L = np.array(cones_left, dtype=float)
    R = np.array(cones_right, dtype=float)
    if len(L) == 0 or len(R) == 0:
        all_pts = (np.vstack([L, R]) if len(L) + len(R) > 0
                   else np.zeros((0, 2)))
        return order_polyline(all_pts)

    if method == 'hungarian':
        return _hungarian_pair_cones(L, R)
    else:
        return _greedy_pair_cones(L, R)


def _greedy_pair_cones(L, R):
    """Original greedy nearest-neighbor pairing."""
    used_R = np.zeros(len(R), dtype=bool)
    mids = []
    for cone_left in L:
        d = np.linalg.norm(R - cone_left, axis=1)
        idx = int(np.argmin(d + used_R * 1e6))
        used_R[idx] = True
        mids.append(0.5 * (cone_left + R[idx]))
    mids = np.array(mids)
    return order_polyline(mids)


def _hungarian_pair_cones(L, R):
    """Hungarian algorithm for optimal cone pairing."""
    # If SciPy is unavailable, fallback to greedy pairing
    if linear_sum_assignment is None:
        return _greedy_pair_cones(L, R)

    # Compute distance matrix (with fallback if cdist unavailable)
    if cdist is not None:
        dist_matrix = cdist(L, R)
    else:
        # Manual distance matrix
        diff = L[:, None, :] - R[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff * diff, axis=2))

    # Handle unequal lengths by padding with high costs
    n_left, n_right = len(L), len(R)
    if n_left != n_right:
        max_dist = np.max(dist_matrix) * 2
        if n_left < n_right:
            # More right cones - pad left
            padding = np.full((n_right - n_left, n_right), max_dist)
            dist_matrix = np.vstack([dist_matrix, padding])
            L = np.vstack([L, np.zeros((n_right - n_left, 2))])
        else:
            # More left cones - pad right
            padding = np.full((n_left, n_left - n_right), max_dist)
            dist_matrix = np.hstack([dist_matrix, padding])
            R = np.vstack([R, np.zeros((n_left - n_right, 2))])

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    # Extract valid pairs (exclude padded entries)
    mids = []
    for i, j in zip(row_ind, col_ind):
        if i < n_left and j < n_right:
            mids.append(0.5 * (L[i] + R[j]))

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
    """Spline-based smoothing with corridor constraints.

    Args:
        path_xy: Input path points
        corridor: Tuple of (left_cones, right_cones) for constraints
        smoothing_factor: Spline smoothing parameter (0=interpolation, higher=smoother)
        num_points: Number of output points (default: same as input)
    """
    if len(path_xy) < 3:
        return path_xy.copy()
    # Parameterize by cumulative distance
    diffs = np.diff(path_xy, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    t = np.concatenate([[0], np.cumsum(distances)])

    # If SciPy spline is unavailable, use linear interpolation fallback
    if UnivariateSpline is None:
        if num_points is None:
            t_new = t
        else:
            t_new = np.linspace(t[0], t[-1], int(num_points))
        smooth_path = np.column_stack([
            np.interp(t_new, t, path_xy[:, 0]),
            np.interp(t_new, t, path_xy[:, 1]),
        ])
        if corridor is not None:
            smooth_path = _apply_corridor_constraints(smooth_path, corridor)
        return smooth_path

    if num_points is None:
        num_points = len(path_xy)

    # Create splines for x and y coordinates
    try:
        spline_x = UnivariateSpline(t, path_xy[:, 0], s=smoothing_factor * len(t))
        spline_y = UnivariateSpline(t, path_xy[:, 1], s=smoothing_factor * len(t))

        # Generate smooth path
        t_new = np.linspace(t[0], t[-1], num_points)
        smooth_path = np.column_stack([spline_x(t_new), spline_y(t_new)])

        # Apply corridor constraints if provided
        if corridor is not None:
            smooth_path = _apply_corridor_constraints(smooth_path, corridor)

        return smooth_path

    except Exception:
        # Fallback to Laplacian smoothing if spline fails
        return laplacian_smooth(path_xy, corridor=corridor)


def _apply_corridor_constraints(path, corridor):
    """Apply corridor constraints to keep path within track bounds."""
    left_cones, right_cones = corridor
    constrained_path = path.copy()

    for i, point in enumerate(path):
        # Find nearest corridor points
        if len(left_cones) > 0 and len(right_cones) > 0:
            left_dists = np.linalg.norm(left_cones - point, axis=1)
            right_dists = np.linalg.norm(right_cones - point, axis=1)

            nearest_left = left_cones[np.argmin(left_dists)]
            nearest_right = right_cones[np.argmin(right_dists)]

            # Project point onto corridor if outside bounds
            to_left = np.dot(point - nearest_right, nearest_left - nearest_right)
            to_right = np.dot(point - nearest_left, nearest_right - nearest_left)

            if to_left < 0:  # Outside left boundary
                constrained_path[i] = nearest_left + 0.1 * (nearest_right - nearest_left)
            elif to_right < 0:  # Outside right boundary
                constrained_path[i] = nearest_right + 0.1 * (nearest_left - nearest_right)

    return constrained_path


def optimization_based_racing_line(
        left_cones, right_cones, num_points=50,
        curvature_weight=1.0, smoothness_weight=0.1):
    """Generate optimal racing line using convex optimization.

    Args:
        left_cones: Left boundary cone positions
        right_cones: Right boundary cone positions
        num_points: Number of points in the racing line
        curvature_weight: Weight for minimizing curvature
        smoothness_weight: Weight for path smoothness
    """
    if len(left_cones) < 2 or len(right_cones) < 2:
        return pair_cones_to_midline(left_cones, right_cones)

    try:
        # Create corridor centerline as initial guess
        midline = pair_cones_to_midline(left_cones, right_cones, method='hungarian')

        # Parameterize corridor boundaries
        t_param = np.linspace(0, 1, num_points)
        left_interp = _interpolate_boundary(left_cones, t_param)
        right_interp = _interpolate_boundary(right_cones, t_param)

        # Optimization variables - path points
        x = cp.Variable(num_points)
        y = cp.Variable(num_points)

        # Objective: minimize curvature and maximize smoothness
        curvature_cost = 0
        smoothness_cost = 0

        for i in range(1, num_points - 1):
            # Approximate curvature using second differences
            d2x = x[i+1] - 2*x[i] + x[i-1]
            d2y = y[i+1] - 2*y[i] + y[i-1]
            curvature_cost += cp.square(d2x) + cp.square(d2y)

            # Smoothness term
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            smoothness_cost += cp.square(dx) + cp.square(dy)

        objective = cp.Minimize(
            curvature_weight * curvature_cost
            + smoothness_weight * smoothness_cost
        )

        # Constraints: stay within corridor
        constraints = []
        for i in range(num_points):
            # Linear interpolation between left and right boundaries
            # Point must be convex combination of left and right
            alpha = cp.Variable()
            constraints.extend([
                x[i] == alpha * left_interp[i, 0] + (1 - alpha) * right_interp[i, 0],
                y[i] == alpha * left_interp[i, 1] + (1 - alpha) * right_interp[i, 1],
                alpha >= 0.1,  # Stay away from boundaries
                alpha <= 0.9
            ])

        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, verbose=False)

        if problem.status == cp.OPTIMAL:
            return np.column_stack([x.value, y.value])
        else:
            # Fallback to midline if optimization fails
            return midline

    except Exception:
        # Fallback to standard pairing if optimization fails
        return pair_cones_to_midline(left_cones, right_cones)


def _interpolate_boundary(cones, t_param):
    """Interpolate boundary cones to parameter values."""
    if len(cones) < 2:
        return np.tile(cones[0] if len(cones) > 0 else [0, 0], (len(t_param), 1))

    # Order cones by x-coordinate for interpolation
    ordered_cones = cones[np.argsort(cones[:, 0])]

    # Create parameter values for cones based on cumulative distance
    diffs = np.diff(ordered_cones, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    t_cones = np.concatenate([[0], np.cumsum(distances)])
    t_cones = t_cones / t_cones[-1]  # Normalize to [0, 1]

    # Interpolate x and y coordinates
    x_interp = np.interp(t_param, t_cones, ordered_cones[:, 0])
    y_interp = np.interp(t_param, t_cones, ordered_cones[:, 1])

    return np.column_stack([x_interp, y_interp])


def curvature_discrete(path_xy):
    """Discrete triangle-based curvature approximation (>=0)."""
    if len(path_xy) < 3:
        return np.zeros(len(path_xy))
    v10 = path_xy[1:-1] - path_xy[:-2]
    v20 = path_xy[2:] - path_xy[:-2]
    v21 = path_xy[2:] - path_xy[1:-1]
    a = np.linalg.norm(v21, axis=1)
    b = np.linalg.norm(v20, axis=1)
    c = np.linalg.norm(v10, axis=1)
    cross = v10[:, 0] * v20[:, 1] - v10[:, 1] * v20[:, 0]
    area = 0.5 * np.abs(cross)
    denom = (a * b * c) + 1e-12
    k_mid = 4.0 * area / denom
    kappa = np.zeros(len(path_xy))
    kappa[1:-1] = k_mid
    kappa[0] = kappa[1]
    kappa[-1] = kappa[-2]
    return kappa


def curvature_geometric(path_xy):
    """Compute geometric curvature using arc-length parameterization."""
    if len(path_xy) < 3:
        return np.zeros(len(path_xy))

    diffs = np.diff(path_xy, axis=0)
    ds = np.sqrt(np.sum(diffs**2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(ds)])

    x = path_xy[:, 0]
    y = path_xy[:, 1]

    dxds = np.gradient(x, s)
    dyds = np.gradient(y, s)
    d2xds2 = np.gradient(dxds, s)
    d2yds2 = np.gradient(dyds, s)

    # Geometric curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    numerator = np.abs(dxds * d2yds2 - dyds * d2xds2)
    denominator = np.power(dxds**2 + dyds**2, 1.5)
    denominator = np.maximum(denominator, 1e-8)

    kappa = numerator / denominator
    if len(kappa) >= 3:
        kappa[0] = kappa[1]
        kappa[-1] = kappa[-2]
    return kappa
