#!/usr/bin/env python3
"""Detailed debug of Stanley controller calculations."""

import numpy as np

def debug_stanley_calculation(state, path_xy, k_e=0.3, k_v=10.0, wheelbase=1.6):
    """Debug version of Stanley controller with detailed printouts."""
    X, Y, yaw, v = state
    if len(path_xy) < 2:
        return 0.0, 0

    # Find closest point on path
    dists = np.linalg.norm(path_xy - np.array([X, Y]), axis=1)
    i_near = int(np.argmin(dists))

    # Get path heading at closest point
    if i_near < len(path_xy) - 1:
        path_vec = path_xy[i_near + 1] - path_xy[i_near]
    else:
        path_vec = path_xy[i_near] - path_xy[i_near - 1]

    path_heading = np.arctan2(path_vec[1], path_vec[0])

    # Heading error
    heading_error = path_heading - yaw
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

    # Cross-track error (signed)
    closest_point = path_xy[i_near]
    front_axle_pos = np.array([X + wheelbase * np.cos(yaw),
                              Y + wheelbase * np.sin(yaw)])
    
    # Calculate signed cross-track error using cross product
    error_vec = front_axle_pos - closest_point
    path_vec_norm = path_vec / (np.linalg.norm(path_vec) + 1e-8)
    
    # Cross product gives signed distance
    # For 2D vectors, np.cross returns a scalar: a_x*b_y - a_y*b_x
    crosstrack_error = np.cross(path_vec_norm, error_vec)
    
    print(f"State: x={X:.3f}, y={Y:.3f}, yaw={yaw:.3f}, v={v:.3f}")
    print(f"Closest point: {closest_point}")
    print(f"Front axle: {front_axle_pos}")
    print(f"Error vector: {error_vec}")
    print(f"Path vector: {path_vec}")
    print(f"Path vector normalized: {path_vec_norm}")
    print(f"Cross-track error: {crosstrack_error:.3f}")
    
    # Stanley control law
    crosstrack_term = np.arctan2(k_e * crosstrack_error, k_v + v)
    steer = heading_error + crosstrack_term
    
    print(f"Heading error: {heading_error:.3f}")
    print(f"Crosstrack term: {crosstrack_term:.3f}")
    print(f"Final steer: {steer:.3f}")
    
    return steer, i_near

if __name__ == "__main__":
    # Test with the same scenario as debug_divergence.py
    path = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]])
    state = np.array([0.0, 0.5, 0.0, 1.0])  # x, y, yaw, v - offset by 0.5 in y
    
    print("=== Stanley Controller Debug ===")
    debug_stanley_calculation(state, path)
