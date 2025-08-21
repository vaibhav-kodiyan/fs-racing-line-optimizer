"""Test Stanley controller convergence behavior."""

import numpy as np
import pytest
from fmsim.models import stanley_control, BicycleKinematic, VehicleParams


def test_stanley_controller_convergence():
    """Test that Stanley controller converges to a straight path."""
    # Create a simple straight path
    path = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]])
    
    # Initialize vehicle offset from the path
    initial_state = np.array([0.0, 0.5, 0.0, 2.0])  # x, y, yaw, v
    
    params = VehicleParams(wheelbase=1.6)
    vehicle = BicycleKinematic(params)
    
    state = initial_state.copy()
    dt = 0.1
    
    # Simulate for enough steps to see convergence
    for i in range(30):
        steer, _ = stanley_control(state, path, wheelbase=params.wheelbase)
        
        # Simple speed control
        accel = (2.0 - state[3]) * 0.5
        control = (steer, accel)
        
        state = vehicle.step(state, control, dt)
    
    # After 30 steps, the vehicle should be closer to the path
    # than it was initially (convergence test)
    initial_cross_track_error = abs(initial_state[1])
    final_cross_track_error = abs(state[1])
    
    assert final_cross_track_error < initial_cross_track_error, \
        f"Vehicle diverged: initial error {initial_cross_track_error:.3f}, " \
        f"final error {final_cross_track_error:.3f}"
    
    # Vehicle should be reasonably close to the path
    assert final_cross_track_error < 0.3, \
        f"Vehicle did not converge sufficiently: final error {final_cross_track_error:.3f}"


def test_stanley_controller_signed_error():
    """Test that Stanley controller correctly handles signed cross-track error."""
    # Simple straight path
    path = np.array([[0, 0], [1, 0], [2, 0]])
    
    # Test vehicle on left side of path (positive y)
    state_left = np.array([1.0, 0.5, 0.0, 1.0])
    steer_left, _ = stanley_control(state_left, path, wheelbase=1.6)
    
    # Test vehicle on right side of path (negative y)  
    state_right = np.array([1.0, -0.5, 0.0, 1.0])
    steer_right, _ = stanley_control(state_right, path, wheelbase=1.6)
    
    # Steering should have opposite signs for opposite sides
    assert steer_left * steer_right < 0, \
        f"Steering signs should be opposite: left={steer_left:.3f}, right={steer_right:.3f}"
    
    # For a vehicle on the left side, steering should be negative (turn right)
    assert steer_left < 0, f"Vehicle on left should steer right: {steer_left:.3f}"
    
    # For a vehicle on the right side, steering should be positive (turn left)
    assert steer_right > 0, f"Vehicle on right should steer left: {steer_right:.3f}"


def test_stanley_controller_no_divergence():
    """Test that Stanley controller doesn't cause divergence over extended simulation."""
    # Create a curved path
    t = np.linspace(0, 2*np.pi, 20)
    path = np.column_stack([t, 0.5 * np.sin(t)])
    
    # Start slightly off the path
    state = np.array([0.0, 0.2, 0.0, 1.0])
    
    params = VehicleParams(wheelbase=1.6)
    vehicle = BicycleKinematic(params)
    
    max_cross_track_error = 0.0
    dt = 0.05
    
    # Simulate for many steps
    for i in range(100):
        steer, _ = stanley_control(state, path, wheelbase=params.wheelbase)
        
        # Simple speed control
        accel = (1.5 - state[3]) * 0.5
        control = (steer, accel)
        
        state = vehicle.step(state, control, dt)
        
        # Find closest point on path to calculate cross-track error
        dists = np.linalg.norm(path - state[:2], axis=1)
        cross_track_error = np.min(dists)
        max_cross_track_error = max(max_cross_track_error, cross_track_error)
    
    # Vehicle should not diverge beyond reasonable bounds
    assert max_cross_track_error < 2.0, \
        f"Vehicle diverged too far from path: max error {max_cross_track_error:.3f}"
