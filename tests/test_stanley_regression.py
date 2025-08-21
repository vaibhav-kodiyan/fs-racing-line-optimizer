#!/usr/bin/env python3
"""Regression tests for Stanley controller cross-track error fix."""

import numpy as np
from fmsim.models import stanley_control


def test_stanley_convergence():
    """Test that Stanley controller converges to path instead of diverging."""
    # Create a simple straight path
    path = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]])
    
    # Initialize vehicle slightly off the path (positive Y offset)
    state = np.array([0.0, 0.5, 0.0, 1.0])  # x, y, yaw, v
    
    # Run simulation for a short period
    states = [state.copy()]
    dt = 0.1
    
    for i in range(20):
        steer, _ = stanley_control(state, path, wheelbase=1.6)
        
        # Simple kinematic update
        x, y, yaw, v = state
        new_x = x + v * np.cos(yaw) * dt
        new_y = y + v * np.sin(yaw) * dt
        new_yaw = yaw + (v / 1.6) * np.tan(steer) * dt
        
        state = np.array([new_x, new_y, new_yaw, v])
        states.append(state.copy())
        
        # Check for divergence
        if abs(state[1]) > 1.0:  # If car diverges too much
            return False, states
    
    # Check if car converged toward the path
    final_y = state[1]
    converged = abs(final_y) < 0.3  # Should be closer to path than initial offset
    
    return converged, states


def test_stanley_sign_consistency():
    """Test that steering direction is consistent with position error."""
    path = np.array([[0, 0], [1, 0], [2, 0]])
    
    # Test above path
    state_above = np.array([1.0, 0.5, 0.0, 1.0])
    steer_above, _ = stanley_control(state_above, path, wheelbase=1.6)
    
    # Test below path
    state_below = np.array([1.0, -0.5, 0.0, 1.0])
    steer_below, _ = stanley_control(state_below, path, wheelbase=1.6)
    
    # Steering should be in opposite directions
    assert steer_above * steer_below < 0, "Steering directions should be opposite"
    
    # Above path should steer left (negative), below should steer right (positive)
    assert steer_above < 0, "Above path should steer left"
    assert steer_below > 0, "Below path should steer right"


def test_stanley_zero_error():
    """Test that Stanley controller gives zero steering when on path."""
    path = np.array([[0, 0], [1, 0], [2, 0]])
    
    # Test exactly on path
    state_on_path = np.array([1.0, 0.0, 0.0, 1.0])
    steer, _ = stanley_control(state_on_path, path, wheelbase=1.6)
    
    # Should be close to zero steering
    assert abs(steer) < 0.01, f"Steering should be near zero when on path, got {steer}"


if __name__ == "__main__":
    print("Running Stanley controller regression tests...")
    
    # Test sign consistency
    test_stanley_sign_consistency()
    print("✓ Sign consistency test passed")
    
    # Test zero error case
    test_stanley_zero_error()
    print("✓ Zero error test passed")
    
    # Test convergence
    converged, states = test_stanley_convergence()
    if converged:
        print("✓ Convergence test passed - car converges to path")
    else:
        print("✗ Convergence test failed - car diverges from path")
        
    print("All regression tests completed!")
