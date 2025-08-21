#!/usr/bin/env python3
"""Test script to demonstrate the Stanley controller bug."""

import numpy as np
from fmsim.models import stanley_control

def test_stanley_sign_bug():
    """Test that demonstrates the sign error in Stanley controller."""
    # Create a straight horizontal path
    path = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
    
    # Test case 1: Car is above the path (positive Y)
    state_above = np.array([2.0, 1.0, 0.0, 1.0])  # x=2, y=1, yaw=0, v=1
    steer_above, _ = stanley_control(state_above, path, k_e=1.0, wheelbase=1.6)
    
    # Test case 2: Car is below the path (negative Y)
    state_below = np.array([2.0, -1.0, 0.0, 1.0])  # x=2, y=-1, yaw=0, v=1
    steer_below, _ = stanley_control(state_below, path, k_e=1.0, wheelbase=1.6)
    
    print(f"Steering when above path: {steer_above:.3f}")
    print(f"Steering when below path: {steer_below:.3f}")
    
    # The steering should be in opposite directions, but currently they're the same
    # This indicates the sign error in cross-track error calculation
    
    if steer_above * steer_below > 0:
        print("BUG CONFIRMED: Steering commands have same sign - car will diverge!")
        return False
    else:
        print("Steering commands have opposite signs - behavior is correct")
        return True

if __name__ == "__main__":
    test_stanley_sign_bug()
