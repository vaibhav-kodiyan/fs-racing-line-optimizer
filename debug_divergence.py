#!/usr/bin/env python3
"""Debug script to analyze the car divergence issue."""

import numpy as np
import matplotlib.pyplot as plt
from fmsim.models import stanley_control, BicycleKinematic, VehicleParams

def test_stanley_controller():
    """Test Stanley controller behavior with a simple straight path."""
    # Create a simple straight path
    path = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]])
    
    # Initialize vehicle slightly off the path
    state = np.array([0.0, 0.5, 0.0, 1.0])  # x, y, yaw, v - offset by 0.5 in y
    
    params = VehicleParams(wheelbase=1.6)
    vehicle = BicycleKinematic(params)
    
    states = [state.copy()]
    steers = []
    
    dt = 0.1
    for i in range(50):
        steer, _ = stanley_control(state, path, wheelbase=params.wheelbase)
        steers.append(steer)
        
        # Simple acceleration to maintain speed
        accel = (2.0 - state[3]) * 0.5
        control = (steer, accel)
        
        state = vehicle.step(state, control, dt)
        states.append(state.copy())
        
        print(f"Step {i}: pos=({state[0]:.3f}, {state[1]:.3f}), yaw={state[2]:.3f}, steer={steer:.3f}")
        
        if i > 10 and abs(state[1]) > 2.0:  # If car diverges too much
            print("Car diverged from path!")
            break
    
    states = np.array(states)
    
    # Plot the trajectory
    plt.figure(figsize=(10, 6))
    plt.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Path')
    plt.plot(states[:, 0], states[:, 1], 'r-', linewidth=2, label='Vehicle trajectory')
    plt.scatter(states[0, 0], states[0, 1], color='green', s=100, label='Start')
    plt.scatter(states[-1, 0], states[-1, 1], color='red', s=100, label='End')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.title('Stanley Controller Test - Car Divergence')
    plt.axis('equal')
    plt.savefig('debug_trajectory.png')
    print("Trajectory plot saved to debug_trajectory.png")
    
    return states, steers

if __name__ == "__main__":
    test_stanley_controller()
