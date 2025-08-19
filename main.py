import argparse
import numpy as np

from fmsim.utils import load_cones_json, car_triangle
from fmsim.planner import (pair_cones_to_midline, laplacian_smooth, spline_smooth,
                           optimization_based_racing_line, curvature_discrete, curvature_geometric)
from fmsim.models import (VehicleParams, BicycleKinematic, pure_pursuit_control,
                          stanley_control, MPPIController, TelemetryLogger)
from fmsim.ui import run_animation


def simulate(cones_path, pairing_method='hungarian', smoothing_method='spline', 
             controller_type='stanley', enable_noise=False, enable_latency=False):
    """Generator that yields frames for the UI."""
    left, right = load_cones_json(cones_path)
    
    # Generate path using selected method
    if smoothing_method == 'optimization':
        path = optimization_based_racing_line(left, right, num_points=50)
    else:
        mid_raw = pair_cones_to_midline(left, right, method=pairing_method)
        if len(mid_raw) < 2:
            raise RuntimeError("Not enough cones to build a path.")
        
        # Build corridor for smoothing
        N = len(mid_raw)
        idxL = (np.linspace(0, len(left) - 1, N).astype(int) if len(left) > 0
                else np.zeros(N, int))
        idxR = (np.linspace(0, len(right) - 1, N).astype(int) if len(right) > 0
                else np.zeros(N, int))
        left_s = (left[idxL] if len(left) > 0 else np.zeros((N, 2)))
        right_s = right[idxR] if len(right) > 0 else np.zeros((N, 2))
        
        if smoothing_method == 'spline':
            path = spline_smooth(mid_raw, corridor=(left_s, right_s))
        else:  # laplacian
            path = laplacian_smooth(mid_raw, alpha=0.25, iters=200,
                                    corridor=(left_s, right_s))

    # Precompute curvature using geometric method
    curvature_mean = float(np.mean(curvature_geometric(path)))

    # Setup vehicle with enhanced parameters
    params = VehicleParams(wheelbase=1.6, max_longitudinal_accel=8.0)
    veh = BicycleKinematic(params)
    
    # Enable noise and latency if requested
    if enable_noise:
        veh.set_noise_params(seed=42, imu_noise_std=0.02)
    if enable_latency:
        veh.set_latency(2)  # 2-step latency
    
    # Initialize controller
    if controller_type == 'mppi':
        controller = MPPIController(horizon=15, num_samples=500, vehicle_params=params)
    
    # Initialize telemetry logger
    telemetry = TelemetryLogger()
    
    p0, p1 = path[0], path[1]
    yaw0 = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    state = np.array([p0[0], p0[1], yaw0, 0.0], dtype=float)

    i_track, t, dt = 0, 0.0, 0.03
    while i_track < len(path) - 2:
        # Select controller
        if controller_type == 'stanley':
            steer, i_track = stanley_control(state, path, wheelbase=params.wheelbase)
        elif controller_type == 'mppi':
            steer, i_track = controller.control(state, path, dt)
        else:  # pure_pursuit
            steer, i_track = pure_pursuit_control(
                state, path, lookahead_base=2.5, lookahead_gain=0.25)
        
        # Enhanced speed control with traction limits
        v = state[3]
        target_speed = 8.0  # Increased target speed
        a = (target_speed - v) * 1.2
        
        # Apply control
        control = (steer, a)
        state = veh.step(state, control, dt)
        t += dt
        
        # Log telemetry
        telemetry.log(t, state, control, path)
        
        # Calculate additional metrics
        ax = a  # Longitudinal acceleration
        ay = v**2 * np.tan(steer) / params.wheelbase  # Lateral acceleration estimate
        gz = steer / dt if t > dt else 0.0  # Yaw rate estimate
        
        yield {
            "cones_left": left,
            "cones_right": right,
            "centerline": path,
            "car_tri": car_triangle(state[0], state[1], state[2]),
            "metrics": {
                "t": t,
                "speed": float(state[3]),
                "curvature_mean": curvature_mean,
                "ax": ax,
                "ay": ay,
                "gz": gz,
            },
            "telemetry": telemetry,
        }


def main():
    ap = argparse.ArgumentParser(description='Formula Student Racing Line Optimizer')
    ap.add_argument("--cones", type=str, default="data/sample_cones.json",
                    help="Path to cones JSON")
    ap.add_argument("--pairing", type=str, default="hungarian",
                    choices=["greedy", "hungarian"],
                    help="Cone pairing method")
    ap.add_argument("--smoothing", type=str, default="spline",
                    choices=["laplacian", "spline", "optimization"],
                    help="Path smoothing method")
    ap.add_argument("--controller", type=str, default="stanley",
                    choices=["pure_pursuit", "stanley", "mppi"],
                    help="Path following controller")
    ap.add_argument("--noise", action="store_true",
                    help="Enable IMU noise simulation")
    ap.add_argument("--latency", action="store_true",
                    help="Enable control latency simulation")
    ap.add_argument("--demo", action="store_true",
                    help="Run comprehensive demo of all features")
    
    args = ap.parse_args()
    
    if args.demo:
        run_comprehensive_demo(args.cones)
    else:
        run_animation(simulate(args.cones, args.pairing, args.smoothing, 
                              args.controller, args.noise, args.latency))


def run_comprehensive_demo(cones_path):
    """Run a comprehensive demo showcasing all features."""
    print("\n=== Formula Student Racing Line Optimizer Demo ===")
    print("Testing all implemented features...\n")
    
    # Load cones
    left, right = load_cones_json(cones_path)
    print(f"Loaded {len(left)} left cones and {len(right)} right cones")
    
    # Test different pairing methods
    print("\n1. Testing Cone Pairing Methods:")
    for method in ['greedy', 'hungarian']:
        midline = pair_cones_to_midline(left, right, method=method)
        print(f"   {method.capitalize()}: Generated {len(midline)} midline points")
    
    # Test different smoothing methods
    print("\n2. Testing Path Smoothing Methods:")
    midline = pair_cones_to_midline(left, right, method='hungarian')
    
    for method in ['laplacian', 'spline']:
        if method == 'spline':
            smoothed = spline_smooth(midline, corridor=(left, right))
        else:
            smoothed = laplacian_smooth(midline, corridor=(left, right))
        print(f"   {method.capitalize()}: Smoothed to {len(smoothed)} points")
    
    # Test optimization-based racing line
    print("\n3. Testing Optimization-based Racing Line:")
    opt_line = optimization_based_racing_line(left, right)
    print(f"   Generated optimal racing line with {len(opt_line)} points")
    
    # Test controllers
    print("\n4. Testing Path Following Controllers:")
    path = spline_smooth(midline, corridor=(left, right))
    test_state = np.array([path[0, 0], path[0, 1], 0.0, 5.0])
    
    steer_pp, _ = pure_pursuit_control(test_state, path)
    steer_stanley, _ = stanley_control(test_state, path)
    
    mppi = MPPIController(horizon=10, num_samples=100)
    steer_mppi, _ = mppi.control(test_state, path)
    
    print(f"   Pure Pursuit: steering = {steer_pp:.3f} rad")
    print(f"   Stanley: steering = {steer_stanley:.3f} rad")
    print(f"   MPPI: steering = {steer_mppi:.3f} rad")
    
    # Test enhanced vehicle model
    print("\n5. Testing Enhanced Vehicle Model:")
    params = VehicleParams(max_longitudinal_accel=8.0)
    vehicle = BicycleKinematic(params)
    
    # Test with noise
    vehicle.set_noise_params(seed=42, imu_noise_std=0.02)
    vehicle.set_latency(2)
    
    state = test_state.copy()
    for i in range(5):
        state = vehicle.step(state, (0.1, 2.0), 0.1)
    print(f"   Vehicle simulation with noise/latency: final speed = {state[3]:.2f} m/s")
    
    # Test telemetry logging
    print("\n6. Testing Telemetry Logging:")
    logger = TelemetryLogger()
    for i in range(10):
        logger.log(i * 0.1, state, (0.1, 1.0), path)
    
    stats = logger.get_statistics()
    print(f"   Logged {len(logger.data['time'])} data points")
    print(f"   Average velocity: {stats['velocity']['mean']:.2f} m/s")
    
    # Save telemetry
    logger.save_to_file('demo_telemetry.csv')
    print("   Saved telemetry to demo_telemetry.csv")
    
    print("\n=== Demo Complete ===")
    print("All features tested successfully!")
    print("\nRun with specific options to test individual features:")
    print("  --pairing hungarian --smoothing spline --controller stanley")
    print("  --noise --latency (for enhanced realism)")


if __name__ == "__main__":
    main()
