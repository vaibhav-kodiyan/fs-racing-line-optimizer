import numpy as np
from typing import Tuple, Optional
import random


class VehicleParams:
    """Container for vehicle parameters including traction limits."""

    def __init__(self, wheelbase=1.6, max_steer=np.deg2rad(35), mass=230.0,
                 max_longitudinal_accel=8.0, max_lateral_accel=12.0, 
                 tire_friction=1.2, drag_coeff=0.3, frontal_area=1.2):
        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.mass = mass
        self.max_longitudinal_accel = max_longitudinal_accel
        self.max_lateral_accel = max_lateral_accel
        self.tire_friction = tire_friction
        self.drag_coeff = drag_coeff
        self.frontal_area = frontal_area


class BicycleKinematic:
    """Enhanced kinematic bicycle model with traction limits."""

    def __init__(self, params: VehicleParams):
        self.p = params
        self.noise_seed = None
        self.imu_noise_std = 0.0
        self.latency_steps = 0
        self.command_buffer = []

    def set_noise_params(self, seed: Optional[int] = None, imu_noise_std: float = 0.0):
        """Set deterministic noise parameters."""
        self.noise_seed = seed
        self.imu_noise_std = imu_noise_std
        if seed is not None:
            np.random.seed(seed)

    def set_latency(self, latency_steps: int):
        """Set control latency in simulation steps."""
        self.latency_steps = latency_steps
        self.command_buffer = [(0.0, 0.0)] * latency_steps

    def step(self, state, u, dt):
        X, Y, yaw, v = state
        delta, a_cmd = u
        
        # Apply latency by buffering commands
        if self.latency_steps > 0:
            self.command_buffer.append((delta, a_cmd))
            delta, a_cmd = self.command_buffer.pop(0)
        
        # Apply traction limits
        a = self._apply_traction_limits(v, a_cmd)
        
        # Clamp steering
        delta = np.clip(delta, -self.p.max_steer, self.p.max_steer)
        
        # Kinematic update
        L = self.p.wheelbase
        Xn = X + v * np.cos(yaw) * dt
        Yn = Y + v * np.sin(yaw) * dt
        yawn = yaw + (v / L) * np.tan(delta) * dt
        vn = max(0.0, v + a * dt)
        
        # Add IMU noise if enabled
        if self.imu_noise_std > 0:
            yaw_noise = np.random.normal(0, self.imu_noise_std)
            v_noise = np.random.normal(0, self.imu_noise_std * 0.1)
            yawn += yaw_noise
            vn += v_noise
            vn = max(0.0, vn)
        
        return np.array([Xn, Yn, yawn, vn], dtype=float)
    
    def _apply_traction_limits(self, velocity: float, accel_cmd: float) -> float:
        """Apply longitudinal traction limits based on tire model."""
        # Simple tire model: max accel decreases with speed
        speed_factor = max(0.3, 1.0 - velocity / 30.0)  # Reduce grip at high speed
        max_accel = self.p.max_longitudinal_accel * speed_factor
        
        # Apply drag force
        drag_decel = 0.5 * self.p.drag_coeff * self.p.frontal_area * velocity**2 / self.p.mass
        
        # Limit acceleration
        if accel_cmd > 0:
            limited_accel = min(accel_cmd, max_accel) - drag_decel
        else:
            limited_accel = max(accel_cmd, -max_accel) - drag_decel
            
        return limited_accel


def pure_pursuit_control(state, path_xy, lookahead_base=2.0,
                         lookahead_gain=0.1):
    """Return (steer, target_index) using a simple Pure Pursuit geometry."""
    X, Y, yaw, v = state
    if len(path_xy) < 2:
        return 0.0, 0
    dists = np.linalg.norm(path_xy - np.array([X, Y]), axis=1)
    i_near = int(np.argmin(dists))
    Ld = max(0.5, lookahead_base + lookahead_gain * max(0.0, v))
    i_target = i_near
    accum = 0.0
    while i_target < len(path_xy) - 1 and accum < Ld:
        step = np.linalg.norm(path_xy[i_target + 1] - path_xy[i_target])
        accum += step
        i_target += 1
    target = path_xy[i_target]
    dx = target[0] - X
    dy = target[1] - Y
    alpha = np.arctan2(dy, dx) - yaw
    steer = np.arctan2(2.0 * np.sin(alpha), Ld)
    return steer, i_target


def stanley_control(state, path_xy, k_e=0.3, k_v=10.0, wheelbase=1.6):
    """Stanley controller for path following.
    
    Args:
        state: Vehicle state [x, y, yaw, v]
        path_xy: Path waypoints
        k_e: Cross-track error gain
        k_v: Velocity-dependent gain
        wheelbase: Vehicle wheelbase
    """
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
    
    # Cross-track error
    closest_point = path_xy[i_near]
    front_axle_pos = np.array([X + wheelbase * np.cos(yaw), 
                              Y + wheelbase * np.sin(yaw)])
    crosstrack_error = np.linalg.norm(front_axle_pos - closest_point)
    
    # Determine sign of cross-track error
    path_normal = np.array([-path_vec[1], path_vec[0]])
    path_normal = path_normal / (np.linalg.norm(path_normal) + 1e-8)
    error_vec = front_axle_pos - closest_point
    if np.dot(error_vec, path_normal) < 0:
        crosstrack_error = -crosstrack_error
    
    # Stanley control law
    crosstrack_term = np.arctan2(k_e * crosstrack_error, k_v + v)
    steer = heading_error + crosstrack_term
    
    return steer, i_near


class MPPIController:
    """Model Predictive Path Integral controller."""
    
    def __init__(self, horizon=20, num_samples=1000, lambda_=1.0, 
                 noise_std=0.5, vehicle_params=None):
        self.horizon = horizon
        self.num_samples = num_samples
        self.lambda_ = lambda_
        self.noise_std = noise_std
        self.vehicle_params = vehicle_params or VehicleParams()
        
    def control(self, state, path_xy, dt=0.1):
        """MPPI control computation."""
        if len(path_xy) < 2:
            return 0.0, 0
            
        X, Y, yaw, v = state
        
        # Generate control samples
        control_samples = np.random.normal(0, self.noise_std, 
                                         (self.num_samples, self.horizon))
        
        # Evaluate each sample trajectory
        costs = np.zeros(self.num_samples)
        
        for i in range(self.num_samples):
            cost = self._evaluate_trajectory(state, control_samples[i], 
                                           path_xy, dt)
            costs[i] = cost
        
        # Compute weights using softmax
        weights = np.exp(-costs / self.lambda_)
        weights = weights / (np.sum(weights) + 1e-8)
        
        # Weighted average of control samples
        optimal_control = np.sum(weights[:, np.newaxis] * control_samples, axis=0)
        
        return optimal_control[0], 0  # Return first control action
    
    def _evaluate_trajectory(self, initial_state, controls, path_xy, dt):
        """Evaluate cost of a control trajectory."""
        state = initial_state.copy()
        total_cost = 0.0
        
        for u in controls:
            # Simulate one step
            delta = np.clip(u, -self.vehicle_params.max_steer, 
                           self.vehicle_params.max_steer)
            
            # Simple kinematic model for prediction
            X, Y, yaw, v = state
            L = self.vehicle_params.wheelbase
            
            X_new = X + v * np.cos(yaw) * dt
            Y_new = Y + v * np.sin(yaw) * dt
            yaw_new = yaw + (v / L) * np.tan(delta) * dt
            v_new = v  # Assume constant velocity for simplicity
            
            state = np.array([X_new, Y_new, yaw_new, v_new])
            
            # Compute cost (distance to path)
            dists = np.linalg.norm(path_xy - np.array([X_new, Y_new]), axis=1)
            path_cost = np.min(dists)
            
            # Control effort penalty
            control_cost = 0.1 * u**2
            
            total_cost += path_cost + control_cost
        
        return total_cost


class TelemetryLogger:
    """Logger for vehicle telemetry data."""
    
    def __init__(self):
        self.data = {
            'time': [],
            'x': [],
            'y': [], 
            'yaw': [],
            'velocity': [],
            'steering': [],
            'acceleration': [],
            'cross_track_error': [],
            'heading_error': []
        }
        
    def log(self, time, state, control, path_xy=None):
        """Log telemetry data point."""
        X, Y, yaw, v = state
        steer, accel = control
        
        self.data['time'].append(time)
        self.data['x'].append(X)
        self.data['y'].append(Y)
        self.data['yaw'].append(yaw)
        self.data['velocity'].append(v)
        self.data['steering'].append(steer)
        self.data['acceleration'].append(accel)
        
        # Compute errors if path provided
        if path_xy is not None and len(path_xy) > 0:
            dists = np.linalg.norm(path_xy - np.array([X, Y]), axis=1)
            cross_track_error = np.min(dists)
            
            i_near = int(np.argmin(dists))
            if i_near < len(path_xy) - 1:
                path_vec = path_xy[i_near + 1] - path_xy[i_near]
            else:
                path_vec = path_xy[i_near] - path_xy[max(0, i_near - 1)]
            
            path_heading = np.arctan2(path_vec[1], path_vec[0])
            heading_error = path_heading - yaw
            heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
            
            self.data['cross_track_error'].append(cross_track_error)
            self.data['heading_error'].append(heading_error)
        else:
            self.data['cross_track_error'].append(0.0)
            self.data['heading_error'].append(0.0)
    
    def save_to_file(self, filename):
        """Save telemetry data to CSV file."""
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(self.data.keys())
            
            # Write data rows
            for i in range(len(self.data['time'])):
                row = [self.data[key][i] for key in self.data.keys()]
                writer.writerow(row)
    
    def get_statistics(self):
        """Get basic statistics of the logged data."""
        if len(self.data['time']) == 0:
            return {}
            
        stats = {}
        for key, values in self.data.items():
            if key != 'time' and len(values) > 0:
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'max': np.max(values),
                    'min': np.min(values)
                }
        
        return stats
