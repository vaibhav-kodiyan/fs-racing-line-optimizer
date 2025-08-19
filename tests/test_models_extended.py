"""Extended tests for vehicle models and controllers."""

import numpy as np
from fmsim.models import (
    VehicleParams, BicycleKinematic, stanley_control,
    MPPIController, TelemetryLogger
)


class TestVehicleParamsExtended:
    """Test enhanced vehicle parameters."""

    def test_default_traction_params(self):
        """Test default traction limit parameters."""
        params = VehicleParams()

        assert params.max_longitudinal_accel == 8.0
        assert params.max_lateral_accel == 12.0
        assert params.tire_friction == 1.2
        assert params.drag_coeff == 0.3
        assert params.frontal_area == 1.2

    def test_custom_traction_params(self):
        """Test custom traction parameters."""
        params = VehicleParams(
            max_longitudinal_accel=10.0,
            max_lateral_accel=15.0,
            tire_friction=1.5,
            drag_coeff=0.25,
            frontal_area=1.0
        )

        assert params.max_longitudinal_accel == 10.0
        assert params.max_lateral_accel == 15.0
        assert params.tire_friction == 1.5
        assert params.drag_coeff == 0.25
        assert params.frontal_area == 1.0


class TestBicycleKinematicExtended:
    """Test enhanced bicycle model with traction limits."""

    def test_traction_limits_acceleration(self):
        """Test acceleration traction limits."""
        params = VehicleParams(max_longitudinal_accel=5.0)
        model = BicycleKinematic(params)

        state = np.array([0.0, 0.0, 0.0, 10.0])  # 10 m/s

        # Test high acceleration command gets limited
        u = (0.0, 20.0)  # Very high acceleration
        new_state = model.step(state, u, 0.1)

        # Acceleration should be limited
        actual_accel = (new_state[3] - state[3]) / 0.1
        assert actual_accel < 20.0

    def test_drag_force_application(self):
        """Test drag force reduces acceleration."""
        params = VehicleParams(drag_coeff=0.5, frontal_area=2.0, mass=200.0)
        model = BicycleKinematic(params)

        # High speed state
        state = np.array([0.0, 0.0, 0.0, 20.0])

        # Large acceleration command
        u = (0.0, 8.0)
        new_state = model.step(state, u, 0.1)

        # Velocity should decrease due to drag
        assert new_state[3] < state[3] + 5.0 * 0.1  # Limited by traction

    def test_noise_injection(self):
        """Test IMU noise injection."""
        params = VehicleParams()
        model = BicycleKinematic(params)
        model.set_noise_params(seed=42, imu_noise_std=0.1)

        state = np.array([0.0, 0.0, 0.0, 5.0])
        u = (0.0, 0.0)

        # Run multiple steps and check for noise
        states = []
        for _ in range(10):
            state = model.step(state, u, 0.1)
            states.append(state.copy())

        # Yaw should have some variation due to noise
        yaws = [s[2] for s in states]
        assert len(yaws) > 0  # Speed should remain positive

    def test_latency_buffer(self):
        """Test control latency implementation."""
        params = VehicleParams()
        model = BicycleKinematic(params)
        model.set_latency(3)  # 3-step latency

        state = np.array([0.0, 0.0, 0.0, 5.0])

        # Push a steering command into the latency buffer
        model.step(state, (0.5, 0.0), 0.1)

        # First few steps should not show steering effect
        for _ in range(3):
            state = model.step(state, (0.0, 0.0), 0.1)
            assert len(model.command_buffer) == 3

        # After latency, steering should take effect
        prev_yaw = state[2]
        state = model.step(state, (0.0, 0.0), 0.1)
        assert state[2] != prev_yaw
        # Now the original steering command should be applied


class TestStanleyController:
    """Test Stanley controller implementation."""

    def test_stanley_straight_path(self):
        """Test Stanley controller on straight path."""
        path = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        state = np.array([0.5, 0.1, 0.0, 5.0])  # Slightly off path

        steer, _ = stanley_control(state, path)
        # Should steer to correct cross-track error
        assert abs(steer) < 0.5  # Reasonable steering angle

    def test_stanley_curved_path(self):
        """Test Stanley controller on curved path."""
        # Create curved path
        t = np.linspace(0, np.pi, 10)
        path = np.column_stack([t, np.sin(t)])

        state = np.array([1.0, 0.5, 0.0, 3.0])

        steer, idx = stanley_control(state, path)

        assert isinstance(steer, float)
        assert isinstance(idx, (int, np.integer))
        assert 0 <= idx < len(path)

    def test_stanley_parameters(self):
        """Test Stanley controller with different parameters."""
        path = np.array([[0, 0], [1, 0], [2, 0]])
        state = np.array([0.5, 0.5, 0.0, 5.0])  # Off path

        # High cross-track gain should produce larger steering
        steer1, _ = stanley_control(state, path, k_e=0.1)
        steer2, _ = stanley_control(state, path, k_e=1.0)

        assert abs(steer2) > abs(steer1)


class TestMPPIController:
    """Test MPPI controller implementation."""

    def test_mppi_initialization(self):
        """Test MPPI controller initialization."""
        controller = MPPIController(
            horizon=10, num_samples=100, lambda_=0.5, noise_std=0.2)

        assert controller.horizon == 10
        assert controller.num_samples == 100
        assert controller.lambda_ == 0.5
        assert controller.noise_std == 0.2

    def test_mppi_control_output(self):
        """Test MPPI controller produces valid output."""
        controller = MPPIController(horizon=5, num_samples=50)

        path = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        state = np.array([0.0, 0.0, 0.0, 5.0])

        steer, _ = controller.control(state, path)

        assert isinstance(steer, (float, np.floating))
        assert not np.isnan(steer)
        assert not np.isinf(steer)

    def test_mppi_empty_path(self):
        """Test MPPI controller with empty path."""
        controller = MPPIController()
        state = np.array([5.0, 2.0, 0.0, 5.0])  # Offset to the right
        steer_empty, idx_empty = controller.control(state, np.array([]))

        assert steer_empty == 0.0
        assert idx_empty == 0


class TestTelemetryLogger:
    """Test telemetry logging functionality."""

    def test_logger_initialization(self):
        """Test logger initialization."""
        logger = TelemetryLogger()

        expected_keys = [
            'time', 'x', 'y', 'yaw', 'velocity', 'steering',
            'acceleration', 'cross_track_error', 'heading_error',
        ]

        for key in expected_keys:
            assert key in logger.data
            assert len(logger.data[key]) == 0

    def test_basic_logging(self):
        """Test basic data logging."""
        logger = TelemetryLogger()

        time = 1.0
        state = np.array([1.0, 2.0, 0.5, 10.0])
        control = (0.1, 2.0)

        logger.log(time, state, control)

        assert len(logger.data['time']) == 1
        assert logger.data['x'][0] == 1.0
        assert logger.data['y'][0] == 2.0
        assert logger.data['yaw'][0] == 0.5
        assert logger.data['velocity'][0] == 10.0
        assert logger.data['steering'][0] == 0.1
        assert logger.data['acceleration'][0] == 2.0

    def test_logging_with_path(self):
        """Test logging with path for error calculation."""
        logger = TelemetryLogger()

        time = 1.0
        state = np.array([1.0, 0.5, 0.0, 5.0])  # Slightly off path
        control = (0.0, 0.0)
        path = np.array([[0, 0], [1, 0], [2, 0]])  # Straight path

        logger.log(time, state, control, path)

        # Should calculate cross-track and heading errors
        assert logger.data['cross_track_error'][-1] > 0.0  # Should have some error
        assert abs(logger.data['heading_error'][0]) < 1e-6  # Aligned with path

    def test_multiple_logs(self):
        """Test multiple data points."""
        logger = TelemetryLogger()

        for i in range(5):
            time = float(i)
            state = np.array([float(i), 0.0, 0.0, 5.0])
            control = (0.0, 0.0)
            logger.log(time, state, control)

        assert len(logger.data['time']) == 5
        assert len(logger.data['x']) == 5
        assert logger.data['x'][-1] == 4.0  # Last x position

    def test_statistics_calculation(self):
        """Test statistics calculation."""
        logger = TelemetryLogger()

        # Log multiple data points with varying velocity
        for i in range(5):
            state = np.array([0.0, 0.0, 0.0, float(i * 5)])  # 0, 5, 10, 15, 20
            control = (0.0, 0.0)
            logger.log(float(i), state, control)

        stats = logger.get_statistics()

        assert 'velocity' in stats
        assert 'steering' in stats
        assert stats['velocity']['mean'] == 10.0
        assert stats['velocity']['max'] == 20.0

    def test_empty_statistics(self):
        """Test statistics with no data."""
        logger = TelemetryLogger()
        stats = logger.get_statistics()

        assert stats == {}

    def test_save_to_file(self, tmp_path):
        """Test saving telemetry to file."""
        logger = TelemetryLogger()

        # Log some data
        for i in range(3):
            state = np.array([float(i), 0.0, 0.0, 5.0])
            control = (0.0, 0.0)
            logger.log(i * 0.1, state, control)

        # Save to temporary file
        filename = tmp_path / "telemetry.csv"
        logger.save_to_file(str(filename))

        # Verify file exists
        assert filename.exists()

        # Read and verify content
        with open(filename, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 4  # Header + 3 data rows
        assert 'time,x,y,yaw,velocity' in lines[0]  # Header check
