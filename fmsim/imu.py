import numpy as np


def imu_from_state(prev_state, state, dt, accel_noise=0.2, gyro_noise=np.deg2rad(0.2)):
    """Synthetic IMU from kinematic state differences."""
    X0, Y0, yaw0, v0 = prev_state
    X1, Y1, yaw1, v1 = state
    a_long = (v1 - v0) / dt
    yaw_rate = (yaw1 - yaw0) / dt
    a_lat = v1 * yaw_rate
    ax = float(a_long + np.random.randn() * accel_noise)
    ay = float(a_lat + np.random.randn() * accel_noise)
    gz = float(yaw_rate + np.random.randn() * gyro_noise)
    return {"ax": ax, "ay": ay, "gz": gz}
