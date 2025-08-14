import matplotlib.pyplot as plt
from matplotlib import animation


def run_animation(sim, telemetry_cb=None):
    """Matplotlib-based HUD + animation around a generator 'sim'."""
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1])

    ax_main = fig.add_subplot(gs[:, 0])
    ax_speed = fig.add_subplot(gs[0, 1])
    ax_curv = fig.add_subplot(gs[1, 1])
    ax_imu = fig.add_subplot(gs[:, 2])

    coneL_sc = ax_main.plot([], [], "^", markersize=6)[0]
    coneR_sc = ax_main.plot([], [], "o", markersize=6)[0]
    center_ln, = ax_main.plot([], [], "-", linewidth=2)
    car_poly, = ax_main.plot([], [], "-", linewidth=2)

    speed_ln, = ax_speed.plot([], [])
    curv_ln, = ax_curv.plot([], [])
    imu_ln1, = ax_imu.plot([], [], label="ax")
    imu_ln2, = ax_imu.plot([], [], label="ay")
    imu_ln3, = ax_imu.plot([], [], label="gz")
    ax_imu.legend(loc="upper right")

    ax_main.set_title("Track & Car")
    ax_speed.set_title("Speed (m/s)")
    ax_curv.set_title("Curvature proxy")
    ax_imu.set_title("IMU (ax, ay, gz)")

    speed_hist, curv_hist, ax_hist, ay_hist, gz_hist, t_hist = [], [], [], [], [], []
    paused = {"v": False}

    def on_key(event):
        if event.key == " ":
            paused["v"] = not paused["v"]
        if event.key == "escape":
            plt.close(fig)
    fig.canvas.mpl_connect("key_press_event", on_key)

    def init():
        ax_main.set_xlim(-5, 55)
        ax_main.set_ylim(-10, 30)
        return (coneL_sc, coneR_sc, center_ln, car_poly,
                speed_ln, curv_ln, imu_ln1, imu_ln2, imu_ln3)

    def update(_):
        if paused["v"]:
            return ()
        frame = next(sim)
        L = frame["cones_left"]; R = frame["cones_right"]; C = frame["centerline"]
        car = frame["car_tri"]; met = frame["metrics"]
        if telemetry_cb:
            telemetry_cb(met)

        coneL_sc.set_data(L[:, 0], L[:, 1])
        coneR_sc.set_data(R[:, 0], R[:, 1])
        center_ln.set_data(C[:, 0], C[:, 1])
        car_poly.set_data(car[:, 0], car[:, 1])

        t_hist.append(met["t"])
        speed_hist.append(met["speed"])
        curv_hist.append(met["curvature_mean"])
        ax_hist.append(met["ax"]); ay_hist.append(met["ay"]); gz_hist.append(met["gz"])

        speed_ln.set_data(t_hist, speed_hist)
        curv_ln.set_data(t_hist, curv_hist)
        imu_ln1.set_data(t_hist, ax_hist)
        imu_ln2.set_data(t_hist, ay_hist)
        imu_ln3.set_data(t_hist, gz_hist)

        for ax in (ax_speed, ax_curv, ax_imu):
            ax.relim(); ax.autoscale_view()
        return (coneL_sc, coneR_sc, center_ln, car_poly,
                speed_ln, curv_ln, imu_ln1, imu_ln2, imu_ln3)

    animation.FuncAnimation(fig, update, init_func=init, interval=30, blit=False)
    plt.tight_layout()
    plt.show()
