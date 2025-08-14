import matplotlib.pyplot as plt
from matplotlib import animation
import traceback
# (same content as the "new" side of the diff above; paste it fully here)
# For brevity: use the entire block from the diff's "new" file
def run_animation(sim, telemetry_cb=None):
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1])
    ax_main = fig.add_subplot(gs[:, 0]); ax_speed = fig.add_subplot(gs[0, 1])
    ax_curv = fig.add_subplot(gs[1, 1]); ax_imu = fig.add_subplot(gs[:, 2])
    coneL_sc = ax_main.plot([], [], "^", markersize=6)[0]
    coneR_sc = ax_main.plot([], [], "o", markersize=6)[0]
    center_ln, = ax_main.plot([], [], "-", linewidth=2)
    car_poly, = ax_main.plot([], [], "-", linewidth=2)
    speed_ln, = ax_speed.plot([], []); curv_ln, = ax_curv.plot([], [])
    imu_ln1, = ax_imu.plot([], [], label="ax"); imu_ln2, = ax_imu.plot([], [], label="ay")
    imu_ln3, = ax_imu.plot([], [], label="gz"); ax_imu.legend(loc="upper right")
    ax_main.set_title("Track & Car"); ax_speed.set_title("Speed (m/s)")
    ax_curv.set_title("Curvature proxy"); ax_imu.set_title("IMU (ax, ay, gz)")
    speed_hist, curv_hist, ax_hist, ay_hist, gz_hist, t_hist = [], [], [], [], [], []
    paused = {"v": False}
    def on_key(event):
        if event.key == " ": paused["v"] = not paused["v"]
        if event.key == "escape": plt.close(fig)
    fig.canvas.mpl_connect("key_press_event", on_key)
    def _set_xy(line, arr):
        try:
            if arr is None: line.set_data([], [])
            elif hasattr(arr, "ndim") and arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] >= 1:
                line.set_data(arr[:, 0], arr[:, 1])
            else: line.set_data([], [])
        except Exception: line.set_data([], [])
    def init():
        ax_main.set_xlim(-5, 55); ax_main.set_ylim(-10, 30)
        return (coneL_sc, coneR_sc, center_ln, car_poly, speed_ln, curv_ln, imu_ln1, imu_ln2, imu_ln3)
    def update(_):
        if paused["v"]: return ()
        try: frame = next(sim)
        except StopIteration:
            plt.close(fig); return ()
        except Exception as e:
            print("Exception in simulation update:", repr(e)); traceback.print_exc()
            fig.suptitle("Error: {}".format(e)); paused["v"] = True; return ()
        L = frame["cones_left"]; R = frame["cones_right"]; C = frame["centerline"]
        car = frame["car_tri"]; met = frame["metrics"]
        _set_xy(coneL_sc, L); _set_xy(coneR_sc, R); _set_xy(center_ln, C); _set_xy(car_poly, car)
        t_hist.append(met["t"]); speed_hist.append(met["speed"]); curv_hist.append(met["curvature_mean"])
        ax_hist.append(met["ax"]); ay_hist.append(met["ay"]); gz_hist.append(met["gz"])
        speed_ln.set_data(t_hist, speed_hist); curv_ln.set_data(t_hist, curv_hist)
        imu_ln1.set_data(t_hist, ax_hist); imu_ln2.set_data(t_hist, ay_hist); imu_ln3.set_data(t_hist, gz_hist)
        for ax in (ax_speed, ax_curv, ax_imu): ax.relim(); ax.autoscale_view()
        return (coneL_sc, coneR_sc, center_ln, car_poly, speed_ln, curv_ln, imu_ln1, imu_ln2, imu_ln3)
    # Set a large but finite number of frames to cache (1000 frames = ~30 seconds at 30fps)
    ani = animation.FuncAnimation(
        fig, update, init_func=init, interval=30, blit=False, 
        save_count=1000, cache_frame_data=True
    )
    fig._ani = ani
    plt.tight_layout()
    plt.show()
