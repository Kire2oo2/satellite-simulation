import sys
import os
import matplotlib.pyplot as plt
import simutils as su
import numpy as np
import matplotlib.image as mpimg


def ground_track_plot(lat_deg, lon_deg, img_path='3DModels/earth.jpg', save_path=None):
    # Ensure longitude in [-180, 180]
    lon_deg = (lon_deg + 180) % 360 - 180

    # Detect jumps > 180 deg (crossing the map edge)
    dlon = np.diff(lon_deg)
    jump_idx = np.where(np.abs(dlon) > 180)[0]

    # Insert NaN to break the line
    lat_fixed = lat_deg.copy()
    lon_fixed = lon_deg.copy()
    for idx in reversed(jump_idx):
        lat_fixed = np.insert(lat_fixed, idx + 1, np.nan)
        lon_fixed = np.insert(lon_fixed, idx + 1, np.nan)

    # Plot
    img = mpimg.imread(img_path)
    plt.figure(figsize=(12, 6))
    plt.imshow(img, extent=(-180, 180, -90, 90))
    plt.plot(lon_fixed, lat_fixed, 'r', linewidth=1)
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.title("Satellite Ground Track")
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def log_ground_track(scenario, t, r_i):
    # Convert ECI -> ECEF using scenario.q_E
    q_conj = su.Quaternion(scenario.q_E)
    q_conj.conjugate()
    r_ecef = q_conj.rotate(r_i)

    lat = np.arcsin(r_ecef[2] / np.linalg.norm(r_ecef))
    lon = np.arctan2(r_ecef[1], r_ecef[0])
    lon = (lon + np.pi) % (2 * np.pi) - np.pi

    entry = np.array([t, lat, lon])

    if scenario.ground_track is None:
        scenario.ground_track = entry.reshape(1, -1)
    else:
        scenario.ground_track = np.vstack((scenario.ground_track, entry))


def plot_ground_track(file_path, img_path='earth_grid.jpg', save_path=None):
    data = np.loadtxt(file_path)

    phi = data[:, 0]
    lam = data[:, 1]
    h = data[:, 2]

    lon_deg = phi * 180.0 / np.pi
    lat_deg = lam * 180.0 / np.pi

    lon_deg = (lon_deg + 180.0) % 360.0 - 180.0

    jump_idx = np.where(np.abs(np.diff(lon_deg)) > 180.0)[0]

    lon_fixed = lon_deg.copy()
    lat_fixed = lat_deg.copy()

    for idx in reversed(jump_idx):
        lon_fixed = np.insert(lon_fixed, idx + 1, np.nan)
        lat_fixed = np.insert(lat_fixed, idx + 1, np.nan)

    img = mpimg.imread(img_path)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(img, extent=[-180, 180, -90, 90])
    ax.plot(lon_fixed, lat_fixed, linewidth=2, color='black')
    ax.grid(True)
    ax.set_yticks(np.arange(-90, 90 + 15, 15))
    ax.set_xticks(np.arange(-180, 180 + 15, 15))
    ax.set_ylim(-90, 90)
    ax.set_xlim(-180, 180)
    ax.set(xlabel='Longitude', ylabel='Latitude', title='Ground track')

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def line_plot(file_path):
    data = np.loadtxt(file_path)
    t = data[:,0]
    _, ax = plt.subplots()
    for col in data[:,1:].T:
        ax.plot(t,col)
    plt.show()

def main(argv):
    if len(argv) == 3:
        plot_type = argv[1]
        file_path = argv[2]
        if plot_type == 'lineplot':
            line_plot(file_path)
        else:
            print("Plot type not supported yet.")
    else:
        print("Wrong number of arguments. Expected 2 (plot_type, file_path) got {}".format(len(argv)-1))

if __name__ == "__main__":
    main(sys.argv)

# Assignment 9 plotting helpers

def save_plot(fig, file_name, plot_dir='plots'):
    os.makedirs(plot_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, file_name), dpi=300)
    plt.close(fig)


def plot_decimated(ax, x, y, label=None, linewidth=0.8, alpha=0.85, max_points=4000):
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) == 0:
        return

    step = max(1, int(np.ceil(len(x) / max_points)))

    ax.plot(
        x[::step],
        y[::step],
        label=label,
        linewidth=linewidth,
        alpha=alpha,
        rasterized=True
    )


def rolling_std(y, window):
    y = np.asarray(y, dtype=float)
    out = np.full(len(y), np.nan)

    if window <= 1 or len(y) == 0:
        return np.zeros_like(y, dtype=float)

    for i in range(window - 1, len(y)):
        out[i] = np.std(y[i - window + 1:i + 1])

    return out


def rolling_rms(y, window):
    y = np.asarray(y, dtype=float)
    out = np.full(len(y), np.nan)

    if window <= 1 or len(y) == 0:
        return np.abs(y)

    for i in range(window - 1, len(y)):
        out[i] = np.sqrt(np.mean(y[i - window + 1:i + 1]**2))

    return out

def wrap_ground_track(lon_rad, lat_rad):
    lon_deg = (lon_rad * 180.0 / np.pi + 180.0) % 360.0 - 180.0
    lat_deg = lat_rad * 180.0 / np.pi

    lon_plot = lon_deg.copy()
    lat_plot = lat_deg.copy()

    jump_idx = np.where(np.abs(np.diff(lon_deg)) > 180.0)[0]

    for idx in reversed(jump_idx):
        lon_plot = np.insert(lon_plot, idx + 1, np.nan)
        lat_plot = np.insert(lat_plot, idx + 1, np.nan)

    return lon_plot, lat_plot


def plot_assignment9_altitude(altitude_log, file_name, plot_dir='plots'):
    fig, ax = plt.subplots()

    plot_decimated(
        ax,
        altitude_log[:, 0] / 365.25,
        altitude_log[:, 1],
        label='radial altitude',
        linewidth=0.8,
        alpha=0.9
    )
    plot_decimated(
        ax,
        altitude_log[:, 0] / 365.25,
        altitude_log[:, 2],
        label='geodetic altitude',
        linewidth=0.8,
        alpha=0.9
    )

    ax.axhline(120.0, linestyle='--', linewidth=1.0, label='atmosphere border')
    ax.set_xlabel('Years after epoch')
    ax.set_ylabel('Altitude [km]')
    ax.set_title('HST 9-year altitude using PKepler')
    ax.grid(True)
    ax.legend()

    save_plot(fig, file_name, plot_dir)


def plot_assignment9_position_difference(diff_log, old_tle_name, file_name, plot_dir='plots'):
    fig, ax = plt.subplots()

    ax.plot(diff_log[:, 0] / 60.0, diff_log[:, 1], linewidth=0.9)
    ax.set_xlabel('Time from current epoch [min]')
    ax.set_ylabel('Position difference [km]')
    ax.set_title('{} propagation error over one orbit'.format(old_tle_name))
    ax.grid(True)

    save_plot(fig, file_name, plot_dir)


def plot_assignment9_ground_tracks(gt_current, gt_old, old_tle_name, img_path, file_name, plot_dir='plots'):
    img = mpimg.imread(img_path)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.imshow(img, extent=[-180.0, 180.0, -90.0, 90.0])

    for data, label in [(gt_current, 'current TLE'), (gt_old, '{} propagated'.format(old_tle_name))]:
        lon_plot, lat_plot = wrap_ground_track(data[:, 1], data[:, 2])
        ax.plot(lon_plot, lat_plot, label=label, linewidth=0.9)

    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.set_title('Ground tracks over one orbit')
    ax.grid(True)
    ax.legend()

    save_plot(fig, file_name, plot_dir)


def plot_assignment9_part2_results(case_data, actuator_limit, columns, plot_dir='plots'):
    col_time = columns['time']
    col_true_arcsec = columns['true_arcsec']
    col_tau_a_norm = columns['tau_a_norm']
    col_tau_g_norm = columns['tau_g_norm']
    col_tau_solar_norm = columns['tau_solar_norm']
    col_tau_d_norm = columns['tau_d_norm']
    col_s_norm = columns['s_norm']

    fig, ax = plt.subplots()
    for label, plot_label, data in case_data:
        plot_decimated(
            ax,
            data[:, col_time] / 60.0,
            data[:, col_true_arcsec],
            label=plot_label,
            linewidth=0.8,
            alpha=0.9,
            max_points=4000
        )

    ax.axhline(0.007, linestyle='--', linewidth=1.0, label='0.007 arcsec HST reference')
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Pointing error [arcsec]')
    ax.set_title('HST pointing error')
    ax.grid(True)
    ax.legend()
    save_plot(fig, 'assignment9_part2_pointing_error.png', plot_dir)

    fig, ax = plt.subplots()
    for label, plot_label, data in case_data:
        plot_decimated(
            ax,
            data[:, col_time] / 60.0,
            np.maximum(data[:, col_true_arcsec], 1e-12),
            label=plot_label,
            linewidth=0.8,
            alpha=0.9,
            max_points=4000
        )

    ax.axhline(0.007, linestyle='--', linewidth=1.0, label='0.007 arcsec HST reference')
    ax.set_yscale('log')
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Pointing error [arcsec]')
    ax.set_title('HST pointing error, log scale')
    ax.grid(True, which='both')
    ax.legend()
    save_plot(fig, 'assignment9_part2_pointing_error_log.png', plot_dir)

    # Zoomed pointing-error plot. The initial slew dominates the normal plot, so this is after 20min
    fig, ax = plt.subplots()
    zoom_start_min = 20.0
    ymax = 0.0

    for label, plot_label, data in case_data:
        t_min = data[:, col_time] / 60.0
        mask = t_min >= zoom_start_min

        if np.any(mask):
            plot_decimated(
                ax,
                t_min[mask],
                data[:, col_true_arcsec][mask],
                label=plot_label,
                linewidth=0.8,
                alpha=0.9,
                max_points=4000
            )
            ymax = max(ymax, np.max(data[:, col_true_arcsec][mask]))

    ax.axhline(0.007, linestyle='--', linewidth=1.0, label='0.007 arcsec HST reference')
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Pointing error [arcsec]')
    ax.set_title('HST pointing error, zoomed after initial transient')
    ax.grid(True)
    ax.legend()

    if ymax > 0.0:
        ax.set_ylim(0.0, 1.1 * ymax)

    save_plot(fig, 'assignment9_part2_pointing_error_zoomed.png', plot_dir)

    fig, ax = plt.subplots()
    for label, plot_label, data in case_data:
        plot_decimated(
            ax,
            data[:, col_time] / 60.0,
            data[:, col_tau_a_norm],
            label=plot_label,
            linewidth=0.45,
            alpha=0.7,
            max_points=4000
        )

    ax.axhline(np.sqrt(3.0) * actuator_limit, linestyle='--', linewidth=1.0, label='3-axis saturation norm')
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Actuator torque norm [Nm]')
    ax.set_title('Applied actuator torque')
    ax.grid(True)
    ax.legend()
    save_plot(fig, 'assignment9_part2_actuator_torque.png', plot_dir)

    fig, ax = plt.subplots()
    for label, plot_label, data in case_data:
        if label.startswith('SM'):
            plot_decimated(
                ax,
                data[:, col_time] / 60.0,
                data[:, col_s_norm],
                label=plot_label,
                linewidth=0.55,
                alpha=0.75,
                max_points=4000
            )

    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Sliding surface norm')
    ax.set_title('Sliding mode surface')
    ax.grid(True)
    ax.legend()
    save_plot(fig, 'assignment9_part2_sliding_surface.png', plot_dir)

    fig, ax = plt.subplots()
    for label, plot_label, data in case_data:
        if label.startswith('SM'):
            plot_decimated(
                ax,
                data[:, col_time] / 60.0,
                data[:, col_s_norm],
                label=plot_label,
                linewidth=0.55,
                alpha=0.75,
                max_points=4000
            )

    ax.set_ylim(0.0, 0.004)
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Sliding surface norm')
    ax.set_title('Sliding mode surface, zoomed view')
    ax.grid(True)
    ax.legend()
    save_plot(fig, 'assignment9_part2_sliding_surface_zoomed.png', plot_dir)

    data = case_data[0][2]
    fig, ax = plt.subplots()
    plot_decimated(
        ax,
        data[:, col_time] / 60.0,
        data[:, col_tau_g_norm],
        label='gravity-gradient',
        linewidth=0.45,
        alpha=0.75,
        max_points=4000
    )
    plot_decimated(
        ax,
        data[:, col_time] / 60.0,
        data[:, col_tau_solar_norm],
        label='solar-array',
        linewidth=0.7,
        alpha=0.85,
        max_points=4000
    )
    plot_decimated(
        ax,
        data[:, col_time] / 60.0,
        data[:, col_tau_d_norm],
        label='total',
        linewidth=0.45,
        alpha=0.75,
        max_points=4000
    )

    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Disturbance torque norm [Nm]')
    ax.set_title('Disturbance torques')
    ax.grid(True)
    ax.legend()
    save_plot(fig, 'assignment9_part2_disturbance_torque.png', plot_dir)

