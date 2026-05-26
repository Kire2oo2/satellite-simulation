import os
import numpy as np

import simutils as su
import simulator as sim
import orbit_lib as ol
import sat_lib as sl
import plotter as pl


VISUALISE = False
TLE_FILE = "TLE.txt"
EARTH_IMAGE = "earth_grid.jpg"
DATA_DIR = "data"
PLOT_DIR = "plots"

CURRENT_TLE = "HST1"
OLD_TLE = "HST2"

RUN_PART_1 = True
RUN_PART_2 = True

ACTUATOR_LIMIT = 1.13
ATTITUDE_DT = 0.25
RANDOM_SEED = 9

PD_K1 = 1.0e-4
PD_K2 = 2.0e-2
SM_K1 = 1.0e-2
SM_K = 1.0e-5
SM_EPS = 1.0e-4

GYRO_MU = 0.0
GYRO_VARIANCE = 1.0e-6
STAR_MU = 0.0
STAR_Q = 1.0e-2

SOLAR_A1 = 0.2
SOLAR_A2 = 0.2
SOLAR_P1 = 0.14 * np.pi
SOLAR_P2 = 1.22 * np.pi
SOLAR_PHI1 = 0.31 * np.pi
SOLAR_PHI2 = -0.05 * np.pi

HST_J = np.array([
    [36046.0,  -706.0,  1491.0],
    [ -706.0, 86868.0,   449.0],
    [ 1491.0,   449.0, 93848.0]
])
HST_J_INV = np.linalg.inv(HST_J)

Q_IB_0 = np.array([1.0, 0.0, 0.0, 0.0])
W_B_IB_0 = 1.0e-3 * np.array([0.3, -0.1, 0.2])
Q_ID = 0.5 * np.array([1.0, 1.0, 1.0, 1.0])

COL_TIME = 0
COL_TRUE_ARCSEC = 1
COL_EST_ARCSEC = 2
COL_TAU_A_NORM = 13
COL_TAU_G_NORM = 17
COL_TAU_SOLAR_NORM = 21
COL_TAU_D_NORM = 25
COL_S_NORM = 29


def make_output_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)


def run_part1_task1(tles):
    print("[1/5] Running Part 1 Task 1")

    tle = tles[CURRENT_TLE]
    orbit = ol.orbit_pkepler_from_tle(tle)
    JD = ol.epoch_to_julian_date(tle["epoch"])
    theta_E = ol.sidereal_angle(JD)

    r_i, v_i = orbit.get_state()
    h, e, theta, raan, inc, arg_perigee = ol.orbit_params_from_state(r_i, v_i)
    a = h**2 / (ol.mu * (1.0 - e**2))
    E = ol.eccentric_anomaly_from_true_anomaly(theta, e)
    n_rad_s = np.sqrt(ol.mu / a**3)
    q_io, w_i_io, dw_i_io = ol.orbit_frame_from_state(r_i, v_i)
    lon, lat, altitude = ol.geodetic_from_eci(r_i, theta_E)

    rows = [
        ["Specific relative angular momentum h", h, "km^2/s"],
        ["True anomaly theta", theta, "rad"],
        ["Eccentric anomaly E", E, "rad"],
        ["Semi-major axis a", a, "km"],
        ["Mean motion n", n_rad_s, "rad/s"],
        ["Derivative of mean motion dn", tle["dn"], "rev/day^2"],
        ["Second derivative of mean motion ddn", tle["ddn"], "rev/day^3"],
        ["Position r_i", r_i, "km"],
        ["Velocity v_i", v_i, "km/s"],
        ["Julian date JD", JD, "days"],
        ["Sidereal angle theta_G0", theta_E, "rad"],
        ["Orbit frame q_io", q_io, "-"],
        ["Orbit frame angular velocity", w_i_io, "rad/s"],
        ["Orbit frame angular acceleration", dw_i_io, "rad/s^2"],
        ["Geodetic latitude", lat, "rad"],
        ["Geodetic / geocentric longitude", lon, "rad"],
        ["Altitude", altitude, "km"]
    ]

    file_name = os.path.join(DATA_DIR, "assignment9_task1_epoch_table.txt")

    with open(file_name, "w") as f:
        for name, value, unit in rows:
            f.write("{:<42s} {} {}\n".format(name, value, unit))

    print("      saved:", file_name)


def run_part1_task2(tles):
    print("[2/5] Running Part 1 Task 2")

    current = tles[CURRENT_TLE]
    orbit = ol.orbit_pkepler_from_tle(current)
    JD0 = ol.epoch_to_julian_date(current["epoch"])

    t = 0.0
    t_end = 9.0 * 365.25 * ol.SECONDS_IN_DAY
    dt = 6.0 * 3600.0

    altitude_log = []
    ground_track = []

    while t <= t_end:
        r_i, v_i = orbit.get_state()
        h, e, theta, raan, inc, arg_perigee = ol.orbit_params_from_state(r_i, v_i)
        a = h**2 / (ol.mu * (1.0 - e**2))
        theta_E = ol.sidereal_angle(JD0 + t / ol.SECONDS_IN_DAY)
        lon, lat, geodetic_altitude = ol.geodetic_from_eci(r_i, theta_E)
        radial_altitude = np.linalg.norm(r_i) - ol.R_E

        altitude_log.append([t / ol.SECONDS_IN_DAY, radial_altitude, geodetic_altitude, a, e])
        ground_track.append([t, lon, lat, geodetic_altitude])

        orbit.propagate(dt)
        t += dt

    altitude_log = np.asarray(altitude_log)
    ground_track = np.asarray(ground_track)

    altitude_file = os.path.join(DATA_DIR, "assignment9_task2_hst_9year_altitude.txt")
    ground_file = os.path.join(DATA_DIR, "assignment9_task2_9year_ground_track.txt")

    np.savetxt(
        altitude_file,
        altitude_log,
        header="days_from_epoch radial_altitude_km geodetic_altitude_km semi_major_axis_km eccentricity"
    )
    np.savetxt(
        ground_file,
        ground_track,
        header="time_s longitude_rad latitude_rad geodetic_altitude_km"
    )

    pl.plot_assignment9_altitude(
        altitude_log,
        "assignment9_task2_9year_altitude.png",
        PLOT_DIR
    )

    compare_old_tle(tles, OLD_TLE)

    print("      final radial altitude:   {:.3f} km".format(altitude_log[-1, 1]))
    print("      final geodetic altitude: {:.3f} km".format(altitude_log[-1, 2]))
    print("      saved:", altitude_file)
    print("      saved:", ground_file)


def compare_old_tle(tles, old_tle_name):
    old = tles[old_tle_name]
    current = tles[CURRENT_TLE]

    old_orbit = ol.orbit_pkepler_from_tle(old)
    current_orbit = ol.orbit_pkepler_from_tle(current)

    old_JD = ol.epoch_to_julian_date(old["epoch"])
    current_JD = ol.epoch_to_julian_date(current["epoch"])
    delta_epoch = (current_JD - old_JD) * ol.SECONDS_IN_DAY

    tau = 0.0
    while tau < delta_epoch:
        h_step = min(3600.0, delta_epoch - tau)
        old_orbit.propagate(h_step)
        tau += h_step

    r_old, v_old = old_orbit.get_state()
    r_current, v_current = current_orbit.get_state()

    h_old, e_old, theta_old, raan_old, inc_old, arg_old = ol.orbit_params_from_state(r_old, v_old)
    h_cur, e_cur, theta_cur, raan_cur, inc_cur, arg_cur = ol.orbit_params_from_state(r_current, v_current)

    a_old = h_old**2 / (ol.mu * (1.0 - e_old**2))
    a_cur = h_cur**2 / (ol.mu * (1.0 - e_cur**2))

    error_file = os.path.join(DATA_DIR, "assignment9_task2_{}_error.txt".format(old_tle_name.lower()))

    with open(error_file, "w") as f:
        f.write("Old propagated TLE: {}\n".format(old_tle_name))
        f.write("Current TLE ground truth: {}\n".format(CURRENT_TLE))
        f.write("Epoch separation days: {:.9f}\n".format(delta_epoch / ol.SECONDS_IN_DAY))
        f.write("Position error vector km: {}\n".format(r_old - r_current))
        f.write("Velocity error vector km/s: {}\n".format(v_old - v_current))
        f.write("Position error norm km: {:.9f}\n".format(np.linalg.norm(r_old - r_current)))
        f.write("Velocity error norm km/s: {:.12f}\n".format(np.linalg.norm(v_old - v_current)))
        f.write("Semi-major axis error km: {:.9f}\n".format(a_old - a_cur))
        f.write("Eccentricity error: {:.12e}\n".format(e_old - e_cur))
        f.write("Inclination error deg: {:.9f}\n".format(ol.angle_error(inc_old - inc_cur) * ol.RTOD))
        f.write("RAAN error deg: {:.9f}\n".format(ol.angle_error(raan_old - raan_cur) * ol.RTOD))
        f.write("Argument of perigee error deg: {:.9f}\n".format(ol.angle_error(arg_old - arg_cur) * ol.RTOD))
        f.write("True anomaly error deg: {:.9f}\n".format(ol.angle_error(theta_old - theta_cur) * ol.RTOD))

    period = ol.SECONDS_IN_DAY / current["revs_per_day"]
    dt = 10.0
    tau = 0.0

    diff_log = []
    gt_current = []
    gt_old = []

    while tau <= period:
        r_c, v_c = current_orbit.get_state()
        r_o, v_o = old_orbit.get_state()
        theta_E = ol.sidereal_angle(current_JD + tau / ol.SECONDS_IN_DAY)

        lon_c, lat_c, alt_c = ol.geodetic_from_eci(r_c, theta_E)
        lon_o, lat_o, alt_o = ol.geodetic_from_eci(r_o, theta_E)

        diff_log.append([tau, np.linalg.norm(r_o - r_c), np.linalg.norm(v_o - v_c)])
        gt_current.append([tau, lon_c, lat_c, alt_c])
        gt_old.append([tau, lon_o, lat_o, alt_o])

        current_orbit.propagate(dt)
        old_orbit.propagate(dt)
        tau += dt

    diff_log = np.asarray(diff_log)
    gt_current = np.asarray(gt_current)
    gt_old = np.asarray(gt_old)

    np.savetxt(
        os.path.join(DATA_DIR, "assignment9_task2_{}_one_orbit_position_difference.txt".format(old_tle_name.lower())),
        diff_log,
        header="time_s position_difference_km velocity_difference_km_s"
    )
    np.savetxt(
        os.path.join(DATA_DIR, "assignment9_task2_ground_track_current.txt"),
        gt_current,
        header="time_s longitude_rad latitude_rad altitude_km"
    )
    np.savetxt(
        os.path.join(DATA_DIR, "assignment9_task2_ground_track_{}_propagated.txt".format(old_tle_name.lower())),
        gt_old,
        header="time_s longitude_rad latitude_rad altitude_km"
    )

    pl.plot_assignment9_position_difference(
        diff_log,
        old_tle_name,
        "assignment9_task2_{}_one_orbit_position_difference.png".format(old_tle_name.lower()),
        PLOT_DIR
    )
    pl.plot_assignment9_ground_tracks(
        gt_current,
        gt_old,
        old_tle_name,
        EARTH_IMAGE,
        "assignment9_task2_{}_ground_tracks.png".format(old_tle_name.lower()),
        PLOT_DIR
    )

    print("      {} max one-orbit position difference: {:.3f} km".format(old_tle_name, np.max(diff_log[:, 1])))
    print("      saved:", error_file)


class Part2Scenario(sim.BaseScenario):
    def __init__(self, tles, label, controller, star_trackers):
        self.tles = tles
        self.label = label
        self.controller = controller
        self.star_trackers = star_trackers

        self.orbit = None
        self.JD0 = None
        self.q_E = su.Quaternion()
        self.q_ib = Q_IB_0.copy()
        self.w_b_ib = W_B_IB_0.copy()
        self.attitude_log = []
        self.rng = np.random.default_rng(RANDOM_SEED)

    def init(self, t):
        tle = self.tles[CURRENT_TLE]
        self.JD0 = ol.epoch_to_julian_date(tle["epoch"])
        self.orbit = ol.orbit_pkepler_from_tle(tle)
        self.q_E = su.Quaternion(ol.sidereal_angle(self.JD0), np.array([0.0, 0.0, 1.0]))
        self.q_ib = Q_IB_0.copy()
        self.w_b_ib = W_B_IB_0.copy()
        self.attitude_log = []
        self.rng = np.random.default_rng(RANDOM_SEED)

    def update(self, t, dt):
        r_i, v_i = self.orbit.get_state()

        q_measurements = []
        for _ in range(self.star_trackers):
            q_measurements.append(sl.star_tracker_measurement(self.q_ib, self.rng, STAR_MU, STAR_Q))

        q_hat_ib = sl.average_star_trackers(q_measurements)
        w_hat_b_ib = self.w_b_ib + self.rng.normal(GYRO_MU, np.sqrt(GYRO_VARIANCE), 3)

        q_err_est = sl.quaternion_error(Q_ID, q_hat_ib)
        tau_c, s = self.control_torque(q_err_est, w_hat_b_ib)
        tau_a = np.clip(tau_c, -ACTUATOR_LIMIT, ACTUATOR_LIMIT)

        tau_g = ol.gravity_gradient(r_i, self.q_ib, HST_J)
        tau_solar = self.solar_array_torque(t)
        tau_d = tau_g + tau_solar
        tau_total = tau_a + tau_d

        q_err_true = sl.quaternion_error(Q_ID, self.q_ib)
        true_arcsec = sl.pointing_error_arcsec(q_err_true)
        est_arcsec = sl.pointing_error_arcsec(q_err_est)

        self.attitude_log.append([
            t,
            true_arcsec,
            est_arcsec,
            np.linalg.norm(q_err_true[1:]),
            np.linalg.norm(self.w_b_ib),
            np.linalg.norm(w_hat_b_ib),
            tau_c[0], tau_c[1], tau_c[2], np.linalg.norm(tau_c),
            tau_a[0], tau_a[1], tau_a[2], np.linalg.norm(tau_a),
            tau_g[0], tau_g[1], tau_g[2], np.linalg.norm(tau_g),
            tau_solar[0], tau_solar[1], tau_solar[2], np.linalg.norm(tau_solar),
            tau_d[0], tau_d[1], tau_d[2], np.linalg.norm(tau_d),
            s[0], s[1], s[2], np.linalg.norm(s)
        ])

        w_dot = HST_J_INV @ (tau_total - np.cross(self.w_b_ib, HST_J @ self.w_b_ib))
        q_dot = 0.5 * sl._q_mul(self.q_ib, np.array([0.0, *self.w_b_ib]))

        self.w_b_ib = self.w_b_ib + dt * w_dot
        self.q_ib = sl._q_array(self.q_ib + dt * q_dot)

        self.orbit.propagate(dt)
        theta_E = ol.sidereal_angle(self.JD0 + (t + dt) / ol.SECONDS_IN_DAY)
        self.q_E = su.Quaternion(theta_E, np.array([0.0, 0.0, 1.0]))

    def control_torque(self, q_err_est, w_hat_b_ib):
        if self.controller == "PD":
            s = np.zeros(3)
            tau_c = np.cross(w_hat_b_ib, HST_J @ w_hat_b_ib) + HST_J @ (
                -PD_K1 * q_err_est[1:] - PD_K2 * w_hat_b_ib
            )
            return tau_c, s

        qv_dot = q_err_est[0] * w_hat_b_ib + np.cross(q_err_est[1:], w_hat_b_ib)
        s = w_hat_b_ib + 2.0 * SM_K1 * q_err_est[1:]
        sat_s = np.clip(s / SM_EPS, -1.0, 1.0)
        tau_c = np.cross(w_hat_b_ib, HST_J @ w_hat_b_ib) + HST_J @ (
            -SM_K1 * qv_dot - SM_K * sat_s
        )
        return tau_c, s

    def solar_array_torque(self, t):
        d = SOLAR_A1 * np.sin(SOLAR_P1 * t + SOLAR_PHI1) + SOLAR_A2 * np.sin(SOLAR_P2 * t + SOLAR_PHI2)
        return np.array([0.0, d, 0.0])

    def get(self):
        r_i, v_i = self.orbit.get_state()
        return [
            ["satellite", r_i, su.Quaternion(self.q_ib)],
            ["body frame", r_i, su.Quaternion(self.q_ib)],
            ["earth", np.zeros(3), self.q_E],
            ["ECEF frame", np.zeros(3), self.q_E],
            ["ECI frame", np.zeros(3), su.Quaternion()]
        ]

    def post_process(self, t, dt):
        data = np.asarray(self.attitude_log)
        file_name = os.path.join(DATA_DIR, "assignment9_part2_{}.txt".format(self.label))

        np.savetxt(
            file_name,
            data,
            header=(
                "time_s true_error_arcsec estimated_error_arcsec true_qv_norm true_w_norm measured_w_norm "
                "tau_c_x tau_c_y tau_c_z tau_c_norm tau_a_x tau_a_y tau_a_z tau_a_norm "
                "tau_g_x tau_g_y tau_g_z tau_g_norm tau_solar_x tau_solar_y tau_solar_z tau_solar_norm "
                "tau_d_x tau_d_y tau_d_z tau_d_norm s_x s_y s_z s_norm"
            )
        )

        steady = data[data[:, COL_TIME] >= data[-1, COL_TIME] / 4.0]
        rms = np.sqrt(np.mean(steady[:, COL_TRUE_ARCSEC]**2))

        print("      {} final = {:.3f} arcsec, RMS after first orbit = {:.3f} arcsec".format(
            self.label,
            data[-1, COL_TRUE_ARCSEC],
            rms
        ))


def run_part2(tles):
    current = tles[CURRENT_TLE]
    period = ol.SECONDS_IN_DAY / current["revs_per_day"]

    cases = [
        ["PD_1ST", "PD", 1],
        ["SM_1ST", "SM", 1],
        ["SM_3ST", "SM", 3]
    ]

    for idx, (label, controller, trackers) in enumerate(cases, start=3):
        print("[{}/5] Running Part 2 case {}".format(idx, label))

        scenario = Part2Scenario(tles, label, controller, trackers)
        sim_config = {
            "t_0": 0.0,
            "t_e": 4.0 * period,
            "t_step": ATTITUDE_DT,
            "speed_factor": 100,
            "anim_dt": 1.0 / 25.0,
            "scale_factor": 1000,
            "visualise": VISUALISE
        }
        sim.create_and_start_simulation(sim_config, scenario)

    make_part2_summary_and_plots(cases)


def make_part2_summary_and_plots(cases):
    case_data = []

    for label, controller, trackers in cases:
        data = np.loadtxt(os.path.join(DATA_DIR, "assignment9_part2_{}.txt".format(label)))
        plot_label = "{}, {} star tracker".format(controller, trackers)

        if trackers != 1:
            plot_label += "s"

        case_data.append([label, plot_label, data])

    summary_file = os.path.join(DATA_DIR, "assignment9_part2_summary.txt")

    with open(summary_file, "w") as f:
        f.write("case final_arcsec rms_arcsec_after_first_orbit max_arcsec_after_first_orbit max_actuator_norm max_disturbance_norm\n")

        for label, plot_label, data in case_data:
            steady = data[data[:, COL_TIME] >= data[-1, COL_TIME] / 4.0]
            f.write("{} {:.12e} {:.12e} {:.12e} {:.12e} {:.12e}\n".format(
                label,
                data[-1, COL_TRUE_ARCSEC],
                np.sqrt(np.mean(steady[:, COL_TRUE_ARCSEC]**2)),
                np.max(steady[:, COL_TRUE_ARCSEC]),
                np.max(data[:, COL_TAU_A_NORM]),
                np.max(data[:, COL_TAU_D_NORM])
            ))

    columns = {
        "time": COL_TIME,
        "true_arcsec": COL_TRUE_ARCSEC,
        "tau_a_norm": COL_TAU_A_NORM,
        "tau_g_norm": COL_TAU_G_NORM,
        "tau_solar_norm": COL_TAU_SOLAR_NORM,
        "tau_d_norm": COL_TAU_D_NORM,
        "s_norm": COL_S_NORM
    }

    pl.plot_assignment9_part2_results(case_data, ACTUATOR_LIMIT, columns, PLOT_DIR)

    print("      saved:", summary_file)
    print("      generated Part 2 plots in", PLOT_DIR)


def main():
    make_output_dirs()
    tles = ol.read_tles(TLE_FILE)

    print("=== Assignment 9 ===")

    if RUN_PART_1:
        run_part1_task1(tles)
        run_part1_task2(tles)

    if RUN_PART_2:
        run_part2(tles)

    print("Done.")
    print("Data folder:", DATA_DIR)
    print("Plot folder:", PLOT_DIR)


if __name__ == "__main__":
    main()
