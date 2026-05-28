import os
import numpy as np

import simutils as su
import simulator as sim
import orbit_lib as ol
import sat_lib as sl
import plotter as pl



TLE_FILE = "tle.txt"
EARTH_IMAGE = "earth_grid.jpg"
DATA_DIR = "data"
PLOT_DIR = "plots"

CURRENT_TLE = "HST1"
OLD_TLE = "HST2"

#-----------------------------------------------------------------
#Variables to change what programs are running for this assignment:

VISUALISE = False
RUN_PART_1 = False
RUN_PART_2 = True

RUN_PD_1ST = True
RUN_PD_3ST = False
RUN_SM_1ST = True
RUN_SM_3ST = False


USE_ASSIGNMENT_SENSOR_NOISE = False
#-----------------------------------------------------------------

ACTUATOR_LIMIT = 1.13
ATTITUDE_DT = 0.25
RANDOM_SEED = 9

PD_K1 = 1.0e-4
PD_K2 = 2.0e-2
SM_K1 = 0.25
SM_K = 3e-4
SM_EPS = 3.0e-5

if USE_ASSIGNMENT_SENSOR_NOISE:
    GYRO_MU = 0.0
    GYRO_VARIANCE = 1.0e-6
    STAR_MU = 0.0
    STAR_Q = 1.0e-2
else:
    GYRO_MU = 0.0
    GYRO_VARIANCE = 0.0
    STAR_MU = 0.0
    STAR_Q = 0.0

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


class Part1Task1(sim.BaseScenario):
    def __init__(self, tles):
        self.tles = tles
        self.rows = []
        self.r_i = np.zeros(3)
        self.q_E = su.Quaternion()

    def init(self, t):
        tle = self.tles[CURRENT_TLE]
        orbit = ol.orbit_pkepler_from_tle(tle)
        JD = ol.epoch_to_julian_date(tle["epoch"])
        theta_E = ol.sidereal_angle(JD)

        r_i, v_i = orbit.get_state()
        self.r_i = r_i
        self.q_E = su.Quaternion(theta_E, np.array([0.0, 0.0, 1.0]))
        h, e, theta, raan, inc, arg_perigee = ol.orbit_params_from_state(r_i, v_i)
        a = h**2 / (ol.mu * (1.0 - e**2))
        E = ol.eccentric_anomaly_from_true_anomaly(theta, e)
        n_rad_s = np.sqrt(ol.mu / a**3)
        q_io, w_i_io, dw_i_io = ol.orbit_frame_from_state(r_i, v_i)
        lon, lat, altitude = ol.geodetic_from_eci(r_i, theta_E)

        self.rows = [
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

    def update(self, t, dt):
        pass

    def get(self):
        return [
            ["satellite", self.r_i, su.Quaternion()],
            ["body frame", self.r_i, su.Quaternion()],
            ["earth", np.zeros(3), self.q_E],
            ["ECEF frame", np.zeros(3), self.q_E],
            ["ECI frame", np.zeros(3), su.Quaternion()]
        ]

    def post_process(self, t, dt):
        file_name = os.path.join(DATA_DIR, "assignment9_task1_epoch_table.txt")

        with open(file_name, "w") as f:
            f.write("Assignment 9 Part 1 Task 1\n")
            f.write("HST data at epoch\n")
            f.write("=" * 90 + "\n")

            print("\n=== Assignment 9 Part 1 Task 1 ===")
            print("HST data at epoch")
            print("=" * 90)

            for name, value, unit in self.rows:
                if isinstance(value, np.ndarray):
                    value_string = np.array2string(value, precision=12, suppress_small=False)
                else:
                    value_string = str(value)

                line = "{:<42s}: {} {}".format(name, value_string, unit)

                print(line)
                f.write(line + "\n")

        print("=" * 90)
        print("      saved:", file_name)


class Part1Task2(sim.BaseScenario):
    def __init__(self, tles):
        self.tles = tles
        self.orbit = None
        self.JD0 = None
        self.altitude_log = []
        self.ground_track = []
        self.r_i = np.zeros(3)
        self.q_E = su.Quaternion()

    def init(self, t):
        current = self.tles[CURRENT_TLE]
        self.orbit = ol.orbit_pkepler_from_tle(current)
        self.JD0 = ol.epoch_to_julian_date(current["epoch"])
        self.altitude_log = []
        self.ground_track = []
        self.r_i, _ = self.orbit.get_state()
        self.q_E = su.Quaternion(ol.sidereal_angle(self.JD0), np.array([0.0, 0.0, 1.0]))

    def update(self, t, dt):
        r_i, v_i = self.orbit.get_state()
        h, e, theta, raan, inc, arg_perigee = ol.orbit_params_from_state(r_i, v_i)
        a = h**2 / (ol.mu * (1.0 - e**2))
        theta_E = ol.sidereal_angle(self.JD0 + t / ol.SECONDS_IN_DAY)
        self.r_i = r_i
        self.q_E = su.Quaternion(theta_E, np.array([0.0, 0.0, 1.0]))
        lon, lat, geodetic_altitude = ol.geodetic_from_eci(r_i, theta_E)
        radial_altitude = np.linalg.norm(r_i) - ol.R_E

        self.altitude_log.append([t / ol.SECONDS_IN_DAY, radial_altitude, geodetic_altitude, a, e])
        self.ground_track.append([t, lon, lat, geodetic_altitude])

        self.orbit.propagate(dt)

    def get(self):
        return [
            ["satellite", self.r_i, su.Quaternion()],
            ["body frame", self.r_i, su.Quaternion()],
            ["earth", np.zeros(3), self.q_E],
            ["ECEF frame", np.zeros(3), self.q_E],
            ["ECI frame", np.zeros(3), su.Quaternion()]
        ]

    def post_process(self, t, dt):
        altitude_log = np.asarray(self.altitude_log)
        ground_track = np.asarray(self.ground_track)

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

        print("      final radial altitude:   {:.3f} km".format(altitude_log[-1, 1]))
        print("      final geodetic altitude: {:.3f} km".format(altitude_log[-1, 2]))
        print("      saved:", altitude_file)
        print("      saved:", ground_file)


class Part1Task2OldTLE(sim.BaseScenario):
    def __init__(self, tles, old_tle_name):
        self.tles = tles
        self.old_tle_name = old_tle_name
        self.old_orbit = None
        self.current_orbit = None
        self.current_JD = None
        self.error_rows = []
        self.diff_log = []
        self.gt_current = []
        self.gt_old = []
        self.r_i = np.zeros(3)
        self.q_E = su.Quaternion()

    def init(self, t):
        old = self.tles[self.old_tle_name]
        current = self.tles[CURRENT_TLE]

        self.old_orbit = ol.orbit_pkepler_from_tle(old)
        self.current_orbit = ol.orbit_pkepler_from_tle(current)

        old_JD = ol.epoch_to_julian_date(old["epoch"])
        self.current_JD = ol.epoch_to_julian_date(current["epoch"])
        delta_epoch = (self.current_JD - old_JD) * ol.SECONDS_IN_DAY

        tau = 0.0
        while tau < delta_epoch:
            h_step = min(60.0, delta_epoch - tau)
            self.old_orbit.propagate(h_step)
            tau += h_step

        r_old, v_old = self.old_orbit.get_state()
        r_current, v_current = self.current_orbit.get_state()

        h_old, e_old, theta_old, raan_old, inc_old, arg_old = ol.orbit_params_from_state(r_old, v_old)
        h_cur, e_cur, theta_cur, raan_cur, inc_cur, arg_cur = ol.orbit_params_from_state(r_current, v_current)

        a_old = h_old**2 / (ol.mu * (1.0 - e_old**2))
        a_cur = h_cur**2 / (ol.mu * (1.0 - e_cur**2))

        self.error_rows = [
            ["Old propagated TLE", self.old_tle_name],
            ["Current TLE ground truth", CURRENT_TLE],
            ["Epoch separation days", "{:.9f}".format(delta_epoch / ol.SECONDS_IN_DAY)],
            ["Position error vector km", r_old - r_current],
            ["Velocity error vector km/s", v_old - v_current],
            ["Position error norm km", "{:.9f}".format(np.linalg.norm(r_old - r_current))],
            ["Velocity error norm km/s", "{:.12f}".format(np.linalg.norm(v_old - v_current))],
            ["Semi-major axis error km", "{:.9f}".format(a_old - a_cur)],
            ["Eccentricity error", "{:.12e}".format(e_old - e_cur)],
            ["Inclination error deg", "{:.9f}".format(ol.angle_error(inc_old - inc_cur) * ol.RTOD)],
            ["RAAN error deg", "{:.9f}".format(ol.angle_error(raan_old - raan_cur) * ol.RTOD)],
            ["Argument of perigee error deg", "{:.9f}".format(ol.angle_error(arg_old - arg_cur) * ol.RTOD)],
            ["True anomaly error deg", "{:.9f}".format(ol.angle_error(theta_old - theta_cur) * ol.RTOD)]
        ]

        self.diff_log = []
        self.gt_current = []
        self.gt_old = []
        self.r_i, _ = self.current_orbit.get_state()
        self.q_E = su.Quaternion(ol.sidereal_angle(self.current_JD), np.array([0.0, 0.0, 1.0]))

    def update(self, t, dt):
        r_c, v_c = self.current_orbit.get_state()
        r_o, v_o = self.old_orbit.get_state()
        theta_E = ol.sidereal_angle(self.current_JD + t / ol.SECONDS_IN_DAY)
        self.r_i = r_c
        self.q_E = su.Quaternion(theta_E, np.array([0.0, 0.0, 1.0]))

        lon_c, lat_c, alt_c = ol.geodetic_from_eci(r_c, theta_E)
        lon_o, lat_o, alt_o = ol.geodetic_from_eci(r_o, theta_E)

        self.diff_log.append([t, np.linalg.norm(r_o - r_c), np.linalg.norm(v_o - v_c)])
        self.gt_current.append([t, lon_c, lat_c, alt_c])
        self.gt_old.append([t, lon_o, lat_o, alt_o])

        self.current_orbit.propagate(dt)
        self.old_orbit.propagate(dt)

    def get(self):
        return [
            ["satellite", self.r_i, su.Quaternion()],
            ["body frame", self.r_i, su.Quaternion()],
            ["earth", np.zeros(3), self.q_E],
            ["ECEF frame", np.zeros(3), self.q_E],
            ["ECI frame", np.zeros(3), su.Quaternion()]
        ]

    def post_process(self, t, dt):
        diff_log = np.asarray(self.diff_log)
        gt_current = np.asarray(self.gt_current)
        gt_old = np.asarray(self.gt_old)

        error_file = os.path.join(DATA_DIR, "assignment9_task2_{}_error.txt".format(self.old_tle_name.lower()))

        with open(error_file, "w") as f:
            for name, value in self.error_rows:
                f.write("{}: {}\n".format(name, value))

        np.savetxt(
            os.path.join(DATA_DIR, "assignment9_task2_{}_one_orbit_position_difference.txt".format(self.old_tle_name.lower())),
            diff_log,
            header="time_s position_difference_km velocity_difference_km_s"
        )
        np.savetxt(
            os.path.join(DATA_DIR, "assignment9_task2_ground_track_current.txt"),
            gt_current,
            header="time_s longitude_rad latitude_rad altitude_km"
        )
        np.savetxt(
            os.path.join(DATA_DIR, "assignment9_task2_ground_track_{}_propagated.txt".format(self.old_tle_name.lower())),
            gt_old,
            header="time_s longitude_rad latitude_rad altitude_km"
        )

        pl.plot_assignment9_position_difference(
            diff_log,
            self.old_tle_name,
            "assignment9_task2_{}_one_orbit_position_difference.png".format(self.old_tle_name.lower()),
            PLOT_DIR
        )
        pl.plot_assignment9_ground_tracks(
            gt_current,
            gt_old,
            self.old_tle_name,
            EARTH_IMAGE,
            "assignment9_task2_{}_ground_tracks.png".format(self.old_tle_name.lower()),
            PLOT_DIR
        )

        print("      {} max one-orbit position difference: {:.3f} km".format(
            self.old_tle_name,
            np.max(diff_log[:, 1])
        ))
        print("      saved:", error_file)


class Part2Case(sim.BaseScenario):
    def __init__(self, tles, label, controller, star_trackers):
        self.tles = tles
        self.label = label
        self.controller = controller
        self.star_trackers = star_trackers
        self.orbit = None
        self.JD0 = None
        self.q_ib = Q_IB_0.copy()
        self.w_b_ib = W_B_IB_0.copy()
        self.q_E = su.Quaternion()
        self.r_i = np.zeros(3)
        self.attitude_log = []
        self.rng = np.random.default_rng(RANDOM_SEED)

    def init(self, t):
        tle = self.tles[CURRENT_TLE]
        self.JD0 = ol.epoch_to_julian_date(tle["epoch"])
        self.orbit = ol.orbit_pkepler_from_tle(tle)
        self.q_ib = Q_IB_0.copy()
        self.w_b_ib = W_B_IB_0.copy()
        self.q_E = su.Quaternion(ol.sidereal_angle(self.JD0), np.array([0.0, 0.0, 1.0]))
        self.r_i, _ = self.orbit.get_state()
        self.attitude_log = []
        self.rng = np.random.default_rng(RANDOM_SEED)

    def update(self, t, dt):
        r_i, v_i = self.orbit.get_state()

        self.q_ib, self.w_b_ib, row = sl.attitude_step(
            t, dt, r_i, self.q_ib, self.w_b_ib, Q_ID, HST_J, HST_J_INV,
            self.controller, self.star_trackers, self.rng, ACTUATOR_LIMIT,
            GYRO_MU, GYRO_VARIANCE, STAR_MU, STAR_Q,
            PD_K1, PD_K2, SM_K1, SM_K, SM_EPS,
            SOLAR_A1, SOLAR_A2, SOLAR_P1, SOLAR_P2, SOLAR_PHI1, SOLAR_PHI2
        )

        self.r_i = r_i
        self.attitude_log.append(row)
        self.orbit.propagate(dt)
        if VISUALISE:
            theta_E = ol.sidereal_angle(self.JD0 + (t + dt) / ol.SECONDS_IN_DAY)
            self.q_E = su.Quaternion(theta_E, np.array([0.0, 0.0, 1.0]))

    def get(self):
        return [
            ["satellite", self.r_i, su.Quaternion(self.q_ib)],
            ["body frame", self.r_i, su.Quaternion(self.q_ib)],
            ["earth", np.zeros(3), self.q_E],
            ["ECEF frame", np.zeros(3), self.q_E],
            ["ECI frame", np.zeros(3), su.Quaternion()]
        ]

    def post_process(self, t, dt):
        data = np.asarray(self.attitude_log)
        file_name = os.path.join(DATA_DIR, "assignment9_part2_{}.txt".format(self.label))

        np.savetxt(file_name, data, header=sl.part2_header())

        steady = data[data[:, COL_TIME] >= data[-1, COL_TIME] / 4.0]
        rms = np.sqrt(np.mean(steady[:, COL_TRUE_ARCSEC]**2))

        print("      {} final = {:.3f} arcsec, RMS after first orbit = {:.3f} arcsec".format(
            self.label,
            data[-1, COL_TRUE_ARCSEC],
            rms
        ))


def make_output_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)


def run_scenario(scenario, t_end, t_step):
    sim_config = {
        "t_0": 0.0,
        "t_e": t_end,
        "t_step": t_step,
        "speed_factor": 100 if VISUALISE else 1.0e12,
        "anim_dt": 1.0 / 25.0,
        "scale_factor": 1000,
        "visualise": VISUALISE
    }
    sim.create_and_start_simulation(sim_config, scenario)


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


def get_part2_cases():
    cases = []

    if RUN_PD_1ST:
        cases.append(["PD_1ST", "PD", 1])

    if RUN_PD_3ST:
        cases.append(["PD_3ST", "PD", 3])

    if RUN_SM_1ST:
        cases.append(["SM_1ST", "SM", 1])

    if RUN_SM_3ST:
        cases.append(["SM_3ST", "SM", 3])

    return cases


def main():
    make_output_dirs()
    tles = ol.read_tles(TLE_FILE)

    print("=== Assignment 9 ===")

    if RUN_PART_1:
        print("[1/5] Running Part 1 Task 1")
        run_scenario(Part1Task1(tles), 0.0, 1.0)

        print("[2/5] Running Part 1 Task 2")
        run_scenario(Part1Task2(tles), 9.0 * 365.25 * ol.SECONDS_IN_DAY, 6.0 * 3600.0)

        current = tles[CURRENT_TLE]
        period = ol.SECONDS_IN_DAY / current["revs_per_day"]
        run_scenario(Part1Task2OldTLE(tles, OLD_TLE), period + 10.0, 10.0)

    if RUN_PART_2:
        current = tles[CURRENT_TLE]
        period = ol.SECONDS_IN_DAY / current["revs_per_day"]

        cases = get_part2_cases()

        if len(cases) == 0:
            print("No Part 2 cases selected.")
        else:
            total_steps = len(cases)

            for idx, (label, controller, trackers) in enumerate(cases, start=1):
                print("[{}/{}] Running Part 2 case {}".format(idx, total_steps, label))
                run_scenario(Part2Case(tles, label, controller, trackers), 4.0 * period, ATTITUDE_DT)

            make_part2_summary_and_plots(cases)

    print("Done.")
    print("Data folder:", DATA_DIR)
    print("Plot folder:", PLOT_DIR)


if __name__ == "__main__":
    main()