import os
import datetime as dt
import numpy as np

import simutils as su
import simulator as sim
import plotter as pl
import orbit_lib as ol
import sat_lib as sl


VISUALISE = True
USE_CURRENT_TIME = True
NOISE_SCALE = 0.004
ATTITUDE_ESTIMATOR = "Davenport"    # Change to "TRIAD" if you want to compare estimators

class ScenarioAssignment7(sim.BaseScenario):

    def __init__(self):
        self.ground_track = None

    def init(self, t):
        np.random.seed(7)

        tle = ol.read_tle_file("Assignment5_TLE.txt")

        name = tle['name']
        epoch = tle['epoch']
        e = tle['e']
        revs_per_day = tle['revs_per_day']
        M_e = tle['mean_anomaly']
        O = tle['raan']
        inc = tle['i']
        w = tle['arg_perigee']
        dn = tle['dn']
        ddn = tle['ddn']
        bstar = tle['bstar']

        self.JD_epoch = ol.epoch_to_julian_date(epoch)

        if USE_CURRENT_TIME:
            tnow = dt.datetime.now(dt.timezone.utc)
            second = tnow.second + tnow.microsecond * 1e-6

            self.JD_now = ol.datetime_to_julian_date(
                tnow.year,
                tnow.month,
                tnow.day,
                tnow.hour,
                tnow.minute,
                second
            )

            self.delta_t_epoch_to_now = (self.JD_now - self.JD_epoch) * 24.0 * 3600.0
        else:
            self.JD_now = self.JD_epoch
            self.delta_t_epoch_to_now = 0.0

        self.q_E = su.Quaternion(ol.sidereal_angle(self.JD_now), np.array([0.0, 0.0, 1.0]))

        self.orbit = ol.orbit_pkepler(
            n=revs_per_day,
            e=e,
            M_e=M_e,
            O=O,
            i=inc,
            w=w,
            dn=dn,
            ddn=ddn,
            bstar=bstar,
            tle_units=True
        )

        if abs(self.delta_t_epoch_to_now) > 0.0:
            self.orbit.propagate(self.delta_t_epoch_to_now)

        self.J = np.array([
            [0.00146519,  0.00001703, -0.00000633],
            [0.00001703,  0.00151512, -0.00001598],
            [-0.00000633, -0.00001598,  0.00146333]
        ])

        if ATTITUDE_ESTIMATOR == "TRIAD":
            estimator = sl.TRIAD()
        else:
            estimator = sl.Davenport()

        self.satellite = sl.Satellite(
            q_ib=np.array([1.0, 0.0, 0.0, 0.0]),
            w_b_ib=np.array([0.0, 0.0, 0.0]),
            J=self.J,
            orbit=self.orbit,
            substeps=50,
            JD0=self.JD_now,
            use_sensors=True,
            noise_scale=NOISE_SCALE,
            attitude_estimator=estimator
        )

        self.name = name
        self.position_plot = self.position_log_row(t)
        self.error_plot = self.error_log_row(t)
        self.estimate_error_plot = self.estimate_error_log_row(t)
        self.omega_plot = self.omega_log_row(t)
        self.torque_plot = self.torque_log_row(t)
        self.sensor_plot = self.sensor_log_row(t)
        self.ground_track = self.ground_track_log_row(t)

    def eci_to_ecef(self, r_i):
        q_conj = su.Quaternion(self.q_E)
        q_conj.conjugate()
        return q_conj.rotate(r_i)

    def ground_track_log_row(self, t):
        r, v, q, wb = self.satellite.get_state()
        r_ecef = self.eci_to_ecef(r)
        phi, lam, h = ol.geodetic_from_xyz(r_ecef)
        return np.array([phi, lam, h])

    def position_log_row(self, t):
        r, v, q, wb = self.satellite.get_state()
        return np.array([t, r[0], r[1], r[2], v[0], v[1], v[2]])

    def error_log_row(self, t):
        q_ob = self.satellite.ADCS.attitude_error.q
        w_ob_b = self.satellite.ADCS.angular_velocity_error

        return np.array([
            t,
            q_ob[0], q_ob[1], q_ob[2], q_ob[3],
            np.linalg.norm(q_ob[1:]),
            w_ob_b[0], w_ob_b[1], w_ob_b[2],
            np.linalg.norm(w_ob_b)
        ])

    def estimate_error_log_row(self, t):
        r, v, q_ib, wb = self.satellite.get_state()
        q_io, w_i_io, _ = self.satellite.get_orbit_frame()
        q_ob_true = sl._q_mul(sl._q_conj(q_io), q_ib)
        q_ob_true = q_ob_true / np.linalg.norm(q_ob_true)
        q_ob_est = self.satellite.ADCS.q_ob_est
        q_est_error = sl._q_mul(sl._q_conj(q_ob_true), q_ob_est)
        q_est_error = q_est_error / np.linalg.norm(q_est_error)

        if q_est_error[0] < 0.0:
            q_est_error = -q_est_error

        return np.array([
            t,
            q_est_error[0], q_est_error[1], q_est_error[2], q_est_error[3],
            np.linalg.norm(q_est_error[1:])
        ])

    def omega_log_row(self, t):
        r, v, q, wb = self.satellite.get_state()
        q_io, w_i_io, _ = self.satellite.get_orbit_frame()
        meas = self.satellite.get_sensor_measurements()
        gyro_m = meas['gyro']

        return np.array([
            t,
            wb[0], wb[1], wb[2],
            gyro_m[0], gyro_m[1], gyro_m[2],
            w_i_io[0], w_i_io[1], w_i_io[2]
        ])

    def torque_log_row(self, t):
        tau = self.satellite.torque
        return np.array([t, tau[0], tau[1], tau[2]])

    def sensor_log_row(self, t):
        meas = self.satellite.get_sensor_measurements()
        gyro_m = meas['gyro']
        mag_m = meas['magnetometer']
        sun_valid = sum(1 for s in meas['sun'] if np.linalg.norm(s) > 1e-12)
        sun_b = self.satellite.ADCS.sun_body
        mag_b = self.satellite.ADCS.mag_body

        return np.array([
            t,
            gyro_m[0], gyro_m[1], gyro_m[2],
            mag_m[0], mag_m[1], mag_m[2],
            sun_b[0], sun_b[1], sun_b[2],
            mag_b[0], mag_b[1], mag_b[2],
            sun_valid
        ])

    def update(self, t, dt_step):
        t_next = t + dt_step

        self.satellite.update(t, dt_step)
        self.q_E = su.Quaternion(ol.sidereal_angle(self.JD_now + t_next / 86400.0), np.array([0.0, 0.0, 1.0]))

        self.position_plot = np.vstack((self.position_plot, self.position_log_row(t_next)))
        self.error_plot = np.vstack((self.error_plot, self.error_log_row(t_next)))
        self.estimate_error_plot = np.vstack((self.estimate_error_plot, self.estimate_error_log_row(t_next)))
        self.omega_plot = np.vstack((self.omega_plot, self.omega_log_row(t_next)))
        self.torque_plot = np.vstack((self.torque_plot, self.torque_log_row(t_next)))
        self.sensor_plot = np.vstack((self.sensor_plot, self.sensor_log_row(t_next)))
        self.ground_track = np.vstack((self.ground_track, self.ground_track_log_row(t_next)))

    def get(self):
        r, v, q, wb = self.satellite.get_state()

        return [
            ['satellite', r, q],
            ['body frame', r, q],
            ['earth', np.zeros(3), self.q_E],
            ['ECI frame', np.zeros(3), su.Quaternion()],
            ['ECEF frame', np.zeros(3), self.q_E]
        ]

    def post_process(self, t, dt_step):
        os.makedirs("data", exist_ok=True)

        su.log_pos("assignment7_position_velocity", self.position_plot)
        su.log_pos("assignment7_tracking_error_from_estimate", self.error_plot)
        su.log_pos("assignment7_attitude_estimation_error", self.estimate_error_plot)
        su.log_pos("assignment7_angular_velocity_and_gyro", self.omega_plot)
        su.log_pos("assignment7_control_torque", self.torque_plot)
        su.log_pos("assignment7_sensor_measurements", self.sensor_plot)
        su.log_pos("assignment7_ground_track", self.ground_track)

        r, v, q, wb = self.satellite.get_state()
        q_io, w_i_io, _ = self.satellite.get_orbit_frame()
        q_ob = self.satellite.ADCS.attitude_error.q
        q_ob_est = self.satellite.ADCS.q_ob_est
        w_ob_b = self.satellite.ADCS.angular_velocity_error
        phi, lam, h = self.ground_track[-1]
        q_est_err = self.estimate_error_plot[-1, 1:5]

        print()
        print("Assignment 7 final state:")
        print("TLE:               {}".format(self.name))
        print("Estimator:         {}".format(ATTITUDE_ESTIMATOR))
        print("Noise scale:       {:.6f}".format(NOISE_SCALE))
        print("Delta epoch-now:   {:.3f} s".format(self.delta_t_epoch_to_now))
        print("r_i:               [{:.6f}, {:.6f}, {:.6f}] km".format(*r))
        print("v_i:               [{:.6f}, {:.6f}, {:.6f}] km/s".format(*v))
        print("q_ib true:         [{:.6f}, {:.6f}, {:.6f}, {:.6f}]".format(*q))
        print("q_ob estimated:    [{:.6f}, {:.6f}, {:.6f}, {:.6f}]".format(*q_ob_est))
        print("q_ob control err:  [{:.6f}, {:.6f}, {:.6f}, {:.6f}]".format(*q_ob))
        print("q estimate error:  [{:.6f}, {:.6f}, {:.6f}, {:.6f}]".format(*q_est_err))
        print("|qv est error|:    {:.6e}".format(self.estimate_error_plot[-1, 5]))
        print("w_b_ib true:       [{:.6e}, {:.6e}, {:.6e}] rad/s".format(*wb))
        print("w_i_io:            [{:.6e}, {:.6e}, {:.6e}] rad/s".format(*w_i_io))
        print("|w error|:         {:.6e}".format(np.linalg.norm(w_ob_b)))
        print("lon, lat, h:       [{:.6f}, {:.6f}, {:.6f}] deg, deg, km".format(phi * ol.RTOD, lam * ol.RTOD, h))
        print()

        pl.line_plot("data/assignment7_tracking_error_from_estimate.txt")
        pl.line_plot("data/assignment7_attitude_estimation_error.txt")
        pl.line_plot("data/assignment7_angular_velocity_and_gyro.txt")
        pl.line_plot("data/assignment7_control_torque.txt")
        pl.line_plot("data/assignment7_sensor_measurements.txt")
        pl.plot_ground_track(
            "data/assignment7_ground_track.txt",
            img_path="earth_grid.jpg",
            save_path="data/assignment7_ground_track.png"
        )


def main():
    scenario = ScenarioAssignment7()

    sim_config = {
        't_0': 0,
        't_e': 5731,
        't_step': 2,
        'speed_factor': 1,
        'anim_dt': 1.0 / 25.0,
        'scale_factor': 1000,
        'visualise': VISUALISE
    }

    sim.create_and_start_simulation(sim_config, scenario)


if __name__ == "__main__":
    main()
