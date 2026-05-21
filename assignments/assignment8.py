import os
import datetime as dt
import numpy as np

import simutils as su
import simulator as sim
import plotter as pl
import orbit_lib as ol
import sat_lib as sl


VISUALISE = True
USE_CURRENT_TIME = False
NOISE_SCALE = 0.004
ATTITUDE_ESTIMATOR = "Davenport"
DISPLAY_CONTROLLER = "SM"

PD_K1 = 0.05
PD_K2 = 0.5
SM_K1 = 0.05
SM_K = 0.03
SM_EPS = 0.02


class ScenarioAssignment8(sim.BaseScenario):

    def __init__(self):
        self.ground_track = None

    def init(self, t):
        np.random.seed(8)

        self.tle = ol.read_tle_file("Assignment5_TLE.txt")
        self.name = self.tle['name']
        self.JD_epoch = ol.epoch_to_julian_date(self.tle['epoch'])

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

            self.delta_t_epoch_to_now = (self.JD_now - self.JD_epoch) * ol.SECONDS_IN_DAY
        else:
            self.JD_now = self.JD_epoch
            self.delta_t_epoch_to_now = 0.0

        self.q_E = su.Quaternion(ol.sidereal_angle(self.JD_now), np.array([0.0, 0.0, 1.0]))

        self.J = np.array([
            [0.00146519,  0.00001703, -0.00000633],
            [0.00001703,  0.00151512, -0.00001598],
            [-0.00000633, -0.00001598,  0.00146333]
        ])

        if ATTITUDE_ESTIMATOR == "TRIAD":
            estimator_pd = sl.TRIAD()
            estimator_sm = sl.TRIAD()
        else:
            estimator_pd = sl.Davenport()
            estimator_sm = sl.Davenport()

        self.orbit_pd = self.make_orbit()
        self.orbit_sm = self.make_orbit()

        r_pd0, v_pd0 = self.orbit_pd.get_state()
        q_io_pd0, w_i_io_pd0, _ = ol.orbit_frame_from_state(r_pd0, v_pd0)
        w_b_ib_pd0 = sl._q_rotate_inverse(q_io_pd0, w_i_io_pd0)

        r_sm0, v_sm0 = self.orbit_sm.get_state()
        q_io_sm0, w_i_io_sm0, _ = ol.orbit_frame_from_state(r_sm0, v_sm0)
        w_b_ib_sm0 = sl._q_rotate_inverse(q_io_sm0, w_i_io_sm0)

        self.sat_pd = sl.Satellite(
            q_ib=q_io_pd0,
            w_b_ib=w_b_ib_pd0,
            J=self.J,
            orbit=self.orbit_pd,
            substeps=50,
            JD0=self.JD_now,
            use_sensors=True,
            use_disturbances=True,
            noise_scale=NOISE_SCALE,
            attitude_estimator=estimator_pd,
            controller='PD',
            k1=PD_K1,
            k2=PD_K2
        )

        self.sat_sm = sl.Satellite(
            q_ib=q_io_sm0,
            w_b_ib=w_b_ib_sm0,
            J=self.J,
            orbit=self.orbit_sm,
            substeps=50,
            JD0=self.JD_now,
            use_sensors=True,
            use_disturbances=True,
            noise_scale=NOISE_SCALE,
            attitude_estimator=estimator_sm,
            controller='SM',
            k1=SM_K1,
            sm_k=SM_K,
            sm_eps=SM_EPS
        )

        self.k_lower_bound = self.sliding_mode_lower_bound()

        self.comparison_plot = self.comparison_log_row(t)
        self.torque_plot = self.torque_log_row(t)
        self.disturbance_plot = self.disturbance_log_row(t)
        self.ground_track = self.ground_track_log_row(t)

    def make_orbit(self):
        orbit = ol.orbit_pkepler(
            n=self.tle['revs_per_day'],
            e=self.tle['e'],
            M_e=self.tle['mean_anomaly'],
            O=self.tle['raan'],
            i=self.tle['i'],
            w=self.tle['arg_perigee'],
            dn=self.tle['dn'],
            ddn=self.tle['ddn'],
            bstar=self.tle['bstar'],
            tle_units=True
        )

        if abs(self.delta_t_epoch_to_now) > 0.0:
            orbit.propagate(self.delta_t_epoch_to_now)

        return orbit

    def sliding_mode_lower_bound(self):
        rp = self.orbit_sm.a * (1.0 - self.orbit_sm.e)
        gravity_gradient_bound = 3.0 * ol.mu / rp**3
        sinusoidal_disturbance_bound = 0.01

        return gravity_gradient_bound + sinusoidal_disturbance_bound

    def eci_to_ecef(self, r_i):
        q_conj = su.Quaternion(self.q_E)
        q_conj.conjugate()
        return q_conj.rotate(r_i)

    def true_tracking_error(self, satellite):
        r, v, q_ib, w_b_ib = satellite.get_state()
        q_io, w_i_io, _ = satellite.get_orbit_frame()

        q_ob = sl._q_mul(sl._q_conj(q_io), q_ib)
        q_ob = q_ob / np.linalg.norm(q_ob)

        if q_ob[0] < 0.0:
            q_ob = -q_ob

        w_b_io = sl._q_rotate_inverse(q_ib, w_i_io)
        w_b_ob = w_b_ib - w_b_io

        return q_ob, w_b_ob

    def ground_track_log_row(self, t):
        if DISPLAY_CONTROLLER == "PD":
            r, v, q, wb = self.sat_pd.get_state()
        else:
            r, v, q, wb = self.sat_sm.get_state()

        r_ecef = self.eci_to_ecef(r)
        phi, lam, h = ol.geodetic_from_xyz(r_ecef)
        return np.array([phi, lam, h])

    def comparison_log_row(self, t):
        q_pd, w_pd = self.true_tracking_error(self.sat_pd)
        q_sm, w_sm = self.true_tracking_error(self.sat_sm)

        s_sm = self.sat_sm.ADCS.sliding_surface

        return np.array([
            t,
            np.linalg.norm(q_pd[1:]),
            np.linalg.norm(q_sm[1:]),
            np.linalg.norm(w_pd),
            np.linalg.norm(w_sm),
            np.linalg.norm(self.sat_pd.ADCS.angular_velocity_error),
            np.linalg.norm(self.sat_sm.ADCS.angular_velocity_error),
            np.linalg.norm(s_sm)
        ])

    def torque_log_row(self, t):
        tau_pd = self.sat_pd.torque
        tau_sm = self.sat_sm.torque

        return np.array([
            t,
            tau_pd[0], tau_pd[1], tau_pd[2], np.linalg.norm(tau_pd),
            tau_sm[0], tau_sm[1], tau_sm[2], np.linalg.norm(tau_sm)
        ])

    def disturbance_log_row(self, t):
        return np.array([
            t,
            np.linalg.norm(self.sat_pd.tau_g),
            np.linalg.norm(self.sat_pd.tau_extra),
            np.linalg.norm(self.sat_pd.tau_d),
            np.linalg.norm(self.sat_sm.tau_g),
            np.linalg.norm(self.sat_sm.tau_extra),
            np.linalg.norm(self.sat_sm.tau_d)
        ])

    def update(self, t, dt_step):
        t_next = t + dt_step

        self.sat_pd.update(t, dt_step)
        self.sat_sm.update(t, dt_step)
        self.q_E = su.Quaternion(ol.sidereal_angle(self.JD_now + t_next / ol.SECONDS_IN_DAY), np.array([0.0, 0.0, 1.0]))

        self.comparison_plot = np.vstack((self.comparison_plot, self.comparison_log_row(t_next)))
        self.torque_plot = np.vstack((self.torque_plot, self.torque_log_row(t_next)))
        self.disturbance_plot = np.vstack((self.disturbance_plot, self.disturbance_log_row(t_next)))
        self.ground_track = np.vstack((self.ground_track, self.ground_track_log_row(t_next)))

    def get(self):
        if DISPLAY_CONTROLLER == "PD":
            r, v, q, wb = self.sat_pd.get_state()
        else:
            r, v, q, wb = self.sat_sm.get_state()

        return [
            ['satellite', r, q],
            ['body frame', r, q],
            ['earth', np.zeros(3), self.q_E],
            ['ECI frame', np.zeros(3), su.Quaternion()],
            ['ECEF frame', np.zeros(3), self.q_E]
        ]

    def post_process(self, t, dt_step):
        os.makedirs("data", exist_ok=True)

        su.log_pos("assignment8_pd_vs_sm_tracking", self.comparison_plot)
        su.log_pos("assignment8_pd_vs_sm_control_torque", self.torque_plot)
        su.log_pos("assignment8_disturbance_torques", self.disturbance_plot)
        su.log_pos("assignment8_ground_track", self.ground_track)

        q_pd, w_pd = self.true_tracking_error(self.sat_pd)
        q_sm, w_sm = self.true_tracking_error(self.sat_sm)

        print()
        print("Assignment 8 final comparison:")
        print("TLE:                         {}".format(self.name))
        print("Estimator:                   {}".format(ATTITUDE_ESTIMATOR))
        print("Noise scale:                 {:.6f}".format(NOISE_SCALE))
        print("PD gains:                    k1 = {:.4f}, k2 = {:.4f}".format(PD_K1, PD_K2))
        print("SM gains:                    k1 = {:.4f}, k = {:.4f}, eps = {:.4f}".format(SM_K1, SM_K, SM_EPS))
        print("SM lower-bound estimate:     k > {:.6e}".format(self.k_lower_bound))
        print("PD true |qv|:                {:.6e}".format(np.linalg.norm(q_pd[1:])))
        print("SM true |qv|:                {:.6e}".format(np.linalg.norm(q_sm[1:])))
        print("PD true |w_ob_b|:            {:.6e}".format(np.linalg.norm(w_pd)))
        print("SM true |w_ob_b|:            {:.6e}".format(np.linalg.norm(w_sm)))
        print("PD final disturbance torque: {:.6e} Nm".format(np.linalg.norm(self.sat_pd.tau_d)))
        print("SM final disturbance torque: {:.6e} Nm".format(np.linalg.norm(self.sat_sm.tau_d)))
        print()

        pl.line_plot("data/assignment8_pd_vs_sm_tracking.txt")
        pl.line_plot("data/assignment8_pd_vs_sm_control_torque.txt")
        pl.line_plot("data/assignment8_disturbance_torques.txt")
        pl.plot_ground_track(
            "data/assignment8_ground_track.txt",
            img_path="earth_grid.jpg",
            save_path="data/assignment8_ground_track.png"
        )


def main():
    scenario = ScenarioAssignment8()

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
