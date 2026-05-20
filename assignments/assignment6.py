import datetime as dt
import os
import numpy as np

import simutils as su
import simulator as sim
import plotter as pl
import orbit_lib as ol
import sat_lib as sl


VISUALISE = True

class ScenarioAssignment6(sim.BaseScenario):

    def __init__(self):
        self.ground_track = None

    def init(self, t):
        tle = su.read_TLE_file("Assignment5_TLE.txt", "HINCUBE")[0]
        name, epoch, e, rev, Me, inc, O, w, dn, ddn, bstar = tle

        self.J = np.array([
            [0.00146519, 0.00001703, -0.00000633],
            [0.00001703, 0.00151512, -0.00001598],
            [-0.00000633, -0.00001598, 0.00146333]
        ])

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

        self.JD_epoch = ol.epoch_to_julian_date(epoch)
        self.delta_t_epoch_to_now = (self.JD_now - self.JD_epoch) * 24.0 * 3600.0

        self.orbit = ol.orbit_pkepler(
            rev,
            e,
            Me * ol.DTOR,
            O * ol.DTOR,
            inc * ol.DTOR,
            w * ol.DTOR,
            dn,
            ddn,
            bstar
        )

        self.orbit.propagate(self.delta_t_epoch_to_now)
        self.q_E = su.Quaternion(ol.sidereal_angle(self.JD_now), np.array([0.0, 0.0, 1.0]))

        self.satellite = sl.Satellite(
            q_ib=np.array([1.0, 0.0, 0.0, 0.0]),
            w_b_ib=np.array([0.0, 0.0, 0.0]),
            J=self.J,
            orbit=self.orbit,
            substeps=50
        )

        r, v, q, wb = self.satellite.get_state()
        q_io, w_i_io, _ = self.satellite.get_orbit_frame()
        self.satellite.ADCS.update(q, wb, q_io, w_i_io)
        self.satellite.torque = self.satellite.ADCS.get_control()

        self.position_plot = self.position_log_row(t)
        self.error_plot = self.error_log_row(t)
        self.omega_plot = self.omega_log_row(t)
        self.torque_plot = self.torque_log_row(t)
        self.ground_track = self.ground_track_log_row(t).reshape(1, -1)

    def ground_track_log_row(self, t):
        r_i, v_i, q, wb = self.satellite.get_state()

        q_conj = su.Quaternion(self.q_E)
        q_conj.conjugate()
        r_E = q_conj.rotate(r_i)

        phi, lam, h = ol.geodetic_from_xyz(r_E)

        return np.array([phi, lam, h])

    def position_log_row(self, t):
        r, v, q, wb = self.satellite.get_state()
        params = self.orbit.get_params()

        return np.array([
            t,
            r[0], r[1], r[2],
            v[0], v[1], v[2],
            np.linalg.norm(r),
            np.linalg.norm(v),
            params[0], params[1], params[2], params[3], params[5]
        ])

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

    def omega_log_row(self, t):
        r, v, q, wb = self.satellite.get_state()
        q_io, w_i_io, _ = self.satellite.get_orbit_frame()

        return np.array([
            t,
            wb[0], wb[1], wb[2],
            w_i_io[0], w_i_io[1], w_i_io[2]
        ])

    def torque_log_row(self, t):
        tau = self.satellite.torque

        return np.array([
            t,
            tau[0], tau[1], tau[2]
        ])

    def update(self, t, dt_step):
        t_next = t + dt_step

        self.satellite.update(t, dt_step)

        JD = self.JD_now + t_next / (24.0 * 3600.0)
        self.q_E = su.Quaternion(ol.sidereal_angle(JD), np.array([0.0, 0.0, 1.0]))

        self.ground_track = np.vstack((self.ground_track, self.ground_track_log_row(t_next)))
        self.position_plot = np.vstack((self.position_plot, self.position_log_row(t_next)))
        self.error_plot = np.vstack((self.error_plot, self.error_log_row(t_next)))
        self.omega_plot = np.vstack((self.omega_plot, self.omega_log_row(t_next)))
        self.torque_plot = np.vstack((self.torque_plot, self.torque_log_row(t_next)))

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

        su.log_pos("assignment6_position_velocity", self.position_plot)
        su.log_pos("assignment6_tracking_error", self.error_plot)
        su.log_pos("assignment6_angular_velocity", self.omega_plot)
        su.log_pos("assignment6_control_torque", self.torque_plot)
        su.log_pos("assignment6_ground_track", self.ground_track)

        r, v, q, wb = self.satellite.get_state()
        q_io, w_i_io, _ = self.satellite.get_orbit_frame()
        q_ob = self.satellite.ADCS.attitude_error.q
        w_ob_b = self.satellite.ADCS.angular_velocity_error
        phi, lam, h = self.ground_track[-1]

        print()
        print("Assignment 6 final state:")
        print("Delta t epoch to now: {:.3f} s".format(self.delta_t_epoch_to_now))
        print("r_i:          [{:.6f}, {:.6f}, {:.6f}] km".format(*r))
        print("v_i:          [{:.6f}, {:.6f}, {:.6f}] km/s".format(*v))
        print("q_ib:         [{:.6f}, {:.6f}, {:.6f}, {:.6f}]".format(*q))
        print("q_io:         [{:.6f}, {:.6f}, {:.6f}, {:.6f}]".format(*q_io))
        print("q_ob error:   [{:.6f}, {:.6f}, {:.6f}, {:.6f}]".format(*q_ob))
        print("|qv_error|:   {:.6e}".format(np.linalg.norm(q_ob[1:])))
        print("w_b_ib:       [{:.6e}, {:.6e}, {:.6e}] rad/s".format(*wb))
        print("w_i_io:       [{:.6e}, {:.6e}, {:.6e}] rad/s".format(*w_i_io))
        print("|w_error|:    {:.6e}".format(np.linalg.norm(w_ob_b)))
        print("lon, lat, h:  [{:.6f}, {:.6f}, {:.6f}] deg, deg, km".format(phi * ol.RTOD, lam * ol.RTOD, h))
        print()

        pl.line_plot("data/assignment6_position_velocity.txt")
        pl.line_plot("data/assignment6_tracking_error.txt")
        pl.line_plot("data/assignment6_angular_velocity.txt")
        pl.line_plot("data/assignment6_control_torque.txt")
        pl.plot_ground_track(
            "data/assignment6_ground_track.txt",
            img_path="earth_grid.jpg",
            save_path="data/assignment6_ground_track.png"
        )


def main():
    scenario = ScenarioAssignment6()

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
