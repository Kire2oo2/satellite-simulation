import os
import numpy as np

import simutils as su
import simulator as sim
import plotter as pl
import orbit_lib as ol
import sat_lib as sl


VISUALISE = True


class ScenarioAssignment5(sim.BaseScenario):

    def __init__(self):
        self.ground_track = None

    def init(self, t):
        tle = ol.read_tle_file("Assignment5_TLE.txt")

        self.q_E = su.Quaternion()

        self.J = np.array([
            [0.00146519, 0.00001703, -0.00000633],
            [0.00001703, 0.00151512, -0.00001598],
            [-0.00000633, -0.00001598, 0.00146333]
        ])

        self.orbit = ol.orbit_tle(
            tle['revs_per_day'],
            tle['e'],
            tle['mean_anomaly'],
            tle['raan'],
            tle['i'],
            tle['arg_perigee']
        )

        self.satellite = sl.Satellite(
            q_ib=np.array([1.0, 0.0, 0.0, 0.0]),
            w_b_ib=np.array([0.0, 0.0, 0.0]),
            J=self.J,
            orbit=self.orbit,
            substeps=50
        )

        r, v, q, w = self.satellite.get_state()
        q_io, w_i_io, _ = self.satellite.get_orbit_frame()
        self.satellite.ADCS.update(q, w, q_io, w_i_io)
        self.satellite.torque = self.satellite.ADCS.get_control()

        self.position_plot = self.position_log_row(t)
        self.error_plot = self.error_log_row(t)
        self.omega_plot = self.omega_log_row(t)
        self.torque_plot = self.torque_log_row(t)
        pl.log_ground_track(self, t, r)

    def position_log_row(self, t):
        r, v, q, w = self.satellite.get_state()

        return np.array([
            t,
            r[0], r[1], r[2],
            v[0], v[1], v[2],
            np.linalg.norm(r),
            np.linalg.norm(v)
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
        r, v, q, w = self.satellite.get_state()
        q_io, w_i_io, _ = self.satellite.get_orbit_frame()

        return np.array([
            t,
            w[0], w[1], w[2],
            w_i_io[0], w_i_io[1], w_i_io[2]
        ])

    def torque_log_row(self, t):
        tau = self.satellite.torque

        return np.array([
            t,
            tau[0], tau[1], tau[2]
        ])

    def update(self, t, dt):
        t_next = t + dt

        self.satellite.update(t, dt)
        self.q_E = su.Quaternion(ol.w_E * t_next, np.array([0.0, 0.0, 1.0]))

        r, v, q, w = self.satellite.get_state()
        pl.log_ground_track(self, t_next, r)

        self.position_plot = np.vstack((self.position_plot, self.position_log_row(t_next)))
        self.error_plot = np.vstack((self.error_plot, self.error_log_row(t_next)))
        self.omega_plot = np.vstack((self.omega_plot, self.omega_log_row(t_next)))
        self.torque_plot = np.vstack((self.torque_plot, self.torque_log_row(t_next)))

    def get(self):
        r, v, q, w = self.satellite.get_state()

        return [
            ['satellite', r, q],
            ['body frame', r, q],
            ['earth', np.zeros(3), self.q_E],
            ['ECI frame', np.zeros(3), su.Quaternion()],
            ['ECEF frame', np.zeros(3), self.q_E]
        ]

    def post_process(self, t, dt):
        os.makedirs("data", exist_ok=True)

        su.log_pos("assignment5_position_velocity", self.position_plot)
        su.log_pos("assignment5_tracking_error", self.error_plot)
        su.log_pos("assignment5_angular_velocity", self.omega_plot)
        su.log_pos("assignment5_control_torque", self.torque_plot)
        su.log_pos("assignment5_ground_track", self.ground_track)

        r, v, q, w = self.satellite.get_state()
        q_io, w_i_io, _ = self.satellite.get_orbit_frame()
        q_ob = self.satellite.ADCS.attitude_error.q
        w_ob_b = self.satellite.ADCS.angular_velocity_error

        print()
        print("Assignment 5 final state:")
        print("r_i:         [{:.6f}, {:.6f}, {:.6f}] km".format(*r))
        print("v_i:         [{:.6f}, {:.6f}, {:.6f}] km/s".format(*v))
        print("q_ib:        [{:.6f}, {:.6f}, {:.6f}, {:.6f}]".format(*q))
        print("q_io:        [{:.6f}, {:.6f}, {:.6f}, {:.6f}]".format(*q_io))
        print("q_ob error:  [{:.6f}, {:.6f}, {:.6f}, {:.6f}]".format(*q_ob))
        print("|qv_error|:  {:.6e}".format(np.linalg.norm(q_ob[1:])))
        print("w_b_ib:      [{:.6e}, {:.6e}, {:.6e}] rad/s".format(*w))
        print("w_i_io:      [{:.6e}, {:.6e}, {:.6e}] rad/s".format(*w_i_io))
        print("|w_error|:   {:.6e}".format(np.linalg.norm(w_ob_b)))
        print()

        pl.line_plot("data/assignment5_position_velocity.txt")
        pl.line_plot("data/assignment5_tracking_error.txt")
        pl.line_plot("data/assignment5_angular_velocity.txt")
        pl.line_plot("data/assignment5_control_torque.txt")
        pl.ground_track_plot(
            self.ground_track[:, 1] * ol.RTOD,
            self.ground_track[:, 2] * ol.RTOD,
            save_path="data/assignment5_ground_track.png"
        )


def main():
    scenario = ScenarioAssignment5()

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
