import os
import numpy as np

import simutils as su
import simulator as sim
import plotter as pl
import sat_lib as sl


RUN_PART = "4.2"       # Change to "4.1" if you want to run the torque-free rigid body
VISUALISE = True       # Set False if you only want the plots quickly


class ScenarioAssignment4(sim.BaseScenario):

    def __init__(self):
        self.part = RUN_PART

    def init(self, t):
        self.pos = np.array([8000.0, 0.0, 0.0])
        self.q_E = su.Quaternion()

        if self.part == "4.1":
            self.J = np.array([
                [2.0, 1.0, 0.0],
                [1.0, 10.0, 0.1],
                [0.0, 0.1, 2.5]
            ])

            self.torque = np.array([0.0, 0.0, 0.0])

            self.body = sl.RigidBody(
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                angular_velocity=np.array([0.0, 0.0, 5.0]),
                inertia_matrix=self.J
            )

            self.omega_plot = self.omega_log_row(t)
            self.euler_plot = self.euler_log_row(t)
            self.conservation_plot = self.conservation_log_row(t)

        else:
            self.J = np.array([
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5]
            ])

            self.satellite = sl.Satellite(
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                angular_velocity=np.array([0.0, 0.0, 0.0]),
                inertia_matrix=self.J,
                desired_orientation=np.array([0.5, 0.5, 0.5, 0.5]),
                desired_angular_velocity=np.array([0.2, -0.1, 0.05]),
                desired_angular_acceleration=np.array([0.0, 0.0, 0.0]),
                k1=1.0,
                k2=2.0
            )

            self.satellite.torque = self.satellite.control_torque()

            self.error_plot = self.error_log_row(t)
            self.omega_plot = self.omega_log_row(t)
            self.torque_plot = self.torque_log_row(t)

    def omega_log_row(self, t):
        if self.part == "4.1":
            q, w = self.body.get_state()
            return np.array([t, w[0], w[1], w[2]])

        q, w = self.satellite.get_state()
        q_d, w_d = self.satellite.get_reference_state()

        return np.array([
            t,
            w[0], w[1], w[2],
            w_d[0], w_d[1], w_d[2]
        ])

    def euler_log_row(self, t):
        if self.part == "4.1":
            q, w = self.body.get_state()
        else:
            q, w = self.satellite.get_state()

        eul = su.quaternion_to_euler(q)

        return np.array([
            t,
            eul[0],
            eul[1],
            eul[2]
        ])

    def conservation_log_row(self, t):
        q, w = self.body.get_state()

        energy = 0.5 * w.T @ self.J @ w
        angular_momentum_norm = np.linalg.norm(self.J @ w)

        return np.array([
            t,
            energy,
            angular_momentum_norm
        ])

    def error_log_row(self, t):
        q_db = self.satellite.attitude_error.q
        w_db = self.satellite.angular_velocity_error

        return np.array([
            t,
            q_db[0], q_db[1], q_db[2], q_db[3],
            np.linalg.norm(q_db[1:]),
            w_db[0], w_db[1], w_db[2],
            np.linalg.norm(w_db)
        ])

    def torque_log_row(self, t):
        tau = self.satellite.torque

        return np.array([
            t,
            tau[0],
            tau[1],
            tau[2]
        ])

    def update(self, t, dt):
        t_next = t + dt

        if self.part == "4.1":
            self.body.update(t, dt, self.torque)

            self.omega_plot = np.vstack((self.omega_plot, self.omega_log_row(t_next)))
            self.euler_plot = np.vstack((self.euler_plot, self.euler_log_row(t_next)))
            self.conservation_plot = np.vstack((self.conservation_plot, self.conservation_log_row(t_next)))

        else:
            self.satellite.update(t, dt)

            self.error_plot = np.vstack((self.error_plot, self.error_log_row(t_next)))
            self.omega_plot = np.vstack((self.omega_plot, self.omega_log_row(t_next)))
            self.torque_plot = np.vstack((self.torque_plot, self.torque_log_row(t_next)))

    def get(self):
        if self.part == "4.1":
            q, w = self.body.get_state()
        else:
            q, w = self.satellite.get_state()

        return [
            ['satellite', self.pos, q],
            ['body frame', self.pos, q],
            ['earth', np.zeros(3), self.q_E],
            ['ECI frame', np.zeros(3), su.Quaternion()]
        ]

    def post_process(self, t, dt):
        os.makedirs("data", exist_ok=True)

        if self.part == "4.1":
            su.log_pos("assignment4_1_angular_velocity", self.omega_plot)
            su.log_pos("assignment4_1_euler_angles", self.euler_plot)
            su.log_pos("assignment4_1_conservation", self.conservation_plot)

            q, w = self.body.get_state()

            print()
            print("Assignment 4.1 final state:")
            print("q_ib:  [{:.6f}, {:.6f}, {:.6f}, {:.6f}]".format(*q))
            print("w_bib: [{:.6f}, {:.6f}, {:.6f}] rad/s".format(*w))
            print("Energy: {:.6f}".format(self.conservation_plot[-1, 1]))
            print("|h|:    {:.6f}".format(self.conservation_plot[-1, 2]))
            print()

            pl.line_plot("data/assignment4_1_angular_velocity.txt")
            pl.line_plot("data/assignment4_1_euler_angles.txt")
            pl.line_plot("data/assignment4_1_conservation.txt")

        else:
            su.log_pos("assignment4_2_tracking_error", self.error_plot)
            su.log_pos("assignment4_2_angular_velocity", self.omega_plot)
            su.log_pos("assignment4_2_control_torque", self.torque_plot)

            q, w = self.satellite.get_state()
            q_d, w_d = self.satellite.get_reference_state()

            q_db = self.satellite.attitude_error.q
            w_db = self.satellite.angular_velocity_error

            print()
            print("Assignment 4.2 final state:")
            print("q_ib:        [{:.6f}, {:.6f}, {:.6f}, {:.6f}]".format(*q))
            print("q_id:        [{:.6f}, {:.6f}, {:.6f}, {:.6f}]".format(*q_d))
            print("q_db error:  [{:.6f}, {:.6f}, {:.6f}, {:.6f}]".format(*q_db))
            print("|qv_error|:  {:.6e}".format(np.linalg.norm(q_db[1:])))
            print("w_bib:       [{:.6f}, {:.6f}, {:.6f}] rad/s".format(*w))
            print("w_desired:   [{:.6f}, {:.6f}, {:.6f}] rad/s".format(*w_d))
            print("|w_error|:   {:.6e}".format(np.linalg.norm(w_db)))
            print()

            pl.line_plot("data/assignment4_2_tracking_error.txt")
            pl.line_plot("data/assignment4_2_angular_velocity.txt")
            pl.line_plot("data/assignment4_2_control_torque.txt")


def main():
    scenario = ScenarioAssignment4()

    if RUN_PART == "4.2":
        t_e = 100
    else:
        t_e = 500

    sim_config = {
        't_0': 0,
        't_e': t_e,
        't_step': 0.01,
        'speed_factor': 1,
        'anim_dt': 1 / 25,
        'scale_factor': 1,
        'visualise': VISUALISE
    }

    sim.create_and_start_simulation(sim_config, scenario)


if __name__ == "__main__":
    main()