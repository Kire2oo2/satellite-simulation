import os
import numpy as np

import simutils as su
import simulator as sim
import plotter as pl


class ScenarioAssignment3(sim.BaseScenario):

    def init(self, t):
        self.R_E = 6378.1363
        self.w_E = 7.292115e-5
        self.G = 6.6742e-11
        self.m_E = 5.974e24
        self.mu = self.G * self.m_E / 1e9

        self.m = 8000
        self.rc = self.R_E + 1500

        self.k1 = 1.2e-3
        self.k2 = 1.2e-3

        self.q = su.Quaternion()
        self.theta_E = 0
        self.q_E = su.Quaternion(self.theta_E, np.array([0, 0, 1]))

        r0 = np.array([self.R_E + 800, 0, 0])
        v0 = np.array([0, np.sqrt(self.mu / np.linalg.norm(r0)), 0])
        x0 = np.concatenate((r0, v0))

        self.euler_x = np.copy(x0)
        self.leapfrog_x = np.copy(x0)
        self.verlet_x = np.copy(x0)
        self.verlet_xm1 = None

        self.solver_t = 0
        self.solver_h = 10
        self.solver_t_end = 20000

        r_start = np.array([7378, 0, 0])
        v_start = np.array([0, 0, 9])
        self.rk4_x = np.concatenate((r_start, v_start))

        self.two_body = lambda t_k, x_k: np.concatenate((
            x_k[3:],
            -self.mu * x_k[:3] / np.linalg.norm(x_k[:3])**3
        ))

        self.solver_plot = self.solver_log_row(0)
        self.transfer_plot = self.transfer_log_row(0)

    def orbit_values(self, x):
        r = x[:3]
        v = x[3:]

        e_vec = np.cross(v, np.cross(r, v)) / self.mu - r / np.linalg.norm(r)
        e = np.linalg.norm(e_vec)

        h = np.linalg.norm(np.cross(r, v))

        ra = h**2 / (self.mu * (1 - e))
        rp = h**2 / (self.mu * (1 + e))

        if e < 1e-12:
            cos_theta = 0
        else:
            cos_theta = np.dot(e_vec, r) / (e * np.linalg.norm(r))

        return ra, rp, e, cos_theta

    def thrust(self, x):
        ra, rp, e, cos_theta = self.orbit_values(x)

        if cos_theta > 0.9:
            return self.k1 * (self.rc - ra)
        elif cos_theta < -0.9:
            return self.k2 * (self.rc - rp)
        else:
            return 0

    def controlled_two_body(self, t_k, x_k):
        r = x_k[:3]
        v = x_k[3:]

        gravity = -self.mu * r / np.linalg.norm(r)**3

        T = self.thrust(x_k)

        if np.linalg.norm(v) > 0:
            thrust_acc = (T / self.m) * v / np.linalg.norm(v)
        else:
            thrust_acc = np.zeros(3)

        return np.concatenate((v, gravity + thrust_acc))

    def solver_log_row(self, t):
        return np.array([
            t,
            np.linalg.norm(self.euler_x[:3]) - self.R_E,
            np.linalg.norm(self.leapfrog_x[:3]) - self.R_E,
            np.linalg.norm(self.verlet_x[:3]) - self.R_E
        ])

    def transfer_log_row(self, t):
        ra, rp, e, cos_theta = self.orbit_values(self.rk4_x)
        T = self.thrust(self.rk4_x)

        return np.array([
            t,
            np.linalg.norm(self.rk4_x[:3]) - self.R_E,
            ra - self.R_E,
            rp - self.R_E,
            e,
            T
        ])

    def update(self, t, dt):
        t_next = t + dt

        while self.solver_t < min(t_next, self.solver_t_end):
            h = min(self.solver_h, self.solver_t_end - self.solver_t)

            self.euler_x = su.step_euler(h, self.solver_t, self.euler_x, self.two_body)
            self.leapfrog_x = su.step_leapfrog(h, self.solver_t, self.leapfrog_x, self.two_body)

            old_verlet_x = np.copy(self.verlet_x)
            self.verlet_x = su.step_verlet(h, self.solver_t, self.verlet_x, self.verlet_xm1, self.two_body)
            self.verlet_xm1 = old_verlet_x

            self.solver_t += h

        self.rk4_x = su.step_RK4(dt, t, self.rk4_x, self.controlled_two_body)

        self.theta_E += dt * self.w_E
        self.q_E = su.Quaternion(self.theta_E, np.array([0, 0, 1]))

        if self.solver_t <= self.solver_t_end:
            self.solver_plot = np.vstack((self.solver_plot, self.solver_log_row(self.solver_t)))

        self.transfer_plot = np.vstack((self.transfer_plot, self.transfer_log_row(t_next)))

    def get(self):
        return [
            ['satellite', self.rk4_x[:3], self.q],
            ['body_frame', self.rk4_x[:3], self.q],
            ['earth', np.zeros(3), self.q_E],
            ['ECEF frame', np.zeros(3), self.q_E],
            ['ECI frame', np.zeros(3), su.Quaternion()]
        ]

    def post_process(self, t, dt):
        os.makedirs("data", exist_ok=True)

        su.log_pos("assignment3_solver_comparison", self.solver_plot)
        su.log_pos("assignment3_transfer", self.transfer_plot)

        ra, rp, e, cos_theta = self.orbit_values(self.rk4_x)

        print()
        print("Final controlled orbit:")
        print("Apoapsis altitude:  {:.2f} km".format(ra - self.R_E))
        print("Periapsis altitude: {:.2f} km".format(rp - self.R_E))
        print("Eccentricity:       {:.6f}".format(e))
        print()

        pl.line_plot("data/assignment3_solver_comparison.txt")
        pl.line_plot("data/assignment3_transfer.txt")


def main():
    scenario = ScenarioAssignment3()

    sim_config = {
        't_0': 0,
        't_e': 53000,
        't_step': 100,
        'speed_factor': 1,
        'anim_dt': 1 / 25,
        'scale_factor': 1000,
        'visualise': True
    }

    sim.create_and_start_simulation(sim_config, scenario)


if __name__ == "__main__":
    main()