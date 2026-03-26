import numpy as np
import simutils as su
import simulator as sim
import plotter as pl

class ScenarioAssignment1(sim.BaseScenario):

    def __init__(self):
        super().__init__()

        #linter attributes
        self.mu = None
        self.r_earth = None
        self.r = None
        self.theta = None
        self.t_orbit = None
        self.theta_dot = None
        self.omega_ie = None
        self.q_E = None
        self.q = None


    def init(self, t):
        self.mu = 398600
        self.r_earth = 6371

        self.r = self.r_earth + 400
        self.theta = 0.0

        self.t_orbit = 2 * np.pi * np.sqrt(self.r**3 / self.mu)
        self.theta_dot = 2 * np.pi / self.t_orbit

        self.omega_ie = 7.2921e-5

        self.q_E = su.Quaternion()
        self.q = su.Quaternion()

        #plot
        self.pos_plot = np.array([t, self.r * np.cos(self.theta), self.r * np.sin(self.theta), 0])

    def update(self, t, dt):
        self.theta += dt * self.theta_dot

        #earth rotation
        theta_E = self.omega_ie * t
        self.q_E = su.Quaternion(theta_E, np.array([0, 0, 1]))

        #plot?
        self.pos_plot = np.vstack((self.pos_plot, np.array([t, self.r * np.cos(self.theta), self.r * np.sin(self.theta), 0])))

    def get(self):
        r_i = self.r * np.array([
            np.cos(self.theta),
            np.sin(self.theta),
            0
        ])

        return [
            ['satellite', r_i, self.q],
            ['body frame', r_i, self.q],
            ['earth', np.zeros(3), self.q_E],
            ['ECEF frame', np.zeros(3), self.q_E],
            ['ECI frame', np.zeros(3), su.Quaternion()]
        ]

    def post_process(self, t, dt):
        su.log_pos('assignment1_position', self.pos_plot)
        pl.line_plot('data/'+'assignment1_position'+'.txt')

def main():
    scenario = ScenarioAssignment1()
    scenario.init(0)  # initialize scenario attributes

    sim_config = {
        't_0': 0,
        't_e': scenario.t_orbit,  # use scenario attribute
        't_step': 1,
        'speed_factor': 100,
        'anim_dt': 0.04,
        'scale_factor': 1000,
        'visualise': True
    }

    sim.create_and_start_simulation(sim_config, scenario)


if __name__ == "__main__":
    main()