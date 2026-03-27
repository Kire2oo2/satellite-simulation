import numpy as np
import simutils as su
import simulator as sim
import plotter as pl
import orbit_lib as ol  # import the orbit library

class ScenarioAssignment2(sim.BaseScenario):

    def __init__(self):
        super().__init__()

        self.tle = ol.read_tle_file("tle.txt")

        # Orbital constants
        self.mu = ol.mu
        self.omega_ie = ol.w_E

        # Orbital elements
        self.e = None
        self.i = None
        self.raan = None
        self.arg_perigee = None
        self.mean_anomaly = None
        self.revs_per_day = None

        # Derived quantities
        self.n = None
        self.t_orbit = None
        self.a = None
        self.h = None

        # Time
        self.epoch = None
        self.jd0 = None

        # State
        self.q_E = None
        self.q = None
        self.r_i = None
        self.v_i = None

        # Logging
        self.pos_plot = None

    def init(self, t):
        self.e = self.tle['e']
        self.i = self.tle['i']
        self.raan = self.tle['raan']
        self.arg_perigee = self.tle['arg_perigee']
        self.mean_anomaly = self.tle['mean_anomaly']
        self.revs_per_day = self.tle['revs_per_day']
        self.epoch = self.tle['epoch']

        # Mean motion
        self.n = 2 * np.pi * self.revs_per_day / (24 * 3600)

        # Orbital period
        self.t_orbit = ol.orbital_period_from_revs_per_day(self.revs_per_day)

        # Semi-major axis
        self.a = (self.mu / self.n**2) ** (1 / 3)

        # Angular momentum
        self.h = np.sqrt(self.a * self.mu * (1 - self.e**2))

        # Epoch
        self.epoch = 17257.12407589
        self.jd0 = ol.epoch_to_julian_date(self.epoch)

        # Earth rotation quaternion
        theta_g0 = ol.sidereal_angle(self.jd0)
        self.q_E = su.Quaternion(theta_g0, np.array([0, 0, 1]))

        self.q = su.Quaternion()

        # Initial state
        self.r_i, self.v_i = ol.state_from_tle_params(
            self.e, self.revs_per_day, self.mean_anomaly, self.raan, self.i, self.arg_perigee
        )

        # Logging
        self.pos_plot = np.array([t, *self.r_i])

    def update(self, t, dt):
        # Propagate mean anomaly
        self.mean_anomaly = ol.angle_wrap_radians(self.mean_anomaly + self.n * dt)

        # Update satellite position
        self.r_i, self.v_i = ol.orbit_propagation(self.r_i, self.v_i, dt)
        # Earth rotation update
        jd = self.jd0 + t / 86400.0
        theta_g = ol.sidereal_angle(jd)
        self.q_E = su.Quaternion(theta_g, np.array([0, 0, 1]))

        # Logging
        self.pos_plot = np.vstack((self.pos_plot, np.array([t, *self.r_i])))

    def get(self):
        return [
            ['satellite', self.r_i, self.q],
            ['body frame', self.r_i, self.q],
            ['earth', np.zeros(3), self.q_E],
            ['ECEF frame', np.zeros(3), self.q_E],
            ['ECI frame', np.zeros(3), su.Quaternion()]
        ]

    def post_process(self, t, dt):
        su.log_pos('assignment2_position', self.pos_plot)
        pl.line_plot('data/assignment2_position.txt')


def main():
    scenario = ScenarioAssignment2()
    scenario.init(0)

    sim_config = {
        't_0': 0,
        't_e': scenario.t_orbit,
        't_step': 1,
        'speed_factor': 100,
        'anim_dt': 0.04,
        'scale_factor': 1000,
        'visualise': True
    }

    sim.create_and_start_simulation(sim_config, scenario)

if __name__ == "__main__":
    main()