import numpy as np
import simutils as su
import simulator as sim
import plotter as pl

class ScenarioAssignment2(sim.BaseScenario):

    def __init__(self):
        super().__init__()

        self.mu = None
        self.omega_ie = None

        # Orbital elements
        self.e = None
        self.i = None
        self.raan = None
        self.arg_perigee = None
        self.mean_anomaly = None

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

        # Logging
        self.pos_plot = None

    def init(self, t):
        self.mu = 398600.0
        self.omega_ie = 7.292115e-5

        # TLE parameters vanguard 1
        self.e = 0.1836663
        self.i = 0.5976199345046304
        self.raan = 3.735161678900544
        self.arg_perigee = 2.0778249464747613
        self.mean_anomaly = 4.548956349227941

        revs_per_day = 10.85960848

        # Mean motion
        self.n = 2 * np.pi * revs_per_day / (24 * 3600)

        # Orbital period
        self.t_orbit = 2 * np.pi / self.n

        # Semi-major axis
        self.a = (self.mu / self.n**2) ** (1 / 3)

        # Angular momentum
        self.h = np.sqrt(self.a * self.mu * (1 - self.e**2))

        # Epoch
        self.epoch = 17257.12407589
        self.jd0 = self.epoch_to_jd(self.epoch)

        # Earth rotation quaternion
        theta_g0 = self.sidereal_angle(self.jd0)
        self.q_E = su.Quaternion(theta_g0, np.array([0, 0, 1]))

        self.q = su.Quaternion()

        # Initial state
        self.r_i = self.compute_state()

        # Logging
        self.pos_plot = np.array([t, *self.r_i])

    @staticmethod
    def solve_kepler(mean_anomaly, e, tol=1e-8, max_iter=100):
        e_anomaly = mean_anomaly + e / 2 if mean_anomaly < np.pi else mean_anomaly - e / 2

        for _ in range(max_iter):
            e_new = e_anomaly - (
                e_anomaly - e * np.sin(e_anomaly) - mean_anomaly
            ) / (1 - e * np.cos(e_anomaly))

            if abs(e_new - e_anomaly) < tol:
                return e_new

            e_anomaly = e_new

        return e_anomaly

    @staticmethod
    def r1(angle):
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])

    @staticmethod
    def r3(angle):
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ])

    @staticmethod
    def epoch_to_jd(epoch):
        year = int(epoch / 1000)
        day = int(epoch - year * 1000)
        utc = epoch - (year * 1000 + day)

        leap = 1 if (year % 4 == 0 and day < 59) else 0

        jd = (
            2451544.5
            + year * 365
            + int(year / 4)
            + day
            - leap
            + utc
        )

        return jd

    def compute_state(self):

        e_anomaly = self.solve_kepler(self.mean_anomaly, self.e)

        true_anomaly = 2 * np.arctan2(
            np.sqrt(1 + self.e) * np.sin(e_anomaly / 2),
            np.sqrt(1 - self.e) * np.cos(e_anomaly / 2)
        )

        r = self.h**2 / self.mu / (1 + self.e * np.cos(true_anomaly))

        r_perifocal = np.array([
            r * np.cos(true_anomaly),
            r * np.sin(true_anomaly),
            0
        ])

        # Rotation to ECI
        rotation = (
            self.r3(self.raan)
            @ self.r1(self.i)
            @ self.r3(self.arg_perigee)
        )

        return rotation @ r_perifocal

    def sidereal_angle(self, jd):
        t0 = (np.trunc(jd) - 2451545) / 36525

        theta_g0 = (
            100.4606184
            + 36000.77005361 * t0
            + 0.00038793 * t0**2
            - 2.6e-8 * t0**3
        )

        frac = (jd + 0.5) - np.trunc(jd + 0.5)

        theta_g = theta_g0 + (180 / np.pi) * self.omega_ie * (86400 * frac)
        theta_g = theta_g - 360 * np.trunc(theta_g / 360)

        return np.deg2rad(theta_g)



    def update(self, t, dt):

        # Propagate mean anomaly
        self.mean_anomaly = (self.mean_anomaly + self.n * dt) % (2 * np.pi)

        # Update satellite position
        self.r_i = self.compute_state()

        # Earth rotation update
        jd = self.jd0 + t / 86400.0
        theta_g = self.sidereal_angle(jd)
        self.q_E = su.Quaternion(theta_g, np.array([0, 0, 1]))

        #save for plot
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