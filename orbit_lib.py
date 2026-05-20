import numpy as np

R_E = 6378.0           # Earth radius [km]
w_E = 7.292115e-5     # Earth's rotation rate [rad/s]
mu = 398600.0          # Earth gravitational parameter [km^3/s^2]
DTOR = np.pi / 180.0   # Degrees to radians
RTOD = 180.0 / np.pi   # Radians to degrees

def mean_anomaly_from_eccentric(E, e):
    return E - e * np.sin(E)

def mean_anomaly_from_true_anomaly(theta, e):
    return mean_anomaly_from_eccentric(eccentric_anomaly_from_true_anomaly(theta, e), e)

def true_anomaly_from_eccentric_anomaly(E, e):
    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                          np.sqrt(1 - e) * np.cos(E / 2))

def eccentric_anomaly_from_true_anomaly(true_anomaly, e):
    E = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(true_anomaly / 2),
                       np.sqrt(1 + e) * np.cos(true_anomaly / 2))
    return E

def orbital_period_from_semi_major_axis(a, u):
    return 2 * np.pi * np.sqrt(a ** 3 / u)

def orbital_period_from_revs_per_day(revs_per_day):
    return 24 * 3600 / revs_per_day

def orbit_params_from_tle_params(e, revs_per_day, Me, raan, i, arg_perigee):
    n = 2 * np.pi * revs_per_day / (24 * 3600)
    a = (mu / n ** 2) ** (1 / 3)
    h = np.sqrt(a * mu * (1 - e ** 2))
    return h, e, Me, raan, i, arg_perigee

def tle_params_from_orbit_params(h, e, true_anomaly, raan, i, arg_perigee):
    a = h ** 2 / mu / (1 - e ** 2)
    n = np.sqrt(mu / a ** 3)
    revs_per_day = n * 86400 / (2 * np.pi)
    Me = mean_anomaly_from_eccentric(eccentric_anomaly_from_true_anomaly(true_anomaly, e), e)
    return e, revs_per_day, Me, raan, i, arg_perigee


def rotation_matrix_from_classical_euler_sequence(raan, i, arg_perigee):
    R3 = lambda angle: np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    R1 = lambda angle: np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])
    return R3(raan) @ R1(i) @ R3(arg_perigee)

def quaternion_from_classical_euler_sequence(raan, i, arg_perigee):
    from scipy.spatial.transform import Rotation as R
    return R.from_euler('zxz', [raan, i, arg_perigee]).as_quat()


def rotation_matrix_from_roll_pitch_yaw_sequence(roll, pitch, yaw):
    R_x = lambda angle: np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])
    R_y = lambda angle: np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    R_z = lambda angle: np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    return R_z(yaw) @ R_y(pitch) @ R_x(roll)

def quaternion_from_roll_pitch_yaw_sequence(roll, pitch, yaw):
    from scipy.spatial.transform import Rotation as R
    return R.from_euler('xyz', [roll, pitch, yaw]).as_quat()

def angle_wrap_radians(angle):
    return angle % (2 * np.pi)

def angle_wrap_degrees(angle):
    return angle % 360


def read_tle_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    if len(lines) < 3:
        raise ValueError("TLE file must have at least 3 lines: name + 2 lines of data")

    name = lines[0].strip()
    line1 = lines[1].strip()
    line2 = lines[2].strip()

    epoch = float(line1[18:32])
    i = float(line2[8:16]) * DTOR
    raan = float(line2[17:25]) * DTOR
    e = float("0." + line2[26:33].strip())
    arg_perigee = float(line2[34:42]) * DTOR
    mean_anomaly = float(line2[43:51]) * DTOR
    revs_per_day = float(line2[52:63])

    return {
        'name': name,
        'epoch': epoch,
        'e': e,
        'i': i,
        'raan': raan,
        'arg_perigee': arg_perigee,
        'mean_anomaly': mean_anomaly,
        'revs_per_day': revs_per_day
    }


# Algorithms from assignment 2

def sidereal_angle(JD):
    t0 = (np.trunc(JD) - 2451545) / 36525
    theta_g0 = 100.4606184 + 36000.77005361 * t0 + 0.00038793 * t0 ** 2 - 2.6e-8 * t0 ** 3
    frac = (JD + 0.5) - np.trunc(JD + 0.5)
    theta_g = np.deg2rad(theta_g0) + w_E * 86400 * frac
    theta_g = theta_g % (2 * np.pi)
    return theta_g


def state_from_orbit_params(h, e, true_anomaly, raan, i, arg_perigee):
    r = h ** 2 / mu / (1 + e * np.cos(true_anomaly))
    r_perifocal = np.array([r * np.cos(true_anomaly), r * np.sin(true_anomaly), 0])
    v_perifocal = mu / h * np.array([-np.sin(true_anomaly), e + np.cos(true_anomaly), 0])
    R = rotation_matrix_from_classical_euler_sequence(raan, i, arg_perigee)
    return R @ r_perifocal, R @ v_perifocal


def state_from_tle_params(e, revs_per_day, Me, raan, i, arg_perigee):
    h, e, Me, raan, i, arg_perigee = orbit_params_from_tle_params(e, revs_per_day, Me, raan, i, arg_perigee)
    E = eccentric_anomaly_from_mean_anomaly(Me, e)
    true_anomaly = true_anomaly_from_eccentric_anomaly(E, e)
    return state_from_orbit_params(h, e, true_anomaly, raan, i, arg_perigee)


def orbit_params_from_state(r_i, v_i):
    h_vec = np.cross(r_i, v_i)
    h = np.linalg.norm(h_vec)
    r = np.linalg.norm(r_i)
    e_vec = (np.cross(v_i, h_vec) / mu - r_i / r)
    e = np.linalg.norm(e_vec)
    i = np.arccos(h_vec[2] / h)
    n_vec = np.cross([0, 0, 1], h_vec)
    n = np.linalg.norm(n_vec)
    raan = np.arctan2(n_vec[1], n_vec[0])

    if n > 1e-12 and e > 1e-12:
        arg_perigee = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1.0, 1.0))
        if e_vec[2] < 0:
            arg_perigee = 2 * np.pi - arg_perigee
    else:
        arg_perigee = 0

    if e > 1e-12:
        true_anomaly = np.arccos(np.clip(np.dot(e_vec, r_i) / (e * r), -1.0, 1.0))
        if np.dot(r_i, v_i) < 0:
            true_anomaly = 2 * np.pi - true_anomaly
    else:
        true_anomaly = 0

    return h, e, true_anomaly, raan, i, arg_perigee


def orbit_propagation(r_i, v_i, dt):
    h, e, theta, raan, i, arg_perigee = orbit_params_from_state(r_i, v_i)
    a = h ** 2 / mu / (1 - e ** 2)
    n = np.sqrt(mu / a ** 3)
    E = eccentric_anomaly_from_true_anomaly(theta, e)
    M = mean_anomaly_from_eccentric(E, e)
    M_new = angle_wrap_radians(M + n * dt)
    E_new = eccentric_anomaly_from_mean_anomaly(M_new, e)
    theta_new = true_anomaly_from_eccentric_anomaly(E_new, e)
    r_next, v_next = state_from_orbit_params(h, e, theta_new, raan, i, arg_perigee)
    return r_next, v_next


def epoch_to_julian_date(epoch):
    YY = int(epoch // 1000)
    DDD = epoch % 1000
    year = 2000 + YY if YY < 57 else 1900 + YY
    day_of_year = DDD
    jd = 1721424.5 + (year - 1) * 365 + (year - 1) // 4 - (year - 1) // 100 + (year - 1) // 400 + day_of_year
    return jd


def eccentric_anomaly_from_mean_anomaly(Me, e, tol=1e-8, max_iter=100):
    E = Me if e < 0.8 else np.pi
    for _ in range(max_iter):
        E_new = E - (E - e * np.sin(E) - Me) / (1 - e * np.cos(E))
        if abs(E_new - E) < tol:
            return E_new
        E = E_new
    return E


# Assignment 5

def _dcm_to_quaternion_array(R):
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)

    if tr > 0.0:
        q0 = 0.5 * np.sqrt(1.0 + tr)
        q1 = (R[2, 1] - R[1, 2]) / (4.0 * q0)
        q2 = (R[0, 2] - R[2, 0]) / (4.0 * q0)
        q3 = (R[1, 0] - R[0, 1]) / (4.0 * q0)
    else:
        i_max = np.argmax(np.diag(R))

        if i_max == 0:
            q1 = 0.5 * np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2]))
            q0 = (R[2, 1] - R[1, 2]) / (4.0 * q1)
            q2 = (R[0, 1] + R[1, 0]) / (4.0 * q1)
            q3 = (R[0, 2] + R[2, 0]) / (4.0 * q1)
        elif i_max == 1:
            q2 = 0.5 * np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2]))
            q0 = (R[0, 2] - R[2, 0]) / (4.0 * q2)
            q1 = (R[0, 1] + R[1, 0]) / (4.0 * q2)
            q3 = (R[1, 2] + R[2, 1]) / (4.0 * q2)
        else:
            q3 = 0.5 * np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2]))
            q0 = (R[1, 0] - R[0, 1]) / (4.0 * q3)
            q1 = (R[0, 2] + R[2, 0]) / (4.0 * q3)
            q2 = (R[1, 2] + R[2, 1]) / (4.0 * q3)

    q = np.array([q0, q1, q2, q3])
    q = q / np.linalg.norm(q)

    if q[0] < 0.0:
        q = -q

    return q


def orbit_frame_from_state(r_i, v_i):
    r_i = np.asarray(r_i, dtype=float)
    v_i = np.asarray(v_i, dtype=float)

    r_norm = np.linalg.norm(r_i)
    h_i = np.cross(r_i, v_i)
    h_norm = np.linalg.norm(h_i)

    if r_norm < 1e-12:
        raise ValueError("Position vector magnitude is zero")

    if h_norm < 1e-12:
        raise ValueError("Angular momentum vector magnitude is zero")

    x_o_i = r_i / r_norm
    z_o_i = h_i / h_norm
    y_o_i = np.cross(z_o_i, x_o_i)
    y_o_i = y_o_i / np.linalg.norm(y_o_i)

    R_io = np.column_stack((x_o_i, y_o_i, z_o_i))
    q_io = _dcm_to_quaternion_array(R_io)

    w_i_io = h_i / r_norm ** 2
    dw_i_io = -2.0 * h_i * np.dot(r_i, v_i) / r_norm ** 4

    return q_io, w_i_io, dw_i_io


class orbit_classic:
    def __init__(self, h, e, theta, O, i, w):
        self.h = h
        self.e = e
        self.theta = angle_wrap_radians(theta)
        self.O = O
        self.i = i
        self.w = w

    def propagate(self, t_step):
        r = self.h ** 2 / mu / (1 + self.e * np.cos(self.theta))
        theta_dot = self.h / r ** 2
        self.theta = angle_wrap_radians(self.theta + theta_dot * t_step)

    def get_params(self):
        return self.h, self.e, self.theta, self.O, self.i, self.w

    def get_state(self):
        return state_from_orbit_params(self.h, self.e, self.theta, self.O, self.i, self.w)

    def get_orbit_frame(self):
        r_i, v_i = self.get_state()
        return orbit_frame_from_state(r_i, v_i)


class orbit_tle:
    def __init__(self, n, e, M_e, O, i, w):
        self.n = n
        self.e = e
        self.M_e = angle_wrap_radians(M_e)
        self.O = O
        self.i = i
        self.w = w

    def propagate(self, t_step):
        n_rad_s = 2 * np.pi * self.n / (24 * 3600)
        self.M_e = angle_wrap_radians(self.M_e + n_rad_s * t_step)

    def get_params(self):
        return self.n, self.e, self.M_e, self.O, self.i, self.w

    def get_state(self):
        return state_from_tle_params(self.e, self.n, self.M_e, self.O, self.i, self.w)

    def get_orbit_frame(self):
        r_i, v_i = self.get_state()
        return orbit_frame_from_state(r_i, v_i)
