import numpy as np
R_E = 6378.0           # Earth radius [km]
w_E = 7.292115e-5       # Earth's rotation rate [rad/s]
mu = 398600.0           # Earth gravitational parameter [km^3/s^2]
DTOR = np.pi / 180.0   # Degrees to radians
RTOD = 180.0 / np.pi   # Radians to degrees

def mean_anomaly_from_eccentric(E, e):
    return E - e * np.sin(E)

def mean_anomaly_from_true_anomaly(theta, e):
    return mean_anomaly_from_eccentric(eccentric_anomaly_from_true_anomaly(theta, e), e)

def true_anomaly_from_eccentric_anomaly(E, e):
    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2),
                          np.sqrt(1 - e) * np.cos(E/2))

def eccentric_anomaly_from_true_anomaly(true_anomaly, e):
    E = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(true_anomaly/2),
                       np.sqrt(1 + e) * np.cos(true_anomaly/2))
    return E

def orbital_period_from_semi_major_axis(a,u):
    return 2 * np.pi * np.sqrt(a**3 / u)

def orbital_period_from_revs_per_day(revs_per_day):
    return 24*3600 / revs_per_day

def orbit_params_from_tle_params(e, revs_per_day, Me, raan, i, arg_perigee):
    n = 2 * np.pi * revs_per_day / (24*3600)
    a = (mu / n**2)**(1/3)
    h = np.sqrt(a * mu * (1 - e**2))
    return h, e, Me, raan, i, arg_perigee

def tle_params_from_orbit_params(h, e, true_anomaly, raan, i, arg_perigee):
    a = h**2 / mu / (1 - e**2)
    n = np.sqrt(mu / a**3)
    revs_per_day = n * 86400 / (2*np.pi)
    Me = mean_anomaly_from_eccentric(eccentric_anomaly_from_true_anomaly(true_anomaly, e), e)
    return e, revs_per_day, Me, raan, i, arg_perigee


def rotation_matrix_from_classical_euler_sequence(raan, i, arg_perigee):
    R3 = lambda angle: np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
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
    return R.from_euler('zxz', [raan, i, arg_perigee]).as_quat()  # x,y,z,w order may vary

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
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])
    return R_z(yaw) @ R_y(pitch) @ R_x(roll)

def quaternion_from_roll_pitch_yaw_sequence(roll, pitch, yaw):
    from scipy.spatial.transform import Rotation as R
    return R.from_euler('xyz', [roll, pitch, yaw]).as_quat()

def angle_wrap_radians(angle):
    return angle % (2*np.pi)

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

    # Line 1: extract epoch
    # Columns 19-32: Epoch (YYDDD.DDDDDDDD)
    epoch = float(line1[18:32])

    # Line 2: orbital parameters
    i = float(line2[8:16]) * DTOR  # Inclination [rad]
    raan = float(line2[17:25]) * DTOR  # RAAN [rad]
    e = float("0." + line2[26:33].strip())  # Eccentricity
    arg_perigee = float(line2[34:42]) * DTOR  # Argument of perigee [rad]
    mean_anomaly = float(line2[43:51]) * DTOR  # Mean anomaly [rad]
    revs_per_day = float(line2[52:63])  # Revolutions per day

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

#algorithm 1
def sidereal_angle(JD):
    t0 = (np.trunc(JD) - 2451545) / 36525
    theta_g0 = 100.4606184 + 36000.77005361*t0 + 0.00038793*t0**2 - 2.6e-8*t0**3
    frac = (JD + 0.5) - np.trunc(JD + 0.5)  # time since 00:00 UTC
    theta_g = np.deg2rad(theta_g0) + w_E * 86400 * frac
    theta_g = theta_g % (2*np.pi)  # wrap to [0,2π]
    return theta_g

#algorithm 2
def state_from_orbit_params(h, e, true_anomaly, raan, i, arg_perigee):
    r = h**2 / mu / (1 + e*np.cos(true_anomaly))
    r_perifocal = np.array([r*np.cos(true_anomaly), r*np.sin(true_anomaly), 0])
    v_perifocal = mu/h * np.array([-np.sin(true_anomaly), e+np.cos(true_anomaly), 0])
    R = rotation_matrix_from_classical_euler_sequence(raan, i, arg_perigee)
    return R @ r_perifocal, R @ v_perifocal

#algorithm 3
def state_from_tle_params(e, revs_per_day, Me, raan, i, arg_perigee):
    h, e, Me, raan, i, arg_perigee = orbit_params_from_tle_params(e, revs_per_day, Me, raan, i, arg_perigee)

    # Solve Kepler's equation
    E = eccentric_anomaly_from_mean_anomaly(Me, e)

    # Convert to true anomaly
    true_anomaly = true_anomaly_from_eccentric_anomaly(E, e)

    return state_from_orbit_params(h, e, true_anomaly, raan, i, arg_perigee)

#algorithm 4
def orbit_params_from_state(r_i, v_i):
    h_vec = np.cross(r_i, v_i)
    h = np.linalg.norm(h_vec)
    r = np.linalg.norm(r_i)
    v = np.linalg.norm(v_i)
    e_vec = (np.cross(v_i, h_vec)/mu - r_i/r)
    e = np.linalg.norm(e_vec)
    i = np.arccos(h_vec[2]/h)
    n_vec = np.cross([0,0,1], h_vec)
    n = np.linalg.norm(n_vec)
    raan = np.arctan2(n_vec[1], n_vec[0])

    # Argument of perigee
    if n != 0 and e != 0:
        arg_perigee = np.arccos(np.dot(n_vec, e_vec)/(n*e))
        if e_vec[2] < 0:
            arg_perigee = 2*np.pi - arg_perigee
    else:
        arg_perigee = 0

    # True anomaly
    if e != 0:
        true_anomaly = np.arccos(np.dot(e_vec, r_i)/(e*r))
        if np.dot(r_i, v_i) < 0:
            true_anomaly = 2*np.pi - true_anomaly
    else:
        true_anomaly = 0

    return h, e, true_anomaly, raan, i, arg_perigee

#algoritm 5
def orbit_propagation(r_i, v_i, dt):
    # Compute orbital elements from state
    h, e, theta, raan, i, arg_perigee = orbit_params_from_state(r_i, v_i)

    # Mean motion
    a = h ** 2 / mu / (1 - e ** 2)
    n = np.sqrt(mu / a ** 3)

    # Current mean anomaly
    E = eccentric_anomaly_from_true_anomaly(theta, e)
    M = mean_anomaly_from_eccentric(E, e)

    # Propagate mean anomaly
    M_new = angle_wrap_radians(M + n * dt)

    # Solve for new eccentric anomaly
    E_new = eccentric_anomaly_from_mean_anomaly(M_new, e)

    # Compute new true anomaly
    theta_new = true_anomaly_from_eccentric_anomaly(E_new, e)

    # Compute new state
    r_next, v_next = state_from_orbit_params(h, e, theta_new, raan, i, arg_perigee)

    return r_next, v_next

#algorithm 6
def epoch_to_julian_date(epoch):
    # TLE epoch: YYDDD.DDDDD
    YY = int(epoch / 1000)
    DDD = epoch % 1000
    # Century handling: TLE uses 1957-2056 convention
    year = 2000 + YY if YY < 57 else 1900 + YY
    day = int(DDD)
    frac_day = DDD - day

    A = int((year - 1) / 4)
    jd = 1721013.5 + 365*(year - 1) + A + day + frac_day
    return jd

#algorithm 7
def eccentric_anomaly_from_mean_anomaly(Me, e, tol=1e-8, max_iter=100):
    E = Me if e < 0.8 else np.pi
    for _ in range(max_iter):
        E_new = E - (E - e*np.sin(E) - Me)/(1 - e*np.cos(E))
        if abs(E_new - E) < tol:
            return E_new
        E = E_new
    return E