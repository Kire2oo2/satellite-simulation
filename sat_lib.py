import numpy as np
import simutils as su
import orbit_lib as ol


def _q_array(q):
    if isinstance(q, su.Quaternion):
        q = q.q

    q = np.asarray(q, dtype=float)

    if q.shape != (4,):
        raise ValueError("Quaternion must have shape (4,)")

    n = np.linalg.norm(q)

    if n < 1e-12:
        raise ValueError("Quaternion magnitude is zero")

    return q / n


def _q_mul(q1, q2):
    q1 = _q_array(q1)

    if isinstance(q2, su.Quaternion):
        q2 = q2.q

    q2 = np.asarray(q2, dtype=float)

    if q2.shape == (3,):
        q2 = np.concatenate(([0.0], q2))

    a0, a1, a2, a3 = q1
    b0, b1, b2, b3 = q2

    return np.array([
        a0*b0 - a1*b1 - a2*b2 - a3*b3,
        a0*b1 + b0*a1 + a2*b3 - a3*b2,
        a0*b2 + b0*a2 + a3*b1 - a1*b3,
        a0*b3 + b0*a3 + a1*b2 - a2*b1
    ])


def _q_conj(q):
    q = _q_array(q)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _q_rotate(q, v):
    q = _q_array(q)
    v = np.asarray(v, dtype=float)
    s = q[0]
    u = q[1:]
    return v + 2.0 * s * np.cross(u, v) + 2.0 * np.cross(u, np.cross(u, v))


def _q_rotate_inverse(q, v):
    q = _q_array(q)
    v = np.asarray(v, dtype=float)
    s = q[0]
    u = q[1:]
    return v - 2.0 * s * np.cross(u, v) + 2.0 * np.cross(u, np.cross(u, v))


def _shape(x):
    return np.asarray(x, dtype=float).shape


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def _noise_sample(mu, Q, size=3):
    if Q is None:
        Q = 0.0

    if np.isscalar(mu):
        mu_vec = np.ones(size) * float(mu)
    else:
        mu_vec = np.asarray(mu, dtype=float)
        if mu_vec.shape == ():
            mu_vec = np.ones(size) * float(mu_vec)

    if np.isscalar(Q):
        std = np.sqrt(max(float(Q), 0.0))
        return np.random.normal(loc=mu_vec, scale=std, size=size)

    Q = np.asarray(Q, dtype=float)

    if Q.shape == (size,):
        std = np.sqrt(np.maximum(Q, 0.0))
        return np.random.normal(loc=mu_vec, scale=std, size=size)

    if Q.shape == (size, size):
        return np.random.multivariate_normal(mu_vec, Q)

    raise ValueError("Q must be scalar, vector, or covariance matrix")


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


class RigidBody:
    def __init__(self, *args, **kwargs):
        self.old_mode = False

        if 'orientation' in kwargs or (len(args) == 3 and _shape(args[0]) == (4,) and _shape(args[1]) == (3,) and _shape(args[2]) == (3, 3)):
            self.old_mode = True

            if 'orientation' in kwargs:
                q = kwargs['orientation']
                w = kwargs['angular_velocity']
                J = kwargs['inertia_matrix']
            else:
                q, w, J = args

            self.p = np.zeros(3)
            self.v = np.zeros(3)
            self.m = 1.0
            self.orientation = _q_array(q)
            self.angular_velocity = np.asarray(w, dtype=float)
            self.inertia_matrix = np.asarray(J, dtype=float)

        else:
            if len(args) >= 6:
                r, v, m, q, w, J = args[:6]
            else:
                r = kwargs.get('r', np.zeros(3))
                v = kwargs.get('v', np.zeros(3))
                m = kwargs.get('m', 1.0)
                q = kwargs.get('q', kwargs.get('q_ib', np.array([1.0, 0.0, 0.0, 0.0])))
                w = kwargs.get('w', kwargs.get('w_b_ib', np.zeros(3)))
                J = kwargs.get('J', kwargs.get('inertia_matrix', np.eye(3)))

            self.p = np.asarray(r, dtype=float)
            self.v = np.asarray(v, dtype=float)
            self.m = float(m)
            self.orientation = _q_array(q)
            self.angular_velocity = np.asarray(w, dtype=float)
            self.inertia_matrix = np.asarray(J, dtype=float)

        self.torque = np.zeros(3)
        self.acceleration = np.zeros(3)

        if self.p.shape != (3,):
            raise ValueError("Position must have shape (3,)")

        if self.v.shape != (3,):
            raise ValueError("Velocity must have shape (3,)")

        if self.angular_velocity.shape != (3,):
            raise ValueError("Angular velocity must have shape (3,)")

        if self.inertia_matrix.shape != (3, 3):
            raise ValueError("Inertia matrix must have shape (3,3)")

    def state(self):
        if self.old_mode:
            return np.concatenate((self.orientation, self.angular_velocity))
        return np.concatenate((self.p, self.v, self.orientation, self.angular_velocity))

    def get_state(self):
        if self.old_mode:
            return self.orientation, self.angular_velocity
        return self.p, self.v, self.orientation, self.angular_velocity

    def update(self, t_k, h, *args):
        if len(args) == 1:
            self.acceleration = np.zeros(3)
            self.torque = np.asarray(args[0], dtype=float)

            x_k = np.concatenate((self.orientation, self.angular_velocity))
            x_next = su.step_RK4(h, t_k, x_k, self.f_attitude)

            self.orientation = _q_array(x_next[:4])
            self.angular_velocity = x_next[4:]

        elif len(args) == 2:
            self.acceleration = np.asarray(args[0], dtype=float)
            self.torque = np.asarray(args[1], dtype=float)

            w = self.angular_velocity
            q = self.orientation

            w_dot = np.linalg.solve(
                self.inertia_matrix,
                self.torque - np.cross(w, self.inertia_matrix @ w)
            )
            q_dot = 0.5 * _q_mul(q, np.concatenate(([0.0], w)))

            self.p = self.p + h * self.v
            self.v = self.v + h * self.acceleration
            self.orientation = _q_array(q + h * q_dot)
            self.angular_velocity = w + h * w_dot

        else:
            raise ValueError("RigidBody.update expects torque or acceleration and torque")

    def f_attitude(self, t, x):
        q = _q_array(x[:4])
        w = np.asarray(x[4:], dtype=float)

        q_dot = 0.5 * _q_mul(q, np.concatenate(([0.0], w)))
        w_dot = np.linalg.solve(
            self.inertia_matrix,
            self.torque - np.cross(w, self.inertia_matrix @ w)
        )

        return np.concatenate((q_dot, w_dot))


class gyro:
    def __init__(self, q_bs=su.Quaternion(), p_b=np.array([0.0, 0.0, 0.0]), mu=0.0, Q=0.0, z0=None, params=None):
        self.q_bs = _q_array(q_bs)
        self.p = np.asarray(p_b, dtype=float)
        self.mu = mu
        self.Q = Q
        self.z = np.zeros(3) if z0 is None else np.asarray(z0, dtype=float)
        self.bg = np.zeros(3)
        self.sigma_bg = 0.0

        if params is not None:
            self.bg = np.asarray(params.get('bg', np.zeros(3)), dtype=float)
            self.sigma_bg = params.get('sigma_bg', params.get('Q_bg', 0.0))

    def update(self, t, t_step, q_ib, w_b_ib, r_i, v_i, JD=None):
        self.bg = self.bg + _noise_sample(0.0, self.sigma_bg, 3) * t_step
        w_s = _q_rotate_inverse(self.q_bs, w_b_ib)
        self.z = w_s + self.bg + _noise_sample(self.mu, self.Q, 3)

    def output(self, body_frame=False):
        if body_frame:
            return _q_rotate(self.q_bs, self.z)
        return self.z


class magnetometer:
    def __init__(self, q_bs=su.Quaternion(), p_b=np.array([0.0, 0.0, 0.0]), mu=0.0, Q=0.0, z0=None, params=None):
        self.q_bs = _q_array(q_bs)
        self.p = np.asarray(p_b, dtype=float)
        self.mu = mu
        self.Q = Q
        self.z = np.zeros(3) if z0 is None else np.asarray(z0, dtype=float)
        self.bB = np.zeros(3)
        self.MB = np.eye(3)

        if params is not None:
            self.bB = np.asarray(params.get('bB', np.zeros(3)), dtype=float)
            self.MB = np.asarray(params.get('MB', np.eye(3)), dtype=float)

    def update(self, t, t_step, q_ib, w_b_ib, r_i, v_i, JD=None):
        if JD is None:
            JD = 2451545.0 + t / 86400.0

        B_i = ol.magnetic_field_dipole(r_i, JD)
        q_is = _q_mul(q_ib, self.q_bs)
        B_s = _unit(_q_rotate_inverse(q_is, B_i))
        self.z = self.MB @ B_s + self.bB + _noise_sample(self.mu, self.Q, 3)

    def output(self, body_frame=False):
        if body_frame:
            return _q_rotate(self.q_bs, self.z)
        return self.z


class fine_sun_sensor:
    def __init__(self, q_bs=su.Quaternion(), p_b=np.array([0.0, 0.0, 0.0]), mu=0.0, Q=0.0, z0=None, params=None):
        self.q_bs = _q_array(q_bs)
        self.p = np.asarray(p_b, dtype=float)
        self.mu = mu
        self.Q = Q
        self.z = np.zeros(3) if z0 is None else np.asarray(z0, dtype=float)
        self.alpha = np.pi
        self.valid = False

        if params is not None:
            self.alpha = params.get('alpha', np.pi)

    def update(self, t, t_step, q_ib, w_b_ib, r_i, v_i, JD=None):
        if JD is None:
            JD = 2451545.0 + t / 86400.0

        s_i = ol.sun_vector(JD)
        s_b = _q_rotate_inverse(q_ib, _unit(s_i))
        s_s = _q_rotate_inverse(self.q_bs, s_b)

        x, y, z = s_s
        angle = np.arctan2(np.sqrt(x**2 + y**2), z)

        if z > 0.0 and angle < self.alpha / 2.0:
            self.z = _unit(s_s + _noise_sample(self.mu, self.Q, 3))
            self.valid = True
        else:
            self.z = np.zeros(3)
            self.valid = False

    def output(self, body_frame=False):
        if np.linalg.norm(self.z) < 1e-12:
            return np.zeros(3)

        if body_frame:
            return _unit(_q_rotate(self.q_bs, self.z))

        return self.z


class TRIAD:
    def __init__(self, params=None):
        self.params = params

    def estimate_attitude(self, M_B, M_A):
        if len(M_B) < 2 or len(M_A) < 2:
            raise ValueError("TRIAD requires at least two vector pairs")

        U_B = _unit(M_B[0])
        V_B = _unit(M_B[1])
        U_A = _unit(M_A[0])
        V_A = _unit(M_A[1])

        t1_B = U_B
        t2_B = np.cross(t1_B, V_B)
        t2_B = _unit(t2_B)
        t3_B = np.cross(t2_B, t1_B)

        t1_A = U_A
        t2_A = np.cross(t1_A, V_A)
        t2_A = _unit(t2_A)
        t3_A = np.cross(t2_A, t1_A)

        if np.linalg.norm(t2_B) < 1e-12 or np.linalg.norm(t2_A) < 1e-12:
            raise ValueError("TRIAD vector pairs are close to parallel")

        R_BA = np.column_stack((t1_B, t2_B, t3_B)) @ np.column_stack((t1_A, t2_A, t3_A)).T
        return _dcm_to_quaternion_array(R_BA)


class Davenport:
    def __init__(self, params=None):
        self.params = params
        self.weights = None

        if params is not None:
            self.weights = params.get('weights', None)

    def estimate_attitude(self, M_B, M_A):
        if len(M_B) != len(M_A) or len(M_B) < 2:
            raise ValueError("Davenport requires at least two matching vector pairs")

        N = len(M_B)

        if self.weights is None:
            weights = np.ones(N) / N
        else:
            weights = np.asarray(self.weights, dtype=float)
            weights = weights / np.sum(weights)

        B = np.zeros((3, 3))
        z = np.zeros(3)

        for w_i, u_B, u_A in zip(weights, M_B, M_A):
            u_B = _unit(u_B)
            u_A = _unit(u_A)

            if np.linalg.norm(u_B) < 1e-12 or np.linalg.norm(u_A) < 1e-12:
                continue

            B += w_i * np.outer(u_B, u_A)
            z += w_i * np.cross(u_B, u_A)

        K = np.zeros((4, 4))
        K[0, 0] = np.trace(B)
        K[0, 1:] = z
        K[1:, 0] = z
        K[1:, 1:] = B + B.T - np.trace(B) * np.eye(3)

        evals, evecs = np.linalg.eigh(K)
        q = evecs[:, np.argmax(evals)]
        q = _q_array(q)

        if q[0] < 0.0:
            q = -q

        return q


class ADCS_PD:
    def __init__(self, k1, k2, f_c=None, J=None, attitude_estimator=None):
        if J is None:
            J = f_c
            f_c = np.zeros(3)

        self.k1 = k1
        self.k2 = k2
        self.f_c = f_c
        self.J = np.asarray(J, dtype=float)
        self.attitude_estimator = attitude_estimator
        self.tau = np.zeros(3)
        self.attitude_error = su.Quaternion()
        self.angular_velocity_error = np.zeros(3)
        self.q_ob_est = np.array([1.0, 0.0, 0.0, 0.0])
        self.q_ib_est = np.array([1.0, 0.0, 0.0, 0.0])
        self.sun_body = np.zeros(3)
        self.mag_body = np.zeros(3)
        self.sliding_surface = np.zeros(3)

    def update(self, *args):
        if len(args) == 4:
            q_ib, w_b_ib, q_io, w_i_io = args
            self._update_from_true_attitude(q_ib, w_b_ib, q_io, w_i_io, np.zeros(3))
            return

        if len(args) == 6:
            _, q_ib, w_b_ib, q_io, w_i_io, dw_i_io = args
            self._update_from_true_attitude(q_ib, w_b_ib, q_io, w_i_io, dw_i_io)
            return

        if len(args) == 7:
            _, _, q_ib, w_b_ib, q_io, w_i_io, dw_i_io = args
            self._update_from_true_attitude(q_ib, w_b_ib, q_io, w_i_io, dw_i_io)
            return

        if len(args) == 9:
            _, _, gyro_measurement, magnetometer_measurement, sun_sensor_measurements, q_io, w_i_io, r_i, JD = args
            self._update_from_sensors(gyro_measurement, magnetometer_measurement, sun_sensor_measurements, q_io, w_i_io, np.zeros(3), r_i, JD)
            return

        if len(args) == 10:
            _, _, gyro_measurement, magnetometer_measurement, sun_sensor_measurements, q_io, w_i_io, dw_i_io, r_i, JD = args
            self._update_from_sensors(gyro_measurement, magnetometer_measurement, sun_sensor_measurements, q_io, w_i_io, dw_i_io, r_i, JD)
            return

        raise ValueError("ADCS_PD.update received unsupported arguments")

    def _sign_q0(self, q_ob):
        s = np.sign(q_ob[0])
        if s == 0.0:
            s = 1.0
        return s

    def _relative_state(self, q_ib, w_b_ib, q_io, w_i_io, dw_i_io):
        q_ib = _q_array(q_ib)
        q_io = _q_array(q_io)
        w_b_ib = np.asarray(w_b_ib, dtype=float)
        w_i_io = np.asarray(w_i_io, dtype=float)
        dw_i_io = np.asarray(dw_i_io, dtype=float)

        q_ob = _q_mul(_q_conj(q_io), q_ib)
        q_ob = _q_array(q_ob)

        if q_ob[0] < 0.0:
            q_ob = -q_ob

        w_b_io = _q_rotate_inverse(q_ib, w_i_io)
        w_b_ob = w_b_ib - w_b_io
        dw_b_io = _q_rotate_inverse(q_ib, dw_i_io) + np.cross(w_b_io, w_b_ob)

        return q_ob, w_b_ob, w_b_io, dw_b_io

    def _update_from_true_attitude(self, q_ib, w_b_ib, q_io, w_i_io, dw_i_io):
        q_ib = _q_array(q_ib)
        w_b_ib = np.asarray(w_b_ib, dtype=float)
        q_ob, w_b_ob, w_b_io, dw_b_io = self._relative_state(q_ib, w_b_ib, q_io, w_i_io, dw_i_io)

        self.q_ob_est = q_ob
        self.q_ib_est = q_ib
        self.attitude_error = su.Quaternion(q_ob)
        self.angular_velocity_error = w_b_ob
        self.tau = self._control_law(q_ib, w_b_ib, q_ob, w_b_ob, w_b_io, dw_b_io)

    def _update_from_sensors(self, gyro_measurement, magnetometer_measurement, sun_sensor_measurements, q_io, w_i_io, dw_i_io, r_i, JD):
        q_io = _q_array(q_io)
        w_i_io = np.asarray(w_i_io, dtype=float)
        dw_i_io = np.asarray(dw_i_io, dtype=float)
        w_b_ib = np.asarray(gyro_measurement, dtype=float)

        mag_b = _unit(magnetometer_measurement)
        sun_vectors = [_unit(s) for s in sun_sensor_measurements if np.linalg.norm(s) > 1e-12]

        if len(sun_vectors) > 0:
            sun_b = _unit(np.sum(sun_vectors, axis=0))
        else:
            sun_b = np.zeros(3)

        self.sun_body = sun_b
        self.mag_body = mag_b

        s_i = _unit(ol.sun_vector(JD))
        B_i = _unit(ol.magnetic_field_dipole(r_i, JD))
        s_o = _unit(_q_rotate_inverse(q_io, s_i))
        B_o = _unit(_q_rotate_inverse(q_io, B_i))

        if self.attitude_estimator is not None and np.linalg.norm(sun_b) > 1e-12 and np.linalg.norm(mag_b) > 1e-12:
            try:
                q_est = self.attitude_estimator.estimate_attitude([s_o, B_o], [sun_b, mag_b])
                q_ob = _q_conj(q_est)
            except ValueError:
                q_ob = self.q_ob_est
        else:
            q_ob = self.q_ob_est

        q_ob = _q_array(q_ob)

        if q_ob[0] < 0.0:
            q_ob = -q_ob

        q_ib_est = _q_mul(q_io, q_ob)
        q_ib_est = _q_array(q_ib_est)

        q_ob, w_b_ob, w_b_io, dw_b_io = self._relative_state(q_ib_est, w_b_ib, q_io, w_i_io, dw_i_io)

        self.q_ob_est = q_ob
        self.q_ib_est = q_ib_est
        self.attitude_error = su.Quaternion(q_ob)
        self.angular_velocity_error = w_b_ob
        self.tau = self._control_law(q_ib_est, w_b_ib, q_ob, w_b_ob, w_b_io, dw_b_io)

    def _control_law(self, q_ib, w_b_ib, q_ob, w_b_ob, w_b_io, dw_b_io):
        q0_sign = self._sign_q0(q_ob)
        qv = q_ob[1:]

        return np.cross(w_b_ib, self.J @ w_b_ib) + self.J @ (
            dw_b_io - self.k1 * q0_sign * qv - self.k2 * w_b_ob
        )

    def get_control(self):
        return self.tau

    def get_actuation(self):
        return self.tau


class ADCS_SM(ADCS_PD):
    def __init__(self, k1, k, eps, f_c=None, J=None, attitude_estimator=None):
        super().__init__(k1, 0.0, f_c, J, attitude_estimator)
        self.k = k
        self.eps = eps

    def _control_law(self, q_ib, w_b_ib, q_ob, w_b_ob, w_b_io, dw_b_io):
        q0_sign = self._sign_q0(q_ob)
        q0 = q_ob[0]
        qv = q_ob[1:]

        qv_dot = q0 * w_b_ob + np.cross(qv, w_b_ob)
        self.sliding_surface = w_b_ob + 2.0 * self.k1 * q0_sign * qv
        sat_s = np.clip(self.sliding_surface / self.eps, a_min=-1.0, a_max=1.0)

        return np.cross(w_b_ib, self.J @ w_b_ib) + self.J @ (
            dw_b_io - self.k1 * q0_sign * qv_dot - self.k * sat_s
        )

class Satellite:
    def __init__(self, q_ib=None, w_b_ib=None, J=None, r=np.zeros(3), v=np.zeros(3), m=1, orbit=None, substeps=0, **kwargs):
        self.old_mode = False

        if 'orientation' in kwargs:
            self.old_mode = True

            self.rigid_body = RigidBody(
                orientation=kwargs['orientation'],
                angular_velocity=kwargs['angular_velocity'],
                inertia_matrix=kwargs['inertia_matrix']
            )

            self.desired_orientation = _q_array(kwargs['desired_orientation'])
            self.desired_angular_velocity = np.asarray(kwargs['desired_angular_velocity'], dtype=float)
            self.desired_angular_acceleration = np.asarray(kwargs.get('desired_angular_acceleration', np.zeros(3)), dtype=float)
            self.k1 = kwargs.get('k1', 1.0)
            self.k2 = kwargs.get('k2', 2.0)
            self.torque = np.zeros(3)
            self.attitude_error = su.Quaternion()
            self.angular_velocity_error = np.zeros(3)
            return

        self.orbit = orbit
        self.JD0 = kwargs.get('JD0', 2451545.0)
        self.use_sensors = kwargs.get('use_sensors', False)
        self.use_disturbances = kwargs.get('use_disturbances', False)
        self.noise_scale = kwargs.get('noise_scale', 1.0)
        self.controller = kwargs.get('controller', 'PD')
        estimator = kwargs.get('attitude_estimator', Davenport())

        if self.orbit is not None:
            r, v = self.orbit.get_state()

        self.body = RigidBody(r, v, m, q_ib, w_b_ib, J)
        self.N = substeps + 1

        if self.controller.upper() == 'SM':
            self.ADCS = ADCS_SM(
                kwargs.get('k1', 0.05),
                kwargs.get('sm_k', 0.03),
                kwargs.get('sm_eps', 0.02),
                J,
                attitude_estimator=estimator
            )
        else:
            self.ADCS = ADCS_PD(
                kwargs.get('k1', 1e-5),
                kwargs.get('k2', 2e-4),
                J,
                attitude_estimator=estimator
            )

        self.torque = np.zeros(3)
        self.tau_d = np.zeros(3)
        self.tau_g = np.zeros(3)
        self.tau_extra = np.zeros(3)
        self.attitude_error = su.Quaternion()
        self.angular_velocity_error = np.zeros(3)
        self.sensors = []
        self.gyro_sensor = None
        self.magnetometer_sensor = None
        self.sun_sensors = []
        self.last_measurements = {
            'gyro': np.zeros(3),
            'magnetometer': np.zeros(3),
            'sun': [],
            'JD': self.JD0
        }

        if self.use_sensors:
            self.init_default_sensors(self.noise_scale)

    def init_default_sensors(self, noise_scale=1.0):
        self.gyro_sensor = gyro(
            q_bs=np.array([1.0, 0.0, 0.0, 0.0]),
            p_b=np.array([0.0, 0.0, 0.0]),
            mu=0.0,
            Q=0.1 * noise_scale,
            params={'bg': np.zeros(3), 'sigma_bg': 0.0}
        )

        self.magnetometer_sensor = magnetometer(
            q_bs=np.array([1.0, 0.0, 0.0, 0.0]),
            p_b=np.array([0.0, 0.0, 0.0]),
            mu=0.0,
            Q=0.4e-8 * noise_scale,
            params={'bB': np.zeros(3), 'MB': np.eye(3)}
        )

        c = np.cos(np.pi / 4.0)
        s = np.sin(np.pi / 4.0)

        sensor_data = [
            (np.array([ c, 0.0,  s, 0.0]), np.array([ 0.1,  0.0,  0.0])),
            (np.array([ c, 0.0, -s, 0.0]), np.array([-0.1,  0.0,  0.0])),
            (np.array([ c,-s, 0.0, 0.0]), np.array([ 0.0,  0.1,  0.0])),
            (np.array([ c, s, 0.0, 0.0]), np.array([ 0.0, -0.1,  0.0])),
            (np.array([1.0,0.0, 0.0, 0.0]), np.array([ 0.0,  0.0,  0.1])),
            (np.array([0.0,1.0, 0.0, 0.0]), np.array([ 0.0,  0.0, -0.1]))
        ]

        self.sun_sensors = []

        for q_bs, p_b in sensor_data:
            self.sun_sensors.append(
                fine_sun_sensor(
                    q_bs=q_bs,
                    p_b=p_b,
                    mu=0.0,
                    Q=0.2 * noise_scale,
                    params={'alpha': np.pi}
                )
            )

        self.sensors = [self.gyro_sensor, self.magnetometer_sensor] + self.sun_sensors

    def update_sensors(self, t_k, t_sub, q_ib, w_b_ib, r_i, v_i):
        JD = self.JD0 + t_k / 86400.0

        for sensor in self.sensors:
            sensor.update(t_k, t_sub, q_ib, w_b_ib, r_i, v_i, JD)

        gyro_measurement = self.gyro_sensor.output(body_frame=True)
        magnetometer_measurement = self.magnetometer_sensor.output(body_frame=True)
        sun_measurements = [sensor.output(body_frame=True) for sensor in self.sun_sensors]

        self.last_measurements = {
            'gyro': gyro_measurement,
            'magnetometer': magnetometer_measurement,
            'sun': sun_measurements,
            'JD': JD
        }

        return gyro_measurement, magnetometer_measurement, sun_measurements, JD

    def disturbance_torque(self, t_k, r_i, q_ib):
        if not self.use_disturbances:
            self.tau_g = np.zeros(3)
            self.tau_extra = np.zeros(3)
            self.tau_d = np.zeros(3)
            return self.tau_d

        self.tau_g = ol.gravity_gradient(r_i, q_ib, self.body.inertia_matrix)
        self.tau_extra = self.body.inertia_matrix @ np.array([
            0.01 * np.sin(0.01 * t_k),
            0.0,
            0.01 * np.cos(0.01 * t_k)
        ])
        self.tau_d = self.tau_g + self.tau_extra

        return self.tau_d

    def update(self, t_k, t_step):
        if self.old_mode:
            self.torque = self.control_torque()
            self.rigid_body.update(t_k, t_step, self.torque)
            self.update_reference(t_step)
            self.torque = self.control_torque()
            return

        if self.orbit:
            self.update_with_orbit(t_k, t_step)
        else:
            self.update_with_dynamics(t_k, t_step)

        self.torque = self.ADCS.get_control()
        self.attitude_error = self.ADCS.attitude_error
        self.angular_velocity_error = self.ADCS.angular_velocity_error

    def update_with_orbit(self, t_k, t_step):
        r_0, v_0 = self.orbit.get_state()
        self.orbit.propagate(t_step)
        r_1, v_1 = self.orbit.get_state()

        t_sub = t_step / self.N

        for n in range(0, self.N):
            r_i = r_0 + n / self.N * (r_1 - r_0)
            v_i = v_0 + n / self.N * (v_1 - v_0)

            _, _, q_ib, w_b_ib = self.body.get_state()
            q_io, w_i_io, dw_i_io = ol.orbit_frame_from_state(r_i, v_i)

            if self.use_sensors:
                gyro_m, mag_m, sun_m, JD = self.update_sensors(t_k, t_sub, q_ib, w_b_ib, r_i, v_i)
                self.ADCS.update(t_k, t_sub, gyro_m, mag_m, sun_m, q_io, w_i_io, dw_i_io, r_i, JD)
            else:
                self.ADCS.update(t_k, t_sub, q_ib, w_b_ib, q_io, w_i_io, dw_i_io)

            tau_u = self.ADCS.get_control()
            tau_d = self.disturbance_torque(t_k, r_i, q_ib)
            self.body.update(t_k, t_sub, np.zeros(3), tau_u + tau_d)
            t_k += t_sub

        self.body.p, self.body.v = self.orbit.get_state()

    def update_with_dynamics(self, t_k, t_step):
        t_sub = t_step / self.N

        for _ in range(0, self.N):
            r_i, v_i, q_ib, w_b_ib = self.body.get_state()
            q_io, w_i_io, dw_i_io = self.get_orbit_frame()

            if self.use_sensors:
                gyro_m, mag_m, sun_m, JD = self.update_sensors(t_k, t_sub, q_ib, w_b_ib, r_i, v_i)
                self.ADCS.update(t_k, t_sub, gyro_m, mag_m, sun_m, q_io, w_i_io, dw_i_io, r_i, JD)
            else:
                self.ADCS.update(t_k, t_sub, q_ib, w_b_ib, q_io, w_i_io, dw_i_io)

            tau_u = self.ADCS.get_control()
            tau_d = self.disturbance_torque(t_k, r_i, q_ib)
            f = -ol.mu / np.linalg.norm(r_i) ** 3 * r_i
            self.body.update(t_k, t_sub, f, tau_u + tau_d)
            t_k += t_sub

    def get_state(self):
        if self.old_mode:
            return self.rigid_body.get_state()
        return self.body.get_state()

    def get_orbit_frame(self):
        if self.orbit:
            return self.orbit.get_orbit_frame()
        else:
            r, v, _, _ = self.body.get_state()
            return ol.orbit_frame_from_state(r, v)

    def get_reference_state(self):
        if self.old_mode:
            return self.desired_orientation, self.desired_angular_velocity

        q_io, w_i_io, _ = self.get_orbit_frame()
        return _q_array(q_io), w_i_io

    def get_sensor_measurements(self):
        return self.last_measurements

    def update_reference(self, h):
        w = self.desired_angular_velocity
        w_norm = np.linalg.norm(w)

        if w_norm > 1e-12:
            dq = su.Quaternion(h * w_norm, w / w_norm).q
            self.desired_orientation = _q_mul(self.desired_orientation, dq)
            self.desired_orientation = _q_array(self.desired_orientation)

        self.desired_angular_velocity = self.desired_angular_velocity + h * self.desired_angular_acceleration

    def control_torque(self):
        q_ob, w_ob_b = self.rigid_body.get_state()

        q_od = self.desired_orientation
        w_od_d = self.desired_angular_velocity

        q_db = _q_mul(_q_conj(q_od), q_ob)
        q_db = _q_array(q_db)

        if q_db[0] < 0.0:
            q_db = -q_db

        w_od_b = _q_rotate_inverse(q_db, w_od_d)
        w_db_b = w_ob_b - w_od_b

        self.attitude_error = su.Quaternion(q_db)
        self.angular_velocity_error = w_db_b

        return -self.k1 * q_db[1:] - self.k2 * w_db_b

# Assignment 9 helper functions

def random_unit_vector(rng):
    u = rng.normal(0.0, 1.0, 3)
    n = np.linalg.norm(u)

    if n < 1.0e-12:
        return np.array([1.0, 0.0, 0.0])

    return u / n


def star_tracker_measurement(q_ib, rng, mu=0.0, Q=1.0e-2):
    """Simplified star tracker quaternion measurement.

    Q is used directly as the standard deviation, matching the assignment
    listing where Q is passed directly to np.random.normal().
    """
    theta_noise = rng.normal(mu, Q)
    u = random_unit_vector(rng)

    q_e = np.array([
        np.cos(theta_noise / 2.0),
        *(np.sin(theta_noise / 2.0) * u)
    ])

    return _q_array(_q_mul(q_ib, q_e))


def _q_rotated_basis_vector(q, idx):
    q = _q_array(q)
    q0, q1, q2, q3 = q

    if idx == 0:
        return np.array([
            1.0 - 2.0 * (q2*q2 + q3*q3),
            2.0 * (q1*q2 + q0*q3),
            2.0 * (q1*q3 - q0*q2)
        ])

    if idx == 1:
        return np.array([
            2.0 * (q1*q2 - q0*q3),
            1.0 - 2.0 * (q1*q1 + q3*q3),
            2.0 * (q2*q3 + q0*q1)
        ])

    return np.array([
        2.0 * (q1*q3 + q0*q2),
        2.0 * (q2*q3 - q0*q1),
        1.0 - 2.0 * (q1*q1 + q2*q2)
    ])


def _cross_basis_with_vector(idx, v):
    if idx == 0:
        return np.array([0.0, -v[2], v[1]])

    if idx == 1:
        return np.array([v[2], 0.0, -v[0]])

    return np.array([-v[1], v[0], 0.0])


def average_star_trackers(q_measurements):
    if len(q_measurements) == 1:
        return _q_array(q_measurements[0])

    B = np.zeros((3, 3))
    z = np.zeros(3)

    weight = 1.0 / (2.0 * len(q_measurements))

    for k, q_m in enumerate(q_measurements):
        idx_a = k % 3
        idx_b = (k + 1) % 3

        m_a = _q_rotated_basis_vector(q_m, idx_a)
        m_b = _q_rotated_basis_vector(q_m, idx_b)

        B[idx_a, :] += weight * m_a
        B[idx_b, :] += weight * m_b

        z += weight * _cross_basis_with_vector(idx_a, m_a)
        z += weight * _cross_basis_with_vector(idx_b, m_b)

    tr = np.trace(B)

    K = np.zeros((4, 4))
    K[0, 0] = tr
    K[0, 1:] = z
    K[1:, 0] = z
    K[1:, 1:] = B + B.T - tr * np.eye(3)

    evals, evecs = np.linalg.eigh(K)
    q_hat = evecs[:, np.argmax(evals)]

    if q_hat[0] < 0.0:
        q_hat = -q_hat

    return _q_array(q_hat)


def quaternion_error(q_desired, q_actual):
    q_err = _q_array(_q_mul(_q_conj(q_desired), q_actual))

    if q_err[0] < 0.0:
        q_err = -q_err

    return q_err


def pointing_error_arcsec(q_err):
    q_err = _q_array(q_err)
    qv_norm = min(1.0, np.linalg.norm(q_err[1:]))
    return 2.0 * 180.0 * 3600.0 / np.pi * np.arcsin(qv_norm)



def solar_array_torque(t, A1=0.2, A2=0.2, p1=0.14*np.pi, p2=1.22*np.pi,
                                   phi1=0.31*np.pi, phi2=-0.05*np.pi):
    d = A1 * np.sin(p1 * t + phi1) + A2 * np.sin(p2 * t + phi2)
    return np.array([0.0, d, 0.0])


def control_torque(controller, q_err_est, w_hat_b_ib, J,
                               pd_k1, pd_k2, sm_k1, sm_k, sm_eps):
    q_err_est = _q_array(q_err_est)
    w_hat_b_ib = np.asarray(w_hat_b_ib, dtype=float)
    J = np.asarray(J, dtype=float)

    if controller.upper() == "PD":
        s = np.zeros(3)
        tau_c = np.cross(w_hat_b_ib, J @ w_hat_b_ib) + J @ (
            -pd_k1 * q_err_est[1:] - pd_k2 * w_hat_b_ib
        )
        return tau_c, s

    qv_dot = q_err_est[0] * w_hat_b_ib + np.cross(q_err_est[1:], w_hat_b_ib)
    s = w_hat_b_ib + 2.0 * sm_k1 * q_err_est[1:]
    sat_s = np.clip(s / sm_eps, -1.0, 1.0)
    tau_c = np.cross(w_hat_b_ib, J @ w_hat_b_ib) + J @ (
        -sm_k1 * qv_dot - sm_k * sat_s
    )
    return tau_c, s


def attitude_step(t, dt, r_i, q_ib, w_b_ib, q_desired, J, J_inv,
                  controller, star_trackers, rng, actuator_limit,
                  gyro_mu, gyro_variance, star_mu, star_Q,
                  pd_k1, pd_k2, sm_k1, sm_k, sm_eps,
                  solar_a1, solar_a2, solar_p1, solar_p2,
                  solar_phi1, solar_phi2):

    q_measurements_all = []

    # Always generate 3 tracker measurements so 1ST and 3ST consume
    # the same random sequence before gyro noise is generated.
    for _ in range(3):
        q_measurements_all.append(star_tracker_measurement(q_ib, rng, star_mu, star_Q))

    q_hat_ib = average_star_trackers(q_measurements_all[:star_trackers])

    w_hat_b_ib = w_b_ib + rng.normal(gyro_mu, np.sqrt(gyro_variance), 3)

    q_err_est = quaternion_error(q_desired, q_hat_ib)
    tau_c, s = control_torque(
        controller, q_err_est, w_hat_b_ib, J,
        pd_k1, pd_k2, sm_k1, sm_k, sm_eps
    )
    tau_a = np.clip(tau_c, -actuator_limit, actuator_limit)

    tau_g = ol.gravity_gradient(r_i, q_ib, J)
    tau_solar = solar_array_torque(
        t, solar_a1, solar_a2, solar_p1, solar_p2, solar_phi1, solar_phi2
    )
    tau_d = tau_g + tau_solar
    tau_total = tau_a + tau_d

    q_err_true = quaternion_error(q_desired, q_ib)
    true_arcsec = pointing_error_arcsec(q_err_true)
    est_arcsec = pointing_error_arcsec(q_err_est)

    row = [
        t,
        true_arcsec,
        est_arcsec,
        np.linalg.norm(q_err_true[1:]),
        np.linalg.norm(w_b_ib),
        np.linalg.norm(w_hat_b_ib),
        tau_c[0], tau_c[1], tau_c[2], np.linalg.norm(tau_c),
        tau_a[0], tau_a[1], tau_a[2], np.linalg.norm(tau_a),
        tau_g[0], tau_g[1], tau_g[2], np.linalg.norm(tau_g),
        tau_solar[0], tau_solar[1], tau_solar[2], np.linalg.norm(tau_solar),
        tau_d[0], tau_d[1], tau_d[2], np.linalg.norm(tau_d),
        s[0], s[1], s[2], np.linalg.norm(s)
    ]

    w_dot = J_inv @ (tau_total - np.cross(w_b_ib, J @ w_b_ib))
    q_dot = 0.5 * _q_mul(q_ib, np.array([0.0, *w_b_ib]))

    w_next = w_b_ib + dt * w_dot
    q_next = _q_array(q_ib + dt * q_dot)

    return q_next, w_next, row

def part2_header():
    return (
        "time_s true_error_arcsec estimated_error_arcsec true_qv_norm true_w_norm measured_w_norm "
        "tau_c_x tau_c_y tau_c_z tau_c_norm tau_a_x tau_a_y tau_a_z tau_a_norm "
        "tau_g_x tau_g_y tau_g_z tau_g_norm tau_solar_x tau_solar_y tau_solar_z tau_solar_norm "
        "tau_d_x tau_d_y tau_d_z tau_d_norm s_x s_y s_z s_norm"
    )
