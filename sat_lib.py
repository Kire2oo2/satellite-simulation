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

    def f_full(self, t, x):
        p = np.asarray(x[:3], dtype=float)
        v = np.asarray(x[3:6], dtype=float)
        q = _q_array(x[6:10])
        w = np.asarray(x[10:13], dtype=float)

        p_dot = v
        v_dot = self.acceleration
        q_dot = 0.5 * _q_mul(q, np.concatenate(([0.0], w)))
        w_dot = np.linalg.solve(
            self.inertia_matrix,
            self.torque - np.cross(w, self.inertia_matrix @ w)
        )

        return np.concatenate((p_dot, v_dot, q_dot, w_dot))


class ADCS_PD:
    def __init__(self, k1, k2, f_c=None, J=None):
        if J is None:
            J = f_c
            f_c = np.zeros(3)

        self.k1 = k1
        self.k2 = k2
        self.f_c = f_c
        self.J = np.asarray(J, dtype=float)
        self.tau = np.zeros(3)
        self.attitude_error = su.Quaternion()
        self.angular_velocity_error = np.zeros(3)

    def update(self, *args):
        if len(args) == 4:
            q_ib, w_b_ib, q_io, w_i_io = args
        elif len(args) == 6:
            _, q_ib, w_b_ib, q_io, w_i_io, _ = args
        else:
            raise ValueError("ADCS_PD.update expects q_ib, w_b_ib, q_io, w_i_io")

        q_ib = _q_array(q_ib)
        q_io = _q_array(q_io)
        w_b_ib = np.asarray(w_b_ib, dtype=float)
        w_i_io = np.asarray(w_i_io, dtype=float)

        q_ob = _q_mul(_q_conj(q_io), q_ib)
        q_ob = _q_array(q_ob)

        if q_ob[0] < 0.0:
            q_ob = -q_ob

        w_o_io = _q_rotate_inverse(q_io, w_i_io)
        w_o_io_b = _q_rotate_inverse(q_ob, w_o_io)
        w_ob_b = w_b_ib - w_o_io_b

        self.attitude_error = su.Quaternion(q_ob)
        self.angular_velocity_error = w_ob_b
        self.tau = -self.k1 * q_ob[1:] - self.k2 * w_ob_b

    def get_control(self):
        return self.tau


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

        if self.orbit is not None:
            r, v = self.orbit.get_state()

        self.body = RigidBody(r, v, m, q_ib, w_b_ib, J)
        self.N = substeps + 1
        self.ADCS = ADCS_PD(1e-5, 2e-4, J)
        self.torque = np.zeros(3)
        self.attitude_error = su.Quaternion()
        self.angular_velocity_error = np.zeros(3)

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
            q_io, w_i_io, _ = ol.orbit_frame_from_state(r_i, v_i)

            self.ADCS.update(q_ib, w_b_ib, q_io, w_i_io)
            tau_u = self.ADCS.get_control()

            self.body.update(t_k, t_sub, np.zeros(3), tau_u)
            t_k += t_sub

        self.body.p, self.body.v = self.orbit.get_state()

    def update_with_dynamics(self, t_k, t_step):
        t_sub = t_step / self.N

        for _ in range(0, self.N):
            r_i, v_i, q_ib, w_b_ib = self.body.get_state()
            q_io, w_i_io, dw_i_io = self.get_orbit_frame()

            self.ADCS.update(t_k, q_ib, w_b_ib, q_io, w_i_io, dw_i_io)
            tau_u = self.ADCS.get_control()

            f = -ol.mu / np.linalg.norm(r_i) ** 3 * r_i
            self.body.update(t_k, t_sub, f, tau_u)
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
