import numpy as np
import simutils as su


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
    return (su.Quaternion(q1) @ su.Quaternion(q2)).q


def _q_conj(q):
    q = _q_array(q)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _q_rotate_inverse(q, v):
    q = _q_array(q)
    return (su.Quaternion(_q_conj(q)) @ su.Quaternion(v) @ su.Quaternion(q)).q[1:]


class RigidBody:
    def __init__(self, orientation, angular_velocity, inertia_matrix):
        self.orientation = _q_array(orientation)
        self.angular_velocity = np.asarray(angular_velocity, dtype=float)
        self.inertia_matrix = np.asarray(inertia_matrix, dtype=float)
        self.torque = np.zeros(3)

        if self.angular_velocity.shape != (3,):
            raise ValueError("Angular velocity must have shape (3,)")

        if self.inertia_matrix.shape != (3, 3):
            raise ValueError("Inertia matrix must have shape (3,3)")

    def state(self):
        return np.concatenate((self.orientation, self.angular_velocity))

    def get_state(self):
        return self.orientation, self.angular_velocity

    def update(self, t_k, h, torque):
        self.torque = np.asarray(torque, dtype=float)

        x_k = self.state()
        x_next = su.step_RK4(h, t_k, x_k, self.f)

        self.orientation = _q_array(x_next[:4])
        self.angular_velocity = x_next[4:]

    def f(self, t, x):
        q = _q_array(x[:4])
        w = np.asarray(x[4:], dtype=float)

        q_dot = 0.5 * _q_mul(q, np.concatenate(([0.0], w)))

        w_dot = np.linalg.solve(
            self.inertia_matrix,
            self.torque - np.cross(w, self.inertia_matrix @ w)
        )

        return np.concatenate((q_dot, w_dot))


class Satellite:
    def __init__(
            self,
            orientation,
            angular_velocity,
            inertia_matrix,
            desired_orientation,
            desired_angular_velocity,
            desired_angular_acceleration=None,
            k1=1.0,
            k2=2.0
    ):
        self.rigid_body = RigidBody(
            orientation,
            angular_velocity,
            inertia_matrix
        )

        self.desired_orientation = _q_array(desired_orientation)
        self.desired_angular_velocity = np.asarray(desired_angular_velocity, dtype=float)

        if desired_angular_acceleration is None:
            self.desired_angular_acceleration = np.zeros(3)
        else:
            self.desired_angular_acceleration = np.asarray(desired_angular_acceleration, dtype=float)

        self.k1 = k1
        self.k2 = k2

        self.torque = np.zeros(3)
        self.attitude_error = su.Quaternion()
        self.angular_velocity_error = np.zeros(3)

    def update(self, t_k, h):
        self.torque = self.control_torque()
        self.rigid_body.update(t_k, h, self.torque)
        self.update_reference(h)
        self.torque = self.control_torque()

    def get_state(self):
        return self.rigid_body.get_state()

    def get_reference_state(self):
        return self.desired_orientation, self.desired_angular_velocity

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