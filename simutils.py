import datetime as dt
import numpy as np
from vispy.scene import MatrixTransform as Mat4
from vispy.util.quaternion import Quaternion as Quat

class Error(Exception):
    pass

class InvalidConstruction(Error):
    def __init__(self,message):
        self.message=message

class Quaternion:
    def __init__(self, arg1 = None, arg2 = None):
        if arg1 is None and arg2 is None:
            self.q = np.array([1,0,0,0])
        elif arg1 is not None and arg2 is None:
            if type(arg1) is Quaternion:
                self.q = np.array(arg1.q)
            elif len(arg1) == 4:
                self.q = np.array(arg1)
            elif len(arg1) == 3:
                self.q = np.array([0,*arg1])
            else:
                raise InvalidConstruction("Wrong initialization, expects one of:\narg1=None,arg2=None\narg1=Quaternion,arg2=None\narg1=list[4],arg2=None\narg1=list[3],arg2=None\narg1=float,arg2=list[3]\n")
        elif arg1 is not None and arg2 is not None:
            if len(arg2) == 3:
                mag = np.sqrt(arg2[0]**2.0+arg2[1]**2.0+arg2[2]**2.0)
                self.q = np.array([np.cos(arg1/2.0),*(np.sin(arg1/2.0)/mag*np.array(arg2))])
            else:
                raise InvalidConstruction("Wrong initialization, expects one of:\narg1=None,arg2=None\narg1=Quaternion,arg2=None\narg1=list[4],arg2=None\narg1=list[3],arg2=None\narg1=float,arg2=list[3]\n")
        else:
            raise InvalidConstruction("Wrong initialization, expects one of:\narg1=None,arg2=None\narg1=Quaternion,arg2=None\narg1=list[4],arg2=None\narg1=list[3],arg2=None\narg1=float,arg2=list[3]\n")

    def __len__(self):
        return len(self.q)

    def __repr__(self):
        return "Quaternion: [{}]".format(",".join([str(x) for x in self.q]))

    def __getitem__(self,index):
        if type(index) == slice:
            if index.stop < index.start:
                raise IndexError("starting index should be smaller than ending index")
            elif index.start in range(0,len(self)+1) and index.stop in range(0,len(self)+1):
                return np.array([self[i] for i in range(index.start,index.stop+1)])
            else:
                raise IndexError("Indexes out of bounds")
        else:
            if index > 3:
                raise IndexError("Index out of bounds")
            else:
                return self.q[index]

    def __add__(self,other):
        return Quaternion(self.q+other.q)

    def __sub__(self,other):
        return Quaternion(self.q-other.q)

    def __mul__(self,other):
        return Quaternion(self.q*other)

    def __rmul__(self,other):
        return self*other

    def __truediv__(self,other):
        return 1/other*self

    def __matmul__(self,other):
        return Quaternion([self[0]*other[0]-np.dot(self[1:3],other[1:3]), *(self[0]*other[1:3]+other[0]*self[1:3]+np.cross(self[1:3],other[1:3]))])

    def inverted(self):
        mag = self.magnitude()
        if mag < 1e-9:
            raise IndexError("Magnitude is zero")
        return 1.0/mag**2.0*self.conjugated()

    def conjugated(self):
        return Quaternion([self[0],*(-self[1:3])])

    def normalized(self):
        mag = self.magnitude()
        return Quaternion(self.q/mag)

    def invert(self):
        mag = self.magnitude()
        if mag < 1e-9:
            raise IndexError("Magnitude is zero")
        self.q /= mag**2.0

    def conjugate(self):
        self.q = np.array([self[0],*(-self[1:3])])

    def normalize(self):
        mag = self.magnitude()
        if mag < 1e-9:
            raise IndexError("Magnitude is zero")
        self.q = self.q/mag

    def magnitude(self):
        return np.linalg.norm(self.q)

    def rotate(self, u):
        v = self@Quaternion(u)@self.conjugated()
        return v[1:3]

def _tle_exp_to_float(s):
    s = s.strip()

    if s in ("", "0", "00000+0", "00000-0"):
        return 0.0

    if "e" in s.lower():
        return float(s)

    mantissa = s[:-2]
    exponent = s[-2:]

    if mantissa[0] in "+-":
        return float(mantissa[0] + "0." + mantissa[1:] + "e" + exponent)

    return float("0." + mantissa + "e" + exponent)


def read_TLE_file(file_name, satellite_name=''):
  def validate_entry(Name, line1, line2):
    if not Name[0].isalpha():
      return False
    if not line1[0].startswith("1") or not len(line1) == 9:
      return False
    if not line2[0].startswith("2") or not len(line2) == 8:
      return False
    return True

  tle_data = []

  with open(file_name) as f:
    file_contents = f.readlines()

  if len(file_contents) < 3:
    print("Error reading file")
    return tle_data

  for i in range(0, len(file_contents), 3):
    if satellite_name in file_contents[i]:
      Name = file_contents[i].strip()
      line1 = file_contents[i + 1].strip().split()
      line2 = file_contents[i + 2].strip().split()

      if validate_entry(Name, line1, line2):
        _, _, _, sepoch, sdn, sddn, sbstar, _, _ = line1
        _, _, si, sO, secc, sw, sM, srev = line2

        epoch = float(sepoch)
        dn = float(sdn)
        ddn = _tle_exp_to_float(sddn)
        bstar = _tle_exp_to_float(sbstar)
        e = float("." + secc)
        rev = float(srev[:-6])
        Me = float(sM)
        inc = float(si)
        O = float(sO)
        w = float(sw)

        tle_data.append((Name, epoch, e, rev, Me, inc, O, w, dn, ddn, bstar))
      else:
        print("Error reading entry")
        break

  return tle_data


def read_obj(fname):
    verts = []
    vcols = []
    faces = []
    with open(fname,'r') as f:
        for line in f:
            if line.startswith('v '):
                d = [float(x) for x in line.split(' ')[1:]]
                verts.append(d[0:3])
                if len(d) > 3:
                    vcols.append(d[3:])
            elif line.startswith('f '):
                faces.append([int(x.split('/')[0])-1 for x in line.split(' ')[1:]])
            else:
                pass
    return np.array(verts),np.array(vcols),np.array(faces)

def rotscaleloc_to_vispy(pos=None,quat=None,Rot=None,Eul=None,scale=None):
    if quat is not None:
        q = Quat(w=quat[0],x=quat[1],y=quat[2],z=quat[3])
        H = Mat4(q.conjugate().get_matrix())
    elif Rot is not None:
        p = np.array([[0,0,0]]).T
        HT = np.vstack(((np.hstack((Rot,p)),np.array([[0,0,0,1]]))))
        H = Mat4(HT.T)
    elif Eul is not None:
        q = Quat.create_from_euler_angles(Eul[2],Eul[1],Eul[0])
        H = Mat4(q.conjugate().get_matrix())
    else:
        H = Mat4()
    if scale is not None:
        H.scale((scale,scale,scale))
    if pos is not None:

        H.translate(pos)
    return H

def H_to_Rp(H):
    return H.matrix[:3,:3].T,H.matrix[-1][:3]

def log_pos(name,pos):
    file_name = 'data/'+name+'.txt'
    print("logged: "+file_name)
    np.savetxt(file_name,pos)


# Assignment 3 numeric solvers

def step_euler(h, t_k, x_k, f):
    x_k = np.asarray(x_k, dtype=float)
    return x_k + h * f(t_k, x_k)


def step_leapfrog(h, t_k, x_k, f):
    x_k = np.asarray(x_k, dtype=float)

    n = len(x_k) // 2
    r_k = x_k[:n]
    v_k = x_k[n:]

    dx_k = f(t_k, x_k)
    a_k = dx_k[n:]

    v_half = v_k + 0.5 * h * a_k
    r_next = r_k + h * v_half

    x_half = np.concatenate((r_next, v_half))
    a_next = f(t_k + h, x_half)[n:]

    v_next = v_half + 0.5 * h * a_next

    return np.concatenate((r_next, v_next))


def step_verlet(h, t_k, x_k, x_km1, f):
    x_k = np.asarray(x_k, dtype=float)

    n = len(x_k) // 2
    r_k = x_k[:n]
    v_k = x_k[n:]

    dx_k = f(t_k, x_k)
    a_k = dx_k[n:]

    if x_km1 is None:
        r_next = r_k + h * v_k + 0.5 * h**2 * a_k
    else:
        x_km1 = np.asarray(x_km1, dtype=float)
        r_prev = x_km1[:n]
        r_next = 2 * r_k - r_prev + h**2 * a_k

    v_half = (r_next - r_k) / h
    x_half = np.concatenate((r_next, v_half))
    a_next = f(t_k + h, x_half)[n:]

    v_next = v_half + 0.5 * h * a_next

    return np.concatenate((r_next, v_next))


def step_RK4(h, t_k, x_k, f):
    x_k = np.asarray(x_k, dtype=float)

    k1 = f(t_k, x_k)
    k2 = f(t_k + 0.5 * h, x_k + 0.5 * h * k1)
    k3 = f(t_k + 0.5 * h, x_k + 0.5 * h * k2)
    k4 = f(t_k + h, x_k + h * k3)

    return x_k + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# Assignment 4 attitude functions


def skew_symmetric(v):
    v = np.asarray(v, dtype=float)
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])


def _as_quat_array(q):
    if isinstance(q, Quaternion):
        q = q.q

    q = np.asarray(q, dtype=float)

    if q.shape != (4,):
        raise ValueError("Quaternion must have shape (4,)")

    n = np.linalg.norm(q)

    if n < 1e-12:
        raise ValueError("Quaternion magnitude is zero")

    return q / n


def _unpack_euler(args):
    if len(args) == 1:
        eul = np.asarray(args[0], dtype=float)

        if eul.shape != (3,):
            raise ValueError("Euler angles must be [roll, pitch, yaw]")

        return eul[0], eul[1], eul[2]

    if len(args) == 3:
        return float(args[0]), float(args[1]), float(args[2])

    raise ValueError("Expected either one Euler vector or roll, pitch, yaw")


def quaternion_to_dcm(q):
    q0, q1, q2, q3 = _as_quat_array(q)

    return np.array([
        [
            q0*q0 + q1*q1 - q2*q2 - q3*q3,
            2.0 * (q1*q2 - q0*q3),
            2.0 * (q1*q3 + q0*q2)
        ],
        [
            2.0 * (q1*q2 + q0*q3),
            q0*q0 - q1*q1 + q2*q2 - q3*q3,
            2.0 * (q2*q3 - q0*q1)
        ],
        [
            2.0 * (q1*q3 - q0*q2),
            2.0 * (q2*q3 + q0*q1),
            q0*q0 - q1*q1 - q2*q2 + q3*q3
        ]
    ])


def axis_angle_to_dcm(angle, axis=None):
    if axis is None:
        axis_angle = np.asarray(angle, dtype=float)
        theta = np.linalg.norm(axis_angle)

        if theta < 1e-12:
            return np.eye(3)

        u = axis_angle / theta
    else:
        theta = float(angle)
        u = np.asarray(axis, dtype=float)
        n = np.linalg.norm(u)

        if n < 1e-12:
            return np.eye(3)

        u = u / n

    S = skew_symmetric(u)

    return np.eye(3) + np.sin(theta) * S + (1.0 - np.cos(theta)) * (S @ S)


def dcm_to_quaternion(R):
    R = np.asarray(R, dtype=float)

    if R.shape != (3, 3):
        raise ValueError("DCM must have shape (3,3)")

    tr = np.trace(R)

    if tr > 0.0:
        q0 = 0.5 * np.sqrt(1.0 + tr)
        q1 = (R[2, 1] - R[1, 2]) / (4.0 * q0)
        q2 = (R[0, 2] - R[2, 0]) / (4.0 * q0)
        q3 = (R[1, 0] - R[0, 1]) / (4.0 * q0)
    else:
        i = np.argmax(np.diag(R))

        if i == 0:
            q1 = 0.5 * np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2]))
            q0 = (R[2, 1] - R[1, 2]) / (4.0 * q1)
            q2 = (R[0, 1] + R[1, 0]) / (4.0 * q1)
            q3 = (R[0, 2] + R[2, 0]) / (4.0 * q1)

        elif i == 1:
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

    return Quaternion(q)


def euler_to_dcm(*args):
    roll, pitch, yaw = _unpack_euler(args)

    R1 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(roll), -np.sin(roll)],
        [0.0, np.sin(roll), np.cos(roll)]
    ])

    R2 = np.array([
        [np.cos(pitch), 0.0, np.sin(pitch)],
        [0.0, 1.0, 0.0],
        [-np.sin(pitch), 0.0, np.cos(pitch)]
    ])

    R3 = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0]
    ])

    return R3 @ R2 @ R1


def dcm_to_euler(R):
    R = np.asarray(R, dtype=float)

    if R.shape != (3, 3):
        raise ValueError("DCM must have shape (3,3)")

    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[1, 0], R[0, 0])

    return np.array([roll, pitch, yaw])


def euler_to_quaternion(*args):
    roll, pitch, yaw = _unpack_euler(args)

    q_roll = Quaternion(roll, np.array([1.0, 0.0, 0.0]))
    q_pitch = Quaternion(pitch, np.array([0.0, 1.0, 0.0]))
    q_yaw = Quaternion(yaw, np.array([0.0, 0.0, 1.0]))

    q = q_yaw @ q_pitch @ q_roll
    q.normalize()

    return q


def quaternion_to_euler(q):
    q0, q1, q2, q3 = _as_quat_array(q)

    roll = np.arctan2(
        2.0 * (q0*q1 + q2*q3),
        q0*q0 + q3*q3 - q1*q1 - q2*q2
    )

    pitch = np.arcsin(
        np.clip(2.0 * (q0*q2 - q1*q3), -1.0, 1.0)
    )

    yaw = np.arctan2(
        2.0 * (q0*q3 + q1*q2),
        q0*q0 + q1*q1 - q2*q2 - q3*q3
    )

    return np.array([roll, pitch, yaw])