# Copyright 2020 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Rigid-body transformations including velocities and static forces."""


from absl import logging
import mujoco._functions as mj_transformations
import numpy as np

# Constants used to determine when a rotation is close to a pole.
_POLE_LIMIT = (1.0 - 1e-6)
_TOL = 1e-10


def _pythonize_2quat_conversion(func):
    """Pythonizes a MuJoCo rotation function.

    Args:
      func: A MuJoCo rotation function.
      *args: Positional arguments to `func`.
      **kwargs: Keyword arguments to `func`.

    Returns:
      An altered version of `func` that does not require a pre-allocated output
    """
    def pythonic_mju_2quat_conversion_func(*args, **kwargs):
        """Pythonized version of `func`."""
        out = np.zeros(4)
        func(out, *args, **kwargs)
        return out

    return pythonic_mju_2quat_conversion_func


ROTATION_2QUAT_CONVERTER = {
  'quat': lambda quat: quat,
  'axisangle': lambda axang: _pythonize_2quat_conversion(mj_transformations.mju_axisAngle2Quat)(axang[:3], axang[3]),
  'euler': lambda euler, eulerseq: euler_to_quat(euler, ordering=eulerseq),
  'xyaxes': lambda xyaxes, _xyaxes_to_quat: _xyaxes_to_quat(xyaxes),
  'zaxis': lambda z: _pythonize_2quat_conversion(mj_transformations.mju_quatZ2Vec)(z),
}

ROTATIONS_3D = list(ROTATION_2QUAT_CONVERTER.keys())


def _xyaxes_to_quat(xyaxes):
  """Converts xyaxes to quaternion.

  Args:
    xyaxes: (np.ndarray) xyaxes with shape (..., 6).

    Returns:
      quaternion with shape (..., 4).

  """
  # separate x and y axes
  x = xyaxes[..., :3]
  y = xyaxes[..., 3:]

  # get new y that is orthogonal to x
  proj_y_on_x = np.dot(y, x) * x
  y = y - proj_y_on_x

  # normalize axes
  x = x / np.linalg.norm(x, axis=-1, keepdims=True)
  y = y / np.linalg.norm(y, axis=-1, keepdims=True)

  # get z as cross product
  # this already follows the right-hand rule and requires no further normalization
  z = np.cross(x, y)

  # get rotation matrix
  m = np.stack([x, y, z], axis=-2)

  # convert to quaternion
  return mat_to_quat(m)


def _clip_within_precision(number, low, high, precision=_TOL):
  """Clips input to provided range, checking precision.

  Args:
    number: (float) number to be clipped.
    low: (float) lower bound.
    high: (float) upper bound.
    precision: (float) tolerance.

  Returns:
    Input clipped to given range.

  Raises:
    ValueError: If number is outside given range by more than given precision.
  """
  if (number < low - precision).any() or (number > high + precision).any():
    raise ValueError(
        'Input {:.12f} not inside range [{:.12f}, {:.12f}] with precision {}'.
        format(number, low, high, precision))

  return np.clip(number, low, high)


def _batch_mm(m1, m2):
  """Batch matrix multiply.

  Args:
    m1: input lhs matrix with shape (batch, n, m).
    m2: input rhs matrix with shape (batch, m, o).

  Returns:
    product matrix with shape (batch, n, o).
  """
  return np.einsum('bij,bjk->bik', m1, m2)


def _rmat_to_euler_xyz(rmat):
  """Converts a 3x3 rotation matrix to XYZ euler angles."""
  # | r00 r01 r02 |   |  cy*cz           -cy*sz            sy    |
  # | r10 r11 r12 | = |  cz*sx*sy+cx*sz   cx*cz-sx*sy*sz  -cy*sx |
  # | r20 r21 r22 |   | -cx*cz*sy+sx*sz   cz*sx+cx*sy*sz   cx*cy |
  if rmat[0, 2] > _POLE_LIMIT:
    logging.log_every_n_seconds(logging.WARNING, 'Angle at North Pole', 60)
    z = np.arctan2(rmat[1, 0], rmat[1, 1])
    y = np.pi/2
    x = 0.0
    return np.array([x, y, z])

  if rmat[0, 2] < -_POLE_LIMIT:
    logging.log_every_n_seconds(logging.WARNING, 'Angle at South Pole', 60)
    z = np.arctan2(rmat[1, 0], rmat[1, 1])
    y = -np.pi/2
    x = 0.0
    return np.array([x, y, z])

  z = -np.arctan2(rmat[0, 1], rmat[0, 0])
  y = np.arcsin(rmat[0, 2])
  x = -np.arctan2(rmat[1, 2], rmat[2, 2])

  # order of return is the order of input
  return np.array([x, y, z])


def _rmat_to_euler_xyx(rmat):
  """Converts a 3x3 rotation matrix to XYX euler angles."""
  # | r00 r01 r02 |   |  cy      sy*sx1               sy*cx1             |
  # | r10 r11 r12 | = |  sy*sx0  cx0*cx1-cy*sx0*sx1  -cy*cx1*sx0-cx0*sx1 |
  # | r20 r21 r22 |   | -sy*cx0  cx1*sx0+cy*cx0*sx1   cy*cx0*cx1-sx0*sx1 |

  if rmat[0, 0] < 1.0:
    if rmat[0, 0] > -1.0:
      y = np.arccos(_clip_within_precision(rmat[0, 0], -1., 1.))
      x0 = np.arctan2(rmat[1, 0], -rmat[2, 0])
      x1 = np.arctan2(rmat[0, 1], rmat[0, 2])
      return np.array([x0, y, x1])
    else:
      # Not a unique solution:  x1_angle - x0_angle = atan2(-r12,r11)
      y = np.pi
      x0 = -np.arctan2(-rmat[1, 2], rmat[1, 1])
      x1 = 0.0
      return np.array([x0, y, x1])
  else:
    # Not a unique solution:  x1_angle + x0_angle = atan2(-r12,r11)
    y = 0.0
    x0 = -np.arctan2(-rmat[1, 2], rmat[1, 1])
    x1 = 0.0
    return np.array([x0, y, x1])


def _rmat_to_euler_zyx(rmat):
  """Converts a 3x3 rotation matrix to ZYX euler angles."""
  if rmat[2, 0] > _POLE_LIMIT:
    logging.warning('Angle at North Pole')
    x = np.arctan2(rmat[0, 1], rmat[0, 2])
    y = -np.pi/2
    z = 0.0
    return np.array([z, y, x])

  if rmat[2, 0] < -_POLE_LIMIT:
    logging.warning('Angle at South Pole')
    x = np.arctan2(rmat[0, 1], rmat[0, 2])
    y = np.pi/2
    z = 0.0
    return np.array([z, y, x])

  x = np.arctan2(rmat[2, 1], rmat[2, 2])
  y = -np.arcsin(rmat[2, 0])
  z = np.arctan2(rmat[1, 0], rmat[0, 0])

  # order of return is the order of input
  return np.array([z, y, x])


def _rmat_to_euler_xzy(rmat):
  """Converts a 3x3 rotation matrix to XZY euler angles."""
  if rmat[0, 1] > _POLE_LIMIT:
    logging.warning('Angle at North Pole')
    y = np.arctan2(rmat[1, 2], rmat[1, 0])
    z = -np.pi/2
    x = 0.0
    return np.array([x, z, y])

  if rmat[0, 1] < -_POLE_LIMIT:
    logging.warning('Angle at South Pole')
    y = np.arctan2(rmat[1, 2], rmat[1, 0])
    z = np.pi/2
    x = 0.0
    return np.array([x, z, y])

  y = np.arctan2(rmat[0, 2], rmat[0, 0])
  z = -np.arcsin(rmat[0, 1])
  x = np.arctan2(rmat[2, 1], rmat[1, 1])

  # order of return is the order of input
  return np.array([x, z, y])


def _rmat_to_euler_yzx(rmat):
  """Converts a 3x3 rotation matrix to YZX euler angles."""
  if rmat[1, 0] > _POLE_LIMIT:
    logging.warning('Angle at North Pole')
    x = -np.arctan2(rmat[0, 2], rmat[0, 1])
    z = np.pi/2
    y = 0.0
    return np.array([y, z, x])

  if rmat[1, 0] < -_POLE_LIMIT:
    logging.warning('Angle at South Pole')
    x = -np.arctan2(rmat[0, 2], rmat[0, 1])
    z = -np.pi/2
    y = 0.0
    return np.array([y, z, x])

  x = -np.arctan2(rmat[1, 2], rmat[1, 1])
  z = np.arcsin(rmat[1, 0])
  y = -np.arctan2(rmat[2, 0], rmat[0, 0])

  # order of return is the order of input
  return np.array([y, z, x])


def _rmat_to_euler_zxy(rmat):
  """Converts a 3x3 rotation matrix to ZXY euler angles."""
  if rmat[2, 1] > _POLE_LIMIT:
    logging.warning('Angle at North Pole')
    y = np.arctan2(rmat[0, 2], rmat[0, 0])
    x = np.pi/2
    z = 0.0
    return np.array([z, x, y])

  if rmat[2, 1] < -_POLE_LIMIT:
    logging.warning('Angle at South Pole')
    y = np.arctan2(rmat[0, 2], rmat[0, 0])
    x = -np.pi/2
    z = 0.0
    return np.array([z, x, y])

  y = -np.arctan2(rmat[2, 0], rmat[2, 2])
  x = np.arcsin(rmat[2, 1])
  z = -np.arctan2(rmat[0, 1], rmat[1, 1])

  # order of return is the order of input
  return np.array([z, x, y])


def _rmat_to_euler_yxz(rmat):
  """Converts a 3x3 rotation matrix to YXZ euler angles."""
  if rmat[1, 2] > _POLE_LIMIT:
    logging.warning('Angle at North Pole')
    z = -np.arctan2(rmat[0, 1], rmat[0, 0])
    x = -np.pi/2
    y = 0.0
    return np.array([y, x, z])

  if rmat[1, 2] < -_POLE_LIMIT:
    logging.warning('Angle at South Pole')
    z = -np.arctan2(rmat[0, 1], rmat[0, 0])
    x = np.pi/2
    y = 0.0
    return np.array([y, x, z])

  z = np.arctan2(rmat[1, 0], rmat[1, 1])
  x = -np.arcsin(rmat[1, 2])
  y = np.arctan2(rmat[0, 2], rmat[2, 2])

  # order of return is the order of input
  return np.array([y, x, z])


def _axis_rotation(theta, full):
  """Returns the theta dim, cos and sin, and blank matrix for axis rotation."""
  n = 1 if np.isscalar(theta) else len(theta)
  ct = np.cos(theta)
  st = np.sin(theta)

  if full:
    rmat = np.zeros((n, 4, 4))
    rmat[:, 3, 3] = 1.
  else:
    rmat = np.zeros((n, 3, 3))

  return n, ct, st, rmat

# map from full rotation orderings to euler conversion functions
_eulermap = {
    'XYZ': _rmat_to_euler_xyz,
    'XYX': _rmat_to_euler_xyx,
    'ZYX': _rmat_to_euler_zyx,
    'XZY': _rmat_to_euler_xzy,
    'YZX': _rmat_to_euler_yzx,
    'ZXY': _rmat_to_euler_zxy,
    'YXZ': _rmat_to_euler_yxz
}


def euler_to_quat(euler_vec, ordering='XYZ'):
  """Returns the quaternion corresponding to the provided euler angles.

  Args:
    euler_vec: The euler angle rotations.
    ordering: (str) Desired euler angle ordering.

  Returns:
    quat: A quaternion [w, i, j, k]
  """
  mat = euler_to_rmat(euler_vec, ordering=ordering)
  return mat_to_quat(mat)


def euler_to_rmat(euler_vec, ordering='ZXZ', full=False):
  """Returns rotation matrix (or transform) for the given Euler rotations.

  Euler*** methods compose a Rotation matrix corresponding to the given
  rotations r1, r2, r3 following the given rotation ordering. Ordering
  specifies the order of rotation matrices in matrix multiplication order.
  E.g. for XYZ we return rotX(r1) * rotY(r2) * rotZ(r3).

  Args:
    euler_vec: The euler angle rotations.
    ordering: euler angle ordering string (see _euler_orderings).
    full: If true, returns a full 4x4 transfom.

  Returns:
    The rotation matrix or homogenous transform corresponding to the given
    Euler rotation.
  """

  # map from partial rotation orderings to rotation functions
  rotmap = {'X': rotation_x_axis, 'Y': rotation_y_axis, 'Z': rotation_z_axis}
  rotations = [rotmap[c] for c in ordering]

  euler_vec = np.atleast_2d(euler_vec)

  rots = []
  for i in range(len(rotations)):
    rots.append(rotations[i](euler_vec[:, i], full))

  if rots[0].ndim == 3:
    result = _batch_mm(_batch_mm(rots[0], rots[1]), rots[2])
    return result.squeeze()
  else:
    return (rots[0].dot(rots[1])).dot(rots[2])


def quat_conj(quat):
  """Return conjugate of quaternion.

  This function supports inputs with or without leading batch dimensions.

  Args:
    quat: A quaternion [w, i, j, k].

  Returns:
    A quaternion [w, -i, -j, -k] representing the inverse of the rotation
    defined by `quat` (not assuming normalization).
  """
  # Ensure quat is an np.array in case a tuple or a list is passed
  quat = np.asarray(quat)
  return np.stack(
      [quat[..., 0], -quat[..., 1],
       -quat[..., 2], -quat[..., 3]], axis=-1).astype(np.float64)


def quat_inv(quat):
  """Return inverse of quaternion.

  This function supports inputs with or without leading batch dimensions.

  Args:
    quat: A quaternion [w, i, j, k].

  Returns:
    A quaternion representing the inverse of the original rotation.
  """
  # Ensure quat is an np.array in case a tuple or a list is passed
  quat = np.asarray(quat)
  return quat_conj(quat) / np.sum(quat * quat, axis=-1, keepdims=True)


def _get_qmat_indices_and_signs():
  """Precomputes index and sign arrays for constructing `qmat` in `quat_mul`."""
  w, x, y, z = range(4)
  qmat_idx_and_sign = np.array([
      [w, -x, -y, -z],
      [x, w, -z, y],
      [y, z, w, -x],
      [z, -y, x, w],
  ])
  indices = np.abs(qmat_idx_and_sign)
  signs = 2 * (qmat_idx_and_sign >= 0) - 1
  # Prevent array constants from being modified in place.
  indices.flags.writeable = False
  signs.flags.writeable = False
  return indices, signs

_qmat_idx, _qmat_sign = _get_qmat_indices_and_signs()


def quat_mul(quat1, quat2):
  """Computes the Hamilton product of two quaternions.

  Any number of leading batch dimensions is supported.

  Args:
    quat1: A quaternion [w, i, j, k].
    quat2: A quaternion [w, i, j, k].

  Returns:
    The quaternion product quat1 * quat2.
  """
  # Construct a (..., 4, 4) matrix to multiply with quat2 as shown below.
  qmat = quat1[..., _qmat_idx] * _qmat_sign

  # Compute the batched Hamilton product:
  # |w1 -i1 -j1 -k1|   |w2|   |w1w2 - i1i2 - j1j2 - k1k2|
  # |i1  w1 -k1  j1| . |i2| = |w1i2 + i1w2 + j1k2 - k1j2|
  # |j1  k1  w1 -i1|   |j2|   |w1j2 - i1k2 + j1w2 + k1i2|
  # |k1 -j1  i1  w1|   |k2|   |w1k2 + i1j2 - j1i2 + k1w2|
  return (qmat @ quat2[..., None])[..., 0]


def quat_diff(source, target):
  """Computes quaternion difference between source and target quaternions.

  This function supports inputs with or without leading batch dimensions.

  Args:
    source: A quaternion [w, i, j, k].
    target: A quaternion [w, i, j, k].

  Returns:
    A quaternion representing the rotation from source to target.
  """
  return quat_mul(quat_conj(source), target)


def quat_log(quat, tol=_TOL):
  """Log of a quaternion.

  This function supports inputs with or without leading batch dimensions.

  Args:
    quat: A quaternion [w, i, j, k].
    tol: numerical tolerance to prevent nan.

  Returns:
    4D array representing the log of `quat`. This is analogous to
    `rmat_to_axisangle`.
  """
  # Ensure quat is an np.array in case a tuple or a list is passed
  quat = np.asarray(quat)
  q_norm = np.linalg.norm(quat + tol, axis=-1, keepdims=True)
  a = quat[..., 0:1]
  v = np.stack([quat[..., 1], quat[..., 2], quat[..., 3]], axis=-1)
  # Clip to 2*tol because we subtract it here
  v_new = v / np.linalg.norm(v + tol, axis=-1, keepdims=True) * np.arccos(
      _clip_within_precision(a - tol, -1., 1., precision=2.*tol)) / q_norm
  return np.stack(
      [np.log(q_norm[..., 0]), v_new[..., 0], v_new[..., 1], v_new[..., 2]],
      axis=-1)


def quat_dist(source, target):
  """Computes distance between source and target quaternions.

  This function assumes that both input arguments are unit quaternions.

  This function supports inputs with or without leading batch dimensions.

  Args:
    source: A quaternion [w, i, j, k].
    target: A quaternion [w, i, j, k].

  Returns:
    Scalar representing the rotational distance from source to target.
  """
  quat_product = quat_mul(source, quat_inv(target))
  quat_product /= np.linalg.norm(quat_product, axis=-1, keepdims=True)
  return np.linalg.norm(quat_log(quat_product), axis=-1, keepdims=True)


def quat_rotate(quat, vec):
  """Rotate a vector by a quaternion.

  Args:
    quat: A quaternion [w, i, j, k].
    vec: A 3-vector representing a position.

  Returns:
    The rotated vector.
  """
  qvec = np.hstack([[0], vec])
  return quat_mul(quat_mul(quat, qvec), quat_conj(quat))[1:]


def quat_to_axisangle(quat):
  """Returns the axis-angle corresponding to the provided quaternion.

  Args:
    quat: A quaternion [w, i, j, k].

  Returns:
    axisangle: A 3x1 numpy array describing the axis of rotation, with angle
        encoded by its length.
  """
  angle = 2 * np.arccos(_clip_within_precision(quat[0], -1., 1.))

  if angle < _TOL:
    return np.zeros(3)
  else:
    qn = np.sin(angle/2)
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    axis = quat[1:4] / qn
    return axis * angle


def quat_to_euler(quat, ordering='XYZ'):
  """Returns the euler angles corresponding to the provided quaternion.

  Args:
    quat: A quaternion [w, i, j, k].
    ordering: (str) Desired euler angle ordering.

  Returns:
    euler_vec: The euler angle rotations.
  """
  mat = quat_to_mat(quat)
  return rmat_to_euler(mat[0:3, 0:3], ordering=ordering)


def quat_to_mat(quat):
  """Return homogeneous rotation matrix from quaternion.

  Args:
    quat: A quaternion [w, i, j, k].

  Returns:
    A 4x4 homogeneous matrix with the rotation corresponding to `quat`.
  """
  q = np.array(quat, dtype=np.float64, copy=True)
  nq = np.dot(q, q)
  if nq < _TOL:
    return np.identity(4)
  q *= np.sqrt(2.0 / nq)
  q = np.outer(q, q)
  return np.array(
      ((1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0),
       (q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0),
       (q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0),
       (0.0, 0.0, 0.0, 1.0)),
      dtype=np.float64)


def rotation_x_axis(theta, full=False):
  """Returns a rotation matrix of a rotation about the X-axis.

  Supports vector-valued theta, in which case the returned array is of shape
  (len(t), 3, 3), or (len(t), 4, 4) if full=True. If theta is scalar the batch
  dimension is squeezed out.

  Args:
    theta: The rotation amount.
    full: (bool) If true, returns a full 4x4 transform.
  """
  n, ct, st, rmat = _axis_rotation(theta, full)

  rmat[:, 0, 0:3] = np.array([[1, 0, 0]])
  rmat[:, 1, 0:3] = np.vstack([np.zeros(n), ct, -st]).T
  rmat[:, 2, 0:3] = np.vstack([np.zeros(n), st, ct]).T

  return rmat.squeeze()


def rotation_y_axis(theta, full=False):
  """Returns a rotation matrix of a rotation about the Y-axis.

  Supports vector-valued theta, in which case the returned array is of shape
  (len(t), 3, 3), or (len(t), 4, 4) if full=True. If theta is scalar the batch
  dimension is squeezed out.

  Args:
    theta: The rotation amount.
    full: (bool) If true, returns a full 4x4 transfom.
  """
  n, ct, st, rmat = _axis_rotation(theta, full)

  rmat[:, 0, 0:3] = np.vstack([ct, np.zeros(n), st]).T
  rmat[:, 1, 0:3] = np.array([[0, 1, 0]])
  rmat[:, 2, 0:3] = np.vstack([-st, np.zeros(n), ct]).T

  return rmat.squeeze()


def rotation_z_axis(theta, full=False):
  """Returns a rotation matrix of a rotation about the z-axis.

  Supports vector-valued theta, in which case the returned array is of shape
  (len(t), 3, 3), or (len(t), 4, 4) if full=True. If theta is scalar the batch
  dimension is squeezed out.

  Args:
    theta: The rotation amount.
    full: (bool) If true, returns a full 4x4 transfom.
  """
  n, ct, st, rmat = _axis_rotation(theta, full)

  rmat[:, 0, 0:3] = np.vstack([ct, -st, np.zeros(n)]).T
  rmat[:, 1, 0:3] = np.vstack([st, ct, np.zeros(n)]).T
  rmat[:, 2, 0:3] = np.array([[0, 0, 1]])

  return rmat.squeeze()


def rmat_to_euler(rmat, ordering='ZXZ'):
  """Returns the euler angles corresponding to the provided rotation matrix.

  Args:
    rmat: The rotation matrix.
    ordering: (str) Desired euler angle ordering.

  Returns:
    Euler angles corresponding to the provided rotation matrix.
  """
  return _eulermap[ordering](rmat)


def mat_to_quat(mat):
  """Return quaternion from homogeneous or rotation matrix.

  Args:
    mat: A homogeneous transform or rotation matrix

  Returns:
    A quaternion [w, i, j, k].
  """
  if mat.shape == (3, 3):
    tmp = np.eye(4)
    tmp[0:3, 0:3] = mat
    mat = tmp

  q = np.empty((4,), dtype=np.float64)
  t = np.trace(mat)
  if t > mat[3, 3]:
    q[0] = t
    q[3] = mat[1, 0] - mat[0, 1]
    q[2] = mat[0, 2] - mat[2, 0]
    q[1] = mat[2, 1] - mat[1, 2]
  else:
    i, j, k = 0, 1, 2
    if mat[1, 1] > mat[0, 0]:
      i, j, k = 1, 2, 0
    if mat[2, 2] > mat[i, i]:
      i, j, k = 2, 0, 1
    t = mat[i, i] - (mat[j, j] + mat[k, k]) + mat[3, 3]
    q[i + 1] = t
    q[j + 1] = mat[i, j] + mat[j, i]
    q[k + 1] = mat[k, i] + mat[i, k]
    q[0] = mat[k, j] - mat[j, k]
  q *= 0.5 / np.sqrt(t * mat[3, 3])
  return q


# ################
# # 2D Functions #
# ################


def rotation_matrix_2d(theta):
  ct = np.cos(theta)
  st = np.sin(theta)
  return np.array([
      [ct, -st],
      [st, ct]
  ])
