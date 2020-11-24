# Copyright 2017 The dm_control Authors.
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

r"""Optimal policy for LQR levels.

LQR control problem is described in
https://en.wikipedia.org/wiki/Linear-quadratic_regulator#Infinite-horizon.2C_discrete-time_LQR
"""


from dm_control.mujoco import wrapper
import numpy as np
import scipy.linalg as scipy_linalg


def solve(env):
  """Returns the optimal value and policy for LQR problem.

  Args:
    env: An instance of `control.EnvironmentV2` with LQR level.

  Returns:
    p: A numpy array, the Hessian of the optimal total cost-to-go (value
      function at state x) is V(x) = .5 * x' * p * x.
    k: A numpy array which gives the optimal linear policy u = k * x.
    beta: The maximum eigenvalue of (a + b * k). Under optimal policy, at
      timestep n the state tends to 0 like beta^n.

  Raises:
    RuntimeError: If the controlled system is unstable.
  """
  n = env.physics.model.nq  # number of DoFs
  m = env.physics.model.nu  # number of controls

  # Compute the mass matrix.
  mass = np.zeros((n, n))
  wrapper.mjbindings.mjlib.mj_fullM(env.physics.model.ptr, mass,
                                    env.physics.data.qM)

  # Compute input matrices a, b, q and r to the DARE solvers.
  # State transition matrix a.
  stiffness = np.diag(env.physics.model.jnt_stiffness.ravel())
  damping = np.diag(env.physics.model.dof_damping.ravel())
  dt = env.physics.model.opt.timestep

  j = np.linalg.solve(-mass, np.hstack((stiffness, damping)))
  a = np.eye(2 * n) + dt * np.vstack(
      (dt * j + np.hstack((np.zeros((n, n)), np.eye(n))), j))

  # Control transition matrix b.
  b = env.physics.data.actuator_moment.T
  bc = np.linalg.solve(mass, b)
  b = dt * np.vstack((dt * bc, bc))

  # State cost Hessian q.
  q = np.diag(np.hstack([np.ones(n), np.zeros(n)]))

  # Control cost Hessian r.
  r = env.task.control_cost_coef * np.eye(m)

  # Solve the discrete algebraic Riccati equation.
  p = scipy_linalg.solve_discrete_are(a, b, q, r)
  k = -np.linalg.solve(b.T.dot(p.dot(b)) + r, b.T.dot(p.dot(a)))

  # Under optimal policy, state tends to 0 like beta^n_timesteps
  beta = np.abs(np.linalg.eigvals(a + b.dot(k))).max()
  if beta >= 1.0:
    raise RuntimeError('Controlled system is unstable.')
  return p, k, beta
