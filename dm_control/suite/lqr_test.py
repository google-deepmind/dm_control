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

"""Tests specific to the LQR domain."""

import  math

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from dm_control.suite import lqr
from dm_control.suite import lqr_solver
import numpy as np


class LqrTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('lqr_2_1', lqr.lqr_2_1),
      ('lqr_6_2', lqr.lqr_6_2))
  def test_lqr_optimal_policy(self, make_env):
    env = make_env()
    p, k, beta = lqr_solver.solve(env)
    self.assertPolicyisOptimal(env, p, k, beta)

  def assertPolicyisOptimal(self, env, p, k, beta):
    tolerance = 1e-3
    n_steps = int(math.ceil(math.log10(tolerance) / math.log10(beta)))
    logging.info('%d timesteps for %g convergence.', n_steps, tolerance)
    total_loss = 0.0

    timestep = env.reset()
    initial_state = np.hstack((timestep.observation['position'],
                               timestep.observation['velocity']))
    logging.info('Measuring total cost over %d steps.', n_steps)
    for _ in range(n_steps):
      x = np.hstack((timestep.observation['position'],
                     timestep.observation['velocity']))
      # u = k*x is the optimal policy
      u = k.dot(x)
      total_loss += 1 - (timestep.reward or 0.0)
      timestep = env.step(u)

    logging.info('Analytical expected total cost is .5*x^T*p*x.')
    expected_loss = .5 * initial_state.T.dot(p).dot(initial_state)
    logging.info('Comparing measured and predicted costs.')
    np.testing.assert_allclose(expected_loss, total_loss, rtol=tolerance)

if __name__ == '__main__':
  absltest.main()
