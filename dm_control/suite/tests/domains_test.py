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

"""Tests for dm_control.suite domains."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.

from absl.testing import absltest
from absl.testing import parameterized

from dm_control import suite

import numpy as np
import six

_NUM_EPISODES = 5
_NUM_STEPS_PER_EPISODE = 10


class DomainTest(parameterized.TestCase):
  """Tests run on all the tasks registered."""

  def test_constants(self):
    num_tasks = sum(len(tasks) for tasks in
                    six.itervalues(suite.TASKS_BY_DOMAIN))

    self.assertEqual(len(suite.ALL_TASKS), num_tasks)

  def _validate_observation(self, observation_dict, observation_spec):
    obs = observation_dict.copy()
    for name, spec in six.iteritems(observation_spec):
      arr = obs.pop(name)
      self.assertEqual(arr.shape, spec.shape)
      self.assertEqual(arr.dtype, spec.dtype)
      self.assertTrue(
          np.all(np.isfinite(arr)),
          msg='{!r} has non-finite value(s): {!r}'.format(name, arr))
    self.assertEmpty(
        obs,
        msg='Observation contains arrays(s) that are not in the spec: {!r}'
        .format(obs))

  def _validate_reward_range(self, time_step):
    if time_step.first():
      self.assertIsNone(time_step.reward)
    else:
      self.assertIsInstance(time_step.reward, float)
      self.assertBetween(time_step.reward, 0, 1)

  def _validate_discount(self, time_step):
    if time_step.first():
      self.assertIsNone(time_step.discount)
    else:
      self.assertIsInstance(time_step.discount, float)
      self.assertBetween(time_step.discount, 0, 1)

  def _validate_control_range(self, lower_bounds, upper_bounds):
    for b in lower_bounds:
      self.assertEqual(b, -1.0)
    for b in upper_bounds:
      self.assertEqual(b, 1.0)

  @parameterized.parameters(*suite.ALL_TASKS)
  def test_components_have_names(self, domain, task):
    env = suite.load(domain, task)
    model = env.physics.model

    object_types_and_size_fields = {
        'body': 'nbody',
        'joint': 'njnt',
        'geom': 'ngeom',
        'site': 'nsite',
        'camera': 'ncam',
        'light': 'nlight',
        'mesh': 'nmesh',
        'hfield': 'nhfield',
        'texture': 'ntex',
        'material': 'nmat',
        'equality': 'neq',
        'tendon': 'ntendon',
        'actuator': 'nu',
        'sensor': 'nsensor',
        'numeric': 'nnumeric',
        'text': 'ntext',
        'tuple': 'ntuple',
    }

    for object_type, size_field in six.iteritems(object_types_and_size_fields):
      for idx in range(getattr(model, size_field)):
        object_name = model.id2name(idx, object_type)
        self.assertNotEqual(object_name, '',
                            msg='Model {!r} contains unnamed {!r} with ID {}.'
                            .format(model.name, object_type, idx))

  @parameterized.parameters(*suite.ALL_TASKS)
  def test_task_runs(self, domain, task):
    """Tests task runs correctly and observation is coherent with spec."""
    is_benchmark = (domain, task) in suite.BENCHMARKING
    env = suite.load(domain, task)

    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    model = env.physics.model

    # Check cameras.
    self.assertGreaterEqual(model.ncam, 2, 'Model {!r} should have at least 2 '
                            'cameras, has {!r}.'.format(model.name, model.ncam))

    # Check action bounds.
    lower_bounds = action_spec.minimum
    upper_bounds = action_spec.maximum

    if is_benchmark:
      self._validate_control_range(lower_bounds, upper_bounds)

    lower_bounds = np.where(np.isinf(lower_bounds), -1.0, lower_bounds)
    upper_bounds = np.where(np.isinf(upper_bounds), 1.0, upper_bounds)

    # Run a partial episode, check observations, rewards, discount.
    for _ in range(_NUM_EPISODES):
      time_step = env.reset()
      for _ in range(_NUM_STEPS_PER_EPISODE):
        self._validate_observation(time_step.observation, observation_spec)
        if is_benchmark:
          self._validate_reward_range(time_step)
        self._validate_discount(time_step)
        action = np.random.uniform(lower_bounds, upper_bounds)
        time_step = env.step(action)

  @parameterized.parameters(*suite.ALL_TASKS)
  def test_visualize_reward(self, domain, task):
    env = suite.load(domain, task)
    env.task.visualise_reward = True
    env.reset()
    action = np.zeros(env.action_spec().shape)
    for _ in range(2):
      env.step(action)


if __name__ == '__main__':
  absltest.main()
