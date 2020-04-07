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

"""Tests for rescaling bodies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from dm_control import mjcf
from dm_control.locomotion.walkers import rescale
import numpy as np


class RescaleTest(absltest.TestCase):

  def setUp(self):
    super(RescaleTest, self).setUp()

    # build a simple three-link chain with an endpoint site
    self._mjcf_model = mjcf.RootElement()
    body = self._mjcf_model.worldbody.add('body', pos=[0, 0, 0])
    body.add('geom', type='capsule', fromto=[0, 0, 0, 0, 0, -0.4], size=[0.06])
    body.add('joint', type='ball')
    body = body.add('body', pos=[0, 0, -0.5])
    body.add('geom', type='capsule', pos=[0, 0, -0.15], size=[0.06, 0.15])
    body.add('joint', type='ball')
    body = body.add('body', pos=[0, 0, -0.4])
    body.add('geom', type='capsule', fromto=[0, 0, 0, 0.3, 0, -0.4],
             size=[0.06])
    body.add('joint', type='ball')
    body.add('site', name='endpoint', type='sphere', pos=[0.3, 0, -0.4],
             size=[0.1])

  def test_rescale(self):
    # verify endpoint is where expected
    physics = mjcf.Physics.from_mjcf_model(self._mjcf_model)
    np.testing.assert_allclose(physics.named.data.site_xpos['endpoint'],
                               np.array([0.3, 0., -1.3]), atol=1e-15)

    # rescale chain and verify endpoint is where expected after modification
    subtree_root = self._mjcf_model
    position_factor = .5
    size_factor = .5
    rescale.rescale_subtree(subtree_root, position_factor, size_factor)
    physics = mjcf.Physics.from_mjcf_model(self._mjcf_model)
    np.testing.assert_allclose(physics.named.data.site_xpos['endpoint'],
                               np.array([0.15, 0., -0.65]), atol=1e-15)

if __name__ == '__main__':
  absltest.main()
