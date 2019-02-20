# Copyright 2019 The dm_control Authors.
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

"""Tests for dm_control.composer.props.position_detector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.
from absl.testing import absltest
from absl.testing import parameterized
from dm_control import composer
from dm_control.entities.props import position_detector
from dm_control.entities.props import primitive
import numpy as np


class PositionDetectorTest(parameterized.TestCase):

  def setUp(self):
    super(PositionDetectorTest, self).setUp()
    self.arena = composer.Arena()
    self.props = [
        primitive.Primitive(geom_type='sphere', size=(0.1,)),
        primitive.Primitive(geom_type='sphere', size=(0.1,))
    ]
    for prop in self.props:
      self.arena.add_free_entity(prop)
    self.task = composer.NullTask(self.arena)

  def assertDetected(self, entity, detector):
    if not self.inverted:
      self.assertIn(entity, detector.detected_entities)
    else:
      self.assertNotIn(entity, detector.detected_entities)

  def assertNotDetected(self, entity, detector):
    if not self.inverted:
      self.assertNotIn(entity, detector.detected_entities)
    else:
      self.assertIn(entity, detector.detected_entities)

  @parameterized.parameters(False, True)
  def test3DDetection(self, inverted):
    self.inverted = inverted

    detector_pos = np.array([0.3, 0.2, 0.1])
    detector_size = np.array([0.1, 0.2, 0.3])
    detector = position_detector.PositionDetector(
        pos=detector_pos, size=detector_size, inverted=inverted)
    detector.register_entities(*self.props)
    self.arena.attach(detector)
    env = composer.Environment(self.task)

    env.reset()
    self.assertNotDetected(self.props[0], detector)
    self.assertNotDetected(self.props[1], detector)

    def initialize_episode(physics, unused_random_state):
      for prop in self.props:
        prop.set_pose(physics, detector_pos)
    self.task.initialize_episode = initialize_episode
    env.reset()
    self.assertDetected(self.props[0], detector)
    self.assertDetected(self.props[1], detector)

    self.props[0].set_pose(env.physics, detector_pos - detector_size)
    env.step([])
    self.assertNotDetected(self.props[0], detector)
    self.assertDetected(self.props[1], detector)

    self.props[0].set_pose(env.physics, detector_pos - detector_size / 2)
    self.props[1].set_pose(env.physics, detector_pos + detector_size * 1.01)
    env.step([])
    self.assertDetected(self.props[0], detector)
    self.assertNotDetected(self.props[1], detector)

  @parameterized.parameters(False, True)
  def test2DDetection(self, inverted):
    self.inverted = inverted

    detector_pos = np.array([0.3, 0.2])
    detector_size = np.array([0.1, 0.2])
    detector = position_detector.PositionDetector(
        pos=detector_pos, size=detector_size, inverted=inverted)
    detector.register_entities(*self.props)
    self.arena.attach(detector)
    env = composer.Environment(self.task)

    env.reset()
    self.assertNotDetected(self.props[0], detector)
    self.assertNotDetected(self.props[1], detector)

    def initialize_episode(physics, unused_random_state):
      # In 2D mode, detection should occur no matter how large |z| is.
      self.props[0].set_pose(physics, [detector_pos[0], detector_pos[1], 1e+6])
      self.props[1].set_pose(physics, [detector_pos[0], detector_pos[1], -1e+6])
    self.task.initialize_episode = initialize_episode
    env.reset()
    self.assertDetected(self.props[0], detector)
    self.assertDetected(self.props[1], detector)

    self.props[0].set_pose(
        env.physics, [detector_pos[0] - detector_size[0], detector_pos[1], 0])
    env.step([])
    self.assertNotDetected(self.props[0], detector)
    self.assertDetected(self.props[1], detector)

    self.props[0].set_pose(
        env.physics, [detector_pos[0] - detector_size[0] / 2,
                      detector_pos[1] + detector_size[1] / 2, 0])
    self.props[1].set_pose(
        env.physics, [detector_pos[0], detector_pos[1] + detector_size[1], 0])
    env.step([])
    self.assertDetected(self.props[0], detector)
    self.assertNotDetected(self.props[1], detector)


if __name__ == '__main__':
  absltest.main()
