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

"""Point-mass domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from dm_env import specs
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np
from scipy.spatial import ConvexHull
import alphashape
import random

_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()
CORNER_INDEX_POSITION = [86, 81, 59, 54]
CORNER_INDEX_ACTION = ['B0_0', 'B0_8', 'B8_0', 'B8_8']


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model('cloth_v0.xml'), common.ASSETS


@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, **kwargs):
    """Returns the easy cloth task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Cloth(randomize_gains=False, random=random, **kwargs)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, special_task=True, **environment_kwargs)


class Physics(mujoco.Physics):
    """physics for the point_mass domain."""


class Cloth(base.Task):
    """A point_mass `Task` to reach target with smooth reward."""

    def __init__(self, randomize_gains, random=None, pixel_size=64, camera_id=0,
                 reward='area', eval=False):
        """Initialize an instance of `PointMass`.

        Args:
          randomize_gains: A `bool`, whether to randomize the actuator gains.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._randomize_gains = randomize_gains
        self.pixel_size = pixel_size
        self.camera_id = camera_id
        self.reward = reward
        print('pixel_size', self.pixel_size, 'camera_id',
              self.camera_id, 'reward', self.reward)
        super(Cloth, self).__init__(random=random)

    def action_spec(self, physics):
        """Returns a `BoundedArray` matching the `physics` actuators."""
        return specs.BoundedArray(
            shape=(12,), dtype=np.float32, minimum=[-1.0] * 12, maximum=[1.0] * 12)

    def initialize_episode(self, physics):
        physics.data.xpos[1:, :2] = physics.data.xpos[1:,
                                                      :2] + self.random.uniform(-.3, .3)
        physics.named.data.xfrc_applied['B3_4', :3] = np.array([0, 0, -2])
        physics.named.data.xfrc_applied['B4_4', :3] = np.array([0, 0, -2])

        physics.named.data.xfrc_applied[CORNER_INDEX_ACTION, :3] = np.random.uniform(-.5, .5, size=3)
        super(Cloth, self).initialize_episode(physics)

    def before_step(self, action, physics):
        """Sets the control signal for the actuators to values in `action`."""
        action = action.reshape(4, 3)
        physics.named.data.xfrc_applied[CORNER_INDEX_ACTION, :3] = action

    def get_observation(self, physics):
        """Returns an observation of the state."""
        obs = collections.OrderedDict()
        obs['position'] = physics.named.data.geom_xpos[6:, :2].astype('float32').reshape(-1)
        return obs

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        diag_reward = self._compute_diagonal_reward(physics)

        if self.reward == 'area':
            pixels = physics.render(width=self.pixel_size, height=self.pixel_size,
                                    camera_id=self.camera_id)
            segmentation = (pixels < 100).any(axis=-1).astype('float32')
            reward = segmentation.mean()
        elif self.reward == 'area_convex':
            area_convex_reward = self._compute_area_convex(physics)
            reward = area_convex_reward
        elif self.reward == 'diagonal':
            reward = diag_reward
        elif self.reward == 'area_concave':
            area_concave_reward = self._compute_area_concave(physics)
            reward = area_concave_reward
        else:
            raise ValueError(self.reward)

        return reward, dict()

    def _compute_diagonal_reward(self, physics):
        pos_ll = physics.data.geom_xpos[86, :2]
        pos_lr = physics.data.geom_xpos[81, :2]
        pos_ul = physics.data.geom_xpos[59, :2]
        pos_ur = physics.data.geom_xpos[54, :2]
        diag_dist1 = np.linalg.norm(pos_ll - pos_ur)
        diag_dist2 = np.linalg.norm(pos_lr - pos_ul)
        diag_reward = diag_dist1 + diag_dist2
        return diag_reward

    def _compute_area_convex(self, physics):
        joints = physics.data.geom_xpos[6:, :2]
        hull = ConvexHull(joints)
        vertices = joints[hull.vertices]

        x, y = vertices[:, 0], vertices[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) -
                            np.dot(y, np.roll(x, 1)))
        return area

    def _compute_area_concave(self, physics):
        points = physics.data.geom_xpos[6:, :2]
        alpha_shape = alphashape.alphashape(points, 20.0)
        return alpha_shape.area

    def _compute_area_reward(self, physics):
        pixels = physics.render(width=self.pixel_size, height=self.pixel_size,
                                camera_id=self.camera_id)
        segmentation = (pixels < 100).any(axis=-1).astype('float32')
        reward = segmentation.mean()
        return reward
