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
import mujoco_py

"""Input action, chooses random n joints"""


_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()
CORNER_INDEX_POSITION = [86, 81, 59, 54]
CORNER_INDEX_ACTION = ['B0_0', 'B0_8', 'B8_0', 'B8_8']
GEOM_INDEX = ['G0_0', 'G0_8', 'G8_0', 'G8_8']


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    # return common.read_model('cloth_v0.xml'), common.ASSETS
    return common.read_model('cloth_v4.xml'), common.ASSETS


@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, **kwargs):
    """Returns the easy cloth task."""

    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Cloth(randomize_gains=False, random=random, **kwargs)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, n_frame_skip=1, special_task=True, **environment_kwargs)


class Physics(mujoco.Physics):
    """physics for the point_mass domain."""


class Cloth(base.Task):
    """A point_mass `Task` to reach target with smooth reward."""

    def __init__(self, randomize_gains, random=None, n_locations=1, pixel_size=64, camera_id=0,
                 reward='area', mode='normal', eval=False):
        """Initialize an instance of `PointMass`.

        Args:
          randomize_gains: A `bool`, whether to randomize the actuator gains.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        assert mode in ['normal', 'corners_xy', 'corners_onehot']

        self._randomize_gains = randomize_gains
        self.n_locations = n_locations
        self.pixel_size = pixel_size
        self.camera_id = camera_id
        self.reward = reward
        self.eval = eval
        self.mode = mode
        print('pixel_size', self.pixel_size, 'camera_id',
              self.camera_id, 'reward', self.reward,
              'eval', self.eval, 'mode', self.mode)
        print('n_locations', self.n_locations)
        self._current_locs = None

        if 'corners' in mode:
            assert 1 <= self.n_locations <= 4

        super(Cloth, self).__init__(random=random)

    def action_spec(self, physics):
        """Returns a `BoundedArray` matching the `physics` actuators."""

        # action force(3) ~[-1,1]
        if self.eval:
            if self.mode in ['normal', 'corners_xy']:
                size = 5 * self.n_locations
            else:
                size = (3 + 4) * self.n_locations
        else:
            size = 3 * self.n_locations
        return specs.BoundedArray(
            shape=(size,), dtype=np.float,
            minimum=[-1.0] * size, maximum=[1.0] * size)

    def initialize_episode(self, physics):
        self._current_locs = self._generate_loc()

        physics.named.data.xfrc_applied['B3_4', :3] = np.array([0, 0, -2])
        physics.named.data.xfrc_applied['B4_4', :3] = np.array([0, 0, -2])
        physics.named.data.xfrc_applied[CORNER_INDEX_ACTION,
                                        :3] = np.random.uniform(-.5, .5, size=3)

        super(Cloth, self).initialize_episode(physics)

    def before_step(self, action, physics):
        """Sets the control signal for the actuators to values in `action`."""
        physics.named.data.xfrc_applied[1:, :3] = np.zeros((3,))

        if self.eval:
            if self.mode in ['normal', 'corners_xy']:
                assert len(action) == 5 * self.n_locations # action + location
                force_actions = action[:3 * self.n_locations]
                locations = action[3 * self.n_locations:]
                assert len(locations) == 2 * self.n_locations
                locations = (locations * 0.5 + 0.5) * 8 # [-1, 1] -> [0, 8]
                locations = np.round(locations).astype('int32')

                for i in range(self.n_locations):
                    force_action = force_actions[3 * i:3 * (i + 1)]
                    x, y = locations[2 * i:2 * (i + 1)]
                    force_id = 'B{}_{}'.format(x, y)
                    physics.named.data.xfrc_applied[force_id, :3] = 5 * force_action
            else:
                assert len(action) == (3 + 4) * self.n_locations
                force_actions = action[:3 * self.n_locations]
                locations = action[3 * self.n_locations:]
                assert len(locations) == 4 * self.n_locations
                assert np.sum(locations) == self.n_locations # Rough sanity check that probably onehot
                locations = locations.astype('int32')
                for i in range(self.n_locations):
                    force_action = force_actions[3 * i:3 * (i + 1)]
                    force_location = locations[4 * i:4 * (i + 1)]
                    force_location = np.argwhere(force_location == 1).reshape(-1)
                    assert len(force_location) == 1
                    force_location = force_location[0]
                    physics.named.data.xfrc_applied[CORNER_INDEX_ACTION[force_location], :3] = 5 * force_action
        else:
            assert len(action) == 3 * self.n_locations
            if self.mode in ['normal', 'corners_xy']:
                xs = self._current_locs % 9
                ys = self._current_locs // 9

                for i, (x, y) in enumerate(zip(xs, ys)):
                    x, y = int(x), int(y)
                    force_id = 'B{}_{}'.format(x, y)
                    physics.named.data.xfrc_applied[force_id, :3] = 5 * action[3 * i:3 * (i + 1)]
            elif self.mode == 'corners_onehot':
                for i in range(self.n_locations):
                    c = CORNER_INDEX_ACTION[self._current_locs[i]]
                    physics.named.data.xfrc_applied[c, :3] = 5 * action[3 * i:3 * (i + 1)]
            else:
                raise Exception(self.mode)

        self._current_locs = self._generate_loc()

    def get_observation(self, physics):
        """Returns an observation of the state."""
        if self._current_locs is None:
            print('current locs None')
            self._current_locs = self._generate_loc()
            obs = self._generate_obs(physics)
            self._current_locs = None
            return obs

        return self._generate_obs(physics)

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        diag_reward = self._compute_diagonal_reward(physics)
        #area_concave_reward = self._compute_area_concave(physics)

        info = dict(reward_diagonal=diag_reward,)
         #           reward_area_concave=area_concave_reward)

        if self.reward == 'area':
            area_reward = self._compute_area_reward(physics)
            reward = area_reward
        elif self.reward == 'diagonal':
            reward = diag_reward
        elif self.reward == 'area_concave':
            area_concave_reward = self._compute_area_concave(physics)
            reward = area_concave_reward
        else:
            raise ValueError(self.reward)

        return reward, info

    def _generate_obs(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.position().astype('float32')
        obs['velocity'] = physics.velocity().astype('float32')

        if not self.eval:
            if self.mode in ['normal', 'corners_xy']:
                xs = self._current_locs % 9
                ys = self._current_locs // 9

                points = np.hstack((xs[:, None], ys[:, None])).astype('float32')
                points /= 8
                points = 2 * points - 1  # [0, 1] -> [-1, 1]
                points = points.reshape(-1)

                obs['force_location'] = points
            elif self.mode == 'corners_onehot':
                onehots = np.zeros((self.n_locations, 4))
                onehots[np.arange(self.n_locations), self._current_locs] = 1
                onehots = onehots.reshape(-1)
                obs['force_location'] = onehots.astype('float32')
            else:
                raise Exception(self.mode)
            obs['force_location'] = np.tile(obs['force_location'], 100)

        return obs

    def _generate_loc(self):
        if self.mode == 'corners_xy':
            loc = np.random.choice([0, 8, 72, 80], size=self.n_locations,
                                                  replace=False)
        elif self.mode == 'normal':
            loc = np.random.choice(
                81, size=self.n_locations, replace=False)
        elif self.mode == 'corners_onehot':
            loc = np.random.choice(4, size=self.n_locations, replace=False)
        else:
            raise Exception(self.mode)
        return loc

    def _compute_diagonal_reward(self, physics):
        pos_ll = physics.data.geom_xpos[86, :2]
        pos_lr = physics.data.geom_xpos[81, :2]
        pos_ul = physics.data.geom_xpos[59, :2]
        pos_ur = physics.data.geom_xpos[54, :2]
        diag_dist1 = np.linalg.norm(pos_ll - pos_ur)
        diag_dist2 = np.linalg.norm(pos_lr - pos_ul)
        diag_reward = diag_dist1 + diag_dist2
        return diag_reward

    def _compute_area_concave(self, physics):
        points = physics.data.geom_xpos[6:, :2]
        alpha_shape = alphashape.alphashape(points, 20.0)
        return alpha_shape.area

    def _compute_area_convex(self, physics):
        joints = physics.data.geom_xpos[6:, :2]
        hull = ConvexHull(joints)
        vertices = joints[hull.vertices]

        x, y = vertices[:, 0], vertices[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) -
                            np.dot(y, np.roll(x, 1)))
        return area

    def _compute_area_reward(self, physics):
        pixels = physics.render(width=self.pixel_size, height=self.pixel_size,
                                camera_id=self.camera_id)
        segmentation = (pixels < 100).any(axis=-1).astype('float32')
        reward = segmentation.mean()
        return reward
