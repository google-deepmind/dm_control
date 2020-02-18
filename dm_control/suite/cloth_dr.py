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

"""Planar Stacker domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from dm_control import mujoco
from dm_env import specs
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import xml_tools
from dm_control.suite.wrappers.modder import LightModder, MaterialModder, CameraModder

import os
from imageio import imsave
from PIL import Image, ImageColor
from lxml import etree
import numpy as np
import math

_TOL = 1e-13
_CLOSE = .01  # (Meters) Distance below which a thing is considered close.
_CONTROL_TIMESTEP = .02  # (Seconds)
_TIME_LIMIT = 30  # (Seconds)

CORNER_INDEX_ACTION = ['B0_0', 'B0_8', 'B8_0', 'B8_8']
CORNER_INDEX_GEOM = ['G0_0', 'G0_8', 'G8_0', 'G8_8']

W = 64

SUITE = containers.TaggedTasks()


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model('cloth_dr.xml'), common.ASSETS


@SUITE.add('hard')
def easy(time_limit=_TIME_LIMIT, random=None, environment_kwargs=None, **kwargs):
    """Returns stacker task with 2 boxes."""

    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Stack(randomize_gains=False, random=random, **kwargs)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, control_timestep=_CONTROL_TIMESTEP, special_task=True, time_limit=time_limit,
        **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics with additional features for the Planar Manipulator domain."""


class Stack(base.Task):
    """A Stack `Task`: stack the boxes."""

    def __init__(self, randomize_gains, random=None, random_pick=True, init_flat=False, use_dr=False):
        """Initialize an instance of `PointMass`.

        Args:
          randomize_gains: A `bool`, whether to randomize the actuator gains.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._randomize_gains = randomize_gains
        self._random_pick = random_pick
        self._init_flat = init_flat
        self._use_dr = use_dr

        super(Stack, self).__init__(random=random)

    def action_spec(self, physics):
        """Returns a `BoundedArraySpec` matching the `physics` actuators."""
        if not self._random_pick:
            return specs.BoundedArray(
                shape=(5,), dtype=np.float, minimum=[-1.0] * 5, maximum=[1.0] * 5)
        else:
            return specs.BoundedArray(
                shape=(3,), dtype=np.float, minimum=[-1.0, -1.0, -1.0], maximum=[1.0, 1.0, 1.0])


    def initialize_episode(self, physics):
        physics.named.data.xfrc_applied['B3_4', :3] = np.array([0, 0, -2])
        physics.named.data.xfrc_applied['B4_4', :3] = np.array([0, 0, -2])

        if not self._init_flat:
            physics.named.data.xfrc_applied[CORNER_INDEX_ACTION, :3] = np.random.uniform(-.3, .3, size=3)

        render_kwargs = {}
        render_kwargs['camera_id'] = 0
        render_kwargs['width'] = W
        render_kwargs['height'] = W
        image = physics.render(**render_kwargs)
        self.image = image

        image_dim_1 = image[:, :, 1].reshape((W, W, 1))
        image_dim_2 = image[:, :, 2].reshape((W, W, 1))
        self.mask = (np.all(image > 200, axis=2) + np.all(image_dim_2 < 40, axis=2) + (
            ~np.all(image_dim_1 > 135, axis=2))).astype(int)

        if self._use_dr:
            # initialize the random parameters
            self.cam_pos = np.array([0, 0, 0.75])
            self.cam_quat = np.array([1, 0, 0, 0])

            self.light_diffuse = np.array([0.6, 0.6, 0.6])
            self.light_specular = np.array([0.3, 0.3, 0.3])
            self.light_ambient = np.array([0, 0, 0])
            self.light_castshadow = np.array([1])
            self.light_dir = np.array([0, 0, -1])
            self.light_pos = np.array([0, 0, 1])

            self.dof_damping = np.concatenate([np.zeros((6)), np.ones((160)) * 0.002], axis=0)
            self.body_mass = np.concatenate([np.zeros(1), np.ones(81) * 0.00309])
            self.body_inertia = np.concatenate(
                [np.zeros((1, 3)), np.tile(np.array([[2.32e-07, 2.32e-07, 4.64e-07]]), (81, 1))], axis=0)
            self.geom_friction = np.tile(np.array([[1, 0.005, 0.001]]), (86, 1))

        super(Stack, self).initialize_episode(physics)

    def before_step(self, action, physics):

        """Sets the control signal for the actuators to values in `action`."""
        # Support legacy internal code.

        # clear previous xfrc_force
        physics.named.data.xfrc_applied[:, :3] = np.zeros((3,))

        if self._use_dr:
            physics.named.model.mat_texid[15] = np.random.choice(3, 1) + 9

            ### visual randomization

            #    #light randomization
            lightmodder = LightModder(physics)
            # ambient_value=lightmodder.get_ambient('light')
            ambient_value = self.light_ambient.copy() + np.random.uniform(-0.4, 0.4, size=3)
            lightmodder.set_ambient('light', ambient_value)

            # shadow_value=lightmodder.get_castshadow('light')
            shadow_value = self.light_castshadow.copy()
            lightmodder.set_castshadow('light', shadow_value + np.random.uniform(0, 40))
            # diffuse_value=lightmodder.get_diffuse('light')
            diffuse_value = self.light_diffuse.copy()
            lightmodder.set_diffuse('light', diffuse_value + np.random.uniform(-0.01, 0.01, ))
            # dir_value=lightmodder.get_dir('light')
            dir_value = self.light_dir.copy()
            lightmodder.set_dir('light', dir_value + np.random.uniform(-0.1, 0.1))
            # pos_value=lightmodder.get_pos('light')
            pos_value = self.light_pos.copy()
            lightmodder.set_pos('light', pos_value + np.random.uniform(-0.1, 0.1))
            # specular_value=lightmodder.get_specular('light')
            specular_value = self.light_specular.copy()
            lightmodder.set_specular('light', specular_value + np.random.uniform(-0.1, 0.1))

            # material randomization#
            #    Material_ENUM=['ground','wall_x','wall_y','wall_neg_x','wall_neg_x','wall_neg_y']
            #    materialmodder=MaterialModder(physics)
            #    for name in Material_ENUM:
            #     materialmodder.rand_all(name)

            # camera randomization
            # cameramodder=CameraModder(physics)
            # # fovy_value=cameramodder.get_fovy('fixed')
            # # cameramodder.set_fovy('fixed',fovy_value+np.random.uniform(-1,1))
            # # pos_value = cameramodder.get_pos('fixed')
            # pos_value=self.cam_pos.copy()
            # cameramodder.set_pos('fixed',np.random.uniform(-0.003,0.003,size=3)+pos_value)
            # # quat_value = cameramodder.get_quat('fixed')
            # quat_value=self.cam_quat.copy()
            # cameramodder.set_quat('fixed',quat_value+np.random.uniform(-0.01,0.01,size=4))

            ### physics randomization

            # damping randomization

            physics.named.model.dof_damping[:] = np.random.uniform(0, 0.0001) + self.dof_damping

            # # friction randomization
            geom_friction = self.geom_friction.copy()
            physics.named.model.geom_friction[5:, 0] = np.random.uniform(-0.5, 0.5) + geom_friction[5:, 0]
            #
            physics.named.model.geom_friction[5:, 1] = np.random.uniform(-0.002, 0.002) + geom_friction[5:, 1]
            #
            physics.named.model.geom_friction[5:, 2] = np.random.uniform(-0.0005, 0.0005) + geom_friction[5:, 2]
            #
            # # inertia randomization
            body_inertia = self.body_inertia.copy()
            physics.named.model.body_inertia[1:] = np.random.uniform(-0.5, 0.5) * 1e-07 + body_inertia[1:]
            #
            # mass randomization
            body_mass = self.body_mass.copy()

            physics.named.model.body_mass[1:] = np.random.uniform(-0.0005, 0.0005) + body_mass[1:]

        # scale the position to be a normal range
        if not self._random_pick:
            location = (action[:2] * 0.5 + 0.5) * 63
            location = np.round(location).astype('int32')
            goal_position = action[2:]
        else:
            location = self.current_loc
            goal_position = action
        goal_position = goal_position * 0.1

        # computing the mapping from geom_xpos to location in image
        cam_fovy = physics.model.cam_fovy[0]
        f = 0.5 * W / math.tan(cam_fovy * math.pi / 360)
        cam_matrix = np.array([[f, 0, W / 2], [0, f, W / 2], [0, 0, 1]])
        cam_mat = physics.data.cam_xmat[0].reshape((3, 3))
        cam_pos = physics.data.cam_xpos[0].reshape((3, 1))
        cam = np.concatenate([cam_mat, cam_pos], axis=1)
        cam_pos_all = np.zeros((81, 3, 1))
        for i in range(81):
            geom_xpos_added = np.concatenate([physics.data.geom_xpos[i+5], np.array([1])]).reshape((4, 1))
            cam_pos_all[i] = cam_matrix.dot(cam.dot(geom_xpos_added)[:3])

        # cam_pos_xy=cam_pos_all[5:,:]
        cam_pos_xy = np.rint(cam_pos_all[:, :2].reshape((81, 2)) / cam_pos_all[:, 2])
        cam_pos_xy = cam_pos_xy.astype(int)
        cam_pos_xy[:, 1] = W - cam_pos_xy[:, 1]

        # hyperparameter epsilon=3(selecting 3 nearest joint) and select the point
        epsilon = 3
        possible_index = []
        possible_z = []
        for i in range(81):
            # flipping the x and y to make sure it corresponds to the real location
            if abs(cam_pos_xy[i][0] - location[1]) < epsilon and abs(
                    cam_pos_xy[i][1] - location[0]) < epsilon and i > 4:
                possible_index.append(i)
                possible_z.append(physics.data.geom_xpos[i, 2])

        if possible_index != []:
            index = possible_index[possible_z.index(max(possible_z))]

            corner_action = index + 1
            corner_geom = index + 5

            # apply consecutive force to move the point to the target position
            position = goal_position + physics.named.data.geom_xpos[corner_geom]
            dist = position - physics.named.data.geom_xpos[corner_geom]

            loop = 0
            while np.linalg.norm(dist) > 0.025:
                loop += 1
                if loop > 40:
                    break
                physics.named.data.xfrc_applied[corner_action, :3] = dist * 20
                physics.step()
                self.after_step(physics)
                dist = position - physics.named.data.geom_xpos[corner_geom]

    def get_observation(self, physics):
        """Returns either features or only sensors (to be used with pixels)."""
        obs = collections.OrderedDict()

        image = self.get_image(physics)
        self.image = image

        if self._random_pick:
            location  = self.sample_location(physics)
        else:
            location = [-1, -1]
            mask = self.segment_image(image)
            location_range = np.transpose(np.where(mask))
            self.location_rate = location_range
            num_loc = np.shape(location_range)[0]
            self.num_loc = num_loc
        self.current_loc = location

        if self.current_loc is None:
            obs['location'] = np.tile([-1, -1], 50).reshape(-1).astype('float32') / 63.
        else:
            obs['location'] = np.tile(self.current_loc, 50).reshape(-1).astype('float32') / 63.
        return obs

    def get_termination(self, physics):
        if self.num_loc < 1:
            return 1.0
        else:
            return None

    def get_image(physics):
        render_kwargs = {}
        render_kwargs['camera_id'] = 0
        render_kwargs['width'] = W
        render_kwargs['height'] = W
        image = physics.render(**render_kwargs)
        return image

    def segment_image(self, image):
        image_dim_1 = image[:, :, [1]]
        image_dim_2 = image[:, :, [2]]
        mask = np.all(image> 200, axis=2) + np.all(image_dim_2 < 40, axis=2) + \
               (~np.all(image_dim_1 > 135, axis=2))
        return mask > 0

    def sample_location(self, physics):
        image = self.image
        location_range = np.transpose(np.where(self.segment_image(image)))
        self.location_range = location_range
        num_loc = np.shape(location_range)[0]
        self.num_loc = num_loc
        if num_loc == 0:
            return None
        index = np.random.randint(num_loc, size=1)
        location = location_range[index][0]
        return location

    def get_reward(self, physics):
        image_dim_1 = self.image[:, :, 1].reshape((W, W, 1))
        image_dim_2 = self.image[:, :, 2].reshape((W, W, 1))
        current_mask = (np.all(self.image > 200, axis=2) + np.all(image_dim_2 < 40, axis=2) + (
            ~np.all(image_dim_1 > 135, axis=2))).astype(int)
        area = np.sum(current_mask * self.mask)
        reward = area / np.sum(self.mask)
        return reward
