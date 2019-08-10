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
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import xml_tools

from lxml import etree
import numpy as np

_TOL = 1e-13
_CLOSE = .01    # (Meters) Distance below which a thing is considered close.
_CONTROL_TIMESTEP = .01  # (Seconds)
_TIME_LIMIT = 10  # (Seconds)
# _ARM_JOINTS = ['arm_root', 'arm_shoulder', 'arm_elbow', 'arm_wrist',
#                'finger', 'fingertip', 'thumb', 'thumbtip']
# _ARM_JOINTS = ['arm_root', 'arm_shoulder', 'arm_elbow', 'arm_wrist',
#             ]
_ARM_JOINTS = ['arm_root', 'arm_shoulder', 'arm_elbow',
            ]


CORNER_INDEX_ACTION=['B0_0','B0_8','B8_0','B8_8']

# _ARM_JOINTS = [
#                'finger', 'fingertip', 'thumb', 'thumbtip']

SUITE = containers.TaggedTasks()

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('stacker.xml'), common.ASSETS

def make_model(n_boxes):
  """Returns a tuple containing the model XML string and a dict of assets."""
  xml_string = common.read_model('stacker.xml')
  parser = etree.XMLParser(remove_blank_text=True)
  mjcf = etree.XML(xml_string, parser)

  # Remove unused boxes
  for b in range(n_boxes, 4):
    box = xml_tools.find_element(mjcf, 'body', 'box' + str(b))
    box.getparent().remove(box)

  return etree.tostring(mjcf, pretty_print=True), common.ASSETS


@SUITE.add('hard')
def easy(fully_observable=True, time_limit=_TIME_LIMIT, random=None,
            environment_kwargs=None):
  """Returns stacker task with 2 boxes."""

  physics=Physics.from_xml_string(*get_model_and_assets())

  task = Stack(randomize_gains=False,random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, control_timestep=_CONTROL_TIMESTEP, time_limit=time_limit,
      **environment_kwargs)


# @SUITE.add('hard')
# def stack_4(fully_observable=True, time_limit=_TIME_LIMIT, random=None,
#             environment_kwargs=None):
#   """Returns stacker task with 4 boxes."""
#   n_boxes = 4
#   physics = Physics.from_xml_string(*make_model(n_boxes=n_boxes))
#   task = Stack(n_boxes=n_boxes,
#                fully_observable=fully_observable,
#                random=random)
#   environment_kwargs = environment_kwargs or {}
#   return control.Environment(
#       physics, task, control_timestep=_CONTROL_TIMESTEP, time_limit=time_limit,
#       **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics with additional features for the Planar Manipulator domain."""

  def bounded_joint_pos(self, joint_names):
    """Returns joint positions as (sin, cos) values."""
    joint_pos = self.named.data.qpos[joint_names]
    return np.vstack([np.sin(joint_pos), np.cos(joint_pos)]).T

  def joint_vel(self, joint_names):
    """Returns joint velocities."""
    return self.named.data.qvel[joint_names]

  def body_2d_pose(self, body_names, orientation=True):
    """Returns positions and/or orientations of bodies."""
    if not isinstance(body_names, str):
      body_names = np.array(body_names).reshape(-1, 1)  # Broadcast indices.
    pos = self.named.data.xpos[body_names, ['x', 'z']]
    if orientation:
      ori = self.named.data.xquat[body_names, ['qw', 'qy']]
      return np.hstack([pos, ori])
    else:
      return pos

  def touch(self):
    return np.log1p(self.data.sensordata)

  def site_distance(self, site1, site2):
    site1_to_site2 = np.diff(self.named.data.site_xpos[[site2, site1]], axis=0)
    return np.linalg.norm(site1_to_site2)


class Stack(base.Task):
  """A Stack `Task`: stack the boxes."""

  def __init__(self, randomize_gains, random=None):
    """Initialize an instance of `PointMass`.

    Args:
      randomize_gains: A `bool`, whether to randomize the actuator gains.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._randomize_gains = randomize_gains

    # self._ref_joint_vel_indexes = [
    #   "right_j" + str(i) for i in range(6)
    # ]
    # self._ref_joint_vel_indexes.append('B0_0')
    # self._ref_joint_vel_indexes+=['r_gripper_l_finger_joint','r_gripper_r_finger_joint']
    self.index = 1
    # self.mujoco_robot_dof = 7
    # self.gripper_dof = 2
    # self.dof = self.mujoco_robot_dof + self.gripper_dof
    super(Stack, self).__init__(random=random)

  def initialize_episode(self, physics):

    randint = self.random.randint
    uniform = self.random.uniform
    model = physics.named.model
    data = physics.named.data
    self.index=1

    # Find a collision-free random initial configuration.
    penetrating = True
    while penetrating:
      # Randomise angles of arm joints.
      is_limited = model.jnt_limited[_ARM_JOINTS].astype(np.bool)
      joint_range = model.jnt_range[_ARM_JOINTS]
      lower_limits = np.where(is_limited, joint_range[:, 0], -np.pi)
      upper_limits = np.where(is_limited, joint_range[:, 1], np.pi)
      angles = uniform(lower_limits, upper_limits)
      data.qpos[_ARM_JOINTS] = angles

      # # Symmetrize hand.
      # data.qpos['finger'] = data.qpos['thumb']
      physics.after_reset()
      penetrating = physics.data.ncon > 1

    physics.data.xpos[6:, :2] = physics.data.xpos[6:, :2] + self.random.uniform(-.3, .3)
    physics.named.data.xfrc_applied['B3_4', :3] = np.array([0, 0, -2])
    physics.named.data.xfrc_applied['B4_4', :3] = np.array([0, 0, -2])
    physics.named.data.xfrc_applied['B0_8', :3] = np.array([0.1,0.1,0.5])

    super(Stack, self).initialize_episode(physics)

  def action_range(self):
    """
    Action lower/upper limits per dimension.
    """
    low = np.ones(self.dof) * -1.
    high = np.ones(self.dof) * 1.
    return low, high

  def format_action(self, action):
    """
    1 => open, -1 => closed
    """
    # assert len(action) == 1
    return np.array([1 * action, -1 * action])

  def before_step(self, action, physics):

    global index
    # pos=physics.named.data.xpos['B4_4']
    # target_pos=action[:3]
    # gripper_action=action[3]
    physics.named.model.eq_active[-1] = 0
    if self.index >20:
      physics.named.data.xfrc_applied['B0_8', :3] = np.zeros(3)

      if self.index >100:

        if self.index < 105:
          target_pos = physics.named.data.xpos['B8_8']

          physics.named.model.eq_active[-1] = 1


        else:

          if self.index < 125:
            target_pos = np.array([0.2, 0.1, 0.1])

            physics.named.model.eq_active[-1] = 1
          else:
            # if self.index < 155:
            #   target_pos = np.array([0.7, 0, 0.9])+np.array([0.2,0.2,0])
            #   gripper_action = 1
            #   physics.named.model.eq_active[-1] = 1
            # else:
            target_pos = np.array([0.1, 0, 0.3]) + np.array([0.1, 0.1, 0])
            gripper_action = 1
            physics.named.model.eq_active[-1] = 0

      # physics.named.data.site_xpos['fingertip_touch'] = physics.named.data.xpos['r_gripper_l_finger_tip']
        print(self.index)
        print(target_pos)
        result = qpos_from_site_pose(physics, site_name='grasp', target_pos=target_pos, max_steps=200,
                                     joint_names=_ARM_JOINTS, inplace=True)
        print(result.success)
        print(result.err_norm)
        # assert result.err_norm <= _TOL

        physics.named.data.qpos[:] = result.qpos

      # gripper_action_actual = self.format_action(gripper_action)
      # physics.data.ctrl[-1] = gripper_action
    self.index += 1
    # gripper_action_actual[]
    # for contact in physics.data.contact[0:physics.data.ncon]:
    #   geom_name1=physics.model.id2name(contact.geom1,'geom')
    #   geom_name2=physics.model.id2name(contact.geom2,'geom')
    #   # geom.append(geom_name1)
    # geom.append(geom_name2)
    # print("geom1:{},geom2,{}".format(geom_name1,geom_name2))

    # gravity compensation
    physics.named.data.qfrc_applied[
      _ARM_JOINTS
    ] = physics.named.data.qfrc_bias[_ARM_JOINTS]

    # physics.named.data.xpos['r_gripper_r_finger_tip']=physics.named.data.xpos['r_gripper_l_finger_tip']=physics.named.data.xpos['B0_0']

  def get_observation(self, physics):
    """Returns either features or only sensors (to be used with pixels)."""
    obs = collections.OrderedDict()
    obs['arm_pos'] = physics.bounded_joint_pos(_ARM_JOINTS)
    obs['arm_vel'] = physics.joint_vel(_ARM_JOINTS)
    obs['touch'] = physics.touch()
    # if self._fully_observable:
    # obs['hand_pos'] = physics.body_2d_pose('hand')
    obs['cloth_pos']=physics.body_2d_pose('B0_0')
    obs['cloth_vel']=physics.joint_vel('J0_0_0')
      # obs['box_pos'] = physics.body_2d_pose(self._box_names)
      # obs['box_vel'] = physics.joint_vel(self._box_joint_names)
      # obs['target_pos'] = physics.body_2d_pose('target', orientation=False)
    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    # box_size = physics.named.model.geom_size['target', 0]
    # min_box_to_target_distance = min(physics.site_distance(name, 'target')
    #                                  for name in self._box_names)
    # box_is_close = rewards.tolerance(min_box_to_target_distance,
    #                                  margin=2*box_size)

    pos_ll =physics.data.geom_xpos[86,:2]
    pos_lr = physics.data.geom_xpos[81, :2]
    pos_ul = physics.data.geom_xpos[59, :2]
    pos_ur = physics.data.geom_xpos[54, :2]
    # print(pos_ll)
    # print(pos_lr)
    # print(pos_ur)
    # print(pos_ul)
    diag_dist1 = np.linalg.norm(pos_ll - pos_ur)
    diag_dist2 = np.linalg.norm(pos_lr - pos_ul)
    reward_dist = diag_dist1 + diag_dist2

    hand_to_target_distance = physics.site_distance('grasp', 'target')
    hand_is_far = rewards.tolerance(hand_to_target_distance,
                                    bounds=(.1, float('inf')),
                                    margin=_CLOSE)
    # return box_is_close * hand_is_far
    return reward_dist*hand_is_far