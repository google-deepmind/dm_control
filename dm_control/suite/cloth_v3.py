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
from dm_env import specs

from lxml import etree
import numpy as np

_TOL = 1e-13
_CLOSE = .01    # (Meters) Distance below which a thing is considered close.
# _CONTROL_TIMESTEP = .01  # (Seconds)
_TIME_LIMIT = 20  # (Seconds)
# _ARM_JOINTS = ['arm_root', 'arm_shoulder', 'arm_elbow', 'arm_wrist',
#                'finger', 'fingertip', 'thumb', 'thumbtip']
_ARM_JOINTS = ['arm_root', 'arm_shoulder', 'arm_elbow', 'arm_wrist',
            ]


CORNER_INDEX_ACTION=['B0_0','B0_8','B8_0','B8_8']

# _ARM_JOINTS = [
#                'finger', 'fingertip', 'thumb', 'thumbtip']

SUITE = containers.TaggedTasks()

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('cloth_v3.xml'), common.ASSETS



# , control_timestep=_CONTROL_TIMESTEP
@SUITE.add('hard')
def easy(time_limit=_TIME_LIMIT, random=None,
            environment_kwargs=None):


  physics=Physics.from_xml_string(*get_model_and_assets())

  task = Cloth(randomize_gains=False,random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit,
      **environment_kwargs)





class Physics(mujoco.Physics):
  """Physics with additional features for the Planar Manipulator domain."""



class Cloth(base.Task):
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

    # self.index = 1

    super(Cloth, self).__init__(random=random)

  def initialize_episode(self, physics):



    physics.data.xpos[6:, :2] = physics.data.xpos[6:, :2] + self.random.uniform(-.3, .3)
    physics.named.data.xfrc_applied['B3_4', :3] = np.array([0, 0, -2])
    physics.named.data.xfrc_applied['B4_4', :3] = np.array([0, 0, -2])

    physics.named.data.xfrc_applied[CORNER_INDEX_ACTION,:3]=np.random.uniform(-.5,.5,size=3)


    super(Cloth, self).initialize_episode(physics)

  def action_spec(self, physics):
    """Returns a `BoundedArray` matching the `physics` actuators."""
    # return specs.BoundedArray(
    # shape=(12,), dtype=np.float, minimum=[-5.0]*12 ,maximum=[5.0]*12)
    # return specs.BoundedArray(
    #     shape=(3,), dtype=np.float, minimum=[-5.0] * 3, maximum=[5.0] * 3)

    return specs.BoundedArray(
      shape=(12,), dtype=np.float, minimum=[-5.0] * 12, maximum=[5.0] * 12)


  def before_step(self, action, physics):
    action = action.reshape(4, 3)
    # physics.named.data.xpos['upper_arm']=np.array([0,0 ,0.4])
    # physics.named.data.xpos['middle_arm']=np.array([0,0,0.58])
    # physics.named.data.xpos['lower_arm']=np.array([0,0,0.73])
    # physics.named.data.xpos['hand']=np.array([0,0,0.85])
    # if self.index > 130:
    physics.named.data.xfrc_applied[CORNER_INDEX_ACTION, :3] = action
    # self.index+=1
    # physics.data.ctrl[:]=0
    # physics.named.data.qfrc_applied[
    #   _ARM_JOINTS
    # ] = physics.named.data.qfrc_bias[_ARM_JOINTS]

    # global index
    # pos=physics.named.data.xpos['B4_4']
    # target_pos=action[:3]
    # gripper_action=action[3]
    # physics.named.model.eq_active[-1] = 0
    # if self.index >20:
    #   physics.named.data.xfrc_applied['B0_8', :3] = np.zeros(3)
    #
    #   if self.index >100:
    #
    #     if self.index < 105:
    #       target_pos = physics.named.data.xpos['B8_8']
    #
    #       physics.named.model.eq_active[-1] = 1
    #
    #
    #     else:
    #
    #       if self.index < 125:
    #         target_pos = np.array([0.2, 0.1, 0.1])
    #
    #         physics.named.model.eq_active[-1] = 1
    #       else:
    #         # if self.index < 155:
    #         #   target_pos = np.array([0.7, 0, 0.9])+np.array([0.2,0.2,0])
    #         #   gripper_action = 1
    #         #   physics.named.model.eq_active[-1] = 1
    #         # else:
    #         target_pos = np.array([0.1, 0, 0.3]) + np.array([0.1, 0.1, 0])
    #         gripper_action = 1
    #         physics.named.model.eq_active[-1] = 0
    #
    #   # physics.named.data.site_xpos['fingertip_touch'] = physics.named.data.xpos['r_gripper_l_finger_tip']
    #     print(self.index)
    #     print(target_pos)
    #     result = qpos_from_site_pose(physics, site_name='grasp', target_pos=target_pos, max_steps=200,
    #                                  joint_names=_ARM_JOINTS, inplace=True)
    #     print(result.success)
    #     print(result.err_norm)
    #     # assert result.err_norm <= _TOL
    #
    #     physics.named.data.qpos[:] = result.qpos
    #
    #   # gripper_action_actual = self.format_action(gripper_action)
    #   # physics.data.ctrl[-1] = gripper_action
    # self.index += 1
    # gripper_action_actual[]
    # for contact in physics.data.contact[0:physics.data.ncon]:
    #   geom_name1=physics.model.id2name(contact.geom1,'geom')
    #   geom_name2=physics.model.id2name(contact.geom2,'geom')
    #   # geom.append(geom_name1)
    # geom.append(geom_name2)
    # print("geom1:{},geom2,{}".format(geom_name1,geom_name2))

    # gravity compensation



  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict()
    # obs['position'] = physics.position()
    # obs['velocity'] = physics.velocity()
    obs['position']=physics.data.qpos[4:].copy()
    obs['velocity']=physics.data.qvel[4:].copy()
    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""

    pos_ll = physics.named.data.geom_xpos['G0_0', :2]
    pos_lr = physics.named.data.geom_xpos['G0_8', :2]
    pos_ul = physics.named.data.geom_xpos['G8_0', :2]
    pos_ur = physics.named.data.geom_xpos['G8_8', :2]

    diag_dist1 = np.linalg.norm(pos_ll - pos_ur)
    diag_dist2 = np.linalg.norm(pos_lr - pos_ul)
    reward_dist = diag_dist1 + diag_dist2
    return reward_dist
