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

# Internal dependencies.

from dm_control import mujoco
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import xml_tools

from lxml import etree
import numpy as np


_CLOSE = .01    # (Meters) Distance below which a thing is considered close.
_CONTROL_TIMESTEP = .01  # (Seconds)
_TIME_LIMIT = 10  # (Seconds)
_ARM_JOINTS = ['arm_root', 'arm_shoulder', 'arm_elbow', 'arm_wrist',
               'finger', 'fingertip', 'thumb', 'thumbtip']

SUITE = containers.TaggedTasks()


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
def stack_2(observable=True, time_limit=_TIME_LIMIT, random=None):
  """Returns stacker task with 2 boxes."""
  n_boxes = 2
  physics = Physics.from_xml_string(*make_model(n_boxes=n_boxes))
  task = Stack(n_boxes, observable, random=random)
  return control.Environment(
      physics, task, control_timestep=_CONTROL_TIMESTEP, time_limit=time_limit)


@SUITE.add('hard')
def stack_4(observable=True, time_limit=_TIME_LIMIT, random=None):
  """Returns stacker task with 4 boxes."""
  n_boxes = 4
  physics = Physics.from_xml_string(*make_model(n_boxes=n_boxes))
  task = Stack(n_boxes, observable, random=random)
  return control.Environment(
      physics, task, control_timestep=_CONTROL_TIMESTEP, time_limit=time_limit)


class Physics(mujoco.Physics):
  """Physics with additional features for the Planar Manipulator domain."""

  def bounded_position(self):
    """Returns the state, with unbounded angles as sine/cosine."""
    state = []
    hinge_joint = enums.mjtJoint.mjJNT_HINGE
    for joint_id in range(self.model.njnt):
      joint_value = self.named.data.qpos[joint_id]
      if (not self.model.jnt_limited[joint_id] and
          self.model.jnt_type[joint_id] == hinge_joint):  # Unbounded hinge.
        state += [np.sin(joint_value), np.cos(joint_value)]
      else:
        state.append(joint_value)
    return np.asarray(state)

  def body_location(self, body):
    """Returns the x,z position and y orientation of a body."""
    body_position = self.named.model.body_pos[body, ['x', 'z']]
    body_orientation = self.named.model.body_quat[body, ['qw', 'qy']]
    return np.hstack((body_position, body_orientation))

  def proprioception(self):
    """Returns the arm state, with unbounded angles as sine/cosine."""
    arm = []
    for joint in _ARM_JOINTS:
      joint_value = self.named.data.qpos[joint]
      if not self.named.model.jnt_limited[joint]:
        arm += [np.sin(joint_value), np.cos(joint_value)]
      else:
        arm.append(joint_value)
    return np.hstack(arm + [self.named.data.qvel[_ARM_JOINTS]])

  def touch(self):
    return np.log1p(self.data.sensordata)

  def site_distance(self, site1, site2):
    site1_to_site2 = np.diff(self.named.data.site_xpos[[site2, site1]], axis=0)
    return np.linalg.norm(site1_to_site2)


class Stack(base.Task):
  """A Stack `Task`: stack the boxes."""

  def __init__(self, n_boxes, observable, random=None):
    """Initialize an instance of the `Stack` task.

    Args:
      n_boxes: An `int`, number of boxes to stack.
      observable: A `bool`, whether the observation contains target info.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._n_boxes = n_boxes
    self._observable = observable
    super(Stack, self).__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode."""
    # Local aliases
    randint = self.random.randint
    uniform = self.random.uniform
    model = physics.named.model
    data = physics.named.data

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

      # Symmetrize hand.
      data.qpos['finger'] = data.qpos['thumb']

      # Randomise target location.
      target_height = 2*randint(self._n_boxes) + 1
      box_size = model.geom_size['target', 0]
      model.body_pos['target', 'z'] = box_size * target_height
      model.body_pos['target', 'x'] = uniform(-.37, .37)

      # Randomise box locations.
      for b in range(self._n_boxes):
        box = 'box' + str(b)
        data.qpos[box + '_x'] = uniform(.1, .3)
        data.qpos[box + '_z'] = uniform(0, .7)
        data.qpos[box + '_y'] = uniform(0, 2*np.pi)

      # Check for collisions.
      physics.after_reset()
      penetrating = physics.data.ncon > 0

  def get_observation(self, physics):
    """Returns either features or only sensors (to be used with pixels)."""
    obs = collections.OrderedDict()
    if self._observable:
      box_locations = [physics.body_location('box' + str(b))
                       for b in range(self._n_boxes)]
      obs['position'] = physics.bounded_position()
      obs['hand'] = physics.body_location('hand')
      obs['boxes'] = np.hstack(box_locations)
      obs['velocity'] = physics.velocity()
      obs['touch'] = physics.touch()
    else:
      obs['proprioception'] = physics.proprioception()
      obs['touch'] = physics.touch()
    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    box_size = physics.named.model.geom_size['target', 0]
    def target_to_box(b):
      return rewards.tolerance(physics.site_distance('box' + str(b), 'target'),
                               margin=2*box_size)
    box_is_close = max(target_to_box(b) for b in range(self._n_boxes))
    hand_to_target = physics.site_distance('grasp', 'target')
    hand_is_far = rewards.tolerance(hand_to_target, (.1, float('inf')), _CLOSE)
    return box_is_close * hand_is_far
