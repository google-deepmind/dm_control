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

"""Cartpole domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Internal dependencies.

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards

from lxml import etree
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin


_DEFAULT_TIME_LIMIT = 10
SUITE = containers.TaggedTasks()


def get_model_and_assets(num_poles=1):
  """Returns a tuple containing the model XML string and a dict of assets."""
  return _make_model(num_poles), common.ASSETS


@SUITE.add('benchmarking')
def balance(time_limit=_DEFAULT_TIME_LIMIT, random=None):
  """Returns the Cartpole Balance task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Balance(swing_up=False, sparse=False, random=random)
  return control.Environment(physics, task, time_limit=time_limit)


@SUITE.add('benchmarking')
def balance_sparse(time_limit=_DEFAULT_TIME_LIMIT, random=None):
  """Returns the sparse reward variant of the Cartpole Balance task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Balance(swing_up=False, sparse=True, random=random)
  return control.Environment(physics, task, time_limit=time_limit)


@SUITE.add('benchmarking')
def swingup(time_limit=_DEFAULT_TIME_LIMIT, random=None, **kwargs):
  """Returns the Cartpole Swing-Up task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Balance(swing_up=True, sparse=False, random=random)
  return control.Environment(physics, task, time_limit=time_limit, **kwargs)


@SUITE.add('benchmarking')
def swingup_sparse(time_limit=_DEFAULT_TIME_LIMIT, random=None):
  """Returns the sparse reward variant of teh Cartpole Swing-Up task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Balance(swing_up=True, sparse=True, random=random)
  return control.Environment(physics, task, time_limit=time_limit)


@SUITE.add()
def two_poles(time_limit=_DEFAULT_TIME_LIMIT, random=None):
  """Returns the Cartpole Balance task."""
  physics = Physics.from_xml_string(*get_model_and_assets(num_poles=2))
  task = Balance(swing_up=True, sparse=False, random=random)
  return control.Environment(physics, task, time_limit=time_limit)


@SUITE.add()
def three_poles(time_limit=_DEFAULT_TIME_LIMIT, random=None):
  """Returns the Cartpole Balance task."""
  physics = Physics.from_xml_string(*get_model_and_assets(num_poles=3))
  task = Balance(swing_up=True, sparse=False, random=random)
  return control.Environment(physics, task, time_limit=time_limit)


def _make_model(n_poles):
  """Generates an xml string defining a cart with `n_poles` bodies."""
  xml_string = common.read_model('cartpole.xml')
  if n_poles == 1:
    return xml_string
  mjcf = etree.fromstring(xml_string)
  parent = mjcf.find('./worldbody/body/body')  # Find first pole.
  # Make chain of poles.
  for pole_index in xrange(2, n_poles+1):
    child = etree.Element('body', name='pole_{}'.format(pole_index),
                          pos='0 0 1', childclass='pole')
    etree.SubElement(child, 'joint', name='hinge_{}'.format(pole_index))
    etree.SubElement(child, 'geom', name='pole_{}'.format(pole_index))
    parent.append(child)
    parent = child
  # Move plane down.
  floor = mjcf.find('./worldbody/geom')
  floor.set('pos', '0 0 {}'.format(1 - n_poles - .05))
  # Move cameras back.
  cameras = mjcf.findall('./worldbody/camera')
  cameras[0].set('pos', '0 {} 1'.format(-1 - 2*n_poles))
  cameras[1].set('pos', '0 {} 2'.format(-2*n_poles))
  return etree.tostring(mjcf, pretty_print=True)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Cartpole domain."""

  def cart_position(self):
    """Returns the position of the cart."""
    return self.named.data.qpos['slider'][0]

  def angular_vel(self):
    """Returns the angular velocity of the pole."""
    return self.data.qvel[1:]

  def pole_angle_cosine(self):
    """Returns the cosine of the pole angle."""
    return self.named.data.xmat[2:, 'zz']

  def bounded_position(self):
    """Returns the state, with pole angle split into sin/cos."""
    return np.hstack((self.cart_position(),
                      self.named.data.xmat[2:, ['zz', 'xz']].ravel()))


class Balance(base.Task):
  """A Cartpole `Task` to balance the pole.

  State is initialized either close to the target configuration or at a random
  configuration.
  """
  _CART_RANGE = (-.25, .25)
  _ANGLE_COSINE_RANGE = (.995, 1)

  def __init__(self, swing_up, sparse, random=None):
    """Initializes an instance of `Balance`.

    Args:
      swing_up: A `bool`, which if `True` sets the cart to the middle of the
        slider and the pole pointing towards the ground. Otherwise, sets the
        cart to a random position on the slider and the pole to a random
        near-vertical position.
      sparse: A `bool`, whether to return a sparse or a smooth reward.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._sparse = sparse
    self._swing_up = swing_up
    super(Balance, self).__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

    Initializes the cart and pole according to `swing_up`, and in both cases
    adds a small random initial velocity to break symmetry.

    Args:
      physics: An instance of `Physics`.
    """
    nv = physics.model.nv
    if self._swing_up:
      physics.named.data.qpos['slider'] = .01*self.random.randn()
      physics.named.data.qpos['hinge_1'] = np.pi + .01*self.random.randn()
      physics.named.data.qpos[2:] = .1*self.random.randn(nv - 2)
    else:
      physics.named.data.qpos['slider'] = self.random.uniform(-.1, .1)
      physics.named.data.qpos[1:] = self.random.uniform(-.034, .034, nv - 1)
    physics.named.data.qvel[:] = 0.01 * self.random.randn(physics.model.nv)

  def get_observation(self, physics):
    """Returns an observation of the (bounded) physics state."""
    obs = collections.OrderedDict()
    obs['position'] = physics.bounded_position()
    obs['velocity'] = physics.velocity()
    return obs

  def _get_reward(self, physics, sparse):
    if sparse:
      cart_in_bounds = rewards.tolerance(physics.cart_position(),
                                         self._CART_RANGE)
      angle_in_bounds = rewards.tolerance(physics.pole_angle_cosine(),
                                          self._ANGLE_COSINE_RANGE).prod()
      return cart_in_bounds * angle_in_bounds
    else:
      upright = (physics.pole_angle_cosine() + 1) / 2
      centered = rewards.tolerance(physics.cart_position(), margin=2)
      centered = (1 + centered) / 2
      small_control = rewards.tolerance(physics.control(), margin=1,
                                        value_at_margin=0,
                                        sigmoid='quadratic')[0]
      small_control = (4 + small_control) / 5
      small_velocity = rewards.tolerance(physics.angular_vel(), margin=5).min()
      small_velocity = (1 + small_velocity) / 2
      return upright.mean() * small_control * small_velocity * centered

  def get_reward(self, physics):
    """Returns a sparse or a smooth reward, as specified in the constructor."""
    return self._get_reward(physics, sparse=self._sparse)
