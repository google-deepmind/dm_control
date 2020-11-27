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

"""Procedurally generated LQR domain."""

import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import xml_tools
from lxml import etree
import numpy as np

from dm_control.utils import io as resources

_DEFAULT_TIME_LIMIT = float('inf')
_CONTROL_COST_COEF = 0.1
SUITE = containers.TaggedTasks()


def get_model_and_assets(n_bodies, n_actuators, random):
  """Returns the model description as an XML string and a dict of assets.

  Args:
    n_bodies: An int, number of bodies of the LQR.
    n_actuators: An int, number of actuated bodies of the LQR. `n_actuators`
      should be less or equal than `n_bodies`.
    random: A `numpy.random.RandomState` instance.

  Returns:
    A tuple `(model_xml_string, assets)`, where `assets` is a dict consisting of
    `{filename: contents_string}` pairs.
  """
  return _make_model(n_bodies, n_actuators, random), common.ASSETS


@SUITE.add()
def lqr_2_1(time_limit=_DEFAULT_TIME_LIMIT, random=None,
            environment_kwargs=None):
  """Returns an LQR environment with 2 bodies of which the first is actuated."""
  return _make_lqr(n_bodies=2,
                   n_actuators=1,
                   control_cost_coef=_CONTROL_COST_COEF,
                   time_limit=time_limit,
                   random=random,
                   environment_kwargs=environment_kwargs)


@SUITE.add()
def lqr_6_2(time_limit=_DEFAULT_TIME_LIMIT, random=None,
            environment_kwargs=None):
  """Returns an LQR environment with 6 bodies of which first 2 are actuated."""
  return _make_lqr(n_bodies=6,
                   n_actuators=2,
                   control_cost_coef=_CONTROL_COST_COEF,
                   time_limit=time_limit,
                   random=random,
                   environment_kwargs=environment_kwargs)


def _make_lqr(n_bodies, n_actuators, control_cost_coef, time_limit, random,
              environment_kwargs):
  """Returns a LQR environment.

  Args:
    n_bodies: An int, number of bodies of the LQR.
    n_actuators: An int, number of actuated bodies of the LQR. `n_actuators`
      should be less or equal than `n_bodies`.
    control_cost_coef: A number, the coefficient of the control cost.
    time_limit: An int, maximum time for each episode in seconds.
    random: Either an existing `numpy.random.RandomState` instance, an
      integer seed for creating a new `RandomState`, or None to select a seed
      automatically.
    environment_kwargs: A `dict` specifying keyword arguments for the
      environment, or None.

  Returns:
    A LQR environment with `n_bodies` bodies of which first `n_actuators` are
    actuated.
  """

  if not isinstance(random, np.random.RandomState):
    random = np.random.RandomState(random)

  model_string, assets = get_model_and_assets(n_bodies, n_actuators,
                                              random=random)
  physics = Physics.from_xml_string(model_string, assets=assets)
  task = LQRLevel(control_cost_coef, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit,
                             **environment_kwargs)


def _make_body(body_id, stiffness_range, damping_range, random):
  """Returns an `etree.Element` defining a body.

  Args:
    body_id: Id of the created body.
    stiffness_range: A tuple of (stiffness_lower_bound, stiffness_uppder_bound).
      The stiffness of the joint is drawn uniformly from this range.
    damping_range: A tuple of (damping_lower_bound, damping_upper_bound). The
      damping of the joint is drawn uniformly from this range.
    random: A `numpy.random.RandomState` instance.

  Returns:
   A new instance of `etree.Element`. A body element with two children: joint
   and geom.
  """
  body_name = 'body_{}'.format(body_id)
  joint_name = 'joint_{}'.format(body_id)
  geom_name = 'geom_{}'.format(body_id)

  body = etree.Element('body', name=body_name)
  body.set('pos', '.25 0 0')
  joint = etree.SubElement(body, 'joint', name=joint_name)
  body.append(etree.Element('geom', name=geom_name))
  joint.set('stiffness',
            str(random.uniform(stiffness_range[0], stiffness_range[1])))
  joint.set('damping',
            str(random.uniform(damping_range[0], damping_range[1])))
  return body


def _make_model(n_bodies,
                n_actuators,
                random,
                stiffness_range=(15, 25),
                damping_range=(0, 0)):
  """Returns an MJCF XML string defining a model of springs and dampers.

  Args:
    n_bodies: An integer, the number of bodies (DoFs) in the system.
    n_actuators: An integer, the number of actuated bodies.
    random: A `numpy.random.RandomState` instance.
    stiffness_range: A tuple containing minimum and maximum stiffness. Each
      joint's stiffness is sampled uniformly from this interval.
    damping_range: A tuple containing minimum and maximum damping. Each joint's
      damping is sampled uniformly from this interval.

  Returns:
    An MJCF string describing the linear system.

  Raises:
    ValueError: If the number of bodies or actuators is erronous.
  """
  if n_bodies < 1 or n_actuators < 1:
    raise ValueError('At least 1 body and 1 actuator required.')
  if n_actuators > n_bodies:
    raise ValueError('At most 1 actuator per body.')

  file_path = os.path.join(os.path.dirname(__file__), 'lqr.xml')
  with resources.GetResourceAsFile(file_path) as xml_file:
    mjcf = xml_tools.parse(xml_file)
  parent = mjcf.find('./worldbody')
  actuator = etree.SubElement(mjcf.getroot(), 'actuator')
  tendon = etree.SubElement(mjcf.getroot(), 'tendon')

  for body in range(n_bodies):
    # Inserting body.
    child = _make_body(body, stiffness_range, damping_range, random)
    site_name = 'site_{}'.format(body)
    child.append(etree.Element('site', name=site_name))

    if body == 0:
      child.set('pos', '.25 0 .1')
    # Add actuators to the first n_actuators bodies.
    if body < n_actuators:
      # Adding actuator.
      joint_name = 'joint_{}'.format(body)
      motor_name = 'motor_{}'.format(body)
      child.find('joint').set('name', joint_name)
      actuator.append(etree.Element('motor', name=motor_name, joint=joint_name))

    # Add a tendon between consecutive bodies (for visualisation purposes only).
    if body < n_bodies - 1:
      child_site_name = 'site_{}'.format(body + 1)
      tendon_name = 'tendon_{}'.format(body)
      spatial = etree.SubElement(tendon, 'spatial', name=tendon_name)
      spatial.append(etree.Element('site', site=site_name))
      spatial.append(etree.Element('site', site=child_site_name))
    parent.append(child)
    parent = child

  return etree.tostring(mjcf, pretty_print=True)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the LQR domain."""

  def state_norm(self):
    """Returns the norm of the physics state."""
    return np.linalg.norm(self.state())


class LQRLevel(base.Task):
  """A Linear Quadratic Regulator `Task`."""

  _TERMINAL_TOL = 1e-6

  def __init__(self, control_cost_coef, random=None):
    """Initializes an LQR level with cost = sum(states^2) + c*sum(controls^2).

    Args:
      control_cost_coef: The coefficient of the control cost.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).

    Raises:
      ValueError: If the control cost coefficient is not positive.
    """
    if control_cost_coef <= 0:
      raise ValueError('control_cost_coef must be positive.')

    self._control_cost_coef = control_cost_coef
    super().__init__(random=random)

  @property
  def control_cost_coef(self):
    return self._control_cost_coef

  def initialize_episode(self, physics):
    """Random state sampled from a unit sphere."""
    ndof = physics.model.nq
    unit = self.random.randn(ndof)
    physics.data.qpos[:] = np.sqrt(2) * unit / np.linalg.norm(unit)
    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict()
    obs['position'] = physics.position()
    obs['velocity'] = physics.velocity()
    return obs

  def get_reward(self, physics):
    """Returns a quadratic state and control reward."""
    position = physics.position()
    state_cost = 0.5 * np.dot(position, position)
    control_signal = physics.control()
    control_l2_norm = 0.5 * np.dot(control_signal, control_signal)
    return 1 - (state_cost + control_l2_norm * self._control_cost_coef)

  def get_evaluation(self, physics):
    """Returns a sparse evaluation reward that is not used for learning."""
    return float(physics.state_norm() <= 0.01)

  def get_termination(self, physics):
    """Terminates when the state norm is smaller than epsilon."""
    if physics.state_norm() < self._TERMINAL_TOL:
      return 0.0
