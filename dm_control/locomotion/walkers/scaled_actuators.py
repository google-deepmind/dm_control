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

"""Position & velocity actuators whose controls are scaled to a given range."""


_DISALLOWED_KWARGS = frozenset(
    ['biastype', 'gainprm', 'biasprm', 'ctrllimited',
     'joint', 'tendon', 'site', 'slidersite', 'cranksite'])
_ALLOWED_TAGS = frozenset(['joint', 'tendon', 'site'])

_GOT_INVALID_KWARGS = 'Received invalid keyword argument(s): {}'
_GOT_INVALID_TARGET = '`target` tag type should be one of {}: got {{}}'.format(
    sorted(_ALLOWED_TAGS))


def _check_target_and_kwargs(target, **kwargs):
  invalid_kwargs = _DISALLOWED_KWARGS.intersection(kwargs)
  if invalid_kwargs:
    raise TypeError(_GOT_INVALID_KWARGS.format(sorted(invalid_kwargs)))
  if target.tag not in _ALLOWED_TAGS:
    raise TypeError(_GOT_INVALID_TARGET.format(target))


def add_position_actuator(target, qposrange, ctrlrange=(-1, 1),
                          kp=1.0, **kwargs):
  """Adds a scaled position actuator that is bound to the specified element.

  This is equivalent to MuJoCo's built-in `<position>` actuator where an affine
  transformation is pre-applied to the control signal, such that the minimum
  control value corresponds to the minimum desired position, and the
  maximum control value corresponds to the maximum desired position.

  Args:
    target: A PyMJCF joint, tendon, or site element object that is to be
      controlled.
    qposrange: A sequence of two numbers specifying the allowed range of target
      position.
    ctrlrange: A sequence of two numbers specifying the allowed range of
      this actuator's control signal.
    kp: The gain parameter of this position actuator.
    **kwargs: Additional MJCF attributes for this actuator element.
      The following attributes are disallowed: `['biastype', 'gainprm',
      'biasprm', 'ctrllimited', 'joint', 'tendon', 'site',
      'slidersite', 'cranksite']`.

  Returns:
    A PyMJCF actuator element that has been added to the MJCF model containing
    the specified `target`.

  Raises:
    TypeError: `kwargs` contains an unrecognized or disallowed MJCF attribute,
      or `target` is not an allowed MJCF element type.
  """
  _check_target_and_kwargs(target, **kwargs)
  kwargs[target.tag] = target

  slope = (qposrange[1] - qposrange[0]) / (ctrlrange[1] - ctrlrange[0])
  g0 = kp * slope
  b0 = kp * (qposrange[0] - slope * ctrlrange[0])
  b1 = -kp
  b2 = 0
  return target.root.actuator.add('general',
                                  biastype='affine',
                                  gainprm=[g0],
                                  biasprm=[b0, b1, b2],
                                  ctrllimited=True,
                                  ctrlrange=ctrlrange,
                                  **kwargs)


def add_velocity_actuator(target, qvelrange, ctrlrange=(-1, 1),
                          kv=1.0, **kwargs):
  """Adds a scaled velocity actuator that is bound to the specified element.

  This is equivalent to MuJoCo's built-in `<velocity>` actuator where an affine
  transformation is pre-applied to the control signal, such that the minimum
  control value corresponds to the minimum desired velocity, and the
  maximum control value corresponds to the maximum desired velocity.

  Args:
    target: A PyMJCF joint, tendon, or site element object that is to be
      controlled.
    qvelrange: A sequence of two numbers specifying the allowed range of target
      velocity.
    ctrlrange: A sequence of two numbers specifying the allowed range of
      this actuator's control signal.
    kv: The gain parameter of this velocity actuator.
    **kwargs: Additional MJCF attributes for this actuator element.
      The following attributes are disallowed: `['biastype', 'gainprm',
      'biasprm', 'ctrllimited', 'joint', 'tendon', 'site',
      'slidersite', 'cranksite']`.

  Returns:
    A PyMJCF actuator element that has been added to the MJCF model containing
    the specified `target`.

  Raises:
    TypeError: `kwargs` contains an unrecognized or disallowed MJCF attribute,
      or `target` is not an allowed MJCF element type.
  """
  _check_target_and_kwargs(target, **kwargs)
  kwargs[target.tag] = target

  slope = (qvelrange[1] - qvelrange[0]) / (ctrlrange[1] - ctrlrange[0])
  g0 = kv * slope
  b0 = kv * (qvelrange[0] - slope * ctrlrange[0])
  b1 = 0
  b2 = -kv
  return target.root.actuator.add('general',
                                  biastype='affine',
                                  gainprm=[g0],
                                  biasprm=[b0, b1, b2],
                                  ctrllimited=True,
                                  ctrlrange=ctrlrange,
                                  **kwargs)
