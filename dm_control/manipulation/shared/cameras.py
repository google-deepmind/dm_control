# Copyright 2018 The dm_control Authors.
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

"""Tools for adding custom cameras to the arena."""

import collections

from dm_control.composer.observation import observable


CameraSpec = collections.namedtuple('CameraSpec', ['name', 'pos', 'xyaxes'])

# Custom cameras that may be added to the arena for particular tasks.
FRONT_CLOSE = CameraSpec(
    name='front_close',
    pos=(0., -0.6, 0.75),
    xyaxes=(1., 0., 0., 0., 0.7, 0.75)
)

FRONT_FAR = CameraSpec(
    name='front_far',
    pos=(0., -0.8, 1.),
    xyaxes=(1., 0., 0., 0., 0.7, 0.75)
)

TOP_DOWN = CameraSpec(
    name='top_down',
    pos=(0., 0., 2.5),
    xyaxes=(1., 0., 0., 0., 1., 0.)
)

LEFT_CLOSE = CameraSpec(
    name='left_close',
    pos=(-0.6, 0., 0.75),
    xyaxes=(0., -1., 0., 0.7, 0., 0.75)
)

RIGHT_CLOSE = CameraSpec(
    name='right_close',
    pos=(0.6, 0., 0.75),
    xyaxes=(0., 1., 0., -0.7, 0., 0.75)
)


def add_camera_observables(entity, obs_settings, *camera_specs):
  """Adds cameras to an entity's worldbody and configures observables for them.

  Args:
    entity: A `composer.Entity`.
    obs_settings: An `observations.ObservationSettings` instance.
    *camera_specs: Instances of `CameraSpec`.

  Returns:
    A `collections.OrderedDict` keyed on camera names, containing pre-configured
    `observable.MJCFCamera` instances.
  """
  obs_dict = collections.OrderedDict()
  for spec in camera_specs:
    camera = entity.mjcf_model.worldbody.add('camera', **spec._asdict())
    obs = observable.MJCFCamera(camera)
    obs.configure(**obs_settings.camera._asdict())
    obs_dict[spec.name] = obs
  return obs_dict
