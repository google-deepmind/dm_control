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

"""Locomotion texture assets."""

import collections
import os
import sys

ROOT_DIR = '../locomotion/arenas/assets'


def get_texturedir(style):
  return os.path.join(ROOT_DIR, style)

SKY_STYLES = ('outdoor_natural')

SkyBox = collections.namedtuple(
    'SkyBox', ('file', 'gridsize', 'gridlayout'))


def get_sky_texture_info(style):
  if style not in SKY_STYLES:
    raise ValueError('`style` should be one of {}: got {!r}'.format(
        SKY_STYLES, style))
  return SkyBox(file='OutdoorSkybox2048.png',
                gridsize='3 4',
                gridlayout='.U..LFRB.D..')


GROUND_STYLES = ('outdoor_natural')

GroundTexture = collections.namedtuple(
    'GroundTexture', ('file', 'type'))


def get_ground_texture_info(style):
  if style not in GROUND_STYLES:
    raise ValueError('`style` should be one of {}: got {!r}'.format(
        GROUND_STYLES, style))
  return GroundTexture(
      file='OutdoorGrassFloorD.png',
      type='2d')




