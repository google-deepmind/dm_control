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

"""Import core names of MuJoCo ctypes bindings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

from dm_control.mujoco.wrapper.mjbindings import constants
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import sizes
from dm_control.mujoco.wrapper.mjbindings import types
from dm_control.mujoco.wrapper.mjbindings import wrappers

# pylint: disable=g-import-not-at-top
try:
  from dm_control.mujoco.wrapper.mjbindings import functions
  from dm_control.mujoco.wrapper.mjbindings.functions import mjlib
except (IOError, OSError):
  logging.warn('mjbindings failed to import mjlib and other functions. '
               'libmujoco.so may not be accessible.')
