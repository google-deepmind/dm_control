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

"""Module containing abstract base classes for Composer environments."""

from dm_control.composer.arena import Arena
from dm_control.composer.constants import *  # pylint: disable=wildcard-import
from dm_control.composer.define import cached_property
from dm_control.composer.define import observable
from dm_control.composer.entity import Entity
from dm_control.composer.entity import FreePropObservableMixin
from dm_control.composer.entity import ModelWrapperEntity
from dm_control.composer.entity import Observables
from dm_control.composer.environment import Environment
from dm_control.composer.environment import HOOK_NAMES
from dm_control.composer.initializer import Initializer
from dm_control.composer.robot import Robot
from dm_control.composer.task import NullTask
from dm_control.composer.task import Task
