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

"""Module for observables in the Composer library."""

from dm_control.composer.observation.observable.base import Generic
from dm_control.composer.observation.observable.base import MujocoCamera
from dm_control.composer.observation.observable.base import MujocoFeature
from dm_control.composer.observation.observable.base import Observable

from dm_control.composer.observation.observable.mjcf import MJCFCamera
from dm_control.composer.observation.observable.mjcf import MJCFFeature
