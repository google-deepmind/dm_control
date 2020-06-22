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

"""Global constants used in manipulation tasks."""

CONTROL_TIMESTEP = 0.04  # Interval between agent actions, in seconds.

# Predefined RGBA values
RED = (1., 0., 0., 0.3)
GREEN = (0., 1., 0., 0.3)
BLUE = (0., 0., 1., 0.3)
CYAN = (0., 1., 1., 0.3)
MAGENTA = (1., 0., 1., 0.3)
YELLOW = (1., 1., 0., 0.3)

TASK_SITE_GROUP = 3  # Invisible group for task-related sites.
