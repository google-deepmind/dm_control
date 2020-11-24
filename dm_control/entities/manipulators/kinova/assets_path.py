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

"""Helper module that specifies the path to Kinova assets."""

import os
import six

# pylint: disable=g-import-not-at-top
if six.PY2:
  import imp
  _DM_CONTROL_ROOT = imp.find_module('dm_control')[1]
else:
  import importlib
  _DM_CONTROL_ROOT = os.path.dirname(
      importlib.util.find_spec('dm_control').origin)
# pylint: enable=g-import-not-at-top

KINOVA_ROOT = os.path.join(_DM_CONTROL_ROOT, 'third_party/kinova')
