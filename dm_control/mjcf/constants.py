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

"""Magic constants used within `dm_control.mjcf`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

PREFIX_SEPARATOR = '/'
PREFIX_SEPARATOR_ESCAPE = '\\'

# Used to disambiguate namespaces between attachment frames.
NAMESPACE_SEPARATOR = '@'

# Magic attribute names
BASEPATH = 'basepath'
CHILDCLASS = 'childclass'
CLASS = 'class'
DEFAULT = 'default'
DCLASS = 'dclass'

# Magic tags
ACTUATOR = 'actuator'
BODY = 'body'
DEFAULT = 'default'
MESH = 'mesh'
SITE = 'site'
SKIN = 'skin'
TENDON = 'tendon'
WORLDBODY = 'worldbody'

MJDATA_TRIGGERS_DIRTY = [
    'qpos', 'qvel', 'act', 'ctrl', 'qfrc_applied', 'xfrc_applied']
MJMODEL_DOESNT_TRIGGER_DIRTY = [
    'rgba', 'matid', 'emission', 'specular', 'shininess', 'reflectance']

# When writing into `model.{body,geom,site}_{pos,quat}` we must ensure that the
# corresponding rows in `model.{body,geom,site}_sameframe` are set to zero,
# otherwise MuJoCo will use the body or inertial frame instead of our modified
# pos/quat values. We must do the same for `body_{ipos,iquat}` and
# `body_simple`.
MJMODEL_DISABLE_ON_WRITE = {
    # Field name in MjModel: (attribute names of Binding instance to be zeroed)
    'body_pos': ('sameframe',),
    'body_quat': ('sameframe',),
    'geom_pos': ('sameframe',),
    'geom_quat': ('sameframe',),
    'site_pos': ('sameframe',),
    'site_quat': ('sameframe',),
    'body_ipos': ('simple', 'sameframe'),
    'body_iquat': ('simple', 'sameframe'),
}

# This is the actual upper limit on VFS filename length, despite what it says
# in the header file (100) or the error message (99).
MAX_VFS_FILENAME_LENGTH = 98

# The prefix used in the schema to denote reference_namespace that are defined
# via another attribute.
INDIRECT_REFERENCE_NAMESPACE_PREFIX = 'attrib:'

INDIRECT_REFERENCE_ATTRIB = {
    'xbody': 'body',
}
