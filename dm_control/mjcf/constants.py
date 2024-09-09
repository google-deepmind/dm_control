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

# Path namespaces.
MESHDIR_NAMESPACE = 'mesh'
TEXTUREDIR_NAMESPACE = 'texture'
ASSETDIR_NAMESPACE = 'asset'

MJDATA_TRIGGERS_DIRTY = [
    'qpos', 'qvel', 'act', 'ctrl', 'qfrc_applied', 'xfrc_applied']
MJMODEL_DOESNT_TRIGGER_DIRTY = [
    'rgba', 'matid', 'emission', 'specular', 'shininess', 'reflectance',
    'needstage',
]

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

MAX_VFS_FILENAME_LENGTH = 998

# The prefix used in the schema to denote reference_namespace that are defined
# via another attribute.
INDIRECT_REFERENCE_NAMESPACE_PREFIX = 'attrib:'

INDIRECT_REFERENCE_ATTRIB = {
    'xbody': 'body',
}

# 17 decimal digits is sufficient to represent a double float without loss
# of precision.
# https://en.wikipedia.org/wiki/IEEE_754#Character_representation
XML_DEFAULT_PRECISION = 17
