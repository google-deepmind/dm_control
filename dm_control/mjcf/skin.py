# Copyright 2020 The dm_control Authors.
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

"""Utilities for parsing and writing MuJoCo skin files.

The file format is described at http://mujoco.org/book/XMLreference.html#skin.
"""

import collections
import io
import struct
import numpy as np

MAX_BODY_NAME_LENGTH = 40

Skin = collections.namedtuple(
    'Skin', ('vertices', 'texcoords', 'faces', 'bones'))

Bone = collections.namedtuple(
    'Bone', ('body', 'bindpos', 'bindquat', 'vertex_ids', 'vertex_weights'))


def parse(contents, body_getter):
  """Parses the contents of a MuJoCo skin file.

  Args:
    contents: a bytes-like object containing the contents of a skin file.
    body_getter: a callable that takes a string and returns the `mjcf.Element`
      instance of a MuJoCo body of the specified name.

  Returns:
    A `Skin` named tuple.
  """
  f = io.BytesIO(contents)
  nvertex, ntexcoord, nface, nbone = struct.unpack('<iiii', f.read(4*4))
  vertices = np.frombuffer(f.read(4*(3*nvertex)), dtype='<f4').reshape(-1, 3)
  texcoords = np.frombuffer(f.read(4*(2*ntexcoord)), dtype='<f4').reshape(-1, 2)
  faces = np.frombuffer(f.read(4*(3*nface)), dtype='<i4').reshape(-1, 3)
  bones = []
  for _ in range(nbone):
    body_name = f.read(MAX_BODY_NAME_LENGTH).decode().split('\0')[0]
    body = lambda body_name=body_name: body_getter(body_name)
    bindpos = np.asarray(struct.unpack('<fff', f.read(4*3)), dtype=float)
    bindquat = np.asarray(struct.unpack('<ffff', f.read(4*4)), dtype=float)
    vertex_count = struct.unpack('<i', f.read(4))[0]
    vertex_ids = np.frombuffer(f.read(4*vertex_count), dtype='<i4')
    vertex_weights = np.frombuffer(f.read(4*vertex_count), dtype='<f4')
    bones.append(Bone(body=body, bindpos=bindpos, bindquat=bindquat,
                      vertex_ids=vertex_ids, vertex_weights=vertex_weights))

  return Skin(vertices=vertices, texcoords=texcoords, faces=faces, bones=bones)


def serialize(skin):
  """Serializes a `Skin` named tuple into the contents of a MuJoCo skin file.

  Args:
    skin: a `Skin` named tuple.

  Returns:
    A `bytes` object representing the content of a MuJoCo skin file.
  """
  out = io.BytesIO()
  nvertex = len(skin.vertices)
  ntexcoord = len(skin.texcoords)
  nface = len(skin.faces)
  nbone = len(skin.bones)
  out.write(struct.pack('<iiii', nvertex, ntexcoord, nface, nbone))
  out.write(skin.vertices.astype('<f4').tobytes())
  out.write(skin.texcoords.astype('<f4').tobytes())
  out.write(skin.faces.astype('<i4').tobytes())
  for bone in skin.bones:
    body_bytes = bone.body().full_identifier.encode('utf-8')
    if len(body_bytes) > MAX_BODY_NAME_LENGTH:
      raise ValueError(
          'body name is longer than  permitted by the skin file format '
          '(maximum 40): {:r}'.format(body_bytes))
    out.write(body_bytes)
    out.write(b'\0'*(MAX_BODY_NAME_LENGTH - len(body_bytes)))
    out.write(bone.bindpos.astype('<f4').tobytes())
    out.write(bone.bindquat.astype('<f4').tobytes())
    assert len(bone.vertex_ids) == len(bone.vertex_weights)
    out.write(struct.pack('<i', len(bone.vertex_ids)))
    out.write(bone.vertex_ids.astype('<i4').tobytes())
    out.write(bone.vertex_weights.astype('<f4').tobytes())
  return out.getvalue()
