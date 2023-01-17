# Copyright 2023 The dm_control Authors.
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

"""Make skin for the dog model."""

import struct

import numpy as np
from scipy import spatial

from dm_control import mjcf
from dm_control.mujoco.wrapper.mjbindings import enums


def create(model, skin_msh):
  """Create and add skin in the dog model.

        Args:
          model: model in which we want to add the skin.
          skin_msh: a binary mesh format of the skin.
  """
  print('Making Skin.')
  # Add skin mesh:
  skinmesh = model.worldbody.add(
    'geom',
    name='skinmesh',
    mesh='skin_msh',
    type='mesh',
    contype=0,
    conaffinity=0,
    rgba=[1, .5, .5, .5],
    group=1,
    euler=(0, 0, 90))
  physics = mjcf.Physics.from_mjcf_model(model)

  # Get skinmesh vertices in global coordinates
  vertadr = physics.named.model.mesh_vertadr['skin_msh']
  vertnum = physics.named.model.mesh_vertnum['skin_msh']
  skin_vertices = physics.model.mesh_vert[vertadr:vertadr + vertnum, :]
  skin_vertices = skin_vertices.dot(
    physics.named.data.geom_xmat['skinmesh'].reshape(3, 3).T)
  skin_vertices += physics.named.data.geom_xpos['skinmesh']
  skin_normals = physics.model.mesh_normal[vertadr:vertadr + vertnum, :]
  skin_normals = skin_normals.dot(
    physics.named.data.geom_xmat['skinmesh'].reshape(3, 3).T)
  skin_normals += physics.named.data.geom_xpos['skinmesh']

  # Get skinmesh faces
  faceadr = physics.named.model.mesh_faceadr['skin_msh']
  facenum = physics.named.model.mesh_facenum['skin_msh']
  skin_faces = physics.model.mesh_face[faceadr:faceadr + facenum, :]

  # Make skin
  skin = model.asset.add(
    'skin',
    name='skin',
    vertex=skin_vertices.ravel(),
    face=skin_faces.ravel())

  # Functions for capsule vertices
  numslices = 10
  numstacks = 10
  numquads = 8

  def hemisphere(radius):
    positions = []
    for az in np.linspace(0, 2 * np.pi, numslices, False):
      for el in np.linspace(0, np.pi, numstacks, False):
        pos = np.asarray(
          [np.cos(el) * np.cos(az),
           np.cos(el) * np.sin(az),
           np.sin(el)])
        positions.append(pos)
    return radius * np.asarray(positions)

  def cylinder(radius, height):
    positions = []
    for az in np.linspace(0, 2 * np.pi, numslices, False):
      for el in np.linspace(-1, 1, numstacks):
        pos = np.asarray(
          [radius * np.cos(az), radius * np.sin(az), height * el])
        positions.append(pos)
    return np.asarray(positions)

  def capsule(radius, height):
    hp = hemisphere(radius)
    cy = cylinder(radius, height)
    offset = np.array((0, 0, height))
    return np.unique(np.vstack((cy, hp + offset, -hp - offset)), axis=0)

  def ellipsoid(size):
    hp = hemisphere(1)
    sphere = np.unique(np.vstack((hp, -hp)), axis=0)
    return sphere * size

  def box(sx, sy, sz):
    positions = []
    for x in np.linspace(-sx, sx, numquads + 1):
      for y in np.linspace(-sy, sy, numquads + 1):
        for z in np.linspace(-sz, sz, numquads + 1):
          if abs(x) == sx or abs(y) == sy or abs(z) == sz:
            pos = np.asarray([x, y, z])
            positions.append(pos)
    return np.unique(np.asarray(positions), axis=0)

  # Find smallest distance between
  # each skin vertex and vertices of all meshes in body i
  distance = np.zeros((skin_vertices.shape[0], physics.model.nbody))
  for i in range(1, physics.model.nbody):
    geom_id = np.argwhere(physics.model.geom_bodyid == i).ravel()
    mesh_id = physics.model.geom_dataid[geom_id]
    body_verts = []
    for k, gid in enumerate(geom_id):
      skip = False
      if physics.model.geom_type[gid] == enums.mjtGeom.mjGEOM_MESH:
        vertadr = physics.model.mesh_vertadr[mesh_id[k]]
        vertnum = physics.model.mesh_vertnum[mesh_id[k]]
        vertices = physics.model.mesh_vert[vertadr:vertadr + vertnum, :]
      elif physics.model.geom_type[gid] == enums.mjtGeom.mjGEOM_CAPSULE:
        radius = physics.model.geom_size[gid, 0]
        height = physics.model.geom_size[gid, 1]
        vertices = capsule(radius, height)
      elif physics.model.geom_type[gid] == enums.mjtGeom.mjGEOM_ELLIPSOID:
        vertices = ellipsoid(physics.model.geom_size[gid])
      elif physics.model.geom_type[gid] == enums.mjtGeom.mjGEOM_BOX:
        vertices = box(*physics.model.geom_size[gid])
      else:
        skip = True
      if not skip:
        vertices = vertices.dot(physics.data.geom_xmat[gid].reshape(3, 3).T)
        vertices += physics.data.geom_xpos[gid]
        body_verts.append(vertices)

    body_verts = np.vstack((body_verts))
    # hull = spatial.ConvexHull(body_verts)
    tree = spatial.cKDTree(body_verts)
    distance[:, i], _ = tree.query(skin_vertices)

    # non-KDTree implementation of the above 2 lines:
    # distance[:, i] = np.amin(
    #     spatial.distance.cdist(skin_vertices, body_verts, 'euclidean'),
    #     axis=1)

  # Calculate bone weights from distances
  sigma = .015
  weights = np.exp(-distance[:, 1:] / sigma)
  threshold = .01
  weights /= np.atleast_2d(np.sum(weights, axis=1)).T
  weights[weights < threshold] = 0
  weights /= np.atleast_2d(np.sum(weights, axis=1)).T

  for i in range(1, physics.model.nbody):
    vertweight = weights[weights[:, i - 1] >= threshold, i - 1]
    vertid = np.argwhere(weights[:, i - 1] >= threshold).ravel()
    if vertid.any():
      skin.add(
        'bone',
        body=physics.model.id2name(i, 'body'),
        bindquat=[1, 0, 0, 0],
        bindpos=physics.data.xpos[i, :],
        vertid=vertid,
        vertweight=vertweight)

  # Remove skinmesh
  skinmesh.remove()

  # Convert skin into *.skn file according to
  # https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-skin
  f = open('dog_skin.skn', 'w+b')
  nvert = skin.vertex.size // 3
  f.write(
    struct.pack('4i', nvert, nvert, skin.face.size // 3,
                physics.model.nbody - 1))
  f.write(struct.pack(str(skin.vertex.size) + 'f', *skin.vertex))
  assert physics.model.mesh_texcoord.shape[0] == physics.bind(
    skin_msh).vertnum
  f.write(
    struct.pack(
      str(2 * nvert) + 'f', *physics.model.mesh_texcoord.flatten()))
  f.write(struct.pack(str(skin.face.size) + 'i', *skin.face))
  for bone in skin.bone:
    name_length = len(bone.body)
    assert name_length <= 40
    f.write(
      struct.pack(str(name_length) + 'c', *[s.encode() for s in bone.body]))
    f.write((40 - name_length) * b'\x00')
    f.write(struct.pack('3f', *bone.bindpos))
    f.write(struct.pack('4f', *bone.bindquat))
    f.write(struct.pack('i', bone.vertid.size))
    f.write(struct.pack(str(bone.vertid.size) + 'i', *bone.vertid))
    f.write(struct.pack(str(bone.vertid.size) + 'f', *bone.vertweight))
  f.close()

  # Remove XML-based skin, add binary skin.
  skin.remove()
