
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_control.mujoco import wrapper
from dm_control.mujoco.wrapper import util
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

import numpy as np

# rendering engine bindings
import enginewrapper

INDEX_TYPE = 0
INDEX_DATA_ID = 1
INDEX_OBJ_TYPE = 2
INDEX_OBJ_ID = 3
INDEX_CATEGORY = 4
INDEX_TEX_ID = 5
INDEX_TEX_UNIFORM = 6

INDEX_TEX_REPEAT = 7
INDEX_SIZE = 8
INDEX_POS = 9
INDEX_MAT = 10
INDEX_RGBA = 11
INDEX_EMISSION = 12
INDEX_SPECULAR = 13
INDEX_SHININESS = 14
INDEX_REFLECTANCE = 15

OBJ_TYPE_GEOMETRY = 5

GEOMETRY_TYPE_PLANE = 0
GEOMETRY_TYPE_HFIELD = 1
GEOMETRY_TYPE_SPHERE = 2
GEOMETRY_TYPE_CAPSULE = 3
GEOMETRY_TYPE_ELLIPSOID = 4
GEOMETRY_TYPE_CYLINDER = 5
GEOMETRY_TYPE_BOX = 6
GEOMETRY_TYPE_MESH = 7

class GeometryInfo(object):

    def __init__(self, gid, gtype, pos, rot, params ):
        self.id = gid
        self.type = gtype
        self.pos = pos
        self.rot = rot
        self.params = params

class Visualizer(object):

    def __init__(self, physics):
        super(Visualizer, self).__init__()
        # initialize rendering engine
        enginewrapper.init()
        # save a reference to the physics
        self._physics = physics
        # create the scene for the abstract visualization stage
        self._scene = wrapper.MjvScene()
        self._scene_option = wrapper.MjvOption()
        # a perturbation object, just for completion
        self._perturb = wrapper.MjvPerturb()
        self._perturb.active = 0
        self._perturb.select = 0

        # create a mjvcamera, as it seems is needed for this stage
        self._render_camera = wrapper.MjvCamera()
        self._render_camera.fixedcamid = -1
        self._render_camera.type_ = enums.mjtCamera.mjCAMERA_FREE

        # a list to store the geometries from the abstract visualization stage
        self._geometries = {}
        # the meshes wrapped by the bindings
        self._meshes = {}

    def scene(self):
        return self._scene

    def render(self):
        # abstract visualization stage - retrieve the viz data
        mjlib.mjv_updateScene(self._physics.model.ptr, self._physics.data.ptr,
                              self._scene_option.ptr, self._perturb.ptr,
                              self._render_camera.ptr, enums.mjtCatBit.mjCAT_ALL,
                              self._scene.ptr)
        self._collect_geometries()
        self._update_geometries_meshes()
        # request rendering to the engine backend
        enginewrapper.update()

    def _collect_geometries(self):
        # collect geometries structs
        _geoms = util.buf_to_npy(self._scene._ptr.contents.geoms,
                                 (self._scene.ngeom, ))
        # parse this geometries into our format
        self._parse_geometries(_geoms)

    def _parse_geometries(self, geo_structs):
        for _geo_struct in geo_structs :
            _id = _geo_struct[INDEX_OBJ_ID]
            _type = _geo_struct[INDEX_TYPE]
            _obj_type = _geo_struct[INDEX_OBJ_TYPE]
            _pos = _geo_struct[INDEX_POS]
            _rot = _geo_struct[INDEX_MAT]
            _size = _geo_struct[INDEX_SIZE]
            _color = _geo_struct[INDEX_RGBA]
            # check if the object is a geometry object
            if _obj_type != OBJ_TYPE_GEOMETRY:
                continue

            if _id in self._geometries:
                self._geometries[_id].pos = _pos
                self._geometries[_id].rot = _rot
                self._geometries[_id].type = _type
                self._geometries[_id].params = {'size' : _size,
                                                'color' : _color}
            else:
                self._geometries[_id] = GeometryInfo(_id, _type,
                                                     _pos, _rot,
                                                     {'size': _size, 'color': _color})
                self._meshes[_id] = self._create_geometry_mesh( self._geometries[_id] )
                if self._meshes[_id] is not None:
                    self._meshes[_id].setColor( _color[0], _color[1], _color[2] )

    def _create_geometry_mesh(self, geometry):
        _mesh = None
        if geometry.type == GEOMETRY_TYPE_PLANE:
            # _mesh = enginewrapper.createPlane(geometry.params['size'][0],
            #                                   geometry.params['size'][1])
            _mesh = None
        elif geometry.type == GEOMETRY_TYPE_SPHERE:
            _mesh = enginewrapper.createSphere(geometry.params['size'][0])
        elif geometry.type == GEOMETRY_TYPE_CAPSULE:
            _mesh = enginewrapper.createCapsule(geometry.params['size'][1],
                                                geometry.params['size'][2])
        elif geometry.type == GEOMETRY_TYPE_BOX:
            _mesh = enginewrapper.createBox(geometry.params['size'][0],
                                            geometry.params['size'][1],
                                            geometry.params['size'][2])
        return _mesh

    def _update_geometries_meshes(self):
        for _id in self._geometries :
            if _id not in self._meshes :
                continue
            if self._meshes[_id] is None :
                continue
            self._meshes[_id].setPosition(self._geometries[_id].pos[0],
                                          self._geometries[_id].pos[1],
                                          self._geometries[_id].pos[2])
            self._meshes[_id].setRotation(self._geometries[_id].rot)