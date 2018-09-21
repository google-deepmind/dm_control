
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

# keycodes from glfw
KEY_SPACE = 32
KEY_A = 65
KEY_B = 66
KEY_C = 67
KEY_D = 68
KEY_E = 69
KEY_F = 70
KEY_G = 71
KEY_H = 72
KEY_I = 73
KEY_J = 74
KEY_K = 75
KEY_L = 76
KEY_M = 77
KEY_N = 78
KEY_O = 79
KEY_P = 80
KEY_Q = 81
KEY_R = 82
KEY_S = 83
KEY_T = 84
KEY_U = 85
KEY_V = 86
KEY_W = 87
KEY_X = 88
KEY_Y = 89
KEY_Z = 90
KEY_ESCAPE      = 256
KEY_ENTER       = 257
KEY_TAB         = 258
KEY_BACKSPACE   = 259
KEY_INSERT      = 260
KEY_DELETE      = 261
KEY_RIGHT       = 262
KEY_LEFT        = 263
KEY_DOWN        = 264
KEY_UP          = 265
# mouse button codes from glfw
MOUSE_BUTTON_1 = 0
MOUSE_BUTTON_2 = 1
MOUSE_BUTTON_3 = 2
MOUSE_BUTTON_4 = 3
MOUSE_BUTTON_5 = 4
MOUSE_BUTTON_6 = 5
MOUSE_BUTTON_7 = 6
MOUSE_BUTTON_8 = 7

RELEASED = 0
PRESSED  = 1
REPEATED = 2

CAMERA_TYPE_FIXED = 0
CAMERA_TYPE_FPS = 1
CAMERA_TYPE_ORBIT = 2
CAMERA_TYPE_FOLLOW = 3

class GeometryInfo(object):

    def __init__(self, gid, name, gtype, pos, rot, params ):
        self.id = gid
        self.name = name
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

        # keys
        self._single_keys = [False for i in range(1024)]

        # make a first update to initialize objects
        mjlib.mjv_updateScene(self._physics.model.ptr, self._physics.data.ptr,
                              self._scene_option.ptr, self._perturb.ptr,
                              self._render_camera.ptr, enums.mjtCatBit.mjCAT_ALL,
                              self._scene.ptr)
        self._collect_geometries()
        self._update_geometries_meshes()

    def scene(self):
        return self._scene

    def meshes(self):
        return self._meshes

    def geometries(self):
        return self._geometries

    def testMeshesNames(self):
        for _id in self._geometries :
            print( 'name: ', self._physics.model.id2name( _id, enums.mjtObj.mjOBJ_GEOM ) )

    def getMeshByName(self, name):
        try:
            _id = self._physics.model.name2id( name, enums.mjtObj.mjOBJ_GEOM )
        except wrapper.core.Error :
            print( 'WARNING> geometry with name: ', name, ' does not exist' )
            return enginewrapper.Mesh()
            
        if _id not in self._meshes :
            print( 'WARNING> mesh with name: ', name, ' does not exist' )
            return enginewrapper.Mesh()

        return self._meshes[_id]

    def update(self):
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

            _name = self._physics.model.id2name( _id, enums.mjtObj.mjOBJ_GEOM )

            if _type == GEOMETRY_TYPE_PLANE :
                _color = [.2, .3, .4, 1.]

            if _id in self._geometries:
                self._geometries[_id].pos = _pos
                self._geometries[_id].rot = _rot
                self._geometries[_id].type = _type
                self._geometries[_id].name = _name
                self._geometries[_id].params = {'size' : _size,
                                                'color' : _color}
            else:
                self._geometries[_id] = GeometryInfo(_id, _name, _type,
                                                     _pos, _rot,
                                                     {'size': _size, 'color': _color})
                self._meshes[_id] = self._create_geometry_mesh( self._geometries[_id] )
                if self._meshes[_id] is not None:
                    self._meshes[_id].setColor( _color[0], _color[1], _color[2] )

    def _create_geometry_mesh(self, geometry):
        _mesh = None
        if geometry.type == GEOMETRY_TYPE_PLANE:
            _mesh = enginewrapper.createPlane(2*geometry.params['size'][1],
                                              2*geometry.params['size'][0],
                                              2 * geometry.params['size'][1] / 10.0,
                                              2 * geometry.params['size'][0] / 10.0)
            # testing texture mapping
            if geometry.name == 'floor' or geometry.name == 'ground' :
                _mesh.setBuiltInTexture( 'chessboard' );
            # print( 'plane params: ', geometry.params )
        elif geometry.type == GEOMETRY_TYPE_SPHERE:
            _mesh = enginewrapper.createSphere(geometry.params['size'][0])
            # print( 'sphere params: ', geometry.params )
        elif geometry.type == GEOMETRY_TYPE_CAPSULE:
            _mesh = enginewrapper.createCapsule(geometry.params['size'][1],
                                                2*geometry.params['size'][2])
            # print( 'capsule params: ', geometry.params )
        elif geometry.type == GEOMETRY_TYPE_BOX:
            _mesh = enginewrapper.createBox(2*geometry.params['size'][0],
                                            2*geometry.params['size'][1],
                                            2*geometry.params['size'][2])
            # print( 'box params: ', geometry.params )
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

    def is_key_down(self, key):
        return enginewrapper.isKeyDown(key)

    def is_mouse_down(self, button):
        return enginewrapper.isMouseDown(button)

    def get_cursor_position(self):
        return enginewrapper.getCursorPosition()

    def check_single_press(self, key):
        if not enginewrapper.isKeyDown(key) :
            self._single_keys[key] = False
            return False
        _res = self._single_keys[key] ^ enginewrapper.isKeyDown(key)
        self._single_keys[key] = enginewrapper.isKeyDown(key)
        return _res