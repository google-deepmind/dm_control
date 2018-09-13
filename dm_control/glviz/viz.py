
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_control.mujoco import wrapper
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

import numpy as np

class Visualizer(object):

    def __init__(self, physics):
        super(Visualizer, self).__init__()
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

    def scene(self):
        return self._scene

    def render(self):
        # abstract visualization stage - retrieve the viz data
        mjlib.mjv_updateScene( self._physics.model.ptr, self._physics.data.ptr,
                               self._scene_option.ptr, self._perturb.ptr,
                               self._render_camera.ptr, enums.mjtCatBit.mjCAT_ALL,
                               self._scene.ptr )