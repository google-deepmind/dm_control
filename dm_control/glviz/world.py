
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_control.mujoco import wrapper
from dm_control.mujoco.wrapper import util
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

import numpy as np


class World(object):

    def __init__(self, physics):
        super(World, self).__init__()
        # save a reference to the physics for later usage
        self._physics = physics
        # collect cube bodies into an object pool for later usage
        self._objects = {'BOX': [], 'CAPSULE': [], 'SPHERE': []}
        
    def spawn_object(self, type_id, position, orientation):
        pass

    def spawn_section(self):
        pass



class World1D(World):

    def __init__(self, physics):
        super(World1D, self).__init__(physics)

    def spawn_section(self):
        # should spawn a 1d section
        pass

class World2D(World):

    def __init__(self, physics):
        super(World1D, self).__init__(physics)

    def spawn_section(self):
        # should spawn a 1d section
        pass