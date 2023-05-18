from dm_control import mujoco
from dm_control import suite
from dm_control import viewer
from dm_control.rl import control
from dm_control.suite import common
import numpy as np
import os
import math
from scipy.signal import savgol_filter
import re

def get_model_and_assets(model_name):
    with open(model_name, mode='rb') as f:
        return f.read(), common.ASSETS


class Task(suite.base.Task):
    def __init__(self, mocap_qpos, random=None):
        super().__init__(random=random)
        self.mocap_qpos = mocap_qpos
        self.init = False

    def initialize_episode(self, physics):
        super().initialize_episode(physics)
        physics.forward()
        physics.data.qpos[:] = self.mocap_qpos[0]
        physics.forward()

    def get_observation(self, physics):
        return {}

    def get_reward(self, physics):
        return 0

    def after_step(self, physics):
        target_pos = physics.model.key_mpos[0]
        target_pos = target_pos.reshape((-1, 3))
        physics.data.mocap_pos[:] = target_pos
        physics.data.qpos[:] = self.mocap_qpos[0]

        physics.data.qacc[:] = 0
        physics.data.qvel[:] = 0


    def before_step(self, action, physics):
        physics.forward()


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_dir = os.path.join(dir_path, "./task.xml")
    physics = mujoco.Physics.from_xml_path(model_dir)

    qpos_dir = os.path.join(dir_path, "./result_qpos.npy")
    mocap_qpos = np.load(qpos_dir)

    env = control.Environment(physics, Task(mocap_qpos=[mocap_qpos]))

    # env = control.Environment(physics, Task(mocap_qpos=[[-0.04934, -0.00198, 1.25512, 0.99691, 0.0161, -0.04859, -0.05959,
    #                                                      0.00806733, -0.0921235, -0.00874875, -0.0488637, -0.0894064,
    #                                                      0.0219203, -0.432015, -0.374747, -0.0965783, 0.0391758, -0.0760225,
    #                                                      0.132042, -0.256406, -0.350065, -0.494566, 0.714612, -0.744779,
    #                                                      -0.922223, 0.818614, -0.698744, -1.00263]]))

    viewer.launch(env)
