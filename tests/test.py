

from dm_control import suite
from dm_control.glviz import viz
from dm_control.glviz import world
import numpy as np

from dm_control.mujoco.wrapper.mjbindings import wrappers
from dm_control.mujoco.wrapper import util

import matplotlib.pyplot as plt
import enginewrapper

# Load one task:
env = suite.load( domain_name = "cartpole", task_name = "swingup" )
visualizer = viz.Visualizer( env.physics )
env_world = world.World( env.physics )

# Iterate over a task set:
for domain_name, task_name in suite.BENCHMARKING:
  print( 'domain: ', domain_name, ' - taskname: ', task_name )

# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
time_step = env.reset()

_paused = False

while not time_step.last():
  action = np.random.uniform(action_spec.minimum,
                             action_spec.maximum,
                             size=action_spec.shape)
  # action = np.zeros( action_spec.shape )
  if not _paused :
    time_step = env.step(action)

  # draw axes
  enginewrapper.drawLine( np.array( [0,0,0] ), np.array( [5, 0, 0] ), np.array( [1,0,0] ) )
  enginewrapper.drawLine( np.array( [0,0,0] ), np.array( [0, 5, 0] ), np.array( [0,1,0] ) )
  enginewrapper.drawLine( np.array( [0,0,0] ), np.array( [0, 0, 5] ), np.array( [0,0,1] ) )

  visualizer.update()

  if visualizer.check_single_press( viz.KEY_SPACE ):
    _paused = not _paused
#   if visualizer.check_single_press( viz.KEY_C ):
#     # env_world.create_cube()
#     _cam = enginewrapper.getCurrentCamera()
#     # _cam.setPosition( _cam.getPosition() + np.array( [0.0, 0.0, 0.1] ) )
#     # print( 'cam-pos> ', _cam.getPosition() )
#     _cam.setTargetDir( _cam.getTargetDir() + np.array( [0.0, 0.0, 0.1] ) )
#     print( 'cam-pos> ', _cam.getTargetDir() )
  if visualizer.check_single_press( viz.KEY_ESCAPE ):
    break

  # _cp = visualizer.get_cursor_position()
  # print('cursor position: ', _cp)

#   pixels = env.physics.render(480, 480, camera_id = 0)
#   plt.imshow(pixels)
#   plt.pause(0.01)
#   plt.draw()
