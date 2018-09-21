

from dm_control import suite
from dm_control.glviz import viz
import numpy as np

from dm_control.mujoco.wrapper.mjbindings import wrappers
from dm_control.mujoco.wrapper import util

# import matplotlib.pyplot as plt
import enginewrapper

# Load one task:
env = suite.load( domain_name = "walker", task_name = "walk" )
# env = suite.load( domain_name = "cartpole", task_name = "balance" )
# env = suite.load( domain_name = "acrobot", task_name = "swingup" )
# env = suite.load( domain_name = "ball_in_cup", task_name = "catch" )
# env = suite.load( domain_name = "cheetah", task_name = "run" )
# env = suite.load( domain_name = "finger", task_name = "spin" )
# env = suite.load( domain_name = "fish", task_name = "swim" )# needs tweaking : ellipsoid support
# env = suite.load( domain_name = "hopper", task_name = "stand" )
# env = suite.load( domain_name = "manipulator", task_name = "bring_ball" )# need tweaking : cylinder support and different lighting position
# env = suite.load( domain_name = "pendulum", task_name = "swingup" )
# env = suite.load( domain_name = "point_mass", task_name = "easy" )
# env = suite.load( domain_name = "reacher", task_name = "easy" )
# env = suite.load( domain_name = "swimmer", task_name = "swimmer6" )
# env = suite.load( domain_name = "primitives", task_name = "test" )

visualizer = viz.Visualizer( env.physics )

# # Show available tasks:
# for domain_name, task_name in suite.BENCHMARKING:
#   print( 'domain: ', domain_name, ' - taskname: ', task_name )

# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
time_step = env.reset()

_paused = False

_mesh = visualizer.getMeshByName( 'cart' )
# _mesh = visualizer.getMeshByName( 'torso' )
_camera1 = enginewrapper.createFollowCamera( 'follow',
                                             np.array( [2.0, 4.0, 2.0] ),
                                             np.array( [0.0, 0.0, 0.0] ) )
_camera2 = enginewrapper.createFixedCamera( 'fixed',
                                            np.array( [2.0, 4.0, 2.0] ),
                                            np.array( [0.0, 0.0, 0.0] ) )

_camera1.setFollowReference( _mesh )
enginewrapper.changeToCameraByName( 'follow' )

# visualizer.testMeshesNames()

while True:
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
  if visualizer.check_single_press( viz.KEY_M ):
    enginewrapper.changeToCameraByName( 'main' )
  if visualizer.check_single_press( viz.KEY_C ):
    enginewrapper.changeToCameraByName( 'fixed' )
  if visualizer.check_single_press( viz.KEY_F ):
    enginewrapper.changeToCameraByName( 'follow' )
  if visualizer.check_single_press( viz.KEY_Q ):
    print( 'qpos: ', env.physics.data.qpos[:] )
  if visualizer.check_single_press( viz.KEY_P ):
    env.physics.data.qpos[:] = np.asarray( [0., 0., 2., 1., 0., 0., 0.] )
  if visualizer.check_single_press( viz.KEY_ESCAPE ):
    break

#   pixels = env.physics.render(480, 480, camera_id = 0)
#   plt.imshow(pixels)
#   plt.pause(0.01)
#   plt.draw()
