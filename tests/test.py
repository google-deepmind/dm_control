
from dm_control import suite
from dm_control.glviz import viz
import numpy as np

# Load one task:
env = suite.load( domain_name = "humanoid", task_name = "walk" )
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

  visualizer.update()

  if visualizer.check_single_press( viz.KEY_SPACE ):
    _paused = not _paused
  if visualizer.check_single_press( viz.KEY_ESCAPE ):
    break