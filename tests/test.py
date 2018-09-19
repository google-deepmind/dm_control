
from dm_control import suite
from dm_control.glviz import viz
from dm_control.glviz import world
import numpy as np

# Load one task:
env = suite.load( domain_name = "humanoid", task_name = "walk" )
# env = suite.load( domain_name = "cartpole", task_name = "balance" )
# env = suite.load( domain_name = "primitives", task_name = "test" )

visualizer = viz.Visualizer( env.physics )
env_world = world.World( env.physics )

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