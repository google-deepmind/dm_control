from dm_control import suite
from dm_control.glviz import viz
import numpy as np

from dm_control.mujoco.wrapper.mjbindings import wrappers
from dm_control.mujoco.wrapper import util

# Load one task:
env = suite.load( domain_name = "humanoid", task_name = "walk" )
visualizer = viz.Visualizer( env.physics )

# Iterate over a task set:
for domain_name, task_name in suite.BENCHMARKING:
  print( 'domain: ', domain_name, ' - taskname: ', task_name )

# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
time_step = env.reset()

while not time_step.last():
  action = np.random.uniform(action_spec.minimum,
                             action_spec.maximum,
                             size=action_spec.shape)
  time_step = env.step(action)

  visualizer.render()
  _scene = visualizer.scene()

  # _geoms = util.buf_to_npy( _scene._ptr.contents.geoms, 
  #                           ( _scene.ngeom, ) )

  # print(time_step.reward, time_step.discount, time_step.observation)