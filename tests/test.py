from dm_control import suite
from dm_control.mujoco import wrapper
from dm_control.mujoco.wrapper.mjbindings import mjlib
import numpy as np

# Load one task:
# env = suite.load(domain_name="cartpole", task_name="swingup")
env = suite.load( domain_name = "humanoid", task_name = "walk" )

# Iterate over a task set:
for domain_name, task_name in suite.BENCHMARKING:
#   env = suite.load(domain_name, task_name)
  print( 'domain: ', domain_name, ' - taskname: ', task_name )

# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
time_step = env.reset()

while not time_step.last():
  action = np.random.uniform(action_spec.minimum,
                             action_spec.maximum,
                             size=action_spec.shape)
  time_step = env.step(action)

  img = env.physics.render( 480, 480, camera_id = 0 )
  scene = wrapper.MjvScene()

  print(time_step.reward, time_step.discount, time_step.observation)



# snippets

    # mjlib.mjv_updateScene(  )
    # video[i] = np.hstack([env.physics.render(height, width, camera_id=0),
    #                       env.physics.render(height, width, camera_id=1)])