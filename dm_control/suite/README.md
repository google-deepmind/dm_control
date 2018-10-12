# DeepMind Control Suite.

This submodule contains the domains and tasks described in the
[DeepMind Control Suite tech report](https://arxiv.org/abs/1801.00690).

## Quickstart

```python
from dm_control import suite
import numpy as np

# Load one task:
env = suite.load(domain_name="cartpole", task_name="swingup")

# Iterate over a task set:
for domain_name, task_name in suite.BENCHMARKING:
  env = suite.load(domain_name, task_name)
  # TODO(b/117645013): This is a temporary workaround for issue #48.
  env.physics.contexts.mujoco.free()

# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
time_step = env.reset()
while not time_step.last():
  action = np.random.uniform(action_spec.minimum,
                             action_spec.maximum,
                             size=action_spec.shape)
  time_step = env.step(action)
  print(time_step.reward, time_step.discount, time_step.observation)
```

## Illustration video

Below is a video montage of solved Control Suite tasks, with reward
visualisation enabled.

[![Video montage](https://img.youtube.com/vi/rAai4QzcYbs/0.jpg)](https://www.youtube.com/watch?v=rAai4QzcYbs)
