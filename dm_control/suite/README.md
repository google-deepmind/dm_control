# DeepMind Control Suite.

This submodule contains the domains and tasks described in the
[DeepMind Control Suite tech report](https://arxiv.org/abs/1801.00690).

# ![all domains](all_domains.png)

## Quickstart

```python
from dm_control import suite
import numpy as np

# Load one task:
env = suite.load(domain_name="cartpole", task_name="swingup")

# Iterate over a task set:
for domain_name, task_name in suite.BENCHMARKING:
  env = suite.load(domain_name, task_name)

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


### Quadruped domain [April 2019]

Roughly based on the 'ant' model introduced by [Schulman et al. 2015](https://arxiv.org/abs/1506.02438). Main modifications to the body are:

- 4 DoFs per leg, 1 constraining tendon.
- 3 actuators per leg: 'yaw', 'lift', 'extend'.
- Filtered position actuators with timescale of 100ms.
- Sensors include an IMU, force/torque sensors, and rangefinders.

Four tasks:

- `walk` and `run`: self-right the body then move forward at a desired speed.
- `escape`: escape a bowl-shaped random terrain (uses rangefinders).
- `fetch`, go to a moving ball and bring it to a target.

All behaviors in the video below were trained with [Abdolmaleki et al's
MPO](https://arxiv.org/abs/1806.06920).

[![Video montage](https://img.youtube.com/vi/RhRLjbb7pBE/0.jpg)](https://www.youtube.com/watch?v=RhRLjbb7pBE)
