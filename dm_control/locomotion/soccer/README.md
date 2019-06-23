# DeepMind MuJoCo Multi-Agent Soccer Environment.

This submodule contains the components and environment described in ICLR 2019
paper [Emergent Coordination through Competition][website].

# ![soccer](soccer.png)

## Installation and requirements

See [dm_control](../../../README.md#installation-and-requirements) for instructions.

## Quickstart

```python
import numpy as np
from dm_control.locomotion import soccer as dm_soccer

# Load the 2-vs-2 soccer environment with episodes of 10 seconds:
env = dm_soccer.load(team_size=2, time_limit=10.)

# Retrieves action_specs for all 4 players.
action_specs = env.action_spec()

# Step through the environment for one episode with random actions.
time_step = env.reset()
while not time_step.last():
  actions = []
  for action_spec in action_specs:
    action = np.random.uniform(
        action_spec.minimum, action_spec.maximum, size=action_spec.shape)
    actions.append(action)
  time_step = env.step(actions)

  for i in range(len(action_specs)):
    print(
        "Player {}: reward = {}, discount = {}, observations = {}.".format(
            i, time_step.reward[i], time_step.discount,
            time_step.observation[i]))
```

## Rewards

The environment provides a reward of +1 to each player when their team
scores a goal, -1 when their team concedes a goal, or 0 if neither team scored
on the current timestep.

In addition to the sparse reward returned the environment, the player
observations also contain various environment statistics that may be used to
derive custom per-player shaping rewards (as was done in
http://arxiv.org/abs/1902.07151, where the environment reward was ignored).

## Episode terminations

Episodes will terminate immediately with a discount factor of 0 when either side
scores a goal. There is also a per-episode `time_limit` (45 seconds by default).
If neither team scores within this time then the episode will terminate with a
discount factor of 1.

## Environment Viewer

To visualize an example 2-vs-2 soccer environment in the `dm_control`
interactive viewer, execute `dm_control/locomotion/soccer/explore.py`.

[website]: https://sites.google.com/corp/view/emergent-coordination/home
