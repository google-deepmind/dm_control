# Interactive environment viewer

# ![policy_in_viewer](policy.gif)

The `dm_control.viewer` library can be used to visualize and interact with a
control environment. The following example shows how to launch the viewer with
an environment from the Control Suite:

```python
from dm_control import suite
from dm_control import viewer

# Load an environment from the Control Suite.
env = suite.load(domain_name="cartpole", task_name="swingup")

# Launch the viewer application.
viewer.launch(env)
```

For convenience we also provide a viewer launch script for the Control Suite in
`dm_control/suite/explore.py`.

## Viewing the environment with a policy in the loop

The viewer is also capable of running the environment with a policy in the loop
to provide actions. This is done by passing the optional `policy` argument to
`viewer.launch`. The `policy` should be a callable that accepts a `TimeStep` and
returns a numpy array of actions conforming to `environment.action_spec()`. The
example below shows how to execute a random uniform policy using the viewer:

```python
from dm_control import suite
from dm_control import viewer
import numpy as np

env = suite.load(domain_name="cartpole", task_name="swingup")
action_spec = env.action_spec()

# Define a uniform random policy.
def random_policy(time_step):
  del time_step  # Unused.
  return np.random.uniform(low=action_spec.minimum,
                           high=action_spec.maximum,
                           size=action_spec.shape)

# Launch the viewer application.
viewer.launch(env, policy=random_policy)
```

## Keyboard and mouse controls

The viewer contains a built in help screen that can be brought up by pressing
`F1`. You will find a comprehensive description of keyboard and mouse controls
there.

## Status view

Displays status of the simulation:

-   State - Current status of the Runtime state machine.
-   Time - Simulation clock accompanied by the current setting of time
    multiplier.
-   CPU - How much time per frame does physics simulation consume.
-   FPS - How many frames per second is the application rendering.
-   Camera - Name of the active camera.
-   Paused - Is the simulation paused?
-   Error - Recently caught error message.
