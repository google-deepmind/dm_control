# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Suite environments viewer package."""


from dm_control.viewer import application


def launch(environment_loader, policy=None, title='Explorer', width=1024,
           height=768):
  """Launches an environment viewer.

  Args:
    environment_loader: An environment loader (a callable that returns an
      instance of dm_control.rl.control.Environment), an instance of
      dm_control.rl.control.Environment.
    policy: An optional callable corresponding to a policy to execute within the
      environment. It should accept a `TimeStep` and return a numpy array of
      actions conforming to the output of `environment.action_spec()`.
    title: Application title to be displayed in the title bar.
    width: Window width, in pixels.
    height: Window height, in pixels.
  Raises:
      ValueError: When 'environment_loader' argument is set to None.
  """
  app = application.Application(title=title, width=width, height=height)
  app.launch(environment_loader=environment_loader, policy=policy)
