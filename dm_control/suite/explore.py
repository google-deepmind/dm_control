# Copyright 2018 The dm_control Authors.
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
"""Control suite environments explorer."""


from absl import app
from absl import flags
from dm_control import suite
from dm_control.suite.wrappers import action_noise
from six.moves import input

from dm_control import viewer


_ALL_NAMES = ['.'.join(domain_task) for domain_task in suite.ALL_TASKS]

flags.DEFINE_enum('environment_name', None, _ALL_NAMES,
                  'Optional \'domain_name.task_name\' pair specifying the '
                  'environment to load. If unspecified a prompt will appear to '
                  'select one.')
flags.DEFINE_bool('timeout', True, 'Whether episodes should have a time limit.')
flags.DEFINE_bool('visualize_reward', True,
                  'Whether to vary the colors of geoms according to the '
                  'current reward value.')
flags.DEFINE_float('action_noise', 0.,
                   'Standard deviation of Gaussian noise to apply to actions, '
                   'expressed as a fraction of the max-min range for each '
                   'action dimension. Defaults to 0, i.e. no noise.')
FLAGS = flags.FLAGS


def prompt_environment_name(prompt, values):
  environment_name = None
  while not environment_name:
    environment_name = input(prompt)
    if not environment_name or values.index(environment_name) < 0:
      print('"%s" is not a valid environment name.' % environment_name)
      environment_name = None
  return environment_name


def main(argv):
  del argv
  environment_name = FLAGS.environment_name
  if environment_name is None:
    print('\n  '.join(['Available environments:'] + _ALL_NAMES))
    environment_name = prompt_environment_name(
        'Please select an environment name: ', _ALL_NAMES)

  index = _ALL_NAMES.index(environment_name)
  domain_name, task_name = suite.ALL_TASKS[index]

  task_kwargs = {}
  if not FLAGS.timeout:
    task_kwargs['time_limit'] = float('inf')

  def loader():
    env = suite.load(
        domain_name=domain_name, task_name=task_name, task_kwargs=task_kwargs)
    env.task.visualize_reward = FLAGS.visualize_reward
    if FLAGS.action_noise > 0:
      env = action_noise.Wrapper(env, scale=FLAGS.action_noise)
    return env

  viewer.launch(loader)


if __name__ == '__main__':
  app.run(main)
