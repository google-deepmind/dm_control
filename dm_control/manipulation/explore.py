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

"""A standalone application for visualizing manipulation tasks."""

import functools

from absl import app
from absl import flags
from dm_control import manipulation

from dm_control import viewer

flags.DEFINE_enum(
    'environment_name', None, manipulation.ALL,
    'Optional name of an environment to load. If unspecified '
    'a prompt will appear to select one.')
FLAGS = flags.FLAGS


# TODO(b/121187817): Consolidate with dm_control/suite/explore.py
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

  all_names = list(manipulation.ALL)

  if environment_name is None:
    print('\n  '.join(['Available environments:'] + all_names))
    environment_name = prompt_environment_name(
        'Please select an environment name: ', all_names)

  loader = functools.partial(
      manipulation.load, environment_name=environment_name)
  viewer.launch(loader)


if __name__ == '__main__':
  app.run(main)
