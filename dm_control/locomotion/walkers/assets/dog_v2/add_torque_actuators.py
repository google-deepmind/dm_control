# Copyright 2023 The dm_control Authors.
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

"""Make torque actuators for the dog model."""

import collections


def add_motors(physics, model, lumbar_joints, cervical_joints, caudal_joints):
  """Add torque motors in model.

  Args:
    physics: an instance of physics for the most updated model.
    model: model in which we want to add motors.
    lumbar_joints: a list of joints objects.
    cervical_joints: a list of joints objects.
    caudal_joints: a list of joints objects.

  Returns:
    A list of actuated joints.
  """
  # Fixed Tendons:
  spinal_joints = collections.OrderedDict()
  spinal_joints['lumbar_'] = lumbar_joints
  spinal_joints['cervical_'] = cervical_joints
  spinal_joints['caudal_'] = caudal_joints
  tendons = []
  for region in spinal_joints.keys():
    for direction in ['extend', 'bend', 'twist']:
      joints = [
          joint for joint in spinal_joints[region] if direction in joint.name
      ]
      if joints:
        tendon = model.tendon.add(
            'fixed', name=region + direction, dclass=joints[0].dclass
        )
        tendons.append(tendon)
        joint_inertia = physics.bind(joints).M0
        coefs = joint_inertia**0.25
        coefs /= coefs.sum()
        coefs *= len(joints)
        for i, joint in enumerate(joints):
          tendon.add('joint', joint=joint, coef=coefs[i])

  # Actuators:
  all_spinal_joints = []
  for region in spinal_joints.values():
    all_spinal_joints.extend(region)
  root_joint = model.find('joint', 'root')
  actuated_joints = [
      joint
      for joint in model.find_all('joint')
      if joint not in all_spinal_joints and joint is not root_joint
  ]
  for tendon in tendons:
    gain = 0.0
    for joint in tendon.joint:
      # joint.joint.user = physics.bind(joint.joint).damping
      def_joint = model.default.find('default', joint.joint.dclass)
      j_gain = def_joint.general.gainprm or def_joint.parent.general.gainprm
      gain += j_gain[0] * joint.coef
    gain /= len(tendon.joint)

    model.actuator.add(
        'general', tendon=tendon, name=tendon.name, dclass=tendon.dclass
    )

  for joint in actuated_joints:
    model.actuator.add(
        'general', joint=joint, name=joint.name, dclass=joint.dclass
    )

  return actuated_joints
