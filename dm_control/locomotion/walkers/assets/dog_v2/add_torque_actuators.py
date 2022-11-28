import collections
from dm_control import mjcf

def add_motors(physics, model, lumbar_joints, cervical_joints, caudal_joints):
    #physics = mjcf.Physics.from_mjcf_model(model)
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
                        'fixed', name=region + direction, dclass=joints[0].dclass)
                tendons.append(tendon)
                joint_inertia = physics.bind(joints).M0
                coefs = joint_inertia ** .25
                coefs /= coefs.sum()
                coefs *= len(joints)
                for i, joint in enumerate(joints):
                    tendon.add('joint', joint=joint, coef=coefs[i])

    # Actuators:
    all_spinal_joints = [
            joint for region in spinal_joints.values() for joint in region  # pylint: disable=g-complex-comprehension
    ]
    root_joint = model.find('joint', 'root')
    actuated_joints = [
        joint for joint in model.find_all('joint')
        if joint not in all_spinal_joints and joint is not root_joint
    ]
    for tendon in tendons:
        gain = 0.
        for joint in tendon.joint:
            # joint.joint.user = physics.bind(joint.joint).damping
            def_joint = model.default.find('default', joint.joint.dclass)
            j_gain = def_joint.general.gainprm or def_joint.parent.general.gainprm
            gain += j_gain[0] * joint.coef
        gain /= len(tendon.joint)

        model.actuator.add(
                'general', tendon=tendon, name=tendon.name, dclass=tendon.dclass)

    for joint in actuated_joints:
        model.actuator.add(
                'general', joint=joint, name=joint.name, dclass=joint.dclass)

    return actuated_joints
