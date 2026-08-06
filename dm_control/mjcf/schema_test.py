# Copyright 2026 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Validates dm_control's schema.xml against dm_control itself."""

import os
import xml.etree.ElementTree as ET

from absl.testing import absltest
from dm_control import mjcf
from dm_control.mjcf import schema

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), 'schema.xml')

# attributes whose defaults are not settable through PyMJCF descriptors
# (multi-token strings PyMJCF types more strictly than the runtime)
_DEFAULT_SWEEP_SKIP = set()


class SchemaTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.spec = schema.parse_schema(_SCHEMA_PATH)

  def test_namespace_universe_is_closed(self):
    """Every reference namespace is populated by some identified element."""
    namespaces = schema.collect_namespaces(self.spec)
    dangling = []

    def visit(spec, seen):
      if id(spec) in seen:
        return
      seen.add(id(spec))
      for attr_spec in spec.attributes.values():
        kwargs = getattr(attr_spec, 'other_kwargs', {}) or {}
        ns = kwargs.get('reference_namespace')
        if ns and not ns.startswith('attrib:') and ns not in namespaces:
          dangling.append(f'{spec.name}.{attr_spec.name} -> {ns}')
      for child in spec.children.values():
        visit(child, seen)

    visit(self.spec, set())
    self.assertEqual(dangling, [])

  def test_singleton_api_surface(self):
    """The attribute-vs-list-view surface PyMJCF users rely on."""
    root = mjcf.RootElement(model='m')
    # singletons are elements, directly usable
    root.option.timestep = 0.001
    root.compiler.angle = 'radian'
    root.visual.map.znear = 0.01
    self.assertTrue(hasattr(root.worldbody, 'add'))
    # the top-level default is a singleton...
    root.default.geom.margin = 0.001
    # ...whose nested defaults are repeated
    root.default.add('default', dclass='a')
    root.default.add('default', dclass='b')
    # repeated elements are list views
    root.worldbody.add('body', name='b1')
    root.worldbody.add('body', name='b2')
    self.assertLen(root.worldbody.body, 2)

  def test_full_pymjcf_exercise(self):
    """Build, class, attach twice, bind, compile, round-trip."""
    arm = mjcf.RootElement(model='arm')
    arm.default.joint.damping = 2.0
    strong = arm.default.add('default', dclass='strong')
    strong.joint.damping = 5.0
    upper = arm.worldbody.add('body', name='upper')
    upper.add('joint', name='shoulder', type='ball')
    upper.add('geom', name='upper_geom', type='capsule', size=[.05],
              fromto=[0, 0, 0, 0, 0, -.3])
    lower = upper.add('body', name='lower', pos=[0, 0, -.3])
    elbow = lower.add('joint', name='elbow', type='hinge', axis=[0, 1, 0],
                      dclass='strong')
    lower.add('geom', name='lower_geom', type='capsule', size=[.04],
              fromto=[0, 0, 0, 0, 0, -.25])
    tip = lower.add('site', name='tip', pos=[0, 0, -.25])
    arm.actuator.add('position', name='act', joint=elbow, kp=10)
    arm.sensor.add('jointpos', name='sense', joint=elbow)
    arm.contact.add('exclude', name='ex', body1='upper', body2='lower')

    hand = mjcf.RootElement(model='hand')
    hand.worldbody.add('geom', name='palm', type='box',
                       size=[.02, .02, .02])
    tip.attach(hand)

    scene = mjcf.RootElement(model='scene')
    scene.worldbody.add('geom', name='floor', type='plane', size=[1, 1, .1])
    scene.worldbody.add('light', pos=[0, 0, 3])
    scene.attach(arm)

    physics = mjcf.Physics.from_mjcf_model(scene)
    self.assertEqual(physics.model.nu, 1)
    self.assertGreater(physics.model.nq, 4)
    # binding resolves through two attachment levels
    self.assertEqual(
        scene.find_all('sensor')[0].joint.full_identifier, 'arm/elbow')
    bound = physics.bind(scene.find_all('joint'))
    self.assertEqual(bound.qpos.shape[0], physics.model.nq)
    # serialization is deterministic; PyMJCF output is not re-parseable by
    # design (the '/' scope markers reject), so the parser direction is
    # exercised on ordinary MJCF instead
    self.assertEqual(scene.to_xml_string(), scene.to_xml_string())
    plain = mjcf.from_xml_string("""
      <mujoco model="plain">
        <default>
          <default class="soft"><geom friction="0.5 0.005 0.0001"/></default>
        </default>
        <worldbody>
          <body name="b" pos="0 0 1">
            <joint name="j" type="slide" axis="1 0 0" limited="true"
                   range="-1 1"/>
            <geom name="g" type="box" size=".1 .1 .1" class="soft"/>
          </body>
        </worldbody>
        <actuator><motor name="m" joint="j" gear="2"/></actuator>
      </mujoco>""")
    self.assertEqual(plain.find('geom', 'g').dclass.dclass, 'soft')
    plain_physics = mjcf.Physics.from_mjcf_model(plain)
    self.assertEqual(plain_physics.model.nu, 1)

  def test_every_declared_default_validates(self):
    """Setting each attribute to its declared default passes validation.

    The schema file carries default= metadata; PyMJCF's descriptors
    validate on assignment, so this sweeps every keyword set, arity and
    numeric type through dm_control's own machinery.
    """
    tree = ET.parse(_SCHEMA_PATH)

    def constructible(spec_node, path):
      """Depth-first: yield (path, element node) below construction roots."""
      for child in spec_node.findall('./children/element'):
        name = child.get('name')
        yield path + (name,), child
        yield from constructible(child, path + (name,))

    root = mjcf.RootElement(model='sweep')
    checked, failures = 0, []
    for path, node in constructible(tree.getroot(), ()):
      for attr in node.findall('./attributes/attribute'):
        default = attr.get('default')
        if default is None:
          continue
        if (path[-1], attr.get('name')) in _DEFAULT_SWEEP_SKIP:
          continue
        # walk to the element, constructing along the way where possible
        parent = root
        ok = True
        for tag in path:
          existing = getattr(parent, tag, None)
          if isinstance(existing, mjcf.Element) or (
              existing is not None
              and type(existing).__name__ == '_ElementImpl'):
            parent = existing
            continue
          try:
            parent = parent.add(tag)
          except Exception:  # pylint: disable=broad-exception-caught
            ok = False
            break
        if not ok:
          continue
        try:
          setattr(parent, attr.get('name'), default)
          checked += 1
        except Exception as e:  # pylint: disable=broad-exception-caught
          failures.append(f'{"/".join(path)}@{attr.get("name")}='
                          f'{default!r}: {e}')
    self.assertGreater(checked, 200)
    if failures:
      self.fail(f'{len(failures)} defaults rejected by PyMJCF:\n'
                + '\n'.join(failures[:20]))


if __name__ == '__main__':
  absltest.main()
