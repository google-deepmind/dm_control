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

"""Tests for `dm_control.mjcf.element`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import hashlib
import itertools
import os
import sys
import traceback

# Internal dependencies.
from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf
from dm_control.mjcf import element
from dm_control.mjcf import namescope
from dm_control.mjcf import parser
from dm_control.mujoco.wrapper import util
import lxml
import numpy as np
import six
from six.moves import range

etree = lxml.etree

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'test_assets')
_TEST_MODEL_XML = os.path.join(_ASSETS_DIR, 'test_model.xml')
_TEXTURE_PATH = os.path.join(_ASSETS_DIR, 'textures/deepmind.png')
_MESH_PATH = os.path.join(_ASSETS_DIR, 'meshes/cube.stl')
_MODEL_WITH_INCLUDE_PATH = os.path.join(_ASSETS_DIR, 'model_with_include.xml')

_MODEL_WITH_INVALID_FILENAMES = os.path.join(
    _ASSETS_DIR, 'model_with_invalid_filenames.xml')
_INCLUDED_WITH_INVALID_FILENAMES = os.path.join(
    _ASSETS_DIR, 'included_with_invalid_filenames.xml')


class ElementTest(parameterized.TestCase):

  def assertIsSame(self, mjcf_model, other):
    self.assertTrue(mjcf_model.is_same_as(other))
    self.assertTrue(other.is_same_as(mjcf_model))

  def assertIsNotSame(self, mjcf_model, other):
    self.assertFalse(mjcf_model.is_same_as(other))
    self.assertFalse(other.is_same_as(mjcf_model))

  def assertHasAttr(self, obj, attrib):
    self.assertTrue(hasattr(obj, attrib))

  def assertNotHasAttr(self, obj, attrib):
    self.assertFalse(hasattr(obj, attrib))

  def _test_properties(self, mjcf_element, parent, root, recursive=False):
    self.assertEqual(mjcf_element.tag, mjcf_element.spec.name)
    self.assertEqual(mjcf_element.parent, parent)
    self.assertEqual(mjcf_element.root, root)
    self.assertEqual(mjcf_element.namescope, root.namescope)
    for child_name, child_spec in six.iteritems(mjcf_element.spec.children):
      if not (child_spec.repeated or child_spec.on_demand):
        child = getattr(mjcf_element, child_name)
        self.assertEqual(child.tag, child_name)
        self.assertEqual(child.spec, child_spec)
        if recursive:
          self._test_properties(child, parent=mjcf_element,
                                root=root, recursive=True)

  def testAttributeError(self):
    mjcf_model = element.RootElement(model='test')
    mjcf_model.worldbody._spec = None
    attribute_error_raised = False
    try:
      _ = mjcf_model.worldbody.tag
    except AttributeError:
      # Test that the error comes from the fact that we've set `_spec = None`.
      attribute_error_raised = True
      _, err, tb = sys.exc_info()
      self.assertEqual(str(err),
                       '\'NoneType\' object has no attribute \'name\'')
      _, _, func_name, _ = traceback.extract_tb(tb)[-1]
      # Test that the error comes from the `root` property, not `__getattr__`.
      self.assertEqual(func_name, 'tag')
    self.assertTrue(attribute_error_raised)

  def testProperties(self):
    mujoco = element.RootElement(model='test')
    self.assertIsInstance(mujoco.namescope, namescope.NameScope)
    self._test_properties(mujoco, parent=None, root=mujoco, recursive=True)

  def _test_attributes(self, mjcf_element,
                       expected_values=None, recursive=False):
    attributes = mjcf_element.get_attributes()
    self.assertNotIn('class', attributes)
    for attribute_name in six.iterkeys(mjcf_element.spec.attributes):
      if attribute_name == 'class':
        attribute_name = 'dclass'
      self.assertHasAttr(mjcf_element, attribute_name)
      self.assertIn(attribute_name, dir(mjcf_element))
      attribute_value = getattr(mjcf_element, attribute_name)
      if attribute_value is not None:
        self.assertIn(attribute_name, attributes)
      else:
        self.assertNotIn(attribute_name, attributes)
      if expected_values:
        if attribute_name in expected_values:
          expected_value = expected_values[attribute_name]
          np.testing.assert_array_equal(attribute_value, expected_value)
        else:
          self.assertIsNone(attribute_value)
    if recursive:
      for child in mjcf_element.all_children():
        self._test_attributes(child, recursive=True)

  def testAttributes(self):
    mujoco = element.RootElement(model='test')
    mujoco.default.dclass = 'main'
    self._test_attributes(mujoco, recursive=True)

  def _test_children(self, mjcf_element, recursive=False):
    children = mjcf_element.all_children()
    for child_name, child_spec in six.iteritems(mjcf_element.spec.children):
      if not (child_spec.repeated or child_spec.on_demand):
        self.assertHasAttr(mjcf_element, child_name)
        self.assertIn(child_name, dir(mjcf_element))
        child = getattr(mjcf_element, child_name)
        self.assertIn(child, children)
        with self.assertRaisesRegexp(AttributeError, 'can\'t set attribute'):
          setattr(mjcf_element, child_name, 'value')
        if recursive:
          self._test_children(child, recursive=True)

  def testChildren(self):
    mujoco = element.RootElement(model='test')
    self._test_children(mujoco, recursive=True)

  def testInvalidAttr(self):
    mujoco = element.RootElement(model='test')
    invalid_attrib_name = 'foobar'
    def test_invalid_attr_recursively(mjcf_element):
      self.assertNotHasAttr(mjcf_element, invalid_attrib_name)
      self.assertNotIn(invalid_attrib_name, dir(mjcf_element))
      with self.assertRaisesRegexp(AttributeError, 'object has no attribute'):
        getattr(mjcf_element, invalid_attrib_name)
      with self.assertRaisesRegexp(AttributeError, 'can\'t set attribute'):
        setattr(mjcf_element, invalid_attrib_name, 'value')
      with self.assertRaisesRegexp(AttributeError, 'object has no attribute'):
        delattr(mjcf_element, invalid_attrib_name)
      for child in mjcf_element.all_children():
        test_invalid_attr_recursively(child)
    test_invalid_attr_recursively(mujoco)

  def testAdd(self):
    mujoco = element.RootElement(model='test')

    # repeated elements
    body_foo_attributes = dict(name='foo', pos=[0, 1, 0], quat=[0, 1, 0, 0])
    body_foo = mujoco.worldbody.add('body', **body_foo_attributes)
    self.assertEqual(body_foo.tag, 'body')
    joint_foo_attributes = dict(name='foo', type='free')
    joint_foo = body_foo.add('joint', **joint_foo_attributes)
    self.assertEqual(joint_foo.tag, 'joint')
    self._test_properties(body_foo, parent=mujoco.worldbody, root=mujoco)
    self._test_attributes(body_foo, expected_values=body_foo_attributes)
    self._test_children(body_foo)
    self._test_properties(joint_foo, parent=body_foo, root=mujoco)
    self._test_attributes(joint_foo, expected_values=joint_foo_attributes)
    self._test_children(joint_foo)

    # non-repeated, on-demand elements
    self.assertIsNone(body_foo.inertial)
    body_foo_inertial_attributes = dict(mass=1.0, pos=[0, 0, 0])
    body_foo_inertial = body_foo.add('inertial', **body_foo_inertial_attributes)
    self._test_properties(body_foo_inertial, parent=body_foo, root=mujoco)
    self._test_attributes(body_foo_inertial,
                          expected_values=body_foo_inertial_attributes)
    self._test_children(body_foo_inertial)

    with self.assertRaisesRegexp(ValueError, '<inertial> child already exists'):
      body_foo.add('inertial', **body_foo_inertial_attributes)

    # non-repeated, non-on-demand elements
    with self.assertRaisesRegexp(ValueError, '<compiler> child already exists'):
      mujoco.add('compiler')
    self.assertIsNotNone(mujoco.compiler)
    with self.assertRaisesRegexp(ValueError, '<default> child already exists'):
      mujoco.add('default')
    self.assertIsNotNone(mujoco.default)

  def testAddWithInvalidAttribute(self):
    mujoco = element.RootElement(model='test')
    with self.assertRaisesRegexp(AttributeError, 'not a valid attribute'):
      mujoco.worldbody.add('body', name='foo', invalid_attribute='some_value')
    self.assertFalse(mujoco.worldbody.body)
    self.assertIsNone(mujoco.worldbody.find('body', 'foo'))

  def testSameness(self):
    mujoco = element.RootElement(model='test')

    body_1 = mujoco.worldbody.add('body', pos=[0, 1, 2], quat=[0, 1, 0, 1])
    site_1 = body_1.add('site', pos=[0, 1, 2], quat=[0, 1, 0, 1])
    geom_1 = body_1.add('geom', pos=[0, 1, 2], quat=[0, 1, 0, 1])

    for elem in (body_1, site_1, geom_1):
      self.assertIsSame(elem, elem)

    # strict ordering NOT required: adding geom and site is different order
    body_2 = mujoco.worldbody.add('body', pos=[0, 1, 2], quat=[0, 1, 0, 1])
    geom_2 = body_2.add('geom', pos=[0, 1, 2], quat=[0, 1, 0, 1])
    site_2 = body_2.add('site', pos=[0, 1, 2], quat=[0, 1, 0, 1])

    elems_1 = (body_1, site_1, geom_1)
    elems_2 = (body_2, site_2, geom_2)
    for i, j in itertools.product(range(len(elems_1)), range(len(elems_2))):
      if i == j:
        self.assertIsSame(elems_1[i], elems_2[j])
      else:
        self.assertIsNotSame(elems_1[i], elems_2[j])

    # on-demand child
    body_1.add('inertial', pos=[0, 0, 0], mass=1)
    self.assertIsNotSame(body_1, body_2)

    body_2.add('inertial', pos=[0, 0, 0], mass=1)
    self.assertIsSame(body_1, body_2)

    # different number of children
    subbody_1 = body_1.add('body', pos=[0, 0, 1])
    self.assertIsNotSame(body_1, body_2)

    # attribute mismatch
    subbody_2 = body_2.add('body')
    self.assertIsNotSame(subbody_1, subbody_2)
    self.assertIsNotSame(body_1, body_2)

    subbody_2.pos = [0, 0, 1]
    self.assertIsSame(subbody_1, subbody_2)
    self.assertIsSame(body_1, body_2)

    # grandchild attribute mismatch
    subbody_1.add('joint', type='hinge')
    subbody_2.add('joint', type='ball')
    self.assertIsNotSame(body_1, body_2)

  def testTendonSameness(self):
    mujoco = element.RootElement(model='test')

    spatial_1 = mujoco.tendon.add('spatial')
    spatial_1.add('site', site='foo')
    spatial_1.add('geom', geom='bar')

    spatial_2 = mujoco.tendon.add('spatial')
    spatial_2.add('site', site='foo')
    spatial_2.add('geom', geom='bar')

    self.assertIsSame(spatial_1, spatial_2)

    # strict ordering is required
    spatial_3 = mujoco.tendon.add('spatial')
    spatial_3.add('site', site='foo')
    spatial_3.add('geom', geom='bar')

    spatial_4 = mujoco.tendon.add('spatial')
    spatial_4.add('geom', geom='bar')
    spatial_4.add('site', site='foo')

    self.assertIsNotSame(spatial_3, spatial_4)

  def testCopy(self):
    mujoco = parser.from_path(_TEST_MODEL_XML)
    self.assertIsSame(mujoco, mujoco)

    copy_mujoco = copy.copy(mujoco)
    copy_mujoco.model = 'copied_model'
    self.assertIsSame(copy_mujoco, mujoco)
    self.assertNotEqual(copy_mujoco, mujoco)

    deepcopy_mujoco = copy.deepcopy(mujoco)
    deepcopy_mujoco.model = 'deepcopied_model'
    self.assertIsSame(deepcopy_mujoco, mujoco)
    self.assertNotEqual(deepcopy_mujoco, mujoco)

    self.assertIsSame(deepcopy_mujoco, copy_mujoco)
    self.assertNotEqual(deepcopy_mujoco, copy_mujoco)

  def testWorldBodyFullIdentifier(self):
    mujoco = parser.from_path(_TEST_MODEL_XML)
    mujoco.model = 'model'
    self.assertEqual(mujoco.worldbody.full_identifier, 'world')

    submujoco = copy.copy(mujoco)
    submujoco.model = 'submodel'
    self.assertEqual(submujoco.worldbody.full_identifier, 'world')

    mujoco.attach(submujoco)
    self.assertEqual(mujoco.worldbody.full_identifier, 'world')
    self.assertEqual(submujoco.worldbody.full_identifier, 'submodel/')

    self.assertNotIn('name', mujoco.worldbody.to_xml_string(self_only=True))
    self.assertNotIn('name', submujoco.worldbody.to_xml_string(self_only=True))

  def testAttach(self):
    mujoco = parser.from_path(_TEST_MODEL_XML)
    mujoco.model = 'model'

    submujoco = copy.copy(mujoco)
    submujoco.model = 'submodel'

    subsubmujoco = copy.copy(mujoco)
    subsubmujoco.model = 'subsubmodel'

    with self.assertRaisesRegexp(ValueError, 'Cannot merge a model to itself'):
      mujoco.attach(mujoco)

    attachment_site = submujoco.find('site', 'attachment')
    attachment_site.attach(subsubmujoco)
    subsubmodel_frame = submujoco.find('attachment_frame', 'subsubmodel')
    for attribute_name in ('pos', 'axisangle', 'xyaxes', 'zaxis', 'euler'):
      np.testing.assert_array_equal(
          getattr(subsubmodel_frame, attribute_name),
          getattr(attachment_site, attribute_name))
    self._test_properties(subsubmodel_frame,
                          parent=attachment_site.parent, root=submujoco)
    self.assertEqual(
        subsubmodel_frame.to_xml_string().split('\n')[0],
        '<body pos="0.1 0.1 0.1" quat="0. 1. 0. 0." name="subsubmodel/">')
    self.assertEqual(subsubmodel_frame.all_children(),
                     subsubmujoco.worldbody.all_children())

    with self.assertRaisesRegexp(ValueError, 'already attached elsewhere'):
      mujoco.attach(subsubmujoco)

    with self.assertRaisesRegexp(ValueError, 'Expected a mjcf.RootElement'):
      mujoco.attach(submujoco.contact)

    submujoco.option.flag.gravity = 'enable'
    with self.assertRaisesRegexp(
        ValueError, 'Conflicting values for attribute `gravity`'):
      mujoco.attach(submujoco)
    submujoco.option.flag.gravity = 'disable'

    mujoco.attach(submujoco)
    self.assertEqual(subsubmujoco.parent_model, submujoco)
    self.assertEqual(submujoco.parent_model, mujoco)
    self.assertEqual(subsubmujoco.root_model, mujoco)
    self.assertEqual(submujoco.root_model, mujoco)

    self.assertEqual(submujoco.full_identifier, 'submodel/')
    self.assertEqual(subsubmujoco.full_identifier, 'submodel/subsubmodel/')

    merged_children = ('contact', 'actuator')
    for child_name in merged_children:
      for grandchild in getattr(submujoco, child_name).all_children():
        self.assertIn(grandchild, getattr(mujoco, child_name).all_children())
      for grandchild in getattr(subsubmujoco, child_name).all_children():
        self.assertIn(grandchild, getattr(mujoco, child_name).all_children())
        self.assertIn(grandchild, getattr(submujoco, child_name).all_children())

    base_contact_content = (
        '<exclude name="{0}exclude" body1="{0}b_0" body2="{0}b_1"/>')
    self.assertEqual(
        mujoco.contact.to_xml_string(pretty_print=False),
        '<contact>' +
        base_contact_content.format('') +
        base_contact_content.format('submodel/') +
        base_contact_content.format('submodel/subsubmodel/') +
        '</contact>')

    actuators_template = (
        '<velocity name="{1}b_0_0" class="{0}" joint="{1}b_0_0"/>'
        '<velocity name="{1}b_1_0" class="{0}" joint="{1}b_1_0"/>')
    self.assertEqual(
        mujoco.actuator.to_xml_string(pretty_print=False),
        '<actuator>' +
        actuators_template.format('/', '') +
        actuators_template.format('submodel/', 'submodel/') +
        actuators_template.format('submodel/subsubmodel/',
                                  'submodel/subsubmodel/') +
        '</actuator>')

    self.assertEqual(mujoco.default.full_identifier, '/')
    self.assertEqual(mujoco.default.default[0].full_identifier, 'big_and_green')
    self.assertEqual(submujoco.default.full_identifier, 'submodel/')
    self.assertEqual(submujoco.default.default[0].full_identifier,
                     'submodel/big_and_green')
    self.assertEqual(subsubmujoco.default.full_identifier,
                     'submodel/subsubmodel/')
    self.assertEqual(subsubmujoco.default.default[0].full_identifier,
                     'submodel/subsubmodel/big_and_green')
    default_xml_lines = (mujoco.default.to_xml_string(pretty_print=False)
                         .replace('><', '>><<').split('><'))
    self.assertEqual(default_xml_lines[0], '<default>')
    self.assertEqual(default_xml_lines[1], '<default class="/">')
    self.assertEqual(default_xml_lines[4], '<default class="big_and_green">')
    self.assertEqual(default_xml_lines[6], '</default>')
    self.assertEqual(default_xml_lines[7], '</default>')
    self.assertEqual(default_xml_lines[8], '<default class="submodel/">')
    self.assertEqual(default_xml_lines[11],
                     '<default class="submodel/big_and_green">')
    self.assertEqual(default_xml_lines[13], '</default>')
    self.assertEqual(default_xml_lines[14], '</default>')
    self.assertEqual(default_xml_lines[15],
                     '<default class="submodel/subsubmodel/">')
    self.assertEqual(default_xml_lines[18],
                     '<default class="submodel/subsubmodel/big_and_green">')
    self.assertEqual(default_xml_lines[-3], '</default>')
    self.assertEqual(default_xml_lines[-2], '</default>')
    self.assertEqual(default_xml_lines[-1], '</default>')

  def testDetach(self):
    root = parser.from_path(_TEST_MODEL_XML)
    root.model = 'model'

    submodel = copy.copy(root)
    submodel.model = 'submodel'

    unattached_xml_1 = root.to_xml_string()
    root.attach(submodel)
    attached_xml_1 = root.to_xml_string()

    submodel.detach()
    unattached_xml_2 = root.to_xml_string()
    root.attach(submodel)
    attached_xml_2 = root.to_xml_string()

    self.assertEqual(unattached_xml_1, unattached_xml_2)
    self.assertEqual(attached_xml_1, attached_xml_2)

  def testRenameAttachedModel(self):
    root = parser.from_path(_TEST_MODEL_XML)
    root.model = 'model'

    submodel = copy.copy(root)
    submodel.model = 'submodel'
    geom = submodel.worldbody.add(
        'geom', name='geom', type='sphere', size=[0.1])

    frame = root.attach(submodel)
    submodel.model = 'renamed'
    self.assertEqual(frame.full_identifier, 'renamed/')
    self.assertIsSame(root.find('geom', 'renamed/geom'), geom)

  def testAttachmentFrames(self):
    mujoco = parser.from_path(_TEST_MODEL_XML)
    mujoco.model = 'model'

    submujoco = copy.copy(mujoco)
    submujoco.model = 'submodel'

    subsubmujoco = copy.copy(mujoco)
    subsubmujoco.model = 'subsubmodel'

    attachment_site = submujoco.find('site', 'attachment')
    attachment_site.attach(subsubmujoco)
    mujoco.attach(submujoco)

    # attachments directly on worldbody can have a <freejoint>
    submujoco_frame = mujoco.find('attachment_frame', 'submodel')
    self.assertStartsWith(submujoco_frame.to_xml_string(pretty_print=False),
                          '<body name="submodel/">')
    self.assertEqual(submujoco_frame.full_identifier, 'submodel/')
    free_joint = submujoco_frame.add('freejoint')
    self.assertEqual(free_joint.to_xml_string(pretty_print=False),
                     '<freejoint name="submodel/"/>')
    self.assertEqual(free_joint.full_identifier, 'submodel/')

    # attachments elsewhere cannot have a <freejoint>
    subsubmujoco_frame = submujoco.find('attachment_frame', 'subsubmodel')
    subsubmujoco_frame_xml = subsubmujoco_frame.to_xml_string(
        pretty_print=False, prefix_root=mujoco.namescope)
    self.assertStartsWith(subsubmujoco_frame_xml,
                          '<body pos="0.1 0.1 0.1" quat="0. 1. 0. 0." '
                          'name="submodel/subsubmodel/">')
    self.assertEqual(subsubmujoco_frame.full_identifier,
                     'submodel/subsubmodel/')
    with self.assertRaisesRegexp(AttributeError, 'not a valid child'):
      subsubmujoco_frame.add('freejoint')
    hinge_joint = subsubmujoco_frame.add('joint', type='hinge', axis=[1, 2, 3])
    hinge_joint_xml = hinge_joint.to_xml_string(
        pretty_print=False, prefix_root=mujoco.namescope)
    self.assertEqual(
        hinge_joint_xml,
        '<joint class="submodel/" type="hinge" axis="1. 2. 3." '
        'name="submodel/subsubmodel/"/>')
    self.assertEqual(hinge_joint.full_identifier, 'submodel/subsubmodel/')

  def testDuplicateAttachmentFrameJointIdentifiers(self):
    mujoco = parser.from_path(_TEST_MODEL_XML)
    mujoco.model = 'model'

    submujoco_1 = copy.copy(mujoco)
    submujoco_1.model = 'submodel_1'

    submujoco_2 = copy.copy(mujoco)
    submujoco_2.model = 'submodel_2'

    frame_1 = mujoco.attach(submujoco_1)
    frame_2 = mujoco.attach(submujoco_2)

    joint_1 = frame_1.add('joint', type='slide', name='root_x', axis=[1, 0, 0])
    joint_2 = frame_2.add('joint', type='slide', name='root_x', axis=[1, 0, 0])

    self.assertEqual(joint_1.full_identifier, 'submodel_1/root_x/')
    self.assertEqual(joint_2.full_identifier, 'submodel_2/root_x/')

  def testAttachmentFrameReference(self):
    root_1 = mjcf.RootElement('model_1')
    root_2 = mjcf.RootElement('model_2')
    root_2_frame = root_1.attach(root_2)
    sensor = root_1.sensor.add(
        'framelinacc', name='root_2', objtype='body', objname=root_2_frame)
    self.assertEqual(
        sensor.to_xml_string(pretty_print=False),
        '<framelinacc name="root_2" objtype="body" objname="model_2/"/>')

  def testAttachmentFrameChildReference(self):
    root_1 = mjcf.RootElement('model_1')
    root_2 = mjcf.RootElement('model_2')
    root_2_frame = root_1.attach(root_2)
    root_2_joint = root_2_frame.add(
        'joint', name='root_x', type='slide', axis=[1, 0, 0])
    actuator = root_1.actuator.add(
        'position', name='root_x', joint=root_2_joint)
    self.assertEqual(
        actuator.to_xml_string(pretty_print=False),
        '<position name="root_x" class="/" joint="model_2/root_x/"/>')

  def testDeletion(self):
    mujoco = parser.from_path(_TEST_MODEL_XML)
    mujoco.model = 'model'

    submujoco = copy.copy(mujoco)
    submujoco.model = 'submodel'

    subsubmujoco = copy.copy(mujoco)
    subsubmujoco.model = 'subsubmodel'

    submujoco.find('site', 'attachment').attach(subsubmujoco)
    mujoco.attach(submujoco)

    with self.assertRaisesRegexp(
        ValueError, r'use remove\(affect_attachments=True\)'):
      del mujoco.option

    mujoco.option.remove(affect_attachments=True)
    for root in (mujoco, submujoco, subsubmujoco):
      self.assertIsNotNone(root.option.flag)
      self.assertEqual(
          root.option.to_xml_string(pretty_print=False), '<option/>')
      self.assertIsNotNone(root.option.flag)
      self.assertEqual(
          root.option.flag.to_xml_string(pretty_print=False), '<flag/>')

    with self.assertRaisesRegexp(
        ValueError, r'use remove\(affect_attachments=True\)'):
      del mujoco.contact

    mujoco.contact.remove(affect_attachments=True)
    for root in (mujoco, submujoco, subsubmujoco):
      self.assertEqual(
          root.contact.to_xml_string(pretty_print=False), '<contact/>')

    b_0 = mujoco.find('body', 'b_0')
    b_0_inertial = b_0.inertial
    self.assertEqual(b_0_inertial.mass, 1)
    self.assertIsNotNone(b_0.inertial)
    del b_0.inertial
    self.assertIsNone(b_0.inertial)

  def testRemoveElementWithRequiredAttribute(self):
    root = mjcf.RootElement()
    body = root.worldbody.add('body')
    # `objtype` is a required attribute.
    sensor = root.sensor.add('framepos', objtype='body', objname=body)
    self.assertIn(sensor, root.sensor.all_children())
    sensor.remove()
    self.assertNotIn(sensor, root.sensor.all_children())

  def testRemoveWithChildren(self):
    root = mjcf.RootElement()
    body = root.worldbody.add('body')
    subbodies = []
    for _ in range(5):
      subbodies.append(body.add('body'))
    body.remove()
    for subbody in subbodies:
      self.assertTrue(subbody.is_removed)

  def testFind(self):
    mujoco = parser.from_path(_TEST_MODEL_XML, resolve_references=False)
    mujoco.model = 'model'

    submujoco = copy.copy(mujoco)
    submujoco.model = 'submodel'

    subsubmujoco = copy.copy(mujoco)
    subsubmujoco.model = 'subsubmodel'

    submujoco.find('site', 'attachment').attach(subsubmujoco)
    mujoco.attach(submujoco)

    self.assertIsNotNone(mujoco.find('geom', 'b_0_0'))
    self.assertIsNotNone(mujoco.find('body', 'b_0').find('geom', 'b_0_0'))
    self.assertIsNone(mujoco.find('body', 'b_1').find('geom', 'b_0_0'))

    self.assertIsNone(mujoco.find('geom', 'nonexistent'))
    self.assertIsNone(mujoco.find('geom', 'nonexistent/b_0_0'))

    self.assertEqual(mujoco.find('geom', 'submodel/b_0_0'),
                     submujoco.find('geom', 'b_0_0'))
    self.assertEqual(mujoco.find('geom', 'submodel/subsubmodel/b_0_0'),
                     submujoco.find('geom', 'subsubmodel/b_0_0'))
    self.assertEqual(submujoco.find('geom', 'subsubmodel/b_0_0'),
                     subsubmujoco.find('geom', 'b_0_0'))

    subsubmujoco.find('geom', 'b_0_0').name = 'foo'
    self.assertIsNone(mujoco.find('geom', 'submodel/subsubmodel/b_0_0'))
    self.assertIsNone(submujoco.find('geom', 'subsubmodel/b_0_0'))
    self.assertIsNone(subsubmujoco.find('geom', 'b_0_0'))
    self.assertEqual(mujoco.find('geom', 'submodel/subsubmodel/foo'),
                     submujoco.find('geom', 'subsubmodel/foo'))
    self.assertEqual(submujoco.find('geom', 'subsubmodel/foo'),
                     subsubmujoco.find('geom', 'foo'))

    self.assertEqual(mujoco.find('actuator', 'b_0_0').root, mujoco)
    self.assertEqual(mujoco.find('actuator', 'b_0_0').tag, 'velocity')
    self.assertEqual(mujoco.find('actuator', 'b_0_0').joint, 'b_0_0')

    self.assertEqual(mujoco.find('actuator', 'submodel/b_0_0').root, submujoco)
    self.assertEqual(mujoco.find('actuator', 'submodel/b_0_0').tag, 'velocity')
    self.assertEqual(mujoco.find('actuator', 'submodel/b_0_0').joint, 'b_0_0')

  def testFindInvalidNamespace(self):
    mjcf_model = mjcf.RootElement()
    with self.assertRaisesRegexp(ValueError, 'not a valid namespace'):
      mjcf_model.find('jiont', 'foo')
    with self.assertRaisesRegexp(ValueError, 'not a valid namespace'):
      mjcf_model.find_all('goem')

  def testEnterScope(self):
    mujoco = parser.from_path(_TEST_MODEL_XML, resolve_references=False)
    mujoco.model = 'model'

    self.assertIsNone(mujoco.enter_scope('submodel'))

    submujoco = copy.copy(mujoco)
    submujoco.model = 'submodel'

    subsubmujoco = copy.copy(mujoco)
    subsubmujoco.model = 'subsubmodel'

    submujoco.find('site', 'attachment').attach(subsubmujoco)
    mujoco.attach(submujoco)

    self.assertIsNotNone(mujoco.enter_scope('submodel'))
    self.assertEqual(mujoco.enter_scope('submodel').find('geom', 'b_0_0'),
                     submujoco.find('geom', 'b_0_0'))

    self.assertEqual(
        mujoco.enter_scope('submodel/subsubmodel/').find('geom', 'b_0_0'),
        subsubmujoco.find('geom', 'b_0_0'))
    self.assertEqual(mujoco.enter_scope('submodel').enter_scope(
        'subsubmodel').find('geom', 'b_0_0'),
                     subsubmujoco.find('geom', 'b_0_0'))

    self.assertEqual(
        mujoco.enter_scope('submodel').find('actuator', 'b_0_0').root,
        submujoco)
    self.assertEqual(
        mujoco.enter_scope('submodel').find('actuator', 'b_0_0').tag,
        'velocity')
    self.assertEqual(
        mujoco.enter_scope('submodel').find('actuator', 'b_0_0').joint,
        'b_0_0')

  def testDefaultIdentifier(self):
    mujoco = element.RootElement(model='test')
    body = mujoco.worldbody.add('body')
    joint_0 = body.add('freejoint')
    joint_1 = body.add('joint', type='hinge')
    self.assertIsNone(body.name)
    self.assertIsNone(joint_0.name)
    self.assertIsNone(joint_1.name)
    self.assertEqual(str(body), '<body>...</body>')
    self.assertEqual(str(joint_0), '<freejoint/>')
    self.assertEqual(str(joint_1), '<joint class="/" type="hinge"/>')
    self.assertEqual(body.full_identifier, '//unnamed_body_0')
    self.assertStartsWith(body.to_xml_string(pretty_print=False),
                          '<body name="{:s}">'.format(body.full_identifier))
    self.assertEqual(joint_0.full_identifier, '//unnamed_joint_0')
    self.assertEqual(joint_0.to_xml_string(pretty_print=False),
                     '<freejoint name="{:s}"/>'.format(joint_0.full_identifier))
    self.assertEqual(joint_1.full_identifier, '//unnamed_joint_1')
    self.assertEqual(joint_1.to_xml_string(pretty_print=False),
                     '<joint name="{:s}" class="/" type="hinge"/>'.format(
                         joint_1.full_identifier))

    submujoco = copy.copy(mujoco)
    submujoco.model = 'submodel'
    mujoco.attach(submujoco)
    submujoco_body = submujoco.worldbody.body[0]
    self.assertEqual(submujoco_body.full_identifier,
                     'submodel//unnamed_body_0')
    self.assertEqual(submujoco_body.freejoint.full_identifier,
                     'submodel//unnamed_joint_0')
    self.assertEqual(submujoco_body.joint[0].full_identifier,
                     'submodel//unnamed_joint_1')

  def testFindAll(self):
    mujoco = parser.from_path(_TEST_MODEL_XML)
    mujoco.model = 'model'

    submujoco = copy.copy(mujoco)
    submujoco.model = 'submodel'

    subsubmujoco = copy.copy(mujoco)
    subsubmujoco.model = 'subsubmodel'

    submujoco.find('site', 'attachment').attach(subsubmujoco)
    mujoco.attach(submujoco)

    geoms = mujoco.find_all('geom')
    self.assertEqual(len(geoms), 6)
    self.assertEqual(geoms[0].root, mujoco)
    self.assertEqual(geoms[1].root, mujoco)
    self.assertEqual(geoms[2].root, submujoco)
    self.assertEqual(geoms[3].root, subsubmujoco)
    self.assertEqual(geoms[4].root, subsubmujoco)
    self.assertEqual(geoms[5].root, submujoco)

    b_0 = submujoco.find('body', 'b_0')
    self.assertEqual(len(b_0.find_all('joint')), 6)
    self.assertEqual(
        len(b_0.find_all('joint', immediate_children_only=True)), 1)
    self.assertEqual(
        len(b_0.find_all('joint', exclude_attachments=True)), 2)

  def testFindAllFrameJoints(self):
    root_model = parser.from_path(_TEST_MODEL_XML)
    root_model.model = 'model'

    submodel = copy.copy(root_model)
    submodel.model = 'submodel'

    frame = root_model.attach(submodel)
    joint_x = frame.add('joint', type='slide', axis=[1, 0, 0])
    joint_y = frame.add('joint', type='slide', axis=[0, 1, 0])

    joints = frame.find_all('joint', immediate_children_only=True)
    self.assertListEqual(joints, [joint_x, joint_y])

  def testDictLikeInterface(self):
    mujoco = element.RootElement(model='test')
    elem = mujoco.worldbody.add('body')
    if six.PY3:
      subscript_error_regex = 'object is not subscriptable'
    else:
      subscript_error_regex = 'no attribute \'__getitem__\''
    with self.assertRaisesRegexp(TypeError, subscript_error_regex):
      _ = elem['foo']
    with self.assertRaisesRegexp(TypeError, 'does not support item assignment'):
      elem['foo'] = 'bar'
    with self.assertRaisesRegexp(TypeError, 'does not support item deletion'):
      del elem['foo']

  def testSetAndGetAttributes(self):
    mujoco = element.RootElement(model='test')

    foo_attribs = dict(name='foo', pos=[1, 2, 3, 4], quat=[0, 1, 0, 0])
    with self.assertRaisesRegexp(ValueError, 'no more than 3 entries'):
      foo = mujoco.worldbody.add('body', **foo_attribs)

    # failed creationg should not cause the identifier 'foo' to be registered
    with self.assertRaises(KeyError):
      mujoco.namescope.get('body', 'foo')
    foo_attribs['pos'] = [1, 2, 3]
    foo = mujoco.worldbody.add('body', **foo_attribs)
    self._test_attributes(foo, expected_values=foo_attribs)

    foo_attribs['name'] = 'bar'
    foo_attribs['pos'] = [1, 2, 3, 4]
    foo_attribs['childclass'] = 'klass'
    with self.assertRaisesRegexp(ValueError, 'no more than 3 entries'):
      foo.set_attributes(**foo_attribs)

    # failed assignment should not cause the identifier 'bar' to be registered
    with self.assertRaises(KeyError):
      mujoco.namescope.get('body', 'bar')
    foo_attribs['pos'] = [1, 2, 3]
    foo.set_attributes(**foo_attribs)
    self._test_attributes(foo, expected_values=foo_attribs)

    actual_foo_attribs = foo.get_attributes()
    for attribute_name, value in six.iteritems(foo_attribs):
      np.testing.assert_array_equal(
          actual_foo_attribs.pop(attribute_name), value)
    for value in six.itervalues(actual_foo_attribs):
      self.assertIsNone(value)

  def testResolveReferences(self):
    resolved_model = parser.from_path(_TEST_MODEL_XML)
    self.assertIs(
        resolved_model.find('geom', 'b_1_0').material,
        resolved_model.find('material', 'mat_texture'))
    unresolved_model = parser.from_path(
        _TEST_MODEL_XML, resolve_references=False)
    self.assertEqual(
        unresolved_model.find('geom', 'b_1_0').material, 'mat_texture')
    unresolved_model.resolve_references()
    self.assertIs(
        unresolved_model.find('geom', 'b_1_0').material,
        unresolved_model.find('material', 'mat_texture'))

  @parameterized.named_parameters(
      ('WithoutInclude', _TEST_MODEL_XML),
      ('WithInclude', _MODEL_WITH_INCLUDE_PATH))
  def testParseFromString(self, model_path):
    with open(model_path) as xml_file:
      xml_string = xml_file.read()
    model_dir, _ = os.path.split(model_path)
    parser.from_xml_string(xml_string, model_dir=model_dir)

  @parameterized.named_parameters(
      ('WithoutInclude', _TEST_MODEL_XML),
      ('WithInclude', _MODEL_WITH_INCLUDE_PATH))
  def testParseFromFile(self, model_path):
    model_dir, _ = os.path.split(model_path)
    with open(model_path) as xml_file:
      parser.from_file(xml_file, model_dir=model_dir)

  @parameterized.named_parameters(
      ('WithoutInclude', _TEST_MODEL_XML),
      ('WithInclude', _MODEL_WITH_INCLUDE_PATH))
  def testParseFromPath(self, model_path):
    parser.from_path(model_path)

  def testGetAssetFromFile(self):
    with open(_TEXTURE_PATH, 'rb') as f:
      contents = f.read()
    _, filename = os.path.split(_TEXTURE_PATH)
    prefix, extension = os.path.splitext(filename)
    vfs_filename = prefix + '-' + hashlib.sha1(contents).hexdigest() + extension
    mujoco = parser.from_path(_TEST_MODEL_XML)
    self.assertDictEqual({vfs_filename: contents}, mujoco.get_assets())

  def testGetAssetFromPlaceholder(self):
    mujoco = parser.from_path(_TEST_MODEL_XML)
    # Add an extra texture asset from a placeholder.
    contents = b'I am a texture bytestring'
    extension = '.png'
    vfs_filename = hashlib.sha1(contents).hexdigest() + extension
    placeholder = mjcf.Asset(contents=contents, extension=extension)
    mujoco.asset.add('texture', name='fake_texture', file=placeholder)
    self.assertDictContainsSubset({vfs_filename: contents}, mujoco.get_assets())

  def testGetAssetsFromDict(self):
    with open(_MODEL_WITH_INVALID_FILENAMES, 'rb') as f:
      xml_string = f.read()
    with open(_TEXTURE_PATH, 'rb') as f:
      texture_contents = f.read()
    with open(_MESH_PATH, 'rb') as f:
      mesh_contents = f.read()
    with open(_INCLUDED_WITH_INVALID_FILENAMES, 'rb') as f:
      included_xml_contents = f.read()
    assets = {
        'invalid_texture_name.png': texture_contents,
        'invalid_mesh_name.stl': mesh_contents,
        'invalid_included_name.xml': included_xml_contents,
    }
    # The paths specified in the main and included XML files are deliberately
    # invalid, so the parser should fail unless the pre-loaded assets are passed
    # in as a dict.
    with self.assertRaises(IOError):
      parser.from_xml_string(xml_string=xml_string)

    mujoco = parser.from_xml_string(xml_string=xml_string, assets=assets)
    expected_assets = {}
    for path, contents in six.iteritems(assets):
      _, filename = os.path.split(path)
      prefix, extension = os.path.splitext(filename)
      if extension != '.xml':
        vfs_filename = ''.join(
            [prefix, '-', hashlib.sha1(contents).hexdigest(), extension])
        expected_assets[vfs_filename] = contents
    self.assertDictEqual(expected_assets, mujoco.get_assets())

  def testAssetsCanBeCopied(self):
    with open(_TEXTURE_PATH, 'rb') as f:
      contents = f.read()
    _, filename = os.path.split(_TEXTURE_PATH)
    prefix, extension = os.path.splitext(filename)
    vfs_filename = prefix + '-' + hashlib.sha1(contents).hexdigest() + extension
    mujoco = parser.from_path(_TEST_MODEL_XML)
    mujoco_copy = copy.copy(mujoco)
    expected = {vfs_filename: contents}
    self.assertDictEqual(expected, mujoco.get_assets())
    self.assertDictEqual(expected, mujoco_copy.get_assets())

  def testAssetInheritance(self):
    parent = element.RootElement(model='parent')
    child = element.RootElement(model='child')
    grandchild = element.RootElement(model='grandchild')

    ext = '.png'
    parent_str = b'I belong to the parent'
    child_str = b'I belong to the child'
    grandchild_str = b'I belong to the grandchild'
    parent_vfs_name, child_vfs_name, grandchild_vfs_name = (
        hashlib.sha1(s).hexdigest() + ext
        for s in (parent_str, child_str, grandchild_str))

    parent_ph = mjcf.Asset(contents=parent_str, extension=ext)
    child_ph = mjcf.Asset(contents=child_str, extension=ext)
    grandchild_ph = mjcf.Asset(contents=grandchild_str, extension=ext)

    parent.asset.add('texture', name='parent_tex', file=parent_ph)
    child.asset.add('texture', name='child_tex', file=child_ph)
    grandchild.asset.add('texture', name='grandchild_tex', file=grandchild_ph)

    parent.attach(child)
    child.attach(grandchild)

    # The grandchild should only return its own assets.
    self.assertDictEqual(
        {grandchild_vfs_name: grandchild_str},
        grandchild.get_assets())

    # The child should return its own assets plus those of the grandchild.
    self.assertDictEqual(
        {child_vfs_name: child_str,
         grandchild_vfs_name: grandchild_str},
        child.get_assets())

    # The parent should return everything.
    self.assertDictEqual(
        {parent_vfs_name: parent_str,
         child_vfs_name: child_str,
         grandchild_vfs_name: grandchild_str},
        parent.get_assets())

  def testActuatorReordering(self):

    def make_model_with_mixed_actuators(name):
      actuators = []
      root = mjcf.RootElement(model=name)
      body = root.worldbody.add('body')
      body.add('geom', type='sphere', size=[0.1])
      slider = body.add('joint', type='slide', name='slide_joint')
      # Third-order `general` actuator.
      actuators.append(
          root.actuator.add(
              'general', dyntype='integrator', biastype='affine',
              dynprm=[1, 0, 0], joint=slider, name='general_act'))
      # Cylinder actuators are also third-order.
      actuators.append(
          root.actuator.add('cylinder', joint=slider, name='cylinder_act'))
      # A second-order actuator, added after the third-order actuators.
      actuators.append(
          root.actuator.add('velocity', joint=slider, name='velocity_act'))
      return root, actuators

    child_1, actuators_1 = make_model_with_mixed_actuators(name='child_1')
    child_2, actuators_2 = make_model_with_mixed_actuators(name='child_2')
    child_3, actuators_3 = make_model_with_mixed_actuators(name='child_3')
    parent = mjcf.RootElement()
    parent.attach(child_1)
    parent.attach(child_2)
    child_2.attach(child_3)

    # Check that the generated XML contains all of the actuators that we expect
    # it to have.
    expected_xml_strings = [
        actuator.to_xml_string(prefix_root=parent.namescope)
        for actuator in actuators_1 + actuators_2 + actuators_3
    ]
    xml_strings = [
        util.to_native_string(etree.tostring(node, pretty_print=True))
        for node in parent.to_xml().find('actuator').getchildren()
    ]
    self.assertSameElements(expected_xml_strings, xml_strings)

    # MuJoCo requires that all 3rd-order actuators (i.e. those with internal
    # dynamics) come after all 2nd-order actuators in the XML. Attempting to
    # compile this model will result in an error unless PyMJCF internally
    # reorders the actuators so that the 3rd-order actuator comes last in the
    # generated XML.
    _ = mjcf.Physics.from_mjcf_model(child_1)

    # Actuator re-ordering should also work in cases where there are multiple
    # attached submodels with mixed 2nd- and 3rd-order actuators.
    _ = mjcf.Physics.from_mjcf_model(parent)


if __name__ == '__main__':
  absltest.main()
