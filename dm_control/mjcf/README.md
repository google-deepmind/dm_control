# PyMJCF

IMPORTANT: If you find yourself stuck while using PyMJCF, check out the various
IMPORTANT boxes on this page and the [Common gotchas](#common-gotchas) section
at the bottom to see if any of them is relevant.

This library provides a Python object model for MuJoCo's XML-based
[MJCF](http://www.mujoco.org/book/modeling.html) physics modeling language. The
goal of the library is to allow users to easily interact with and modify MJCF
models in Python, similarly to what the JavaScript DOM does for HTML.

A key feature of this library is the ability to easily compose multiple separate
MJCF models into a larger one. Disambiguation of duplicated names from different
models, or multiple instances of the same model, is handled automatically.

The following snippet provides a quick example of this library's typical use
case. Here, the `UpperBody` class can simply instantiate two copies of `Arm`,
thus reducing code duplication. The names of bodies, joints, or geoms of each
`Arm` are automatically prefixed by their parent's names, and so no name
collision occurs.

```python
from dm_control import mjcf

class Arm:

  def __init__(self, name):
    self.mjcf_model = mjcf.RootElement(model=name)

    self.upper_arm = self.mjcf_model.worldbody.add('body', name='upper_arm')
    self.shoulder = self.upper_arm.add('joint', name='shoulder', type='ball')
    self.upper_arm.add('geom', name='upper_arm', type='capsule',
                       pos=[0, 0, -0.15], size=[0.045, 0.15])

    self.forearm = self.upper_arm.add('body', name='forearm', pos=[0, 0, -0.3])
    self.elbow = self.forearm.add('joint', name='elbow',
                                  type='hinge', axis=[0, 1, 0])
    self.forearm.add('geom', name='forearm', type='capsule',
                     pos=[0, 0, -0.15], size=[0.045, 0.15])

class UpperBody:

  def __init__(self):
    self.mjcf_model = mjcf.RootElement()
    self.mjcf_model.worldbody.add(
        'geom', name='torso', type='box', size=[0.15, 0.045, 0.25])
    left_shoulder_site = self.mjcf_model.worldbody.add(
        'site', size=[1e-6]*3, pos=[-0.15, 0, 0.25])
    right_shoulder_site = self.mjcf_model.worldbody.add(
        'site', size=[1e-6]*3, pos=[0.15, 0, 0.25])

    self.left_arm = Arm(name='left_arm')
    left_shoulder_site.attach(self.left_arm.mjcf_model)

    self.right_arm = Arm(name='right_arm')
    right_shoulder_site.attach(self.right_arm.mjcf_model)

body = UpperBody()
physics = mjcf.Physics.from_mjcf_model(body.mjcf_model)
```

## Basic operations

### Creating an MJCF model

In PyMJCF, the basic building block of a model is an `mjcf.Element`. This
corresponds to an element in the generated XML. However, user code _cannot_
instantiate a generic `mjcf.Element` object directly.

A valid model always consists of a single root `<mujoco>` element. This is
represented as the special `mjcf.RootElement` type in PyMJCF, which _can_ be
instantiated in user code to create an empty model.

```python
from dm_control import mjcf

mjcf_model = mjcf.RootElement()
print(mjcf_model)  # MJCF Element: <mujoco/>
```

### Adding new elements

Attributes of the new element can be passed as kwargs:

```python
my_box = mjcf_model.worldbody.add('geom', name='my_box',
                                  type='box', pos=[0, .1, 0])
print(my_box)  # MJCF Element: <geom name="my_box" type="box" pos="0. 0.1 0."/>
```

### Parsing an existing XML document

Alternatively, if an existing XML file already exists, PyMJCF can parse it to
create a Python object:

```python
from dm_control import mjcf

# Parse from path
mjcf_model = mjcf.from_path(filename)

# Parse from file
with open(filename) as f:
  mjcf_model = mjcf.from_file(f)

# Parse from string
with open(filename) as f:
  xml_string = f.read()
mjcf_model = mjcf.from_xml_string(xml_string)

print(type(mjcf_model))  # <type 'mjcf.RootElement'>
```

### Traversing through a model

Consider the following MJCF model:

```xml
<mujoco model="test">
  <default>
    <default class="brick">
      <geom rgba="1 0 0 1"/>
    </default>
  </default>
  <worldbody>
    <body name="foo">
      <freejoint/>
      <inertial pos="0 0 0" mass="1"/>
      <body name="bar">
        <joint name="my_hinge" type="hinge"/>
        <geom name="my_geom" pos="0 1 2" class="brick"/>
      </body>
    </body>
  </worldbody>
</mujoco>
```

The child elements and XML attributes of an `Element` object are exposed as
Python attributes. These attributes all have the same names as their XML
counterparts, with one exception: the `class` XML attribute is named `dclass` in
order to avoid a clash with the Python `class` keyword:

```python
my_geom = mjcf_model.worldbody.body['foo'].body['bar'].geom['my_geom']
print(isinstance(mjcf_model, mjcf.Element)) # True
print(my_geom.name)    # 'my_geom'
print(my_geom.pos)     # np.array([0., 1., 2.], dtype=float)
print(my_geom.class)   # SyntaxError
print(my_geom.dclass)  # 'brick'
```

Note that attribute values in the object model are **not** affected by defaults:

```python
print(mjcf_model.default.default['brick'].geom.rgba)  # [1, 0, 0, 1]
print(my_geom.rgba)  # None
```

### Finding elements without traversing

We can also find elements directly without having to traverse through the object
hierarchy:

```python
found_geom = mjcf_model.find('geom', 'my_geom')
print(found_geom == my_geom)  # True
```

Find all elements of a given type:

```python
# Note that <freejoint> is also considered a joint
joints = mjcf_model.find_all('joint')
print(len(joints))  # 2
print(joints[0] == mjcf_model.worldbody.body['foo'].freejoint)  # True
print(joints[1] == mjcf_model.worldbody.body['foo'].body['bar'].joint[0])  # True
```

Note that the order of elements returned by `find_all` is the same as the order
in which they are declared in the model.

### Modifying XML attributes

Attributes can be modified, added, or removed:

```python
my_geom.pos = [1, 2, 3]
print(my_geom.pos)   # np.array([1., 2., 3.], dtype=float)
my_geom.quat = [0, 1, 0, 0]
print(my_geom.quat)  # np.array([0., 1., 0., 0.], dtype=float)
del my_geom.quat
print(my_geom.quat)   # None
```

Schema violations result in errors:

```python
print(my_geom.poss)  # raise AttributeError (no child or attribute called poss)
my_geom.pos = 'invalid'  # raise ValueError (assigning string to array)
my_geom.pos = [1, 2, 3, 4, 5, 6]  # raise ValueError (array length is too long)

# raise ValueError (mass is a required attribute of <inertial>)
del mjcf_model.find('body', 'foo').inertial.mass
```

### Uniqueness of identifiers

PyMJCF enforces the uniqueness of "identifier" attributes within a model.
Identifiers consist of the `class` attribute of a `<default>`, and all `name`
attributes. Their uniqueness is only enforced within a particular namespace. For
example, a `<body>` is allowed to have the same name as a `<geom>`, whereas
`<position>` and `<velocity>` actuators cannot have the same name.

```python
mjcf_model.worldbody.add('geom', name='my_geom')
foo = mjcf_model.worldbody.find('body', 'foo')
foo.add('my_geom')  # Error, duplicated geom name
foo.add('foo')  # OK, a geom can have the same name as a body
mjcf_model.find('geom', 'foo').name = 'my_geom'  # Error, duplicated geom name
```

### Reference attributes

Some attributes are references to other elements. For example, the `joint`
attribute of an actuator refers to a `<joint>` element in the model.

An `mjcf.Element` can be directly assigned to these reference attributes:

```python
my_hinge = mjcf_model.find('joint', 'my_hinge')
my_actuator = mjcf_model.actuator.add('velocity', joint=my_hinge)
```

This is the recommended way to assign reference attributes, since it guarantees
that the reference is not invalidated if the referenced element is renamed.
Alternatively, a string can also be assigned to reference attributes. In this
case, PyMJCF does **not** attempt to verify that the named element actually
exists in the model.

IMPORTANT: If the element being referenced is in a different model to the
reference attribute (e.g. in an attached model), the reference **must** be
created by directly assigning an `mjcf.Element` object to the attribute rather
than a string. Strings assigned to reference attributes cannot contain '/',
since they are automatically scoped by PyMJCF upon attachment.

## Attaching models

In this section we will refer to an `mjcf.RootElement` simply as a "model".
Models can be _attached_ to other models in order to create compositional
scenes.

```python
arena = mjcf.RootElement()
arena.worldbody.add('geom', name='ground', type='plane', size=[10, 10, 1])

robot = mjcf.from_xml_file('robot.xml')
arena.attach(robot)
```

We refer to `arena` as the _parent model_, and `robot` as the _child model_ (or
the _attached model_).

### Attachment frames

When a model is attached to a site, an empty body is created in the parent
model. This empty body is called an _attachment frame_.

The attachment frame is created as a child of the body that contains the
attachment site, and it has the same position and orientation as the site. When
the XML is generated, the attachment frame's contents shadow the contents of the
attached model's `<worldbody>`. The attachment frame's name in the generated XML
is the child's `fully/qualified/prefix/`. The trailing slash ensures that the
attachment frame's name never collides with a user-defined body.

More concretely, if we have the following parent and child models:

```xml
<mujoco model="parent">
  <worldbody>
    <body>
      <geom name="foo" type="box" pos="-0.2 0 0.3" size="0.5 0.3 0.1"/>
      <site name="attachment_site" pos="1. 2. 3." quat="1. 0. 0. 1."/>
    </body>
  </worldbody>
</mujoco>

<mujoco model="child">
  <worldbody>
    <geom name="bar" type="box" pos="0.5 0.25 1." size="0.1 0.2 0.3"/>
  </worldbody>
</mujoco>
```

Then the final generated XML will be:

```xml
<!-- PyMJCF-generated XML, contains implementation details -->
<mujoco model="parent">
  <worldbody>
    <body>
      <geom name="foo" type="box" pos="-0.2 0 0.3" size="0.5 0.3 0.1"/>
      <site name="attachment_site" pos="1. 2. 3." quat="1. 0. 0. 1."/>
      <body name="child/" pos="1. 2. 3." quat="1. 0. 0. 1.">
        <geom name="child/my_box" type="box" pos="0.5 0.25 1." size="0.1 0.2 0.3"/>
      </body>
    </body>
  </worldbody>
</mujoco>
```

IMPORTANT: The attachment frame is created _transparently_ to the user. In
particular, it is NOT treated as a regular `body` by PyMJCF. Its name in the
generated XML should be considered implementation detail and should NOT be
relied on.

Having said that, it is sometimes necessary to access the attachment frame, for
example to add a joint between the parent and the child model. The easiest way
to do this is to hold a reference to the object returned by a call to `attach`:

```python
attachment_frame = parent_model.attach('child')
attachment_frame.add('freejoint')
```

Alternatively, if a model has already been attached, the `find` function can be
used with the `attachment_frame` namespace in order to retrieve the attachment
frame. The `get_attachment_frame` convenience function in `mjcf.traversal_utils`
can find the child model's attachment frame without needing access to the parent
model.

```python
frame_1 = parent_model.find('attachment_frame', 'child')

# Convenience function: get the attachment frame directly from a child model
frame_2 = mjcf.traversal_utils.get_attachment_frame(child_model)
print(frame_1 == frame_2)  # True
```

IMPORTANT: To encourage good modeling practices, the only allowed direct
children of an attachment frame are `<joint>` and `<inertial>`. Other types of
elements should instead add be added to the `<worldbody>` of the attached model.

### Element ownership

IMPORTANT: Elements of child models do **not** appear when traversing through
the parent model.

### Default classes

PyMJCF ensures that default classes of a parent model _never_ affect any of its
child models. This minimises the possibility that two models become subtly
"incompatible", as a model always behaves in the same way regardless of what it
is attached to.

The way that PyMJCF achieves this in practice is to move everything in a model's
global `<default>` context into a default class named `/`. In other words, a
PyMJCF-generated model never has anything in the global default context.
Instead, the generated model always looks like:

```xml
<!-- PyMJCF-generated XML, contains implementation details -->
<mujoco>
  <default>
    <default class="/">
      <!-- "global defaults" go here -->
      <geom rgba="1. 0. 0. 1."/>
    </default>
  </default>
</mujoco>
```

IMPORTANT: This transformation is _transparent_ to the user. Within Python, the
above geom rgba setting is accessed as if it were a global default, i.e.
`mjcf_model.default.geom.rgba`. Generally speaking, users should never have to
worry about PyMJCF's internal handling of defaults.

When a model is attached, its `/` default class turns into
`fully/qualified/prefix/`. The trailing slash ensures that this transformation
never conflicts with a user-named default class. More specifically, if we have
the following parent and child models:

```xml
<mujoco model="parent">
  <default>
    <geom rgba="1. 0. 0. 1."/>
    <default class="green">
      <geom rgba="0. 1. 0. 1."/>
    </default>
  </default>
</mujoco>

<mujoco model="child">
  <default>
    <joint range="0. 1."/>
    <default class="stiff">
      <joint stiffness="0.1"/>
    </default>
  </default>
</mujoco>
```

Then the final generated XML will be:

```xml
<!-- PyMJCF-generated XML, contains implementation details -->
<mujoco model="parent">
  <default>
    <default class="/">
      <geom rgba="1. 0. 0. 1."/>
      <default class="green">
        <geom rgba="0. 1. 0. 1."/>
      </default>
    </default>
    <default class="child/">
      <joint range="0. 1."/>
      <default class="child/stiff">
        <joint stiffness="0.1"/>
      </default>
    </default>
  </default>
</mujoco>
```

### Global options

A model cannot be attached to another model if _any_ of the global options are
different. Global options consist of attributes of `<compiler>`, `<option>`,
`<size>`, and `<visual>`. As with the handling of default classes, this is to
ensure that two models do not become subtly "incompatible". For example:

```python
model_1 = mjcf.RootElement()
model_1.compiler.angle = 'radian'

model_2 = mjcf.RootElement()
model_2.compiler.angle = 'degree'

model_1.attach(model_2)  # Error!
```

An option is considered to be conflicting only if _both_ models _explicitly_
assign different values to it. An example of where conflicting options can
become problematic is:

```python
model_1 = mjcf.RootElement()

model_2 = mjcf.RootElement()
model_2.compiler.angle = 'degree'

model_1.attach(model_2)  # No error, but all angles in model_1 are now wrong!
```

Here, `model_1` assumes MuJoCo's default angle unit of radians. Since it does
not explicitly assign a value to `compiler.angle`, PyMJCF does not detect a
conflict with `angle=degree` in `model_2`. All angles in `model_1` are now
incorrectly interpreted as degrees.

### Elements outside of `<worldbody>`

All children of non-worldbody elements, e.g. actuators or tendons, are
automatically merged in to appropriate places when a model is attached. Named
elements are prefixed as previously described.

## Common gotchas {#common-gotchas}

### Use `foo.dclass`, not ~~`foo.class`~~

The `class` XML attribute corresponds to the `dclass` Python attribute in
PyMJCF. This is because `class` is a reserved keyword in Python. However, it is
OK to use `'class'` in `getattr`.

```xml
<geom name="my_geom" class="red"/>
```

```python
print(my_geom.class)   # SyntaxError
print(my_geom.dclass)  # 'red'
print(getattr(my_geom, 'class'))  # 'red'
```

### `foo.type` and `foo.range` are fine

The `type` and `range` attributes will trigger syntax highlighting in Cider, but
they are NOT reserved words in Python.

```python
my_geom.type = 'capsule'  # OK!
my_joint.range = [-1, 1]  # OK!
```

### A model can only be attached once

A model cannot be attached twice. If you require multiple copies of the same
model in your simulation, make a `deepcopy`. Preferably, though, define a class
that constructs the model and just call the constructor as many times as
required.
