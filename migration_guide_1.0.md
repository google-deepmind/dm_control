# dm_control: 1.0.0 update guide

With 1.0.0, we changed the way dm_control uses the MuJoCo physics simulator, and
migrated to new python bindings. For most users, this will require no code
changes. However, some more advanced users will need to change their code
slightly. Below is a list of known changes that need to be made. Please contact
us if you've had to make any further changes that are not listed below.

## Required changes

`dm_control.mujoco.wrapper.mjbindings.types` should not be used This module was
specific to the previous implementation of the dm_control python bindings. For
example, `types.MJRRECT` should be replaced with `mujoco.MjrRect`.

### `MjData.contact` has changed

`MjData.contact` (often accessed as Physics.data.contact) used to offer an
interface similar to a numpy structured array. For example, `data.contact.geom1`
used to be a numpy array of geom IDs.

With the recent update, `MjData.contact` will appear as a list of MjContact
structs. Code that used to operate on the structured array will have to change.
For example, the following code would get an array containing the contact
distance for all contacts that involve geom_id:

```
contact = physics.data.contact
involves_geom = (contact.geom1 == geom_id) | (contact.geom2 == geom_id)
dists = contact[involves_geom].dist
```

After the upgrade:

```
contacts = physics.data.contact
dists = [
    c.dist for c in contacts if c.geom1 == geom_id or c.geom2 == geom_id
]
```

### `.ptr.contents` will not work

Code that accesses `.ptr.contents` on objects such as `MjvScene` will need to be
updated. In most cases, simply using `scene` instead of `scene.ptr.contents`
will work.

### Different exceptions will be thrown from MuJoCo

Code (mostly in tests) that expects `dm_control.mujoco.wrapper.core.Error`
exceptions, will receive different exceptions, thrown by the `mujoco` library.
These will often be `ValueError` (for errors caused by input parameters), or
`mujoco.FatalError` (for low level errors in MuJoCo).

### Better error handling

The Python interpreter no longer crashes out when
[`mju_error`](https://mujoco.readthedocs.io/en/latest/APIreference.html#mju-error)
is called. Instead, `mju_error` calls are translated into `mujoco.FatalError`
exceptions in Python.

When Python callables are used as user-defined MuJoCo callbacks, they are now
permitted to raise exceptions, which will be correctly propagated back down the
Python call stack.

### Change of signature for `mj_saveModel`

[`mj_saveModel`](https://mujoco.readthedocs.io/en/latest/APIreference.html#mj-savemodel)
now expects a numpy `uint8` array rather than a ctypes string buffer, and
doesn't require a "size" parameter (it's inferred from the numpy array size).

Before:
```
model_size = mjlib.mj_sizeModel(model.ptr)
buf = ctypes.create_string_buffer(model_size)
mjlib.mj_saveModel(model.ptr, None, buf, model_size)
```

After:
```
model_size = mujoco.mj_sizeModel(model)
buf = np.empty(model_size, np.uint8)
mjlib.mj_saveModel(model.ptr, None, buf)
```

## Optional changes

The following are some changes that can make your code more concise, but are not
required for it to continue working.

### Use the mujoco module directly, instead of mjlib

Existing code that uses `dm_control.mujoco.wrapper.mjbindings.mjlib` can
directly replace these modules with mujoco. Code that uses `enums` or
`constants` from `dm_control.mujoco.wrapper.mjbindings` can also use mujoco,
with slight type changes. All mujoco functions will accept the old enum values
or the new ones.

Before:
```
import dm_control.mujoco.wrapper.mjbindings
mjlib = mjbindings.mjlib

mjlib.mj_objectVelocity(
    physics.model.ptr, physics.data.ptr,
    enums.mjtObj.mjOBJ_SITE,
    site_id, vel, 0)
```

After:
```
import mujoco

mujoco.mj_objectVelocity(
    physics.model.ptr, physics.data.ptr,
    mujoco.mjtObj.mjOBJ_SITE,
    site_id, vel, 0)
```

### Assume structs are correctly initialized and memory is managed

The MuJoCo C API includes functions that manage the memory for certain structs.
Those include functions that allocate memory (e.g. `mj_makeModel`,
`mj_makeData`, `mjv_makeScene`), functions that free memory (e.g.
`mj_deleteModel`, `mj_deleteData`, `mjv_freeScene`), and functions that reset a
struct to its default value (e.g. `mjv_defaultOption`, `mj_defaultVisual`).

The new Python bindings take care of this. Wrapper classes like
`mujoco.MjvScene` will automatically allocate memory when they're created, and
release it when they're deleted, and be created with default values set.

As such, allocating and freeing functions are not available through the mujoco
Python bindings. The "default" functions are still available, but in most cases
the calls can simply be removed.

Before:
```
from dm_control.mujoco import wrapper
from dm_control.mujoco.wrapper import mjbindings
mjlib = mjbindings.mjlib

scene_option = wrapper.core.MjvOption()
mjlib.mjv_defaultOption(scene_option.ptr)
```

After:
```
from dm_control.mujoco import wrapper

scene_option = wrapper.core.MjvOption()
```
