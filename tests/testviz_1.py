
import enginewrapper

# initialize viz
enginewrapper.init()

# add some meshes, just for fun
_sphere = enginewrapper.createSphere( 0.5 )
_box = enginewrapper.createBox( 0.5, 1.0, 0.25 ) 
_plane = enginewrapper.createPlane( 10.0, 10.0 )
_capsule = enginewrapper.createCapsule( 0.25, 1.0 )

_sphere.setPosition( -2.0, 1.0, 0.0 )
_box.setPosition( 0.0, 1.0, 0.0 )
_capsule.setPosition( 2.0, 1.0, 0.0 )

# loop
while enginewrapper.isActive() :
    enginewrapper.update()
