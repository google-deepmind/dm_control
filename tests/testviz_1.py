
import glviz

# initialize viz
glviz.init()

# add some meshes, just for fun
_sphere = glviz.createSphere( 0.5 )
_box = glviz.createBox( 0.5, 1.0, 0.25 ) 
_plane = glviz.createPlane( 10.0, 10.0 )
_capsule = glviz.createCapsule( 0.25, 1.0 )

_sphere.setPosition( -2.0, 1.0, 0.0 )
_box.setPosition( 0.0, 1.0, 0.0 )
_capsule.setPosition( 2.0, 1.0, 0.0 )

# loop
while glviz.isActive() :
    glviz.update()
