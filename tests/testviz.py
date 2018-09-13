
import glviz

# initialize viz
glviz.init()
# create a mesh
_mesh = glviz.addMesh()

# # just for fun -> Check TODOs
# del _mesh

# loop
while glviz.isActive() :
    glviz.update()
