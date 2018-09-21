
import enginewrapper

# initialize viz
enginewrapper.init()
# create a mesh
_mesh = enginewrapper.createMesh()

# # just for fun -> Check TODOs
# del _mesh

# loop
while enginewrapper.isActive() :
    enginewrapper.update()
