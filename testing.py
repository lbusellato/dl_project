from dl_project.utils.graphics import show_images_grid
from dl_project.dataset.shapes3D import Shapes3D

ds = Shapes3D()
pair = ds.pairs[0]
show_images_grid([ds.images[pair[0],:,:,:],ds.images[pair[1],:,:,:]])