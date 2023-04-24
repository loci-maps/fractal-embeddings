import pandas as pd
import numpy as np
import pyvista as pv
from sklearn.preprocessing import MinMaxScaler
import pyvista as pv
from matplotlib.colors import ListedColormap

# Loads npz 
embeddings_npz = np.load('./sample_data/cohere_memory_embeddings.npz')

rgb = embeddings_npz['pca5'][:, 2:5]
xyz = embeddings_npz['pca5'][:,:3]

# simply pass the numpy points to the PolyData constructor
cloud = pv.PolyData(xyz)
cloud.plot(point_size=15)


# points = np.column_stack((xy_array, np.zeros(xy_array.shape[0])))
# cloud = pv.PolyData(points)
# cloud.point_data['colors'] = rgb_array

# surf = cloud.delaunay_2d()

# norm_rgb = rgb_array
# colormap = ListedColormap(norm_rgb)

# plotter = pv.Plotter()
# plotter.add_mesh(surf, scalars='colors', cmap=colormap, show_edges=True)
# plotter.show()

