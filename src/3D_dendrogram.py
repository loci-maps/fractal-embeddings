# loads in ../sample_data/combined_reduced_embeddings.npz and visualizes umap2 with PCA5 2:5 colors
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, to_tree
from vedo import Points, Line, show, Group, Lines, vector, Box, Assembly

reduced_npz = np.load('fractal-embeddings-main/demo_data/combined_reduced_embeddings.npz', allow_pickle=True)
embedding_npz = np.load('fractal-embeddings-main/demo_data/combined_embeddings.npz', allow_pickle=True)

filenames = embedding_npz['filenames']
embeddings = embedding_npz['embeddings']
umap2 = reduced_npz['umap2']
pca5 = reduced_npz['pca5']
tsne2 = reduced_npz['tsne2']

# Perform hierarchical clustering using the 'ward' method
linked = linkage(embeddings, method='ward')

colors = pca5[:, 2:5]
num_samples = len(filenames)


class ClusterLine(Line):
    def __init__(self, p1, p2, cluster_info):
        super().__init__(p1, p2)
        self.cluster_info = cluster_info

filenames = embedding_npz['filenames']

linked = linkage(embeddings, method='ward')
dendro = dendrogram(linked, no_plot=True)
#dendro = dendrogram(linked, truncate_mode='lastp')

edges = []
leaf_points_coords = []  # to store the leaf points
points_filenames = []  # to store filenames corresponding to the leaf points

# Loop over each set of coordinates and create lines
for i in range(len(dendro['icoord'])):
    x_coords = dendro['icoord'][i]
    y_coords = dendro['dcoord'][i]
   
    for j in range(1, len(x_coords)):
        start_point = [x_coords[j-1], y_coords[j-1], 0]
        end_point = [x_coords[j], y_coords[j], 0]
      
        cluster_info = linked[i]
        edges.append(ClusterLine(start_point, end_point, cluster_info))

    leaf_points_coords.append([x_coords[0], y_coords[0], 0])
    leaf_points_coords.append([x_coords[-1], y_coords[-1], 0])

# Create Points object for leaf nodes
#leaf_points = Points(leaf_points_coords, r=5, c="red5")

# Add filenames to leaf points
# for i in range(len(leaf_points_coords)):
#     points_filenames.append(filenames[i])


box_list = []

for point_coord in leaf_points_coords:
    
    leaf_box = Box(pos=point_coord,
	length=5.0,
	width=5.0,
	height=5.0,
	c='g4',
	alpha=1.0)

    box_list.append(leaf_box)


txt = "filename"
# caption_list = [box.caption(txt, size=(0.04,0.03), font="LogoType", c='tomato')
#                 for box in box_list]

show(edges, bg='lightblue')