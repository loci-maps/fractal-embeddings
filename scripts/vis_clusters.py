import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, to_tree
import matplotlib.pyplot as plt


reduced_npz = np.load('./demo_data/combined_reduced_embeddings.npz', allow_pickle=True)
embedding_npz = np.load('./demo_data/combined_embeddings.npz', allow_pickle=True)

filenames = embedding_npz['filenames']
embeddings = embedding_npz['embeddings']
umap2 = reduced_npz['umap2']
pca5 = reduced_npz['pca5']
tsne2 = reduced_npz['tsne2']



# Perform hierarchical clustering using the 'ward' method
linked = linkage(embeddings, method='ward')

import matplotlib.colors as mcolors

# Adds colors for linkages by blending the leave colors together, recursively
# if t is 1, color2 is returned, if it is 0, color1 is
def blend_colors(color1, color2, t):
    r1, g1, b1 = color1
    r2, g2, b2 = color2

    r = r1 * (1 - t) + r2 * t
    g = g1 * (1 - t) + g2 * t
    b = b1 * (1 - t) + b2 * t

    return (r, g, b)

colors = pca5[:, 2:5]
num_samples = len(filenames)

for row in linked:
    cluster1_id = int(row[0])
    cluster2_id = int(row[1])

    if cluster1_id < len(filenames): # it is a sample/leaf
        cluster1_size = 1
    else:
        cluster1_size = linked[num_samples - cluster1_id][-1]
    
    if cluster2_id < len(filenames): # it is a sample/leaf
        cluster2_size = 1
    else:
        cluster2_size = linked[num_samples - cluster2_id][-1]

    # If a cluster is huge and is blended with a leaf, the color should be mostly the leaf color
    # Can try different weighting here
    t = cluster2_size / (cluster1_size + cluster2_size)

    color1 = colors[cluster1_id]
    color2 = colors[cluster2_id]

    merged_cluster_color = blend_colors(color1, color2, t)
    
    colors = np.vstack((colors, merged_cluster_color))

# converts rgb (0-1) to hex
colors = [mcolors.to_hex(color) for color in colors]

# Plot the dendrogram with custom labels and link colors
plt.figure(figsize=(15, 7))
dendrogram(linked, leaf_rotation=45, labels=[name for name in filenames], link_color_func=lambda link_id: colors[link_id])
plt.title('Hierarchical Clustering Dendrogram with Filenames and PCA5 Link Colors')
plt.show()

