# loads in ../sample_data/combined_reduced_embeddings.npz and visualizes umap2 with PCA5 2:5 colors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Loads npz
embeddings_npz = np.load('../sample_data/combined_reduced_embeddings.npz')

# filenames csv
filenames = pd.read_csv('../sample_data/combined_filenames.csv', header=None)

rgb = embeddings_npz['pca5'][:, :3]
xy = embeddings_npz['umap2']

plt.scatter(xy[:,0], xy[:,1], c=rgb)
# adds filenames to points
for i, txt in enumerate(filenames[0]):
    plt.annotate(txt, (xy[i,0], xy[i,1]), fontsize=10, color = rgb[i])

plt.show()


rgb = embeddings_npz['pca5'][:, :3]
xy = embeddings_npz['tsne2']

plt.scatter(xy[:,0], xy[:,1], c=rgb)
# adds filenames to points
for i, txt in enumerate(filenames[0]):
    plt.annotate(txt, (xy[i,0], xy[i,1]), fontsize=10, color = rgb[i])

plt.show()