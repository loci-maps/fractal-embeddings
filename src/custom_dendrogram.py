import numpy as np
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
import matplotlib.pyplot as plt


def augmented_dendrogram(*args, **kwargs):

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        for i, d in zip(ddata['icoord'], ddata['dcoord']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            plt.plot(x, y, 'ro')
            plt.annotate("%.3g" % y, (x, y), xytext=(0, -8),
                         textcoords='offset points',
                         va='top', ha='center')
    return ddata

# Generate a random sample of `n` points in 2-d.
np.random.seed(12312)
n = 100
x = np.random.multivariate_normal([0, 0], np.array([[4.0, 2.5], [2.5, 1.4]]),
                                  size=(n,))

plt.figure(1, figsize=(6, 5))
plt.clf()
plt.scatter(x[:, 0], x[:, 1])
plt.axis('equal')
plt.grid(True)

linkage_matrix = linkage(x, "single")

plt.figure(2, figsize=(10, 4))
plt.clf()

plt.subplot(1, 2, 1)
show_leaf_counts = False
ddata = augmented_dendrogram(linkage_matrix,
               color_threshold=1,
               p=6,
               truncate_mode='lastp',
               show_leaf_counts=show_leaf_counts,
               )
plt.title("show_leaf_counts = %s" % show_leaf_counts)

plt.subplot(1, 2, 2)
show_leaf_counts = True
ddata = augmented_dendrogram(linkage_matrix,
               color_threshold=1,
               p=6,
               truncate_mode='lastp',
               show_leaf_counts=show_leaf_counts,
               )
plt.title("show_leaf_counts = %s" % show_leaf_counts)

plt.show()