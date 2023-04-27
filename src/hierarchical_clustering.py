import json
import numpy as np
import pandas as pd
import argparse
import queue
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree

class MyNode:
    def __init__(self):
        self.files = []
        self.left = None
        self.right = None

    def get_children(self):
        return len(self.files)

def read_embeddings(embedding_file, filenames_file):
    embeddings_npz = np.load(embedding_file)
    embeddings = embeddings_npz['embeddings']
    filenames = pd.read_csv(filenames_file, header=None).squeeze().tolist()
    return embeddings, filenames

def traverse_CF_tree2(tree, vector_names):
    if tree.is_leaf():
        new_node = MyNode()
        new_node.files.append(vector_names[tree.get_id()])
        return new_node
    else:
        new_node = MyNode()
        if tree.get_left() is not None:
            left = traverse_CF_tree2(tree.get_left(), vector_names)
            new_node.files += left.files
            new_node.left = left
        if tree.get_right() is not None:
            right = traverse_CF_tree2(tree.get_right(), vector_names)
            new_node.files += right.files
            new_node.right = right
        return new_node
 

def create_tree(vectors, vector_names):
    linked = linkage(vectors, 'single')
    rootnode = to_tree(linked)
    return traverse_CF_tree2(rootnode, vector_names)

def build_tree(embedding_file = "../sample_embeddings.csv"):
    vectors, vector_to_file = read_embeddings(embedding_file)
    vectors_names = [vector_to_file[tuple(x)] for x in vectors]
    tree = create_tree(vectors, vectors_names)
    return tree

def get_centroid(files, df):
    centroid = df.loc[df['filename'].isin(files)].mean(numeric_only=True)
    return centroid

def get_all_centroids(tree, df):
    q = queue.Queue()
    q.put(tree)
    q.put("M")
    levels = []
    centroids = []
    pca_results = []

    while not q.empty():
        val = q.get()
        if val == "M":
            levels.append(centroids)

            # Perform PCA on the current level
            if centroids:
                level_data = pd.concat(centroids, axis=1).T
                pca = PCA(n_components=2)
                reduced_data = pca.fit_transform(level_data)
                pca_results.append({"reduced_data": reduced_data,
                                    "explained_variance_ratio": pca.explained_variance_ratio_})

            centroids = []
            if not q.empty():
                q.put("M")
        else:
            centroid = get_centroid(val.files, df)
            centroids.append(centroid)
            if val.left is not None:
                q.put(val.left)
            if val.right is not None:
                q.put(val.right)

    return levels, pca_results



def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def level_statistics(tree, levels):
    stats = []
    level_nodes = get_level_nodes(tree)
    for level, centroids in enumerate(levels):
        level_stats = {}
        level_stats["level"] = level
        level_stats["nodes_per_centroid"] = [node.get_children() for node in level_nodes[level]]
        # Calculate centroid distances
        distances = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                distances.append(euclidean_distance(centroids[i], centroids[j]))
        # Adds mean distance to level stats if there are any distances
        if len(distances) > 0:
            level_stats["mean_distance"] = np.mean(distances)
        else: 
            level_stats["mean_distance"] = 0
        stats.append(level_stats)
    return stats

def get_level_nodes(tree):
    q = queue.Queue()
    q.put(tree)
    q.put("M")
    levels = []
    nodes = []
    while not q.empty():
        val = q.get()
        if val == "M":
            levels.append(nodes)
            nodes = []
            if not q.empty():
                q.put("M")
        else:
            nodes.append(val)
            if val.left is not None:
                q.put(val.left)
            if val.right is not None:
                q.put(val.right)
    return levels


def save_statistics_to_file(stats, filename="level_statistics.txt"):
    with open(filename, "w") as f:
        for level_stats in stats:
            f.write(f"Nodes per centroid: {level_stats['nodes_per_centroid']}\n")
            f.write(f"Mean distance: {level_stats['mean_distance']}\n")
            # f.write("Centroid distances:\n")
            # for dist in level_stats["centroid_distances"]:
            #     f.write(f"{dist}\n")
            f.write("\n")

def main(args):
    embeddings, filenames = read_embeddings(args.embeddings, args.filenames)
    df = pd.DataFrame(embeddings)
    df['filename'] = filenames

    tree = build_tree(embeddings, filenames)
    levels, pca_results = get_all_centroids(tree, df)
    stats = level_statistics(tree, levels)
    save_statistics_to_file(stats, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical clustering analysis of embeddings.")
    parser.add_argument("embeddings", help="Path to the embeddings NPZ file.")
    parser.add_argument("filenames", help="Path to the filenames CSV file.")
    parser.add_argument("-o", "--output", help="Path to save the output statistics file.", default="level_statistics.txt")

    args = parser.parse_args()
    main(args)