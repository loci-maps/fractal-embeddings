import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
import pickle
import argparse
import os

class TreeNode:
    def __init__(self, left=None, right=None, centroid=None, dr_centroids=None, filenames=None):
        self.left = left
        self.right = right
        self.centroid = centroid
        self.dr_centroids = dr_centroids
        self.filenames = filenames

def load_data(embeddings_file, filenames_file):
    embeddings = np.load(embeddings_file)
    filenames = pd.read_csv(filenames_file, header=None)
    return embeddings, filenames.values.flatten()

def build_tree(Z, embeddings, dr_embeddings, filenames):
    n = len(embeddings)
    nodes = [TreeNode(centroid=embeddings[i], dr_centroids=[dr[i] for dr in dr_embeddings], filenames=[filenames[i]]) for i in range(n)]

    for i, (idx1, idx2, _, _) in enumerate(Z):
        n1 = len(nodes[int(idx1)].filenames)
        n2 = len(nodes[int(idx2)].filenames)

        new_centroid = (n1 * nodes[int(idx1)].centroid + n2 * nodes[int(idx2)].centroid) / (n1 + n2)
        new_dr_centroids = [(n1 * nodes[int(idx1)].dr_centroids[j] + n2 * nodes[int(idx2)].dr_centroids[j]) / (n1 + n2) for j in range(len(dr_embeddings))]

        new_node = TreeNode(left=nodes[int(idx1)], right=nodes[int(idx2)], centroid=new_centroid, dr_centroids=new_dr_centroids, filenames=nodes[int(idx1)].filenames + nodes[int(idx2)].filenames)
        nodes.append(new_node)

    return nodes[-1]

def save_tree(tree, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(tree, f)

def load_tree(input_file):
    with open(input_file, 'rb') as f:
        return pickle.load(f)

def main(args):
    # Load the data
    embeddings, filenames = load_data(args.embeddings_file, args.filenames_file)

    # Extract dimensionality reduced embeddings
    dr_embeddings = [embeddings[key] for key in ['pca5', 'umap5', 'umap2', 'tsne2']]

    # Perform hierarchical clustering
    Z = linkage(embeddings['embeddings'], method='ward')

    # Build the dendrogram tree
    tree = build_tree(Z, embeddings['embeddings'], dr_embeddings, filenames)

    # Save the tree
    save_tree(tree, args.output_tree)

    # Load and test the tree
    loaded_tree = load_tree(args.output_tree)
    print("Loaded tree has", len(loaded_tree.filenames), "filenames")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform hierarchical clustering on embeddings and store the dendrogram as a tree")
    parser.add_argument('-e', '--embeddings_file', type=str, help="Path to the input embeddings npz file")
    parser.add_argument('-f', '--filenames_file', type=str, help="Path to the input filenames csv file")
    parser.add_argument('-o', '--output_tree', type=str, help="Path to the output dendrogram tree file")
    args = parser.parse_args()

    main(args)
