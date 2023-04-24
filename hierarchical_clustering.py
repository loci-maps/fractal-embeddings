import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
import queue

class MyNode:
    def __init__(self):
        self.files = []
        self.left = None
        self.right = None

    def get_children(self):
        return len(self.files)

def read_embeddings(embedding_file):
    df = pd.read_csv(embedding_file)
    vectors = df.iloc[:, :-1].values.tolist()
    filenames = df['filename'].values.tolist()
    vector_to_file = {tuple(vectors[i]): filenames[i] for i in range(len(filenames))}
    return vectors, vector_to_file

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

def build_tree(embedding_file = "./sample_embeddings.csv"):
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

import numpy as np

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
            
def plot_tree(tree, levels):
    G = nx.DiGraph()
    labels = {}
    pos = {}
    
    def add_node(node, level=0, x=0, num_nodes=0):
        nonlocal pos
        nonlocal labels

        node_id = num_nodes
        G.add_node(node_id)
        labels[node_id] = f"Level: {level}<br>Files: {node.get_children()}"
        pos[node_id] = (x, -level)
        num_nodes += 1

        if node.left is not None:
            num_nodes = add_node(node.left, level + 1, x - 2**(-level-1), num_nodes)
            G.add_edge(node_id, num_nodes - 1)
        if node.right is not None:
            num_nodes = add_node(node.right, level + 1, x + 2**(-level-1), num_nodes)
            G.add_edge(node_id, num_nodes - 1)

        return num_nodes

    add_node(tree)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[labels[node] for node in G.nodes()],
        textposition='top center',
        hoverinfo='none',
        marker=dict(
            showscale=False,
            colorscale='Viridis',
            reversescale=True,
            color=[],
            size=10,
            line_width=2
        )
    )

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    layout = go.Layout(
        title=dict(text='Hierarchical Clustering Tree', x=0.5, y=0.95),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        width=1000,
        height=1000,
        plot_bgcolor='white',
        hovermode='closest'
    )

    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
    fig.show()

def main():
    tree = build_tree()
    levels, pca_results = get_all_centroids(tree, df)
    stats = level_statistics(tree, levels)
    save_statistics_to_file(stats)
    # plot_tree(tree, levels)



if __name__ == "__main__":
    main()
