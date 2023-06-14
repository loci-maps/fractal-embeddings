import configparser
# import sembed_text_cohere as emb
from scipy.cluster.hierarchy import linkage, dendrogram # , fcluster, to_tree
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

import colorsys

class CustomDendrogram:

    def __init__(self, data_to_cluster, labels, colors=None):
        self.data_to_cluster = data_to_cluster
        self.labels = labels
        self.colors = colors

    # if t is 1, color2 is returned, if it is 0, color1 is
    def blend_colors(self, color1, color2, weight):
        # Convert RGB colors to HSV
        hsv1 = colorsys.rgb_to_hsv(*color1)
        hsv2 = colorsys.rgb_to_hsv(*color2)

        # Blend the colors based on the weight
        blended_hsv = (
            (1 - weight) * hsv1[0] + weight * hsv2[0],    # Weighted average of the hues
            (1 - weight) * hsv1[1] + weight * hsv2[1],    # Weighted average of the saturations
            (1 - weight) * hsv1[2] + weight * hsv2[2]     # Weighted average of the values/brightness
        )

        # Convert the blended color back to RGB
        blended_rgb = colorsys.hsv_to_rgb(*blended_hsv)

        return blended_rgb

    # https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    #  Creates value of links, such as color, by blending together leaf values
    def create_link_values(self, linkage_matrix, leaf_values,
                        merge_function = lambda cluster1_val, cluster2_val, weight: cluster1_val * (1 - weight) + cluster2_val * weight,
                        weight_function = lambda cluster1_size, cluster2_size: cluster2_size / (cluster1_size + cluster2_size)):
        # Merge function takes val1, val2, and a weight (0-1, 0 preferences val1, 1 preferences val2) and returns a merged value

        num_leaves = len(leaf_values)
        
        new_values = leaf_values.copy()
        for row in linkage_matrix:
            cluster1_id = int(row[0])
            cluster2_id = int(row[1])

            if cluster1_id < num_leaves: # it is a sample/leaaf
                cluster1_size = 1
            else:
                cluster1_size = linkage_matrix[num_leaves - cluster1_id][-1]
            
            if cluster2_id < num_leaves: # it is a sample/leaf
                cluster2_size = 1
            else:
                cluster2_size = linkage_matrix[num_leaves - cluster2_id][-1]

            # If a cluster is huge and is merged with a leaf, the weight should preference the cluster
            # Can try different weighting here
            weight = weight_function(cluster1_size, cluster2_size)

            merged_cluster_value = np.array([merge_function(new_values[cluster1_id], new_values[cluster2_id], weight)])

            new_values = np.concatenate((new_values, merged_cluster_value), axis=0)

        return new_values

    def generate_3d_dendrogram(self):
        linkage_matrix = linkage(self.data_to_cluster, method='ward')
        
        link_labels = self.create_link_values(linkage_matrix, self.data_to_cluster[ddata['leaves']])
        
        if self.colors is not None:
            link_colors = self.create_link_values(linkage_matrix, self.colors, merge_function=self.blend_colors)
            link_colors = [mcolors.to_hex(color) for color in link_colors]
        else:
            # black
            link_colors = ['#000000'] * len(link_labels)
        
        plt.figure(figsize=(4, 5))
        ddata = dendrogram(linkage_matrix, labels=self.labels, link_color_func=lambda id: link_colors[id])

        for i in range(len(linkage_matrix)):
            x = ddata['icoord'][i]
            y = ddata['dcoord'][i]
    
            label = 

        for icoord, dcoord, label in zip(ddata['icoord'], ddata['dcoord'], link_labels[len(self.data_to_cluster):]):
            x = 0.5 * sum(icoord[1:3])
            y = dcoord[1]
            plt.text(x, y, label, va='center', ha='center', size=10)

        # Colors the leaves 
        # ax = plt.gca()
        # xlbls = ax.get_xmajorticklabels()
        # for lbl in xlbls:
        #     lbl.set_color(self.colors[int(lbl.get_text())])

        plt.show()


leaf_values = np.array([[0,1,2,3,4,5]]).T
# leaf_colors = np.array([[1,0,0], [0,1,0], [0,0,1], [1,0,0]])
my_dendrogram = CustomDendrogram(leaf_values, leaf_values)#, leaf_colors)

my_dendrogram.generate_3d_dendrogram()

# def show_example():
#     leaf_values = [0,1,2]
#     leaf_colors = np.array([[1,0,0], [0,1,0], [0,0,1]])

#     linkage_matrix = linkage(leaf_values, method='ward')

#     link_labels = create_link_values(linkage_matrix, leaf_values)
#     link_colors = create_link_values(linkage_matrix, leaf_colors, merge_function=blend_colors)
#     # converts to matplotlib colors
#     link_colors = [mcolors.to_hex(color) for color in link_colors]


# plt.figure(figsize=(4, 5))

# ddata = dendrogram(linkage_matrix, labels=leaf_values, link_color_func=lambda id: link_colors[id])

# for icoord, dcoord, label in zip(ddata['icoord'], ddata['dcoord'], link_labels[len(leaf_values):]):
#     x = 0.5 * sum(icoord[1:3])
#     y = dcoord[1]
#     plt.text(x, y, label, va='center', ha='center', size=10)

# # Colors the leaves 
# ax = plt.gca()
# xlbls = ax.get_xmajorticklabels()
# for lbl in xlbls:
#     lbl.set_color(leaf_colors[int(lbl.get_text())])

# plt.show()


# # from vedo import show, Line, Points, Box

# # # Create Points object for leaf nodes
# # leaf_points = Points(leaf_points_coords, r=5, c="red5")

# # # Add filenames to leaf points
# # for i in range(len(leaf_points_coords)):
# #     points_filenames.append(filenames[i])

# # # Plot the dendrogram edges
# # lines = []
# # for edge in edges:
# #     lines.append(Line(edge.start_point, edge.end_point))

# # # Set the line colors based on the cluster information
# # for i in range(len(lines)):
# #     lines[i].color(link_colors[i])

# # # Create the plot
# # plot = show(leaf_points, lines, axes=1)

# # # Set the plot background color
# # plot.background("lightblue")

# # # Display the plot
# # plot.show()