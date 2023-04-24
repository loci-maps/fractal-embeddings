import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import argparse


def normalize(embeddings):
    return (embeddings - embeddings.min()) / (embeddings.max() - embeddings.min())


def plot_embeddings(embeddings, colors, title, filenames=None):
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=colors)

    if filenames is not None:
        for i, txt in enumerate(filenames):
            plt.annotate(txt, (embeddings[i, 0], embeddings[i, 1]), fontsize=8, color = colors[i])

    plt.title(title)
    plt.show()


def run_dimensionality_reduction(embeddings):
    # PCA
    pca5 = PCA(n_components=5)
    embeddings_pca5 = normalize(pca5.fit_transform(embeddings))

    # t-SNE
    tsne2 = TSNE(n_components=2)
    embeddings_tsne2 = tsne2.fit_transform(embeddings)

    # UMAP
    umap_reducer5 = umap.UMAP(n_components=5)
    embeddings_umap5 = normalize(umap_reducer5.fit_transform(embeddings))

    umap_reducer2 = umap.UMAP(n_components=2)
    embeddings_umap2 = umap_reducer2.fit_transform(embeddings)

    return {
        "pca5": embeddings_pca5,
        "tsne2": embeddings_tsne2,
        "umap5": embeddings_umap5,
        "umap2": embeddings_umap2
    }


def main():
    parser = argparse.ArgumentParser(description="Perform dimensionality reduction on embeddings and visualize them.")
    parser.add_argument("-e", "--embeddings", type=str, required=True, help="Embeddings NPZ file.")
    parser.add_argument("-f", "--filenames", type=str, help="Optional CSV file containing filenames.")
    parser.add_argument("-o", "--output_npz", type=str, default="reduced_embeddings.npz", help="Output NPZ file for reduced embeddings.")

    args = parser.parse_args()

    # Load embeddings
    embeddings = np.load(args.embeddings)['embeddings']

    # Load filenames if provided
    # filenames is a csv with one coumn labeled "filename" 
    filenames = None
    if args.filenames is not None:
        filenames = pd.read_csv(args.filenames)['filename'].values

    reduced_embeddings = run_dimensionality_reduction(embeddings)

    # Save reduced embeddings
    np.savez(args.output_npz,
             pca5=reduced_embeddings["pca5"],
             tsne2=reduced_embeddings["tsne2"],
             umap5=reduced_embeddings["umap5"],
             umap2=reduced_embeddings["umap2"])

    # Plot PCA
    plot_embeddings(reduced_embeddings["pca5"], reduced_embeddings["pca5"][:, 2:5], "PCA", filenames)

    # Plot t-SNE
    plot_embeddings(reduced_embeddings["tsne2"], reduced_embeddings["pca5"][:, 2:5], "t-SNE", filenames)

    # Plot UMAP 5 components
    plot_embeddings(reduced_embeddings["umap5"], reduced_embeddings["umap5"][:, 2:5], "UMAP 5 components", filenames)

    # Plot UMAP 2 components
    plot_embeddings(reduced_embeddings["umap2"], reduced_embeddings["pca5"][:, 2:5], "UMAP 2 components", filenames)

if __name__ == "__main__":
    main()
