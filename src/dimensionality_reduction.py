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


def run_dimensionality_reduction(embeddings, reductions):
    results = {}
    if "pca5" in reductions:
        # PCA
        print("Running PCA 5")
        pca5 = PCA(n_components=5)
        embeddings_pca5 = normalize(pca5.fit_transform(embeddings))
        results["pca5"] = embeddings_pca5

    if "tsne2" in reductions:
        # t-SNE
        print("Running t-SNE 2")
        tsne2 = TSNE(n_components=2)
        embeddings_tsne2 = tsne2.fit_transform(embeddings)
        results["tsne2"] = embeddings_tsne2

    if "umap5" in reductions:
        # UMAP
        print("Running UMAP 5")
        umap_reducer5 = umap.UMAP(n_components=5)
        embeddings_umap5 = normalize(umap_reducer5.fit_transform(embeddings))
        results["umap5"] = embeddings_umap5

    if "umap2" in reductions:
        # UMAP
        print("Running UMAP 2")
        umap_reducer2 = umap.UMAP(n_components=2)
        embeddings_umap2 = umap_reducer2.fit_transform(embeddings)
        results["umap2"] = embeddings_umap2

    return results


def main(args):
    # Load embeddings
    npz_file = np.load(args.embeddings, allow_pickle=True)
    embeddings = npz_file['embeddings']

    # Load filenames if provided
    filenames = None
    if 'filenames' in npz_file:
        filenames = npz_file['filenames']

    if args.filenames is not None:
        filenames = pd.read_csv(args.filenames).values

    reductions = ["pca5", "tsne2", "umap5", "umap2"]
    if args.reductions is not None:
        reductions = args.reductions.split(",")
        for reduction in reductions:
            if reduction not in ["pca5", "tsne2", "umap5", "umap2"]:
                print("Invalid reduction: {}".format(reduction))
                return

    reduced_embeddings = run_dimensionality_reduction(embeddings, reductions)

    if args.output_npz is None:
        print("No output file provided. Skipping saving reduced embeddings.")
        return
    else:
        # Save the reduced embeddings that were requested
        np.savez(args.output_npz, **reduced_embeddings)

    if args.plot is None:
        print("No plot provided. Skipping plotting reduced embeddings.")
        return
    else:
        # Plot the reduced embeddings that were requested in reductions 
        if "pca5" in reductions:
            # Plot PCA
            plot_embeddings(reduced_embeddings["pca5"], reduced_embeddings["pca5"][:, 2:5], "PCA", filenames)

        if "tsne2" in reductions:
            # Plot t-SNE
            plot_embeddings(reduced_embeddings["tsne2"], reduced_embeddings["pca5"][:, 2:5], "t-SNE", filenames)
        
        if "umap5" in reductions:
            # Plot UMAP 5 components
            plot_embeddings(reduced_embeddings["umap5"], reduced_embeddings["umap5"][:, 2:5], "UMAP 5 components", filenames)

        if "umap2" in reductions:
            # Plot UMAP 2 components
            plot_embeddings(reduced_embeddings["umap2"], reduced_embeddings["pca5"][:, 2:5], "UMAP 2 components", filenames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform dimensionality reduction on embeddings and visualize them.")
    parser.add_argument("-e", "--embeddings", type=str, required=True, help="Embeddings NPZ file.")
    parser.add_argument("-f", "--filenames", type=str, required=False, help="Optional CSV file containing filenames.")
    parser.add_argument("-o", "--output_npz", type=str, required=False, help="Output NPZ file for reduced embeddings.")
    parser.add_argument("-p", "--plot", action="store_true", required=False, help="Plot reduced embeddings.")
    parser.add_argument("-r", "--reductions", type=str, required=False, help="Comma-separated list of reductions to perform. Defaults to all.")

    args = parser.parse_args()
    main(args)
