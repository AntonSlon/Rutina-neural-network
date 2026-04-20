import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from src.ruBERT import RuBERT  # Импортируем твой класс


def plot_embeddings():
    rubert = RuBERT()
    vectors = rubert.load_vectors("ruBERT_dataset.npz")

    X = vectors["texts"]
    y = vectors["labels"]

    sample_size = min(3000, len(X))
    indices = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]

    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_sample)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X_pca)

    plt.figure(figsize=(10, 8))

    sns.scatterplot(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        hue=y_sample,
        palette="viridis",
        s=40,
        alpha=0.7
    )

    plt.title("RuBERT")
    plt.xlabel("t-SNE компонента 1")
    plt.ylabel("t-SNE компонента 2")

    plt.legend(title="Классы", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.show()


if __name__ == "__main__":
    plot_embeddings()
