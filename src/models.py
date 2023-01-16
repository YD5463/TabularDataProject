import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE, SpectralEmbedding, Isomap, MDS
import scipy.spatial as sp

random_state = 0
np.random.seed(random_state)
sns.set_theme()


def plot_clusters(labels, X, figname: str):
    plt.figure(figsize=(10, 5))
    X_embeddings = TSNE(n_components=2).fit_transform(X)
    plt.scatter(X_embeddings[:, 0], X_embeddings[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    plt.savefig(figname)
    plt.clf()


def get_clusters(model_obj, k, clean_X, X):
    model = model_obj(k)
    good_labels = model.fit_predict(clean_X)
    nan_X = X[np.isnan(X).any(axis=1)]
    nan_columns = np.unique(np.argwhere(np.isnan(nan_X))[:, 1])
    consine = 1 - sp.distance.cdist(np.delete(nan_X, nan_columns, 1), np.delete(clean_X, nan_columns, 1), 'cosine')
    nan_cosine_per_label = np.zeros((k, nan_X.shape[0]))
    for label_id in np.unique(good_labels):
        nan_cosine_per_label[label_id] = consine[:, good_labels == label_id].mean(axis=1)
    labels = np.zeros(X.shape[0])
    labels[~np.isnan(X).any(axis=1)] = good_labels.copy()
    labels[np.isnan(X).any(axis=1)] = nan_cosine_per_label.argmax(axis=0)
    return labels, good_labels


def _cluster_data(X: np.ndarray, model_obj, model_name: str, min_k: int, max_k: int):
    scores = []
    possible_k = list(range(min_k, max_k))
    clean_X = X[~np.isnan(X).any(axis=1)]
    for k in tqdm(possible_k):
        labels, good_labels = get_clusters(model_obj, k, clean_X, X)
        scores.append(metrics.silhouette_score(clean_X, good_labels))

    plt.plot(possible_k, scores)
    plt.title(f"silhouette_score over K - {model_name}")
    plt.savefig(f"{model_name}_scores.png")
    plt.clf()
    best_k = possible_k[np.argmax(scores)]
    best_clusters, good_labels = get_clusters(model_obj, best_k, clean_X, X)
    plot_clusters(good_labels, clean_X, f"{model_name} - {best_k}")
    print(f"{model_name} - {best_k}, score: {np.max(scores)}")
    return best_clusters, np.max(scores)


def cluster_data(X: np.ndarray, min_k=10, max_k=20):
    possible_models = [
        # (lambda k: KMeans(k, n_init="auto", random_state=random_state), "Kmean"),
        # (lambda k: GaussianMixture(k, max_iter=3000), "GaussianMixture"),
        (lambda k: AgglomerativeClustering(k), "AgglomerativeClustering"),
        # (lambda k: DBSCAN(), "DBSCAN"),
        # (lambda k: Birch(n_clusters=k), "Birch")
    ]
    best_model = None
    best_score = 0
    for model, model_name in possible_models:
        model, score = _cluster_data(X, model, model_name, min_k, max_k)
        if score > best_score:
            best_score = score
            best_model = model
    return best_model


def reduce_dimension(X):
    models = [TSNE(n_components=2), Isomap(n_components=2), MDS(n_components=2), SpectralEmbedding(n_components=2)]
    fig, axs = plt.subplots(nrows=len(models), ncols=1, figsize=(8, 8))
    fig.tight_layout()
    for i, model in tqdm(enumerate(models)):
        X_embedded_tsne = model.fit_transform(X)
        axs[i].scatter(X_embedded_tsne[:, 0], X_embedded_tsne[:, 1], s=40, cmap='viridis')
        axs[i].set_title(f"{model} dimensionality reduction")
    plt.show()
    plt.savefig("reduce_dimension.png")
    plt.clf()


def knn_imputer(clusters: np.ndarray, X: np.ndarray, y_true: np.ndarray, k=15):
    scores = {}
    missing_y = X[:, 2]
    for imputer, imputer_name in [(KNNImputer(n_neighbors=k, weights="uniform"), "KNNImputer")]:
        y_pred = imputer.fit_transform(np.hstack((X, missing_y.reshape(-1, 1))))[:, -1]
        scores[imputer_name] = metrics.mean_squared_error(y_true, y_pred)
    y_pred = missing_y.copy()
    for cluster_id in tqdm(np.unique(clusters)):
        imputer = KNNImputer(n_neighbors=k, weights="uniform")
        curr_X_cluster = X[clusters == cluster_id]
        curr_y_cluster = missing_y[clusters == cluster_id]
        curr_data = np.hstack((curr_X_cluster, curr_y_cluster.reshape(-1, 1)))
        y_pred[clusters == cluster_id] = imputer.fit_transform(curr_data)[:, -1]
    scores["CBKnn"] = metrics.mean_squared_error(y_true, y_pred)
    return scores
