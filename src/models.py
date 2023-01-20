from collections import defaultdict

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.manifold import TSNE, SpectralEmbedding, Isomap, MDS
import scipy.spatial as sp
import time
from pathlib import Path

# from fcmeans import FCM
# my_model = FCM(n_clusters=2) # we use two cluster as an example
# my_model.fit(X) ## X, numpy array. rows:samples columns:features

random_state = 0
np.random.seed(random_state)
sns.set_theme()
BASE_PATH = f'./plots/{time.strftime("%m_%d_%H_%M_%S", time.gmtime())}'


Path(BASE_PATH).mkdir(parents=True, exist_ok=True)


def plot_clusters(labels, X, figname: str):
    plt.figure(figsize=(10, 5))
    X_embeddings = TSNE(n_components=2).fit_transform(X)
    plt.scatter(X_embeddings[:, 0], X_embeddings[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    plt.title("Cluster Visualization using TSNE")
    plt.savefig(f"{BASE_PATH}/{figname}")
    plt.legend()
    # plt.show()
    plt.clf()


def cluster_with_nan_by_cosine_sim(X, model):
    clean_X = X[~np.isnan(X).any(axis=1)]
    good_labels = model.fit_predict(clean_X)
    unique_labels = np.unique(good_labels[good_labels != -1])
    k = len(unique_labels)
    nan_X = X[np.isnan(X).any(axis=1)]
    nan_columns = np.unique(np.argwhere(np.isnan(nan_X))[:, 1])
    consine = 1 - sp.distance.cdist(np.delete(nan_X, nan_columns, 1), np.delete(clean_X, nan_columns, 1), 'cosine')
    nan_cosine_per_label = np.zeros((k, nan_X.shape[0]))
    for label_id in unique_labels:
        nan_cosine_per_label[label_id] = consine[:, good_labels == label_id].mean(axis=1)
    labels = np.zeros(X.shape[0])
    labels[~np.isnan(X).any(axis=1)] = good_labels.copy()
    if len(nan_cosine_per_label) != 0:
        labels[np.isnan(X).any(axis=1)] = nan_cosine_per_label.argmax(axis=0)
    else:
        labels[np.isnan(X).any(axis=1)] = 0
    return labels, good_labels


def _cluster_data(X: np.ndarray, model_obj, model_name: str, min_k: int, max_k: int):
    scores = []
    possible_k = list(range(min_k, max_k))
    clean_X = X[~np.isnan(X).any(axis=1)]
    for k in tqdm(possible_k):
        curr_model = model_obj(k)
        labels, good_labels = cluster_with_nan_by_cosine_sim(X, curr_model)
        scores.append(metrics.silhouette_score(clean_X, good_labels))
    plt.plot(possible_k, scores)
    plt.title(f"silhouette_score over K - {model_name}")
    plt.savefig(f"{BASE_PATH}/{model_name}_scores.png")
    plt.xlabel("K over time")
    plt.ylabel("Silhouette Score")
    plt.clf()
    best_k = possible_k[np.argmax(scores)]
    best_model = model_obj(best_k)
    best_clusters, good_labels = cluster_with_nan_by_cosine_sim(X, best_model)
    plot_clusters(good_labels, clean_X, f"{model_name} - {best_k}")
    print(f"{model_name} - {best_k}, score: {np.max(scores)}")
    return best_clusters, np.max(scores)


def cluster_data(X: np.ndarray, min_k=3, max_k=40):
    possible_models = [
        (lambda k: KMeans(k, n_init="auto", random_state=random_state), "Kmean"),
        (lambda k: GaussianMixture(k, max_iter=3000), "GaussianMixture"),
        (lambda k: AgglomerativeClustering(k), "AgglomerativeClustering"),
        (lambda k: Birch(n_clusters=k), "Birch")
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
    plt.savefig(f"{BASE_PATH}/reduce_dimension.png")
    plt.clf()


def knn_imputer(clusters: np.ndarray, X: np.ndarray, y_true: np.ndarray, k=15):
    scores = {}
    nan_column = np.argwhere(np.any(np.isnan(X), axis=0))[0][0]
    missing_y = X[:, nan_column]
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


def eval_methods_over_k(X, y_true):
    best_clusters = cluster_data(X)
    all_scores = defaultdict(list)
    k_range = list(range(15, 60))

    for k in k_range:
        scores = knn_imputer(best_clusters, X, y_true.copy(), k)
        for key, value in scores.items():
            all_scores[key].append(value)
    for method_name, scores in all_scores.items():
        plt.plot(k_range, scores, label=method_name)
    plt.legend()
    plt.xlabel("Number Of Neighborhood")
    plt.ylabel("Mean Squared Error")
    plt.title("Method scores over K")
    plt.savefig(f"{BASE_PATH}/methods_over_k")
    # plt.show()


def density_based_knn(X: np.ndarray, y_true):
    min_samples = 2
    clean_X = X[~np.isnan(X).any(axis=1)]
    # range_eps = np.linspace(0.1, 10, 20)
    # scores = []
    # for eps in range_eps:
    #     model = DBSCAN(eps=eps, min_samples=min_samples)
    #     labels, good_labels = cluster_with_nan_by_cosine_sim(X, model)
    #     noisy_data_count = len(good_labels[good_labels == -1])
    #     if noisy_data_count == good_labels.shape[0]:
    #         scores.append(0)
    #         continue
    #     print(f"noisy data count with eps={eps}: {noisy_data_count}")
    #     score = metrics.silhouette_score(clean_X, good_labels)
    #     scores.append(score)
    # plt.plot(range_eps, scores)
    # plt.savefig(f"{BASE_PATH}/silhouette_score_over_eps.png")
    best_eps = 3#range_eps[np.argmax(scores)]
    best_model = DBSCAN(
        eps=best_eps,
        metric="euclidean",
        min_samples=min_samples,
        n_jobs=-1
    )
    all_labels, labels = cluster_with_nan_by_cosine_sim(X, best_model)
    print(f"noisy data count: {len(labels[labels == -1])}")
    plot_clusters(labels, clean_X, "DBSCAN")
    clusters_ids, counts = np.unique(labels, return_counts=True)
    densities = []
    for label, count in zip(clusters_ids, counts):
        if label == -1:
            continue
        points = clean_X[labels == label]
        volume = (points.max(axis=0) - points.min(axis=0) + 0.0001).prod()
        density = count / volume
        densities.append(density)
    densities = np.array(densities)
    plt.bar(list(range(len(densities))),densities)
    plt.xlabel("cluster id")
    plt.ylabel("weight of k")
    plt.savefig(f"{BASE_PATH}/k_weights_by_cluster_id.png")
    plt.clf()
    scores = defaultdict(list)
    k_range = list(range(10, 60))
    for k in tqdm(k_range):
        min_k = 0.1 * k
        max_k = k
        curr_densities = (((densities - densities.min()) * (max_k - min_k))/ (densities.max() - densities.min()) + min_k).astype(int) + 1
        print(curr_densities, k)
        nan_column = np.argwhere(np.any(np.isnan(X), axis=0))[0][0]
        missing_y = X[:, nan_column]
        for imputer, imputer_name in [(KNNImputer(n_neighbors=k, weights="uniform"), "KNNImputer")]:
            y_pred = imputer.fit_transform(np.hstack((X, missing_y.reshape(-1, 1))))[:, -1]
            scores[imputer_name].append(metrics.mean_squared_error(y_true, y_pred))
        y_pred = missing_y.copy()
        for cluster_id in np.unique(all_labels):
            cluster_k = curr_densities[int(cluster_id)] if cluster_id != -1 else 1
            imputer = KNNImputer(n_neighbors=cluster_k, weights="uniform")
            curr_X_cluster = X[all_labels == cluster_id]
            curr_y_cluster = missing_y[all_labels == cluster_id]
            curr_data = np.hstack((curr_X_cluster, curr_y_cluster.reshape(-1, 1)))
            y_pred[all_labels == cluster_id] = imputer.fit_transform(curr_data)[:, -1]
        scores["CBKnn"].append(metrics.mean_squared_error(y_true, y_pred))
    for method_name, method_scores in scores.items():
        plt.plot(k_range, method_scores, label=method_name)
    plt.legend()
    plt.title("Method scores over K")
    plt.xlabel("Number Of Neighborhood")
    plt.ylabel("Mean Squared Error")
    plt.savefig(f"{BASE_PATH}/methods_over_k_density_based.png")
    return scores
