from src.preprocess import load_titanic, load_data
from src.models import cluster_data, reduce_dimension, knn_imputer, BASE_PATH
import matplotlib.pyplot as plt
from collections import defaultdict

if __name__ == '__main__':
    # df, missing_values = load_data()
    df, y_true = load_titanic(nan_percentage=0.1)
    X = df.values
    # plt.scatter(X[:,0], X[:, 1])
    # plt.show()
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
    plt.title("Method scores over K")
    plt.savefig(f"{BASE_PATH}/methods_over_k")
    plt.show()
